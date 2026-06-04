// SIE-driven fraud risk scoring for a Stripe Link checkout.
//
// Pipeline:
//   1. Build a natural-language summary of the order context (cart + customer).
//   2. Extract entities with GLiNER (used to surface signals to the UI; not
//      directly fed into the score).
//   3. Encode the summary with a small dense encoder (MiniLM).
//   4. Cosine-rank against a pre-encoded fraud-pattern corpus.
//   5. Rerank the top K with a cross-encoder for a sharper final score.
//   6. Map the top reranker score to a low/medium/high band.

import fs from "node:fs";
import path from "node:path";
import { SIEClient } from "@superlinked/sie-sdk";
import { config, riskBand } from "./config.js";
import type { RiskEvent } from "./events.js";
import type { CartItem, Customer, FraudPattern, IndexRecord, SampleOrder } from "./types.js";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");

function loadJson<T>(rel: string): T {
  return JSON.parse(fs.readFileSync(path.join(ROOT, rel), "utf8")) as T;
}

export function buildOrderSummary(
  cart: CartItem[],
  customer: Customer,
  context?: string,
): string {
  const total = cart.reduce((s, i) => s + i.qty * i.unit_price_usd, 0);
  const skus = cart.map((i) => `${i.qty}x ${i.name} ($${i.unit_price_usd})`).join(", ");
  const linkStatus = customer.link_returning
    ? "returning Link customer"
    : "new Link signup (no prior history)";
  const geo =
    customer.billing_country === customer.shipping_country &&
    customer.shipping_country === customer.ip_country
      ? `all signals from ${customer.billing_country}`
      : `billing=${customer.billing_country} shipping=${customer.shipping_country} ip=${customer.ip_country}`;
  return [
    `Order total $${total} for ${customer.name} (${customer.email}).`,
    `Cart: ${skus}.`,
    `Shipping to: ${customer.shipping_address}.`,
    `Customer: account age ${customer.account_age_days} days, ${customer.prior_orders} prior orders, ${linkStatus}.`,
    `Geography: ${geo}.`,
    context ? `Context: ${context}` : "",
  ]
    .filter(Boolean)
    .join(" ");
}

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = b[i]!;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

export interface RiskInput {
  cart: CartItem[];
  customer: Customer;
  context?: string;
}

export interface RiskRunDeps {
  client: SIEClient;
  emit: (event: RiskEvent) => void;
}

export async function runRisk(input: RiskInput, deps: RiskRunDeps): Promise<void> {
  const { client, emit } = deps;
  const start = Date.now();

  emit({
    type: "models",
    data: {
      encoder: config.models.encoder,
      reranker: config.models.reranker,
      extractor: config.models.extractor,
    },
  });

  const summary = buildOrderSummary(input.cart, input.customer, input.context);

  // 1. Extract entities (surface signals to the UI; not fed into the score).
  emit({ type: "extracting" });
  const tEx = Date.now();
  const extractOut = await client.extract(
    config.models.extractor,
    { text: summary },
    { labels: [...config.extractLabels] },
  );
  const entities = (extractOut.entities ?? []).map((e) => ({
    label: e.label ?? "",
    text: e.text ?? "",
    score: Number(e.score ?? 0),
  }));
  emit({ type: "extracted", data: { entities, ms: Date.now() - tEx } });

  // 2. Encode the summary.
  emit({ type: "encoding" });
  const tEn = Date.now();
  const enc = await client.encode(config.models.encoder, { text: summary });
  const queryVec = enc.dense;
  if (!queryVec) throw new Error("encoder returned no dense vector for the order");
  emit({
    type: "encoded",
    data: { dim: queryVec.length, ms: Date.now() - tEn },
  });

  // 3. Cosine-rank the corpus to pick the top-K candidates.
  const patterns: FraudPattern[] = loadJson(config.paths.fraudPatterns);
  const index: IndexRecord[] = loadJson(config.paths.index);
  const byId = new Map(patterns.map((p) => [p.id, p]));
  const cosineRanked = index
    .map((rec) => ({ id: rec.id, score: cosine(Array.from(queryVec), rec.vector) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, config.risk.topKRerank);
  const cosineById = new Map(cosineRanked.map((c) => [c.id, c.score]));

  // 4. Rerank with the cross-encoder. The reranker output orders the top-K
  // for display; the band signal stays on cosine because the cosine score
  // has a usable 0-1 spread, whereas raw cross-encoder logits cluster near
  // zero for this kind of "order-vs-fraud-pattern" similarity.
  emit({ type: "scoring" });
  const tSc = Date.now();
  const docs = cosineRanked.map((c) => {
    const p = byId.get(c.id);
    return { id: c.id, text: p ? `${p.label}. ${p.summary}` : "" };
  });
  const score = await client.score(config.models.reranker, { text: summary }, docs);
  const rerankById = new Map<string, number>();
  for (const s of score.scores ?? []) rerankById.set(s.itemId, s.score);
  const hits = cosineRanked.map((c) => {
    const p = byId.get(c.id)!;
    return {
      id: p.id,
      label: p.label,
      summary: p.summary,
      outcome: p.outcome,
      loss_usd: p.loss_usd,
      // Reranker score is the display-time order; cosine drives the band.
      score: rerankById.get(c.id) ?? 0,
    };
  });
  hits.sort((a, b) => b.score - a.score);
  const topScore = cosineById.get(cosineRanked[0]?.id ?? "") ?? 0;
  const band = riskBand(topScore);
  emit({
    type: "scored",
    data: { hits, topScore, band, ms: Date.now() - tSc },
  });

  emit({ type: "done", data: { totalMs: Date.now() - start } });
}

export function loadSampleOrders(): SampleOrder[] {
  return loadJson<SampleOrder[]>(config.paths.sampleOrders);
}
