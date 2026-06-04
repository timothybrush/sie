import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { SIEClient } from "@superlinked/sie-sdk";
import Stripe from "stripe";
import { config } from "../src/config.js";
import type { RiskEvent } from "../src/events.js";
import { loadSampleOrders, runRisk } from "../src/risk.js";
import type { CartItem, Customer } from "../src/types.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");
const PUBLIC_DIR = path.join(ROOT, "web", "public");
const INDEX_PATH = path.join(ROOT, config.paths.index);

const STRIPE_ENABLED =
  config.stripe.publishableKey.startsWith("pk_") && config.stripe.secretKey.startsWith("sk_");

const stripe: Stripe | null = STRIPE_ENABLED
  ? new Stripe(config.stripe.secretKey, { apiVersion: config.stripe.apiVersion })
  : null;

function send(
  res: http.ServerResponse,
  status: number,
  body: string | Buffer,
  contentType = "text/plain",
): void {
  res.writeHead(status, { "content-type": contentType });
  res.end(body);
  return;
}

function serveFile(res: http.ServerResponse, file: string): void {
  if (!fs.existsSync(file)) return send(res, 404, "not found");
  const ext = path.extname(file).toLowerCase();
  const ct =
    {
      ".html": "text/html",
      ".css": "text/css",
      ".js": "text/javascript",
      ".json": "application/json",
      ".png": "image/png",
      ".svg": "image/svg+xml",
    }[ext] ?? "application/octet-stream";
  res.writeHead(200, { "content-type": ct });
  fs.createReadStream(file).pipe(res);
}

function setupSse(res: http.ServerResponse) {
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
  return (event: RiskEvent) => {
    res.write(`event: ${event.type}\n`);
    const payload = "data" in event ? event.data : null;
    res.write(`data: ${JSON.stringify(payload)}\n\n`);
  };
}

async function ensureIndex(): Promise<void> {
  if (fs.existsSync(INDEX_PATH)) return;
  console.log("building fraud-pattern index...");
  const result = spawnSync(
    process.execPath,
    [
      path.join(ROOT, "node_modules/.bin/tsx"),
      path.join(ROOT, "src/index-build.ts"),
    ],
    { cwd: ROOT, encoding: "utf8", stdio: "inherit" },
  );
  if (result.status !== 0) throw new Error("index-build failed");
}

async function checkSie(): Promise<boolean> {
  try {
    const r = await fetch(`${config.sieUrl}/healthz`, {
      signal: AbortSignal.timeout(2000),
    });
    return r.ok;
  } catch {
    return false;
  }
}

async function fetchRegistered(): Promise<{ ok: boolean; names: string[] }> {
  try {
    const r = await fetch(`${config.sieUrl}/v1/models`, { signal: AbortSignal.timeout(3000) });
    if (!r.ok) return { ok: false, names: [] };
    const json = (await r.json()) as { models?: { name: string }[] };
    return { ok: true, names: (json.models ?? []).map((m) => m.name) };
  } catch {
    return { ok: false, names: [] };
  }
}

async function readBody(req: http.IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString("utf8");
}

async function handleRun(
  _req: http.IncomingMessage,
  res: http.ServerResponse,
  orderId: string,
): Promise<void> {
  const push = setupSse(res);
  const samples = loadSampleOrders();
  const order = samples.find((s) => s.id === orderId);
  if (!order) {
    push({ type: "error", data: { stage: "lookup", message: `unknown order id: ${orderId}` } });
    res.end();
    return;
  }

  const sieOk = await checkSie();
  if (!sieOk) {
    push({
      type: "error",
      data: { stage: "sie", message: `SIE not reachable at ${config.sieUrl}` },
    });
    res.end();
    return;
  }

  try {
    await ensureIndex();
  } catch (e) {
    push({
      type: "error",
      data: { stage: "index", message: e instanceof Error ? e.message : String(e) },
    });
    res.end();
    return;
  }

  const client = new SIEClient(config.sieUrl, {
    apiKey: config.sieApiKey,
    timeout: 600_000,
    waitForCapacity: true,
    provisionTimeout: 900_000,
  });

  try {
    await runRisk(
      { cart: order.cart, customer: order.customer, context: order.description },
      { client, emit: push },
    );
  } catch (e) {
    push({
      type: "error",
      data: { stage: "pipeline", message: e instanceof Error ? e.message : String(e) },
    });
  } finally {
    res.end();
  }
}

async function handlePaymentIntent(
  req: http.IncomingMessage,
  res: http.ServerResponse,
): Promise<void> {
  if (!stripe) {
    return send(
      res,
      200,
      JSON.stringify({ mode: "mock", reason: "STRIPE_*_KEY not set" }),
      "application/json",
    );
  }
  type Body = { cart: CartItem[]; customer: Customer; riskBand?: string };
  let body: Body;
  try {
    body = JSON.parse(await readBody(req));
  } catch {
    return send(res, 400, "invalid json");
  }
  const amount = body.cart.reduce((s, i) => s + i.qty * i.unit_price_usd, 0) * 100;
  const intent = await stripe.paymentIntents.create({
    amount: Math.round(amount),
    currency: "usd",
    automatic_payment_methods: { enabled: true },
    metadata: {
      sie_risk_band: body.riskBand ?? "unknown",
      customer_email: body.customer.email,
    },
    receipt_email: body.customer.email,
  });
  return send(
    res,
    200,
    JSON.stringify({
      mode: "live",
      clientSecret: intent.client_secret,
      publishableKey: config.stripe.publishableKey,
    }),
    "application/json",
  );
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
  const p = url.pathname;

  if (p === "/" || p === "/index.html") return serveFile(res, path.join(PUBLIC_DIR, "index.html"));
  if (p.startsWith("/static/")) return serveFile(res, path.join(PUBLIC_DIR, p.slice("/static/".length)));

  if (p === "/api/health") {
    const { ok, names } = await fetchRegistered();
    return send(
      res,
      200,
      JSON.stringify({
        sie: ok,
        sieUrl: config.sieUrl,
        registeredModels: names.length,
        registered: names,
        stripe: STRIPE_ENABLED ? "live" : "mock",
        publishableKey: STRIPE_ENABLED ? config.stripe.publishableKey : null,
      }),
      "application/json",
    );
  }

  if (p === "/api/samples") {
    return send(res, 200, JSON.stringify(loadSampleOrders()), "application/json");
  }

  if (p === "/api/run") {
    const id = url.searchParams.get("id");
    if (!id) return send(res, 400, "missing id");
    return handleRun(req, res, id);
  }

  if (p === "/api/payment-intent" && req.method === "POST") {
    return handlePaymentIntent(req, res);
  }

  return send(res, 404, "not found");
});

server.listen(config.port, () => {
  const url = `http://localhost:${config.port}`;
  console.log(`stripe-link-fraud ui: ${url}`);
  console.log(`stripe mode: ${STRIPE_ENABLED ? "live (test keys detected)" : "mock (no keys set)"}`);
  if (process.env.OPEN_BROWSER !== "0") {
    const opener =
      process.platform === "darwin"
        ? "open"
        : process.platform === "win32"
          ? "start"
          : "xdg-open";
    spawnSync(opener, [url], { stdio: "ignore" });
  }
});
