// One-time index build: encode the fraud-pattern corpus into dense vectors
// and dump to data/risk_index.json. Re-run with `npm run index` after
// editing data/fraud_patterns.json.
import fs from "node:fs";
import path from "node:path";
import { SIEClient } from "@superlinked/sie-sdk";
import { config } from "./config.js";
import type { FraudPattern, IndexRecord } from "./types.js";

async function main(): Promise<void> {
  const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
  const inPath = path.join(root, config.paths.fraudPatterns);
  const outPath = path.join(root, config.paths.index);

  const patterns: FraudPattern[] = JSON.parse(fs.readFileSync(inPath, "utf8"));
  console.log(`encoding ${patterns.length} fraud patterns`);

  const client = new SIEClient(config.sieUrl, {
    apiKey: config.sieApiKey,
    timeout: 600_000,
    waitForCapacity: true,
    provisionTimeout: 900_000,
  });

  const items = patterns.map((p) => ({ id: p.id, text: `${p.label}. ${p.summary}` }));
  const results = await client.encode(config.models.encoder, items);

  const records: IndexRecord[] = patterns.map((p, i) => {
    const single = results[i];
    if (!single?.dense) throw new Error(`encoder returned no dense vector for ${p.id}`);
    return { id: p.id, vector: Array.from(single.dense) };
  });

  fs.writeFileSync(outPath, JSON.stringify(records, null, 2));
  console.log(`wrote ${outPath} (${records.length} vectors, dim=${records[0]?.vector.length})`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
