import type { SIEClient } from "@superlinked/sie-sdk";
import type { ExtractedField } from "./types.js";

export async function extractFields(
  client: SIEClient,
  model: string,
  text: string,
  labels: string[],
): Promise<ExtractedField[]> {
  if (!text.trim()) return [];
  const result = await client.extract(model, { text }, { labels, threshold: 0.4 });
  return (result.entities ?? []).map((e) => ({
    label: e.label,
    text: e.text,
    score: e.score,
  }));
}
