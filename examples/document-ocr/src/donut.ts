import { detectImageFormat, type SIEClient } from "@superlinked/sie-sdk";
import type { DonutEntity } from "./types.js";

/** Run any image-input "structured" extractor (Donut variants, etc.). */
export async function structuredExtract(
  client: SIEClient,
  model: string,
  imageBytes: Uint8Array,
  options?: Record<string, unknown>,
): Promise<{ entities: DonutEntity[]; data: unknown }> {
  const format = detectImageFormat(imageBytes);
  if (format === "unknown") throw new Error("could not detect image format");
  const wire = { data: imageBytes, format };
  const result = await client.extract(
    model,
    { images: [wire] as unknown as Uint8Array[] },
    { labels: [], options } as unknown as Parameters<typeof client.extract>[2],
  );
  const entities = (result.entities ?? []).map((e) => ({
    label: e.label,
    text: e.text,
  }));
  const data = (result as unknown as { data?: unknown }).data;
  return { entities, data };
}
