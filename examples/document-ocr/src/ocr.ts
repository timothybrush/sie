import { detectImageFormat, type SIEClient } from "@superlinked/sie-sdk";

export async function recognize(
  client: SIEClient,
  model: string,
  imageBytes: Uint8Array,
  options?: Record<string, unknown>,
): Promise<string> {
  const format = detectImageFormat(imageBytes);
  if (format === "unknown") throw new Error("could not detect image format");
  const wire = { data: imageBytes, format };
  // The TS SDK types declare images as Uint8Array[], but the wire format
  // expects {data, format} dicts. Cast around the typing gap.
  const result = await client.extract(
    model,
    { images: [wire] as unknown as Uint8Array[] },
    // The TS SDK's ExtractOptions doesn't declare `options`, but the wire
    // protocol forwards it to the adapter. Cast to bridge the typing gap.
    { labels: [], options } as unknown as Parameters<typeof client.extract>[2],
  );
  if (!result.entities || result.entities.length === 0) return "";
  const text = result.entities[0]?.text;
  return typeof text === "string" ? text : "";
}
