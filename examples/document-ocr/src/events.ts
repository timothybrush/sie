import type { DonutEntity, ExtractedField } from "./types.js";

export type PipelineEvent =
  | { type: "models"; data: { extractor: string; recognition: string; structured: string } }
  | { type: "recognition_start"; data: { model: string } }
  | { type: "recognition_chunk"; data: { textLen: number } }
  | { type: "recognition_done"; data: { markdown: string; ms: number } }
  | { type: "donut_start" }
  | { type: "donut_done"; data: { entities: DonutEntity[]; rawData: unknown; ms: number } }
  | { type: "gliner_start"; data: { labels: string[] } }
  | { type: "gliner_done"; data: { fields: ExtractedField[]; ms: number } }
  | { type: "done"; data: { totalMs: number } }
  | { type: "error"; data: { message: string; stage: string } };
