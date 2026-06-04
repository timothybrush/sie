// Typed SSE events streamed to the browser during a risk-score run.

export type RiskEvent =
  | { type: "models"; data: { encoder: string; reranker: string; extractor: string } }
  | { type: "extracting" }
  | { type: "extracted"; data: { entities: ExtractedEntity[]; ms: number } }
  | { type: "encoding" }
  | { type: "encoded"; data: { dim: number; ms: number } }
  | { type: "scoring" }
  | {
      type: "scored";
      data: {
        hits: RiskHit[];
        topScore: number;
        band: "low" | "medium" | "high";
        ms: number;
      };
    }
  | { type: "done"; data: { totalMs: number } }
  | { type: "error"; data: { stage: string; message: string } };

export interface ExtractedEntity {
  label: string;
  text: string;
  score: number;
}

export interface RiskHit {
  id: string;
  label: string;
  summary: string;
  outcome: string;
  loss_usd: number;
  score: number;
}
