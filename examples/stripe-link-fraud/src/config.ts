export const config = {
  sieUrl: process.env.SIE_URL ?? "http://localhost:8080",
  sieApiKey: process.env.SIE_API_KEY,

  stripe: {
    publishableKey: process.env.STRIPE_PUBLISHABLE_KEY ?? "",
    secretKey: process.env.STRIPE_SECRET_KEY ?? "",
    apiVersion: "2025-02-24.acacia" as const,
  },

  models: {
    extractor: "urchade/gliner_multi-v2.1",
    encoder: "sentence-transformers/all-MiniLM-L6-v2",
    reranker: "BAAI/bge-reranker-base",
  },

  extractLabels: ["product", "shipping_address", "email_domain", "amount"],

  risk: {
    // Thresholds are tuned for the bundled synthetic fraud-pattern corpus
    // (MiniLM-L6 cosine similarities). Below blockThreshold => low. Between
    // block and review => medium. Above reviewThreshold => high. Stripe
    // Link still authorizes in every case; the demo just shows the band.
    blockThreshold: 0.47,
    reviewThreshold: 0.52,
    topKRerank: 3,
  },

  paths: {
    fraudPatterns: "data/fraud_patterns.json",
    sampleOrders: "data/sample_orders.json",
    index: "data/risk_index.json",
  },

  port: Number(process.env.PORT ?? 3033),
} as const;

export type RiskBand = "low" | "medium" | "high";

export function riskBand(score: number): RiskBand {
  if (score >= config.risk.reviewThreshold) return "high";
  if (score >= config.risk.blockThreshold) return "medium";
  return "low";
}
