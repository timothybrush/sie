export type SampleDoc = {
  id: string;
  filename: string;
  label: string;
  description: string;
  labels: string[];
};

export type ExtractedField = {
  label: string;
  text: string;
  score: number;
};

export type DonutEntity = {
  label: string;
  text: string;
};

export type TriageResult = {
  sampleId: string;
  recognitionModel: string;
  markdown: string;
  donutEntities: DonutEntity[];
  donutData: unknown;
  glinerFields: ExtractedField[];
  timings: {
    recognitionMs: number;
    donutMs: number;
    glinerMs: number;
    totalMs: number;
  };
};
