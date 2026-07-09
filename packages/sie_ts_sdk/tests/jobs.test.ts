import { beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { buildJobBody, connectionName } from "../src/jobs.js";
import { packMessage } from "../src/msgpack.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const SUBMIT_RESP = {
  id: "job-1",
  object: "job",
  operation: "encode",
  model: "BAAI/bge-m3",
  state: "queued",
  total_items: 2,
  chunks: 1,
  preflight: { estimated_credits: 64 },
};

describe("buildJobBody (pure slot mapping)", () => {
  it("maps an inline list to items", () => {
    expect(buildJobBody({ source: ["a", "b"], model: "m" })).toEqual({
      operation: "encode",
      model: "m",
      items: [{ text: "a" }, { text: "b" }],
    });
  });

  it("routes a connector URI to src + derived connection", () => {
    const body = buildJobBody({
      source: "postgres://warehouse?query=x",
      model: "m",
      sink: "postgres://warehouse?table=t",
    });
    expect(body.src).toBe("postgres://warehouse?query=x");
    expect(body.connection).toBe("warehouse");
    expect(body.sink).toBe("postgres://warehouse?table=t");
    expect(body.sink_connection).toBeUndefined();
  });

  it("threads a distinct sink connection", () => {
    const body = buildJobBody({
      source: "postgres://wh?query=x",
      model: "m",
      sink: "s3://out/vecs",
    });
    expect(body.sink_connection).toBe("out");
  });

  it("maps schedule / watch triggers", () => {
    expect(buildJobBody({ source: ["a"], model: "m", when: "schedule:*/5 * * * *" })).toMatchObject(
      {
        when: "schedule",
        schedule: "*/5 * * * *",
      },
    );
    expect(buildJobBody({ source: ["a"], model: "m", when: "watch:s3://in" })).toMatchObject({
      when: "watch",
      watch: "s3://in",
    });
  });

  it("derives connection names from URIs", () => {
    expect(connectionName("postgres://warehouse?query=x")).toBe("warehouse");
    expect(connectionName("s3://customer-bucket/in/")).toBe("customer-bucket");
  });

  // field_map / output_field + the internal upload:// scheme

  it("rides field_map + output_field on connector jobs", () => {
    const body = buildJobBody({
      source: "postgres://wh?query=select id, body, source_url from docs",
      model: "BAAI/bge-m3",
      sink: "postgres://wh?table=doc_vectors",
      fieldMap: {
        id_field: "id",
        input_field: "body",
        carry: ["source_url"],
        input_type: "text",
      },
      outputField: "embedding",
    });
    expect(body.field_map).toEqual({
      id_field: "id",
      input_field: "body",
      input_type: "text",
      carry: ["source_url"],
    });
    expect(body.output_field).toBe("embedding");
  });

  it("rejects field_map on inline items and bad slots", () => {
    expect(() =>
      buildJobBody({ source: ["a"], model: "m", fieldMap: { id_field: "id" } }),
    ).toThrowError(/connector-src/);
    expect(() =>
      buildJobBody({
        source: "postgres://wh?query=x",
        model: "m",
        sink: "postgres://wh?table=t",
        fieldMap: { id_column: "id" } as never,
      }),
    ).toThrowError(/unknown field_map key/);
    expect(() =>
      buildJobBody({
        source: "postgres://wh?query=x",
        model: "m",
        sink: "postgres://wh?table=t",
        fieldMap: { input_type: "rows" } as never,
      }),
    ).toThrowError(/input_type/);
  });

  it("derives no connection for the internal upload:// scheme", () => {
    const body = buildJobBody({
      source: "upload://file-abc?format=csv",
      model: "m",
      sink: "upload://file-out",
      fieldMap: { id_field: "doc_id", input_field: "text" },
    });
    expect(body.src).toBe("upload://file-abc?format=csv");
    expect(body.sink).toBe("upload://file-out");
    expect(body.connection).toBeUndefined();
    expect(body.sink_connection).toBeUndefined();
    // upload source → external sink still threads the sink's connection.
    const cross = buildJobBody({
      source: "upload://file-abc",
      model: "m",
      sink: "postgres://wh?table=doc_vectors",
      sinkConnection: "wh",
    });
    expect(cross.connection).toBeUndefined();
    expect(cross.sink_connection).toBe("wh");
  });

  it("forwards op inputs via options as-is (op matrix)", () => {
    // score: options.query rides untouched (connector form).
    const score = buildJobBody({
      source: "postgres://wh?query=x",
      model: "m",
      operation: "score",
      sink: "postgres://wh?table=scores",
      options: { query: "rank these documents" },
    });
    expect(score.options).toEqual({ query: "rank these documents" });

    // extract: labels + output_schema forwarded (inline form).
    const extract = buildJobBody({
      source: ["some text"],
      model: "m",
      operation: "extract",
      options: { labels: ["PERSON", "ORG"], output_schema: { type: "object" } },
    });
    expect(extract.options).toEqual({
      labels: ["PERSON", "ORG"],
      output_schema: { type: "object" },
    });

    // Absent / empty options stay off the wire (additive-only).
    expect(buildJobBody({ source: ["a"], model: "m" }).options).toBeUndefined();
    expect(buildJobBody({ source: ["a"], model: "m", options: {} }).options).toBeUndefined();
  });
});

describe("client.jobs", () => {
  beforeEach(() => mockFetch.mockClear());

  it("submit posts the inline body to /v1/jobs", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(SUBMIT_RESP, 201));
    const client = new SIEClient("http://gw:8080", { apiKey: "sk-sie-x" });
    const result = await client.jobs.submit({ source: ["a", "b"], model: "BAAI/bge-m3" });
    expect(result.id).toBe("job-1");
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://gw:8080/v1/jobs");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      operation: "encode",
      model: "BAAI/bge-m3",
      items: [{ text: "a" }, { text: "b" }],
    });
    expect(init.headers.Authorization).toBe("Bearer sk-sie-x");
  });

  it("submit maps a connector job body", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(SUBMIT_RESP, 201));
    const client = new SIEClient("http://gw:8080");
    await client.jobs.submit({
      source: "postgres://warehouse?query=x",
      model: "m",
      sink: "s3://out/vecs",
    });
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.src).toBe("postgres://warehouse?query=x");
    expect(body.connection).toBe("warehouse");
    expect(body.sink_connection).toBe("out");
  });

  it("submit forwards score query and extract labels via options", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(SUBMIT_RESP, 201));
    const client = new SIEClient("http://gw:8080");
    await client.jobs.submit({
      source: "postgres://wh?query=x",
      model: "BAAI/bge-m3",
      operation: "score",
      sink: "postgres://wh?table=scores",
      options: { query: "rank these documents" },
    });
    const scoreBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(scoreBody.operation).toBe("score");
    expect(scoreBody.options).toEqual({ query: "rank these documents" });

    mockFetch.mockResolvedValueOnce(jsonResponse(SUBMIT_RESP, 201));
    await client.jobs.submit({
      source: ["some text"],
      model: "urchade/gliner_small-v2.1",
      operation: "extract",
      options: { labels: ["PERSON", "ORG"] },
    });
    const extractBody = JSON.parse(mockFetch.mock.calls[1][1].body);
    expect(extractBody.operation).toBe("extract");
    expect(extractBody.options).toEqual({ labels: ["PERSON", "ORG"] });
  });

  it("get and cancel hit the expected URLs", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ id: "job-1", state: "running" }));
    const client = new SIEClient("http://gw:8080");
    await client.jobs.get("job-1");
    expect(mockFetch.mock.calls[0][0]).toBe("http://gw:8080/v1/jobs/job-1");

    mockFetch.mockResolvedValueOnce(jsonResponse({ id: "job-1", state: "cancelled" }));
    const out = await client.jobs.cancel("job-1");
    expect(out.state).toBe("cancelled");
    expect(mockFetch.mock.calls[1][0]).toBe("http://gw:8080/v1/jobs/job-1/cancel");
    expect(mockFetch.mock.calls[1][1].method).toBe("POST");
  });

  it("list returns the data array", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ object: "list", data: [{ id: "job-1" }, { id: "job-2" }] }),
    );
    const client = new SIEClient("http://gw:8080");
    const jobs = await client.jobs.list();
    expect(jobs.map((j) => j.id)).toEqual(["job-1", "job-2"]);
    expect(mockFetch.mock.calls[0][0]).toBe("http://gw:8080/v1/jobs");
  });

  it("results reads http refs and decodes per-item embeddings", async () => {
    const chunkBytes = packMessage([
      {
        success: true,
        id: "0",
        units: { input_tokens: 5 },
        result_msgpack: packMessage({ dense: { dims: 4, values: [0.1, 0.2, 0.3, 0.4] } }),
      },
    ]);
    const job = {
      id: "job-1",
      state: "succeeded",
      total_items: 1,
      settled_credits: 5,
      output: {
        kind: "refs",
        chunks: [{ seq: 0, items: 1, state: "succeeded", ref: "http://refs.local/c0" }],
      },
    };
    mockFetch
      .mockResolvedValueOnce(jsonResponse(job))
      .mockResolvedValueOnce(new Response(chunkBytes, { status: 200 }));

    const client = new SIEClient("http://gw:8080");
    const results = await client.jobs.results("job-1");
    // snake_case throughout (matches the wire + the Python SDK + JobStatus).
    expect(results.job_id).toBe("job-1");
    expect(results.total_items).toBe(1);
    expect(results.settled_credits).toBe(5);
    expect(results.retrieved).toBe(1);
    expect(results.dims).toBe(4);
    expect(Array.from(results.items[0].dense as number[])).toEqual([0.1, 0.2, 0.3, 0.4]);
  });

  it("wait polls until the job reaches a terminal state", async () => {
    mockFetch
      .mockResolvedValueOnce(jsonResponse({ id: "job-1", state: "running" }))
      .mockResolvedValueOnce(jsonResponse({ id: "job-1", state: "succeeded" }));
    const client = new SIEClient("http://gw:8080");
    const job = await client.jobs.wait("job-1", { pollMs: 0 });
    expect(job.state).toBe("succeeded");
    expect(mockFetch.mock.calls.length).toBe(2);
  });

  it("wait throws a job_wait_timeout RequestError when the deadline passes", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "job-1", state: "running" }));
    const client = new SIEClient("http://gw:8080");
    await expect(client.jobs.wait("job-1", { timeoutMs: 0, pollMs: 0 })).rejects.toMatchObject({
      code: "job_wait_timeout",
    });
  });

  it("submit floors the abort timeout to 120s (survives the 30s default)", () => {
    vi.useFakeTimers();
    try {
      let capturedInit: RequestInit | undefined;
      mockFetch.mockImplementationOnce((_url: string, init: RequestInit) => {
        capturedInit = init;
        return new Promise<Response>(() => {}); // never settles; we only inspect the signal
      });
      const client = new SIEClient("http://gw:8080");
      client.jobs.submit({ source: ["a"], model: "m" }).catch(() => {}); // swallow the eventual abort
      expect(capturedInit?.signal?.aborted).toBe(false);
      vi.advanceTimersByTime(30_000);
      expect(capturedInit?.signal?.aborted).toBe(false); // past the 30s default, still alive
      vi.advanceTimersByTime(90_001);
      expect(capturedInit?.signal?.aborted).toBe(true); // aborts at the 120s floor
    } finally {
      vi.useRealTimers();
    }
  });
});
