import { beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const GW = "http://gw:8080";
const CP = "http://cp:9000";

describe("client.connections", () => {
  beforeEach(() => mockFetch.mockClear());

  it("add posts to the control plane with the secret in the body", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ org: "acme", account_id: 7, id: 1, type: "postgres", name: "wh" }, 201),
    );
    const client = new SIEClient(GW, { apiKey: "sk-sie-x", controlPlaneUrl: CP, org: "acme" });
    const out = await client.connections.add("wh", "postgres", "postgres://u:p@h/db");
    expect(out.name).toBe("wh");
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://cp:9000/internal/orgs/acme/connections");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      type: "postgres",
      name: "wh",
      secret: "postgres://u:p@h/db",
    });
  });

  it("list returns the connections array", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        org: "acme",
        account_id: 7,
        connections: [{ id: 1, type: "postgres", name: "wh" }],
      }),
    );
    const client = new SIEClient(GW, { controlPlaneUrl: CP, org: "acme" });
    const conns = await client.connections.list();
    expect(conns.map((c) => c.name)).toEqual(["wh"]);
    expect(mockFetch.mock.calls[0][0]).toBe("http://cp:9000/internal/orgs/acme/connections");
  });

  it("revoke deletes the named connection", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ org: "acme", name: "wh", state: "revoked" }));
    const client = new SIEClient(GW, { controlPlaneUrl: CP, org: "acme" });
    const out = await client.connections.revoke("wh");
    expect(out.state).toBe("revoked");
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://cp:9000/internal/orgs/acme/connections/wh");
    expect(init.method).toBe("DELETE");
  });

  it("throws when controlPlaneUrl is not configured", async () => {
    const client = new SIEClient(GW, { apiKey: "sk-sie-x" });
    await expect(client.connections.list()).rejects.toThrow(/controlPlaneUrl/);
  });

  it("throws when org is not configured", async () => {
    const client = new SIEClient(GW, { controlPlaneUrl: CP });
    await expect(client.connections.list()).rejects.toThrow(/org/);
  });
});
