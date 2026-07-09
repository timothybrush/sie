/**
 * The TS SDK's wire enums must match the shared golden fixtures.
 *
 * Round-trips packages/wire-fixtures/model_state.json against the runtime
 * MODEL_STATES array (the single source the ModelState type is derived from),
 * so drift fails in CI rather than shipping. See
 * packages/wire-fixtures/README.md.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { MODEL_STATES } from "../src/types.js";

const fixturesDir = fileURLToPath(new URL("../../wire-fixtures/", import.meta.url));

function loadFixture(name: string): Record<string, unknown> {
  return JSON.parse(readFileSync(`${fixturesDir}${name}`, "utf8"));
}

describe("wire contract golden fixtures", () => {
  it("ModelState matches the golden fixture", () => {
    const fixture = loadFixture("model_state.json");
    const states = fixture.model_states as string[];
    expect([...MODEL_STATES].sort()).toEqual([...states].sort());
  });
});
