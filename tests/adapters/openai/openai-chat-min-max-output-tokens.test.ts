import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { removeTreeWithRetry } from "../../helpers/remove-tree";
import { buildOpenAIChatPassthroughRequest, createOpenAIChatAdapter } from "../../../src/adapters/openai-chat";
import { resolveMaxTokens } from "../../../src/adapters/openai-chat/summary-budget";
import { createAnthropicAdapter } from "../../../src/adapters/anthropic";
import { handleChatCompletions } from "../../../src/server/chat-completions";
import { acquireOwnedSpendHome } from "../../helpers/owned-spend-home";
import type { OcxConfig, OcxParsedRequest, OcxProviderConfig } from "../../../src/types";

function provider(overrides: Partial<OcxProviderConfig> = {}): OcxProviderConfig {
  return {
    adapter: "openai-chat",
    baseUrl: "https://example.test/v1",
    apiKey: "sk-test",
    authMode: "key",
    ...overrides,
  };
}

function parsed(maxOutputTokens?: number): OcxParsedRequest {
  return {
    modelId: "test-model",
    context: { messages: [{ role: "user", content: "hi", timestamp: 0 }] },
    stream: false,
    options: maxOutputTokens === undefined ? {} : { maxOutputTokens },
  };
}

describe("resolveMaxTokens minMaxOutputTokens floor", () => {
  test("raises a declared maxOutputTokens below the floor", () => {
    const p = provider({ minMaxOutputTokens: 4096 });
    expect(resolveMaxTokens(p, parsed(512))).toBe(4096);
  });

  test("leaves a declared maxOutputTokens at or above the floor untouched", () => {
    const p = provider({ minMaxOutputTokens: 4096 });
    expect(resolveMaxTokens(p, parsed(8192))).toBe(8192);
    expect(resolveMaxTokens(p, parsed(4096))).toBe(4096);
  });

  test("uses the floor itself when nothing is declared", () => {
    const p = provider({ minMaxOutputTokens: 4096 });
    expect(resolveMaxTokens(p, parsed())).toBe(4096);
  });

  test("a modelMaxOutputTokens record value counts as declared", () => {
    const p = provider({ minMaxOutputTokens: 4096, modelMaxOutputTokens: { "test-model": 2048 } });
    expect(resolveMaxTokens(p, parsed())).toBe(4096);
  });

  test("a defaultMaxOutputTokens value below the floor is raised", () => {
    const p = provider({ minMaxOutputTokens: 4096, defaultMaxOutputTokens: 1024 });
    expect(resolveMaxTokens(p, parsed())).toBe(4096);
  });

  test.each([0, -1, undefined] as const)("min %s is off and passes the declared value through", min => {
    const p = provider(min === undefined ? {} : { minMaxOutputTokens: min });
    expect(resolveMaxTokens(p, parsed(512))).toBe(512);
  });

  test("the floor reaches the wire on buildRequest", () => {
    const body = JSON.parse(
      createOpenAIChatAdapter(provider({ minMaxOutputTokens: 4096 }))
        .buildRequest(parsed(512)).body as string,
    ) as Record<string, unknown>;
    expect(body.max_tokens).toBe(4096);
  });
});


type BudgetCase = {
  name: string;
  config: Partial<OcxProviderConfig>;
  caller: Record<string, number>;
  expected: Record<string, number>;
};
const floor = { minMaxOutputTokens: 4096 };
const budgetCases: BudgetCase[] = [
  { name: "legacy caller below floor", config: floor, caller: { max_tokens: 512 }, expected: { max_tokens: 4096 } },
  { name: "modern caller below floor", config: floor, caller: { max_completion_tokens: 512 }, expected: { max_completion_tokens: 4096 } },
  { name: "larger legacy caller", config: floor, caller: { max_tokens: 8192 }, expected: { max_tokens: 8192 } },
  { name: "larger modern caller", config: floor, caller: { max_completion_tokens: 8192 }, expected: { max_completion_tokens: 8192 } },
  { name: "modern wins over larger legacy", config: floor, caller: { max_tokens: 16384, max_completion_tokens: 512 }, expected: { max_completion_tokens: 4096 } },
  { name: "larger modern wins over tiny legacy", config: floor, caller: { max_tokens: 512, max_completion_tokens: 8192 }, expected: { max_completion_tokens: 8192 } },
  { name: "floor with no caller budget", config: floor, caller: {}, expected: { max_tokens: 4096 } },
  { name: "small provider default", config: { ...floor, defaultMaxOutputTokens: 1024 }, caller: {}, expected: { max_tokens: 4096 } },
  { name: "larger provider default", config: { ...floor, defaultMaxOutputTokens: 8192 }, caller: {}, expected: { max_tokens: 8192 } },
  { name: "model default wins over provider default", config: { ...floor, modelMaxOutputTokens: { "test-model": 6144 }, defaultMaxOutputTokens: 12288 }, caller: {}, expected: { max_tokens: 6144 } },
  { name: "small model default is floored", config: { ...floor, modelMaxOutputTokens: { "test-model": 1024 }, defaultMaxOutputTokens: 12288 }, caller: {}, expected: { max_tokens: 4096 } },
  { name: "caller wins over configured defaults", config: { ...floor, modelMaxOutputTokens: { "test-model": 12288 }, defaultMaxOutputTokens: 16384 }, caller: { max_completion_tokens: 6144 }, expected: { max_completion_tokens: 6144 } },
  { name: "unset preserves conflicting aliases", config: {}, caller: { max_tokens: 512, max_completion_tokens: 1024 }, expected: { max_tokens: 512, max_completion_tokens: 1024 } },
  { name: "unset preserves absent budget despite defaults", config: { modelMaxOutputTokens: { "test-model": 8192 }, defaultMaxOutputTokens: 16384 }, caller: {}, expected: {} },
];
function expectBudget(body: Record<string, unknown>, expected: Record<string, number>): void {
  for (const key of ["max_tokens", "max_completion_tokens"]) {
    if (expected[key] === undefined) expect(body).not.toHaveProperty(key);
    else expect(body[key]).toBe(expected[key]);
  }
}

describe("native Chat minMaxOutputTokens", () => {
  const previousHome = process.env.OPENCODEX_HOME;
  const home = mkdtempSync(join(tmpdir(), "ocx-chat-floor-"));
  beforeAll(() => { process.env.OPENCODEX_HOME = home; });
  afterAll(() => {
    if (previousHome === undefined) delete process.env.OPENCODEX_HOME;
    else process.env.OPENCODEX_HOME = previousHome;
    removeTreeWithRetry(home);
  });
  for (const mode of ["stream", "buffered", "buffered SSE"] as const) {
    for (const fixture of budgetCases) {
      test(`${mode}: ${fixture.name} reaches the upstream wire`, async () => {
        const captured: Record<string, unknown>[] = [];
        const upstream = Bun.serve({
          hostname: "127.0.0.1", port: 0,
          async fetch(req) {
            expect(new URL(req.url).pathname).toBe("/v1/chat/completions");
            captured.push(await req.json() as Record<string, unknown>);
            if (mode !== "buffered") {
              return new Response('data: {"id":"chatcmpl_floor","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}\n\ndata: {"id":"chatcmpl_floor","object":"chat.completion.chunk","model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n', {
                headers: { "content-type": "text/event-stream" },
              });
            }
            return Response.json({ id: "chatcmpl_floor", object: "chat.completion", model: "test-model", choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }] });
          },
        });
        const release = acquireOwnedSpendHome();
        try {
          const target = provider({ ...fixture.config, allowPrivateNetwork: true, baseUrl: `http://127.0.0.1:${upstream.port}/v1` });
          const caller = { model: "floor-fixture/test-model", messages: [{ role: "user", content: "hi" }], stream: mode === "stream", ...fixture.caller };
          const original = structuredClone(caller);
          const response = await handleChatCompletions(new Request("http://localhost/v1/chat/completions", {
            method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(caller),
          }), { port: 0, defaultProvider: "floor-fixture", providers: { "floor-fixture": target } } as OcxConfig, { model: "", provider: "" });
          const output = await response.text();
          expect({ status: response.status, error: response.status === 200 ? undefined : output }).toEqual({ status: 200, error: undefined });
          if (mode === "stream") expect(output).toContain("[DONE]");
          else expect(JSON.parse(output).choices[0].message.content).toBe("ok");
          expect(captured).toHaveLength(1);
          expect(captured[0].stream).toBe(mode === "stream");
          expectBudget(captured[0], fixture.expected);
          expect(caller).toEqual(original);
        } finally {
          release();
          await upstream.stop(true);
        }
      });
    }
  }

  test("serializer does not mutate the caller or provider", () => {
    const target = provider({ ...floor, defaultMaxOutputTokens: 12288 });
    const caller = { messages: [{ role: "user", content: "hi" }], max_tokens: 8192, max_completion_tokens: 512 };
    const before = structuredClone({ caller, target });
    Object.freeze(caller);
    Object.freeze(target);
    for (const stream of [true, false]) {
      const request = buildOpenAIChatPassthroughRequest(target, caller, "test-model", stream);
      expectBudget(JSON.parse(request.body), { max_completion_tokens: 4096 });
      expect({ caller, target }).toEqual(before);
    }
  });

  test("non-chat adapter keeps its own budget resolution", async () => {
    const target = provider({ adapter: "anthropic", ...floor });
    const request = parsed(512);
    const before = structuredClone({ target, request });
    const wire = await createAnthropicAdapter(target).buildRequest(request);
    const body = JSON.parse(wire.body);
    expect(body.max_tokens).toBe(512);
    expect({ target, request }).toEqual(before);
  });
});
