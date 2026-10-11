import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as spendLedger from "../../src/lib/spend-reservation-ledger";
import { createRequestExecutionBudget, CODEX_TEXT_GUARDED_BUDGET_POLICY } from "../../src/lib/request-execution-budget";
import { createRequestSpendTracker } from "../../src/server/responses/request-spend";
import { handleResponses } from "../../src/server/responses";
import { clearCopilotAutoModelsForTests } from "../../src/providers/github-copilot-auto";
import { clearGenericFailoverHealth } from "../../src/oauth/generic-account-failover";
import { clearKeyCooldowns } from "../../src/providers/key-failover";
import { clearComboSelectionState, clearComboTargetCooldowns } from "../../src/combos";
import { clearComboRecallForTests } from "../../src/server/responses/combo-session-recall";
import { resetProviderRequestPacingForTest } from "../../src/providers/request-pacing";
import { flushConfigDirHardeningForTests } from "../../src/config/paths";
import { flushWindowsSecretAclReapsBeforeRemoval } from "../../src/lib/windows-secret-acl";
import type { OcxConfig } from "../../src/types";
import type { RequestLogContext } from "../../src/server/request-log";
import { acquireOwnedSpendHome } from "../helpers/owned-spend-home";
import { installIsolatedCodexHome, type IsolatedCodexHome } from "../helpers/isolated-codex-home";
import { removeTreeWithRetry } from "../helpers/remove-tree";

const PROVIDER = "github-copilot";
const INPUT_ESTIMATE = 10;
let previousHome: string | undefined;
let previousFetch: typeof fetch;
let home: string;
let codexHome: IsolatedCodexHome;
let release: (() => void) | undefined;
let spies: Array<{ mockRestore(): void }>;

function resetTransportState(): void {
  clearCopilotAutoModelsForTests();
  clearGenericFailoverHealth();
  clearKeyCooldowns();
  clearComboSelectionState();
  clearComboTargetCooldowns();
  clearComboRecallForTests();
  resetProviderRequestPacingForTest();
}

beforeEach(() => {
  previousHome = process.env.OPENCODEX_HOME;
  previousFetch = globalThis.fetch;
  spies = [];
  codexHome = installIsolatedCodexHome("ocx-copilot-floor-codex-");
  home = mkdtempSync(join(tmpdir(), "ocx-copilot-floor-"));
  process.env.OPENCODEX_HOME = home;
  resetTransportState();
  release = acquireOwnedSpendHome();
});

afterEach(async () => {
  try {
    for (const spy of spies) spy.mockRestore();
    release?.();
    await flushConfigDirHardeningForTests();
    await flushWindowsSecretAclReapsBeforeRemoval(home);
    await flushWindowsSecretAclReapsBeforeRemoval(codexHome.path);
  } finally {
    release = undefined;
    globalThis.fetch = previousFetch;
    resetTransportState();
    if (previousHome === undefined) delete process.env.OPENCODEX_HOME;
    else process.env.OPENCODEX_HOME = previousHome;
    codexHome.restoreEnvironment();
    try { removeTreeWithRetry(home); }
    finally { codexHome.removeAfterDrain(); }
  }
});

function fixture(limit: number, floor: number | undefined) {
  const ledger = spendLedger.createSpendReservationLedger({
    salt: "copilot-floor-fixture",
    policy: { ...spendLedger.DEFAULT_SPEND_RESERVATION_POLICY, pool: { maxTokens: limit } },
    journal: { read: () => [], append: () => {} },
  });
  // The handler's workflow admission and the inherited tracker consult the same ledger.
  spies.push(spyOn(spendLedger, "sharedSpendLedger").mockImplementation(() => ledger));
  spies.push(spyOn(spendLedger, "sharedSpendPolicy").mockImplementation(() => ledger.policy));
  const log: RequestLogContext = { model: "", provider: "", spendInputEstimateTokens: INPUT_ESTIMATE };
  const tracker = createRequestSpendTracker(log, undefined, ledger);
  log.spendTracker = tracker;
  const budget = createRequestExecutionBudget({ ...CODEX_TEXT_GUARDED_BUDGET_POLICY }, undefined, tracker);
  const paths: string[] = [];
  const inference: Array<{ body: Record<string, unknown>; reserved: number }> = [];
  // Same Auto-only discovery/session/intent shapes as github-copilot-auto-runtime.test.ts.
  const offline = (async (input, init) => {
    const url = new URL(input instanceof Request ? input.url : String(input));
    expect(url.origin).toBe("https://api.githubcopilot.com");
    const headers = new Headers(init?.headers);
    expect(headers.get("authorization")).toBe("Bearer synthetic-copilot-key");
    const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : {};
    paths.push(url.pathname);
    if (url.pathname === "/models") return Response.json({ data: [{
      id: "gpt-4o", model_picker_enabled: false, supported_endpoints: ["/chat/completions"],
    }] });
    if (url.pathname === "/models/session") {
      expect(body).toEqual({ auto_mode: { model_hints: ["auto"] } });
      return Response.json({ session_token: "synthetic-session", expires_at: Date.now() / 1000 + 60, available_models: ["gpt-4o"] });
    }
    expect(headers.get("copilot-session-token")).toBe("synthetic-session");
    if (url.pathname === "/models/session/intent") {
      expect(body).toEqual({ prompt: "hello", available_models: ["gpt-4o"] });
      return Response.json({ candidate_models: ["gpt-4o"] });
    }
    if (url.pathname !== "/chat/completions") throw new Error(`Unexpected offline endpoint: ${url.pathname}`);
    inference.push({ body, reserved: ledger.snapshot("pool", PROVIDER)?.reserved ?? 0 });
    // Missing usage deliberately exercises conservative liability after the physical send.
    if (!body.stream) return Response.json({ id: "chat-floor", model: "gpt-4o", choices: [{
      index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop",
    }] });
    const chunks = [
      { id: "chat-floor", model: "gpt-4o", choices: [{ index: 0, delta: { role: "assistant", content: "ok" }, finish_reason: null }] },
      { id: "chat-floor", model: "gpt-4o", choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
    ];
    return new Response(chunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join("") + "data: [DONE]\n\n", {
      headers: { "content-type": "text/event-stream" },
    });
  }) as typeof fetch;
  // Fence both provider executors and any built-in transport against live network access.
  globalThis.fetch = offline;
  const config = {
    port: 0, defaultProvider: PROVIDER, spend: { pool: { maxTokens: limit } },
    providers: { [PROVIDER]: {
      adapter: "openai-responses", authMode: "key", apiKey: "synthetic-copilot-key",
      baseUrl: "https://api.githubcopilot.com", models: ["auto", "gpt-4o"], defaultModel: "auto",
      modelAdapters: { auto: "openai-responses" }, minMaxOutputTokens: floor, fetch: offline,
    } },
    combos: { pair: { strategy: "failover", targets: [{ provider: PROVIDER, model: "auto" }] } },
  } as OcxConfig;
  return { ledger, tracker, budget, log, config, paths, inference };
}

describe("Copilot Auto Chat floor after Responses combo prepayment", () => {
  for (const stream of [false, true]) {
    for (const combo of [true, false]) {
      for (const scenario of [
        { name: "refuses a floor above the pool limit", limit: 1000, floor: 4096, status: 429, cap: 4096 },
        { name: "keeps the caller cap without a floor", limit: 1000, floor: undefined, status: 200, cap: 512 },
        { name: "reserves and retains the full raised liability", limit: 8192, floor: 4096, status: 200, cap: 4096 },
      ]) {
        test(`${combo ? "combo" : "direct"} ${stream ? "stream" : "buffered"}: ${scenario.name}`, async () => {
          const f = fixture(scenario.limit, scenario.floor);
          try {
            const response = await handleResponses(new Request("http://localhost/v1/responses", {
              method: "POST", headers: { "content-type": "application/json" },
              body: JSON.stringify({ model: combo ? "combo/pair" : `${PROVIDER}/auto`, input: "hello", max_output_tokens: 512, stream }),
            }), f.config, f.log, { sendBudget: f.budget });
            const text = await response.text();
            expect(response.status, text).toBe(scenario.status);
            expect(f.paths.slice(0, 3)).toEqual(["/models", "/models/session", "/models/session/intent"]);
            if (scenario.status === 429) {
              expect(f.inference).toHaveLength(0);
              expect(f.budget.physicalStarted).toBe(0);
            } else {
              expect(f.inference).toHaveLength(1);
              expect(f.inference[0]!.body).toMatchObject({ model: "gpt-4o", max_tokens: scenario.cap, stream });
              expect(f.inference[0]!.reserved).toBe(INPUT_ESTIMATE + scenario.cap);
              expect(f.budget.physicalStarted).toBe(1);
              if (stream) expect(text).toContain("event: response.completed");
              else expect(JSON.parse(text).output[0].content[0].text).toBe("ok");
            }
          } finally { f.tracker.settle(undefined); }
          const pool = f.ledger.snapshot("pool", PROVIDER);
          expect(pool?.reserved ?? 0).toBe(0);
          expect(pool?.unresolved ?? 0).toBe(scenario.status === 429 ? 0 : INPUT_ESTIMATE + scenario.cap);
          await f.ledger.waitForReporterDrain();
        });
      }
    }
  }
});
