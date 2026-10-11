import { resolveChatMaxTokens } from "../adapters/openai-chat/summary-budget";
import type { TransientSendBudget } from "../lib/upstream-retry";
import type { OcxProviderConfig } from "../types";
import type { RequestLogContext } from "./request-log";

/** Resolve each concrete target before admission, never inherit another combo child's floor. */
export function recordChatOutputSpendCeiling(
  logCtx: RequestLogContext, provider: OcxProviderConfig, modelId: string, requested: unknown, budget?: TransientSendBudget,
): void {
  const caller = typeof requested === "number" && requested > 0 ? requested : undefined;
  const effective = provider.adapter === "openai-chat" && typeof provider.minMaxOutputTokens === "number" && provider.minMaxOutputTokens > 0
    ? resolveChatMaxTokens(provider, modelId, caller) : caller;
  logCtx.spendOutputCeilingTokens = effective === undefined ? undefined : Math.trunc(effective);
  if (budget && "setSpendOutputCeiling" in budget && typeof budget.setSpendOutputCeiling === "function") {
    budget.setSpendOutputCeiling(logCtx.spendOutputCeilingTokens);
  }
}
