import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const dataPath = path.resolve(
  here,
  "../../concept/09_in_context_execution_calibration_design_inspector.data.json",
);
const review = JSON.parse(fs.readFileSync(dataPath, "utf8"));

assert.equal(review.layout, "design_transaction_inspector");
assert.match(review.subtitle, /候选研究设计/);
assert.equal(review.cards.length, 8);

const expectedCards = [
  ["ICA-DP-01", "Perturbation Condition"],
  ["ICA-DP-02", "Frozen Planner"],
  ["ICA-DP-03", "Frozen Tracker"],
  ["ICA-DP-04", "Support Rollout"],
  ["ICA-DP-05", "Context Encoder"],
  ["ICA-DP-06", "Latent Calibration"],
  ["ICA-DP-07", "Calibration Learning"],
  ["ICA-DP-08", "Frozen Query Execution"],
];
assert.deepEqual(
  review.cards.map(({ designId, title }) => [designId, title]),
  expectedCards,
);

const stepIds = new Set(review.transaction.steps.map(({ id }) => id));
assert.equal(stepIds.size, review.transaction.steps.length);
for (const card of review.cards) {
  assert.ok(card.responsibility.length > 0, `${card.designId} needs responsibility`);
  assert.ok(card.details.length >= 4, `${card.designId} needs atomic decisions`);
  assert.ok(card.details.length <= 8, `${card.designId} has too many atomic decisions`);
  for (const stepId of card.highlightSteps) {
    assert.ok(stepIds.has(stepId), `${card.designId} references unknown step ${stepId}`);
  }
}

const text = JSON.stringify(review);
for (const required of [
  "c_t = [Y_target,t, Y_realized,t, A_exec,t]",
  "C = [c_0, …, c_T]",
  "只输出一个 condition-level Δzξ",
  "同一个 Δzξ 被固定复用",
  "只允许更新 Context Encoder 参数",
  "不使用 task Reward、Privileged Teacher",
]) {
  assert.ok(text.includes(required), `missing required decision: ${required}`);
}

for (const forbidden of ["implementation-confirmed", "runtime-confirmed", "DESIGN-CONFIRMED"]) {
  assert.ok(!text.includes(forbidden), `candidate inspector overclaims ${forbidden}`);
}

console.log("candidate in-context calibration Design Inspector: PASS");
