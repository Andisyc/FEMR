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
assert.equal(review.cards.length, 7);

const expectedCards = [
  ["ICA-DP-01", "Perturbation Condition"],
  ["ICA-DP-02", "Frozen Planner"],
  ["ICA-DP-03", "Frozen Tracker"],
  ["ICA-DP-04", "Support Rollout"],
  ["ICA-DP-05", "Context Encoder"],
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
  assert.ok(card.details.length >= 3, `${card.designId} needs atomic decisions`);
  assert.ok(card.details.length <= 8, `${card.designId} has too many atomic decisions`);
  for (const stepId of card.highlightSteps) {
    assert.ok(stepIds.has(stepId), `${card.designId} references unknown step ${stepId}`);
  }
}

const contextCard = review.cards.find(({ designId }) => designId === "ICA-DP-05");
assert.deepEqual(contextCard.details, [
  {
    heading: "1. 输入完整时序，而非单帧",
    text: "它读取第一次 Rollout 从开始到结束的轨迹。因为单个状态无法说明机器人为何逐渐偏离。",
  },
  {
    heading: "2. 整条轨迹只输出一个 Δz",
    text: "它不为每个时刻输出 Action，而是总结出一个代表当前执行条件的校准量。",
  },
  {
    heading: "3. 第二次 Rollout 固定复用 Δz",
    text: "第二次执行期间不再重复读取旧轨迹，也不动态修改 Δz。实时 Action 仍由 Tracker 输出。",
  },
]);

const supportCard = review.cards.find(({ designId }) => designId === "ICA-DP-04");
assert.deepEqual(supportCard.details, [
  {
    heading: "1. 明确三类信息的来源",
    text: "Y_target 是 Planner 想让机器人做什么；Y_realized 是机器人实际做成什么；A_exec 是真正发送给机器人的动作。",
  },
  {
    heading: "2. 三路数据必须正确对齐",
    text: "某个 Action 的效果可能延迟若干帧才出现。因此要明确哪个 Action 对应后面的哪段 Motion，不能简单按相同帧号拼接。",
  },
  {
    heading: "3. Support 中不能包含答案",
    text: "不输入缺陷参数、Teacher Action、Query Action 或人为构造的正确 Δz，避免模型利用部署时不存在的信息。",
  },
]);

const trackerCard = review.cards.find(({ designId }) => designId === "ICA-DP-03");
assert.deepEqual(trackerCard.details, [
  {
    heading: "1. Tracker Encoder 生成基础运动表示 z_t",
    text: "它根据当前状态、历史和 Planner 的目标 Motion，判断当前应当怎样执行。",
  },
  {
    heading: "2. Tracker Decoder 实时输出 Action",
    text: "Decoder 将基础运动表示 z_t 与 Context Encoder 输出的 Δz 组合，并在每个控制时刻生成 Action。",
  },
  {
    heading: "3. Encoder 和 Decoder 始终冻结",
    text: "训练 Context Encoder 时，Tracker 参数不更新，避免它直接记住各种缺陷。",
  },
  {
    heading: "4. 部署时不进行任何训练",
    text: "真机上不会更新参数、优化器或 normalizer，只进行前向计算。",
  },
  {
    heading: "5. Tracker 是唯一的 Action 生成器",
    text: "Context Encoder 只能提供低维 Δz，不能绕开 Decoder 直接输出 Action。",
  },
]);

const plannerCard = review.cards.find(({ designId }) => designId === "ICA-DP-02");
assert.equal(
  plannerCard.responsibility,
  "始终决定机器人想做什么，为第一次和第二次 Rollout 提供目标 Motion。",
);
assert.deepEqual(plannerCard.details, [
  {
    heading: "1. Planner 始终冻结",
    text: "无论离线训练还是部署，Planner 参数都不更新。",
  },
  {
    heading: "2. Planner 只输出目标 Motion",
    text: "它描述机器人应该怎样运动；具体 Action 仍由 Tracker 生成。",
  },
  {
    heading: "3. 第一次 Rollout 记录当次 Planner 输出",
    text: "轨迹直接保存当时交给 Tracker 的目标 Motion，不需要额外提供预先保存的 clean rollout。",
  },
  {
    heading: "4. 校准不能改变 Intent",
    text: "Context 只能调整目标 Motion 的执行方式，不能修改 Planner 原本要求完成的任务。",
  },
]);

const learningCard = review.cards.find(({ designId }) => designId === "ICA-DP-07");
assert.equal(learningCard.details.length, 8);
assert.deepEqual(
  learningCard.details.map(({ heading }) => heading),
  [
    "1. 按动力学条件构造 Episode",
    "2. 在相同条件下预先采集两条未校准 Rollout",
    "3. Support 与 Query 承担不同角色",
    "4. Support Rollout 生成校准量",
    "5. Query 使用真实未来运动",
    "6. 冻结 Tracker 重建 Query Action",
    "7. 使用 Executed Action 监督，只更新 Context Encoder",
    "8. 跨条件训练，并保持训练—部署接口一致",
  ],
);
assert.ok(learningCard.details.every((detail) => typeof detail === "object"));
assert.equal(learningCard.details.filter((detail) => detail.latex).length, 7);

const text = JSON.stringify(review);
for (const required of [
  "Y_target 是 Planner 想让机器人做什么",
  "输入完整时序，而非单帧",
  "整条轨迹只输出一个 Δz",
  "第二次 Rollout 固定复用 Δz",
  "在相同条件下预先采集两条未校准 Rollout",
  "训练中的 Δzξ 不得重新执行或改写 Query",
  "训练时，Tracker Encoder 输入 Query 中真实发生的 Future Motion",
  "梯度经过冻结 Tracker，但只更新 Context Encoder",
  "不使用 clean Action、Teacher Action、task Reward",
]) {
  assert.ok(text.includes(required), `missing required decision: ${required}`);
}

for (const forbidden of ["implementation-confirmed", "runtime-confirmed", "DESIGN-CONFIRMED"]) {
  assert.ok(!text.includes(forbidden), `candidate inspector overclaims ${forbidden}`);
}

console.log("candidate in-context calibration Design Inspector: PASS");
