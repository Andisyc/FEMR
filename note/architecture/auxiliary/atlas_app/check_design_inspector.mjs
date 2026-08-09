import fs from "node:fs";
import path from "node:path";

const appRoot = process.cwd();
const repoRoot = path.resolve(appRoot, "../../../..");
const viewer = fs.readFileSync("architecture_atlas.html", "utf8");
const registry = fs.readFileSync(path.join(repoRoot, "note/frontres_core/contracts/README.md"), "utf8");
const review = JSON.parse(fs.readFileSync("../../runtime/04_frontres_design_inspector.data.json", "utf8"));

if (!viewer.includes("function renderInspector(data)")
 || !viewer.includes('layout === "design_transaction_inspector"')) {
throw new Error("viewer does not route the shared Transaction inspector");
}
if (!viewer.includes("wrapText(card.title, x + 13, y + 44")) {
throw new Error("Transaction inspector does not render the compact design index");
}
if (!viewer.includes("const detailRows = Array.isArray(activeCard.details)")) {
 throw new Error("Transaction inspector does not render the selected detail reading card");
}
if (!viewer.includes("const textLines = measuredLineCount(row?.text || \"\", detailTextWidth")
  || !viewer.includes("detailFormulasFor(row)")
  || !viewer.includes("detailTableFor(row)")
  || !viewer.includes("addDetailTable(table")
  || !viewer.includes("formulas.length * detailFormulaHeight")
  || !viewer.includes("const detailFormulaHeight = 76")
  || viewer.includes("charWidth: 7.4, maxLines: 1")) {
 throw new Error("Design Inspector must size detail rows from complete text, tables and formulas");
}
if (!viewer.includes('layout === "module_test_inspector") ? 0.8 : 0.65')) {
 throw new Error("Design Inspector must open at a readable 80% zoom instead of auto-fit");
}
if (!viewer.includes('import katex from "./node_modules/katex/dist/katex.mjs";')
  || !viewer.includes("function addLatex(latex, x, y, width, options = {})")) {
 throw new Error("Transaction inspector does not own local KaTeX rendering");
}
if (!viewer.includes('console.error(`[Atlas LaTeX] ${error.message}`, { latex });')
  || !viewer.includes('container.setAttribute("class", "atlas-equation atlas-equation-fallback")')
  || !viewer.includes("container.textContent = latex")) {
 throw new Error("Transaction inspector does not fail readable on malformed LaTeX");
}
if (viewer.includes("const selectionTop = selectorTop")) {
throw new Error("Transaction inspector still hides the complete design decision behind selection");
}
if (viewer.includes("function renderDesignSpecificationReview(data)")
  || viewer.includes('layout === "design_specification_review"')
  || viewer.includes("function renderDesignPointReview(data)")
  || viewer.includes('layout === "design_point_review"')) {
 throw new Error("obsolete long-card design-review route is still reachable");
}
if (review.layout !== "design_transaction_inspector" || !Array.isArray(review.cards)
  || !review.transaction || !Array.isArray(review.transaction.steps)) {
 throw new Error("Transaction inspector schema is invalid");
}
if (Object.keys(review).sort().join(",") !== "cards,defaultZoom,layout,subtitle,title,transaction") {
 throw new Error("Atlas 04 root must contain only the index and shared Transaction spine");
}
if (review.defaultZoom !== 0.8) {
 throw new Error("Atlas 04 must preserve the approved 80% default zoom");
}

const registryRows = [...registry.matchAll(
 /\| `(FRS-DP-\d+)` \| ([^|]+) \| ([^|]+) \| `([^`]+)` \|/g,
)].map((match) => ({
 designId: match[1],
 title: match[2].trim(),
 blockId: match[4],
}));
const cardsById = new Map(review.cards.map((card) => [card.designId, card]));
if (registryRows.length !== 10 || cardsById.size !== registryRows.length) {
 throw new Error(`Transaction index/register mismatch cards=${cardsById.size} registry=${registryRows.length}`);
}

const perturbationDetails = cardsById.get("FRS-DP-01").details;
const expectedPerturbationHeadings = [
  "1. 扰动对象",
  "2. 扰动时序",
  "3. 固定情境",
  "4. GMT 扰动边界与 Probing",
  "5. DR 与 K 的关系",
  "6. Segment 选择",
  "7. 四类训练分布",
  "8. 为什么恢复四档",
];
if (perturbationDetails.length !== expectedPerturbationHeadings.length
  || perturbationDetails.some((detail, index) => detail.heading !== expectedPerturbationHeadings[index])) {
  throw new Error("Perturbation Data must preserve the eight human-approved details in order");
}
const strengthRows = perturbationDetails[6].table?.rows ?? [];
if (strengthRows.map((row) => row[0]).join(",") !== "Easy,Medium,Hard,Broken tail") {
  throw new Error("Perturbation Data must preserve the restored four-class strength distribution");
}

const allowedCardKeys = "blockId,chips,color,designId,details,highlightSteps,responsibility,title";
for (const row of registryRows) {
 const card = cardsById.get(row.designId);
 if (!card) throw new Error(`missing registered design point ${row.designId}`);
 if (card.blockId !== row.blockId || card.title !== row.title) {
  throw new Error(`${row.designId} title/block mapping drift`);
 }
 if (Object.keys(card).sort().join(",") !== allowedCardKeys) {
  throw new Error(`${row.designId} contains non-Transaction card fields`);
 }
 if (typeof card.responsibility !== "string" || !card.responsibility.trim()) {
  throw new Error(`${row.designId} missing concise Transaction responsibility`);
 }
 if (!Array.isArray(card.highlightSteps) || !card.highlightSteps.length) {
  throw new Error(`${row.designId} must highlight at least one Transaction step`);
 }
 if (!Array.isArray(card.chips) || card.chips.length !== 0) {
  throw new Error(`${row.designId} must keep parameter chips empty; details belong in the bottom reading card`);
 }
 const validDetail = (detail) => {
 if (typeof detail === "string") return Boolean(detail.trim());
  if (!detail || typeof detail !== "object") return false;
  const allowedDetailKeys = new Set(["heading", "latex", "table", "text"]);
  if (Object.keys(detail).some((key) => !allowedDetailKeys.has(key))) return false;
  if (typeof detail.text !== "string" || !detail.text.trim()) return false;
  if ("heading" in detail && (typeof detail.heading !== "string" || !detail.heading.trim())) return false;
  if ("table" in detail) {
   const table = detail.table;
   if (!table || Object.keys(table).sort().join(",") !== "columns,rows"
     || !Array.isArray(table.columns) || table.columns.length !== 3
     || !table.columns.every((value) => typeof value === "string" && Boolean(value.trim()))
     || !Array.isArray(table.rows) || !table.rows.length
     || table.rows.some((row) => !Array.isArray(row) || row.length !== table.columns.length
       || row.some((value) => typeof value !== "string" || !value.trim()))) return false;
  }
  if (!("latex" in detail)) return true;
  if (typeof detail.latex === "string") return Boolean(detail.latex.trim());
  return Array.isArray(detail.latex) && detail.latex.length > 0
   && detail.latex.every((formula) => typeof formula === "string" && Boolean(formula.trim()));
 };
 if (!Array.isArray(card.details) || card.details.length < 4 || card.details.length > 8
   || card.details.some((detail) => !validDetail(detail))) {
  throw new Error(`${row.designId} must expose four to eight atomic detail decisions`);
 }
}

const cardTitles = review.cards.map((card) => card.title);
if (new Set(cardTitles).size !== cardTitles.length) {
 throw new Error("Transaction index contains duplicate parent design points");
}

const expectedStepIds = [
 "init-hsl",
 "resolve-km",
 "select-segments",
 "seal-scenarios",
 "restore-xt",
 "sample-attempts",
 "frontres-action",
 "gmt-execute-k",
 "paired-evidence",
 "build-objectives",
 "seal-policy-rows",
 "grouped-update",
 "commit-state",
];
const actualStepIds = review.transaction.steps.map((step) => step.id);
if (actualStepIds.join(",") !== expectedStepIds.join(",")) {
 throw new Error(`Transaction step order drift: ${actualStepIds.join(" -> ")}`);
}
const stepIdSet = new Set(actualStepIds);
for (const step of review.transaction.steps) {
 if (Object.keys(step).sort().join(",") !== "detail,id,label,phase") {
  throw new Error(`${step.id} contains fields outside the concise Transaction grammar`);
 }
 if (!new Set(["pre", "collection", "update"]).has(step.phase)) {
  throw new Error(`${step.id} has invalid Transaction phase ${step.phase}`);
 }
 if (!step.label?.trim() || !step.detail?.trim()) {
  throw new Error(`${step.id} must contain one action label and one concise boundary detail`);
 }
}
for (const card of review.cards) {
 for (const stepId of card.highlightSteps) {
  if (!stepIdSet.has(stepId)) throw new Error(`${card.designId} highlights unknown step ${stepId}`);
 }
}

const segmentReplay = cardsById.get("FRS-DP-02");
for (const requiredStep of [
 "select-segments", "seal-scenarios", "restore-xt", "sample-attempts",
 "seal-policy-rows", "grouped-update",
]) {
 if (!segmentReplay.highlightSteps.includes(requiredStep)) {
  throw new Error(`Segment Replay does not highlight ${requiredStep}`);
  }
}
const expectedSegmentReplayHeadings = [
  "Transaction：完整收集后原子提交",
  "Segment 数量：方法允许 N_segment > 1，当前取 2",
  "每个 Segment 单独封存一个 scenario",
  "同一 Segment 的尝试从相同 x_t 开始",
  "exact M 次尝试只改变一次 Repair 动作",
  "全部有效尝试共同进入 PPO",
  "Gain 评分，Segment Replay 组织排序",
  "当前一次更新封存 2 x M 条 policy row",
];
if (segmentReplay.details.length !== expectedSegmentReplayHeadings.length
  || segmentReplay.details.some((detail, index) => detail.heading !== expectedSegmentReplayHeadings[index])) {
  throw new Error("Segment Replay details must preserve the reviewed atomic order");
}
const kStepDetails = cardsById.get("FRS-DP-03").details;
const expectedKStepHeadings = [
  "训练顺序：K8/M4 → K16/M4 → K32/M4",
  "K 延长后果，M 增加候选",
  "K64 当前不启用",
  "同一 Transaction 使用相同 K/M",
  "K 不是未来输入帧数",
  "一个 attempt 始终只有一条 PPO row",
  "每个 Segment 使用 1 Clean + 1 Noisy + M Repair",
  "双层 Curriculum：外层推进 K，内层重新推进 DR",
];
if (kStepDetails.length !== expectedKStepHeadings.length
  || kStepDetails.some((detail, index) => detail.heading !== expectedKStepHeadings[index])) {
  throw new Error("K-step Curriculum details must preserve the reviewed atomic order");
}
const pairedDetails = cardsById.get("FRS-DP-06").details;
const expectedPairedHeadings = [
 "Clean Rollout：定义正确动作",
 "Noisy Rollout：定义不修复零点",
 "Repair Rollout：记录一次修复的 K-step 后果",
 "Clean 与 Noisy 只执行一次",
 "只有 Repair 进入 PPO",
 "共享 baseline 减少可避免噪声",
];
if (pairedDetails.length !== expectedPairedHeadings.length
 || pairedDetails.some((detail, index) => detail.heading !== expectedPairedHeadings[index])) {
 throw new Error("Paired Rollouts must preserve the six human-approved details in order");
}
const transactionText = JSON.stringify(review);
for (const required of [
 "同一 Clean replay state x_t",
 "冻结的 pi_old",
 "exact M 个修复动作",
 "FrontRES 在 t 仅输出一次 Delta SE(3)",
 "冻结 FrontRES，由 GMT 执行 K 步",
 "收集期间不执行 optimizer update",
 "完整封存后执行一次 grouped update",
 "全局 schedule 固定为 K8/M4、K16/M4、K32/M4",
 "K64 当前未激活",
 "928D", "158D", "770D", "q29[t+1]", "q29[t+2]",
"Future Motion Context",
"未来窗口固定为 t+1 与 t+2 两帧",
"每帧只提取 29D 内部关节 Intent，共 58D",
"两帧均来自同一条 sealed Noisy/deployment reference",
"封存 2 x M 条 policy row 后只执行一次 grouped scalar update",
"固定位置权重 tau_k=k/K",
"能区分正在恢复和正在恶化",
"每个 z_j 就是一项归一化后的 r_j",
"避免自由平均掩盖明显坏姿态或支撑错误",
"每个 sealed scenario 只执行一次 Clean Rollout",
"每个 sealed scenario 只执行一次 zero-action Noisy Rollout",
"先让每条动作序列拥有相同发言权",
"不能因为某组数据行更多就压倒其他组",
"不使用 winner-only、argmax 或 best-of-M 权重",
"所有有效 Segment 始终保留非零采样概率",
"能够从缓存直接恢复 x_t 的 Segment 不因起始帧靠后而被降权",
"平移与旋转都解释为 world-frame residual",
"向上 dz 不做硬裁剪",
"首轮 bounded live test 使用 beta_init=0.02",
"只有‘恢复收益更高，同时动作也更大’的同 Segment trade-off pair",
"live test 不自动修改 beta",
"Critic 不判断哪一个动作更好",
"Critic input 为当前 289D privileged observation、同一 future Intent 58D 与动作前支撑上下文 102D，共 449D",
"先分别计算 U(G_m)",
"HSL 只初始化 proposal Actor",
"Critic-only 保持 Actor 与固定 std 不变",
"Actor-ramp 再逐步增加 Actor loss weight",
"EMA target std 只调节 Critic loss 梯度",
"固定 symlog 先把 raw G_total 映射为 Actor/Critic 共同预测的 robust utility",
"分别计算并裁剪各自的 gradient norm",
"checkpoint-v13 不能 resume",
"每个 K 都拥有一轮独立的 DR Curriculum",
"降低 DR 后重新进入 critic-only",
"不再建立独立 Physics projection",
]) {
 if (!transactionText.includes(required)) throw new Error(`Transaction inspector missing exact fact: ${required}`);
}
const warmupDetails = cardsById.get("FRS-DP-09").details;
const expectedWarmupHeadings = [
"Critic 的职责：预测状态难度，不预测指定动作",
"阶段轮次：每个 K 先校准 Critic，再逐步释放 Actor",
"Critic 输入：347D 原状态 + 102D 动作前支撑条件",
"Critic target：先逐 attempt 变换，再取 exact-M 平均",
"Critic-only 与 Actor-ramp：参考线稳定后再释放 Actor",
"固定 utility 与自适应 loss scale 分工",
"独立梯度裁剪：Critic 误差不能压缩 Actor",
"Recalibrate 与 checkpoint：新数值状态必须冷启动",
];
if (warmupDetails.length !== expectedWarmupHeadings.length
 || warmupDetails.some((detail, index) => detail.heading !== expectedWarmupHeadings[index])) {
 throw new Error("Actor & Critic Warmup must preserve the eight human-readable titled explanations in order");
}
const warmupText = JSON.stringify(warmupDetails);
const warmupSchedule = warmupDetails[1].table;
const expectedWarmupSchedule = [
 ["K8 / M4", "Critic-only", "200"],
 ["K8 / M4", "Actor-ramp", "500"],
 ["K8 / M4", "Joint Optimize", "1300"],
 ["K16 / M4", "Critic-only", "300"],
 ["K16 / M4", "Actor-ramp", "300"],
 ["K16 / M4", "Joint Optimize", "900"],
 ["K32 / M4", "Critic-only", "400"],
 ["K32 / M4", "Actor-ramp", "300"],
 ["K32 / M4", "Joint Optimize", "625"],
];
if (!warmupSchedule || JSON.stringify(warmupSchedule.rows) !== JSON.stringify(expectedWarmupSchedule)) {
 throw new Error("Actor & Critic Warmup must expose the active per-K Critic-only, Actor-ramp, and Joint iteration counts");
}
for (const required of [
"Critic-only → Actor-ramp → Joint Optimize",
"Actor LR = 3e-6",
"Critic LR = 1e-5",
"同一个 Adam",
"checkpoint-v13",
"至少为 1",
"EMA target std 只调节 Critic loss 梯度",
"exact-one commit 后提交",
"V(s)",
"6D Repair action 不进入 Critic",
"449D Critic",
"mean_m U(G_m)",
]) {
 if (!warmupText.includes(required)) {
  throw new Error(`Actor & Critic Warmup missing phase/LR fact: ${required}`);
 }
}
for (const forbidden of ["M 只表示", "Multi-Critic", "checkpoint identity"]) {
 if (warmupText.includes(forbidden)) {
  throw new Error(`Actor & Critic Warmup must keep maintenance-only detail out of Atlas: ${forbidden}`);
 }
}
for (const forbidden of ["新的 G_total", "当前环境数量为 4 x M", "S_j 初始范围"]) {
 if (transactionText.includes(forbidden)) {
  throw new Error(`Transaction inspector retains superseded wording: ${forbidden}`);
 }
}
if (!viewer.includes('addText(String(index + 1), left + 22, rowY')) {
 throw new Error("Design Inspector must render natural row numbering");
}
const conceptFigure = JSON.parse(fs.readFileSync(path.join(repoRoot, "note/architecture/concept/03_frontres_concept_tabs.data.json"), "utf8"));
const warmupNode = conceptFigure.nodes.find((node) => node.id === "M-05");
if (!warmupNode || warmupNode.summary !== "每次 K 增长先重新校准同一 Critic, 再释放 Actor") {
 throw new Error("Concept Figure must bind Critic recalibration to K growth rather than a K x M curriculum");
}
if (!conceptFigure.edges.some((edge) => edge.from === "M-06" && edge.to === "M-05" && edge.label === "换 K 后重校准")) {
 throw new Error("Concept Figure must preserve the K-step Curriculum to Critic recalibration interaction");
}
if (viewer.includes('addText("设计索引"')) {
 throw new Error("Transaction inspector must not render the redundant 设计索引 label");
}
for (const forbidden of [
 "implementation", "evidenceLevel", "openRisk", "reviewQuestion", "specSections",
 "mappingGaps", "discrepancies", "sourceHref",
]) {
 if (transactionText.includes(`\"${forbidden}\"`)) {
  throw new Error(`Transaction inspector still exposes ${forbidden}`);
 }
}
if (cardsById.get("FRS-DP-08").highlightSteps.join(",") !== "init-hsl") {
 throw new Error("HSL must remain a pre-Transaction initializer");
}
if (!cardsById.get("FRS-DP-10").highlightSteps.includes("frontres-action")) {
throw new Error("two-frame Noisy internal Intent must attach to the one action-sampling step");
}
if (!cardsById.get("FRS-DP-10").highlightSteps.includes("seal-scenarios")) {
throw new Error("two-frame Noisy internal Intent must be sealed before actor consumption");
}
if (!cardsById.get("FRS-DP-10").highlightSteps.includes("grouped-update")) {
throw new Error("two-frame Noisy internal Intent must condition the shared state-value update");
}

const repairGainDetails = cardsById.get("FRS-DP-07").details;
const expectedRepairGainHeadings = [
"Intent、Physics 与 Segment Replay",
"Recovery-Aware 分类规则",
"Clean 锚点与归一化",
"K-step 时间方向累积",
"Intent 项",
"Physics 项",
"组内有限补偿、Recovery 压力与 Total",
"β 初值与 live 校准",
];
if (repairGainDetails.length !== expectedRepairGainHeadings.length
 || repairGainDetails.some((detail, index) => detail?.heading !== expectedRepairGainHeadings[index])) {
 throw new Error("Repair Gain must expose the approved eight-section derivation order");
}
const intentTable = repairGainDetails.find((detail) => detail.heading === "Intent 项")?.table;
const physicsTable = repairGainDetails.find((detail) => detail.heading === "Physics 项")?.table;
if (intentTable?.rows.length !== 6 || physicsTable?.rows.length !== 4) {
 throw new Error("Repair Gain must expose the complete Intent and Physics evidence tables");
}
for (const required of [
 "Root orientation", "Joint pose", "Key-body pose", "Linear velocity",
 "Angular velocity", "Root height", "Contact phase", "Support-foot drift",
 "Phase-ZMP", "Survival",
]) {
 if (!JSON.stringify([intentTable, physicsTable]).includes(required)) {
  throw new Error(`Repair Gain evidence tables missing ${required}`);
 }
}
const fixedScaleText = JSON.stringify([intentTable, physicsTable]);
for (const required of [
 "0.087 rad（5°）", "0.10 m", "0.75 m/s", "2.0 rad/s", "0.05 m",
 "0.10 exposure", "0.03 m", "0.02 m", "0.10 horizon fraction",
]) {
 if (!fixedScaleText.includes(required)) {
  throw new Error(`Repair Gain fixed semantic scales missing ${required}`);
 }
}
const repairGainFormulaRows = repairGainDetails.flatMap(detail => (
 Array.isArray(detail?.latex) ? detail.latex : (typeof detail?.latex === "string" ? [detail.latex] : [])
));
if (repairGainFormulaRows.length !== 12) {
 throw new Error(`Repair Gain must expose exactly twelve independent LaTeX lines, got ${repairGainFormulaRows.length}`);
}
const repairGainLatex = repairGainFormulaRows.join("\n");
for (const requiredLatex of [
"r_j", "D_j^{\\rightarrow}", "\\tau_k", "\\mathcal{M}", "I_X", "P_X",
"G_I^{(m)}", "G_P^{(m)}", "P_N", "P_R^{(m)}",
"\\lambda_{RA}^{(m)}", "C_{\\mathrm{repair}}^{(m)}", "\\Delta t^{(m)}",
"0.10\\,\\mathrm m", "\\Delta\\theta^{(m)}", "5^\\circ",
"G_{\\mathrm{total}}^{(m)}", "-\\beta C_{\\mathrm{repair}}^{(m)}",
"R^{(m)}", "\\beta_{\\mathrm{init}}=0.02", "\\beta_{ab}^{\\star}",
]) {
 if (!repairGainLatex.includes(requiredLatex)) {
  throw new Error(`Repair Gain LaTeX missing ${requiredLatex}`);
 }
}
for (const retiredLatex of [
"\\max_j", "\\eta", "\\begin{aligned}", "G_{P,j}^{(m)}",
"\\sum_j\\lambda_{RA,j}^{(m)}",
]) {
 if (repairGainLatex.includes(retiredLatex)) {
  throw new Error(`Repair Gain LaTeX still exposes retired term ${retiredLatex}`);
 }
}

const launcher = fs.readFileSync(path.join(repoRoot, "note/architecture/open_atlas.command"), "utf8");
const server = fs.readFileSync("serve_architecture.mjs", "utf8");
if (!launcher.includes("/healthz") || !launcher.includes("02_frontres_design_inspector.html")) {
 throw new Error("durable Atlas launcher does not own health/readiness and Atlas 04");
}
if (!server.includes('requestUrl.pathname === "/healthz"') || !server.includes('service: "mosaic-frontres-atlas"')) {
 throw new Error("Atlas server health identity is missing");
}
if (!server.includes('resolved !== atlasRoot') || !server.includes('`${atlasRoot}${path.sep}`')) {
 throw new Error("Atlas static-file owner must reject sibling-prefix path traversal");
}

console.log(`design_inspector OK design_points=${review.cards.length} steps=${review.transaction.steps.length}`);
