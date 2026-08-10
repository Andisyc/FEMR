import fs from "node:fs";
import path from "node:path";

const appRoot = process.cwd();
const repoRoot = path.resolve(appRoot, "../../../..");
const viewer = fs.readFileSync("architecture_atlas.html", "utf8");
const atlas = JSON.parse(fs.readFileSync("../../testing/05_frontres_module_test_atlas.data.json", "utf8"));
const repoMap = JSON.parse(fs.readFileSync("../../architecture/01_repo_architecture.data.json", "utf8"));
const wrapper = fs.readFileSync(path.join(repoRoot, "note/architecture/05_frontres_module_test_atlas.html"), "utf8");
const index = fs.readFileSync(path.join(repoRoot, "note/architecture/index.html"), "utf8");

if (atlas.layout !== "module_test_inspector" || atlas.defaultZoom !== 0.8) {
 throw new Error("Module Test Atlas must use module_test_inspector at 80% zoom");
}
if (!viewer.includes('layout === "design_transaction_inspector" || layout === "module_test_inspector"')
 || !viewer.includes("const selectorRows = Math.ceil(cards.length / selectorColumns)")
 || !viewer.includes("const stageCards = Array.isArray(data.stageCards) ? data.stageCards : []")
 || !viewer.includes("const phaseTitles = inspectorTransaction.phaseTitles || {}")
 || !viewer.includes('ready: "当前阶段"')
 || !viewer.includes("data.footerText ||")) {
 throw new Error("shared Inspector renderer does not support the Module Test Atlas contract");
}

const expectedSteps = ["purpose", "oracle", "cases", "confirm", "run", "classify", "evidence"];
const actualSteps = atlas.transaction?.steps?.map((step) => step.id) || [];
if (actualSteps.join(",") !== expectedSteps.join(",")) {
 throw new Error(`testing spine drift: ${actualSteps.join(",")}`);
}
for (const phase of ["pre", "collection", "update"]) {
 if (!atlas.transaction.phaseTitles?.[phase]) throw new Error(`missing phase title ${phase}`);
}

if (!Array.isArray(atlas.stageCards) || atlas.stageCards.length !== 1) {
 throw new Error("Module Test Atlas must expose one Formal Runtime Audit stage card");
}
const formalStage = atlas.stageCards[0];
if (formalStage.designId !== "STAGE-02"
 || formalStage.cardKind !== "stage-reading"
 || formalStage.title !== "Formal Runtime Audit"
 || formalStage.executionStatus !== "not-run") {
 throw new Error("Formal Runtime Audit stage identity drift");
}
const formalText = JSON.stringify(formalStage);
for (const required of [
 "Phase A：Method-Code Alignment",
 "Phase B：Formal Runtime Audit",
 "模块自身算错，退回 Module Test",
 "只证明系统按设计运行，不证明策略已经有效",
]) {
 if (!formalText.includes(required)) throw new Error(`Formal Runtime Audit stage card missing: ${required}`);
}

const expectedModuleIds = repoMap.runtimeOrder;
const cardModuleIds = atlas.cards.map((card) => card.blockId);
const outerReplayModuleIds = ["MOD-OUTER-IDENTITY", "MOD-OUTER-SELECTION", "MOD-OUTER-COMMIT", "MOD-OUTER-PERSISTENCE"];
if (atlas.cards.length !== expectedModuleIds.length + outerReplayModuleIds.length
 || new Set(cardModuleIds).size !== atlas.cards.length
 || expectedModuleIds.some((id) => !cardModuleIds.includes(id))) {
 throw new Error(`Module Test Atlas must cover all ${expectedModuleIds.length} runtime modules plus four outer replay boundaries exactly once`);
}
if (outerReplayModuleIds.some((id) => !cardModuleIds.includes(id))) {
 throw new Error("Module Test Atlas is missing an outer replay boundary card");
}
for (const supportId of repoMap.supportOrder) {
 if (cardModuleIds.includes(supportId)) throw new Error(`support module leaked into primary Test Atlas: ${supportId}`);
}

const requiredHeadings = ["要验证的设计规则", "伪样本测试"];
const confirmedPendingCards = new Set();
for (const card of atlas.cards) {
 if (!/^TEST-\d{2}$/.test(card.designId) || !card.title || !card.color) {
  throw new Error(`invalid test card identity for ${card.blockId}`);
 }
 if (card.highlightSteps.join(",") !== "purpose,oracle,cases") {
  throw new Error(`${card.blockId} must keep the human-readable rule/oracle/cases projection`);
 }
const expectedHumanStatus = "confirmed";
const expectedExecutionStatus = confirmedPendingCards.has(card.designId) ? "not-run" : "passed";
 if (card.humanStatus !== expectedHumanStatus) {
  throw new Error(`${card.blockId} humanStatus must be ${expectedHumanStatus}`);
 }
 if (card.executionStatus !== expectedExecutionStatus) {
  throw new Error(`${card.blockId} has invalid executionStatus ${card.executionStatus}`);
 }
 if (typeof card.executionSummary !== "string" || !card.executionSummary.trim()) {
  throw new Error(`${card.blockId} is missing its execution summary`);
 }
 const headings = card.details?.map((detail) => detail.heading) || [];
 if (headings.join(",") !== requiredHeadings.join(",")) {
  throw new Error(`${card.blockId} does not expose the complete human test card`);
 }
const table = card.details.at(-1)?.table;
if (!table || table.columns?.join(",") !== "伪样本,正确结果,证明什么" || !Array.isArray(table.rows) || table.rows.length < 5) {
 throw new Error(`${card.blockId} must expose at least five concrete pseudo-sample tests with independent answers`);
}
for (const [index, row] of table.rows.entries()) {
 if (!Array.isArray(row) || row.length !== 3 || row.some((cell) => typeof cell !== "string" || !cell.trim())) {
  throw new Error(`${card.blockId} invalid pseudo-sample row ${index + 1}`);
 }
}
 const primaryText = JSON.stringify(card.details);
 for (const forbidden of ["sourcePath", "sourceHref", "S0", "S1", "S2", "S3", "S4", "PASS"]) {
  if (primaryText.includes(forbidden)) throw new Error(`${card.blockId} primary card exposes engineering/status token ${forbidden}`);
 }
}

const gainText = JSON.stringify(atlas.cards.find((card) => card.blockId === "MOD-GAIN"));
for (const required of [
"Repair 更接近 Clean 时 Gain 提高",
"失衡状态获得更高 Physics influence",
"Repair cost 增大时 Gain 降低",
"排序不依赖行位置",
"缺失证据不能补零",
]) {
 if (!gainText.includes(required)) throw new Error(`Repair Gain card missing independent case: ${required}`);
}

if (!wrapper.includes("../../testing/05_frontres_module_test_atlas.data.json")
 || !index.includes("05_frontres_module_test_atlas.html")) {
 throw new Error("Module Test Atlas wrapper or index entry is missing");
}

const counts = Object.fromEntries(
 ["passed", "partial", "blocked", "not-run"].map((status) => [
  status,
  atlas.cards.filter((card) => card.executionStatus === status).length,
 ])
);
if (counts.passed !== 22 || counts.partial !== 0 || counts.blocked !== 0 || counts["not-run"] !== 0) {
 throw new Error(`module execution count drift: ${JSON.stringify(counts)}`);
}
for (const token of ["22 passed"]) {
 if (!atlas.subtitle.includes(token)) throw new Error(`Module Test Atlas subtitle missing ${token}`);
}
console.log(
 `module_test_inspector OK cards=${atlas.cards.length} pseudo_cases>=5 `
 + `passed=${counts.passed} partial=${counts.partial} blocked=${counts.blocked} not-run=${counts["not-run"]}`
);
