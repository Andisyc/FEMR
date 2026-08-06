import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../../../..");
const dataPath = path.resolve(here, "../../architecture/02_code_quality_evidence.data.json");
const rendererPath = path.resolve(here, "layouts/code_quality_evidence.js");
const viewerPath = path.resolve(here, "architecture_atlas.html");
const data = JSON.parse(fs.readFileSync(dataPath, "utf8"));
const renderer = fs.readFileSync(rendererPath, "utf8");
const viewer = fs.readFileSync(viewerPath, "utf8");
new vm.Script(renderer.replace(/^export\s+/gm, ""), { filename: "code_quality_evidence.js" });

if (data.layout !== "code_quality_evidence_atlas") throw new Error("invalid code quality atlas layout");
if (!Array.isArray(data.stages) || data.stages.length !== 7) throw new Error("expected seven training stages");
if (!Array.isArray(data.modules) || data.modules.length === 0) throw new Error("missing scanned modules");
if (!viewer.includes("renderCodeQualityEvidenceAtlas")) throw new Error("viewer does not route the new layout");
if (!renderer.includes("drawStages") || !renderer.includes("drawFunctionTree")) {
  throw new Error("renderer must preserve the stage spine and one function tree");
}
if (!renderer.includes("fileName(fn.sourcePath)") || !renderer.includes("treeContentHeight")) {
  throw new Error("function list must expose real source-file boundaries and content height");
}
if (!renderer.includes("BLOCK_TOP_GAP") || !renderer.includes("↳ ${block.id}")) {
  throw new Error("every B block must render as a visually separate row");
}
if (renderer.includes("clipped(fn.purpose") || !renderer.includes("functionPurposeLabel")) {
  throw new Error("Chinese function-purpose labels must render without clipping");
}
if (!renderer.includes("FUNCTION_COLUMN_COUNT = 2") || !renderer.includes("functionColumnLayout")) {
  throw new Error("function reading card must use two measured balanced columns");
}
if (!renderer.includes("drawReviewLayer") || !renderer.includes("Review 标注层")) {
 throw new Error("review judgments must render in a separate atlas layer");
}
if (
 !renderer.includes("drawChainOverview")
 || !renderer.includes("Evaluation 链路索引")
 || !renderer.includes("点击链路只强调对应分区, 其他函数仍保留")
 || !renderer.includes("chainMuted")
) {
 throw new Error("02 must add non-hiding Evaluation chain context above the Review layer");
}
for (const rendererFact of [
 "activeFunctionProjection",
 "drawProjectionToggle",
 "functionProjectionSections",
 "按链路",
 "按文件",
 "共享函数",
 "未归链函数",
 "守恒",
]) {
 if (!renderer.includes(rendererFact)) {
  throw new Error(`02 dual Evaluation projection is missing ${rendererFact}`);
 }
}
if (!renderer.includes("finding.functionNames") || !renderer.includes("fileName(finding.sourcePath)")) {
  throw new Error("every review row must expose its exact file and function targets");
}
if (!renderer.includes("functionStatusLabel") || !renderer.includes("fn.annotationClass")) {
  throw new Error("every function row must expose its annotation-review class");
}
const moduleIds = new Set(data.modules.map((module) => module.id));
for (const stage of data.stages) {
  if (!stage.moduleIds.length) throw new Error(`${stage.id} has no modules`);
  for (const id of stage.moduleIds) if (!moduleIds.has(id)) throw new Error(`${stage.id} references ${id}`);
}
let functionCount = 0;
let blockCount = 0;
for (const module of data.modules) {
  if (!Array.isArray(module.functions)) throw new Error(`${module.id} functions missing`);
  const findings = module.reviewState?.findings || [];
  const findingIds = new Set();
  for (const finding of findings) {
    if (!finding.id || findingIds.has(finding.id)) throw new Error(`${module.id} has duplicate review finding ids`);
    findingIds.add(finding.id);
    if (!['open', 'resolved', 'stale', 'accepted'].includes(finding.currentStatus)) {
      throw new Error(`${finding.id} has invalid current review status`);
    }
    if (!finding.basicType || !finding.title || !finding.sourcePath || !finding.sourceHref) {
      throw new Error(`${finding.id} has an incomplete source-linked review record`);
    }
  }
  for (const fn of module.functions) {
    functionCount += 1;
    const sourcePath = path.resolve(repoRoot, fn.sourcePath);
    if (!fs.statSync(sourcePath).isFile()) throw new Error(`missing source ${fn.sourcePath}`);
    if (!(Number.isInteger(fn.sourceLine) && fn.sourceLine > 0)) throw new Error(`invalid line for ${fn.name}`);
    if (!fn.sourceHref.includes(encodeURIComponent(fn.sourcePath))) throw new Error(`invalid href for ${fn.name}`);
    if (!fn.purpose) throw new Error(`missing purpose state for ${fn.name}`);
    if (!["annotated", "trivial", "legacy", "candidate"].includes(fn.annotationClass)) {
      throw new Error(`${fn.name} has an invalid annotation-review class`);
    }
    if (fn.annotationClass === "trivial" && !["字段代理", "派生指标", "薄接口", "纯工具"].includes(fn.simpleKind)) {
      throw new Error(`${fn.name} has no auditable simple-function kind`);
    }
    for (const reviewRef of fn.reviewRefs || []) {
      if (!findingIds.has(reviewRef)) throw new Error(`${fn.name} references missing review finding ${reviewRef}`);
    }
    for (const block of fn.blocks || []) {
      blockCount += 1;
      if (!/^B\d+$/.test(block.id) || !block.purpose) throw new Error(`invalid block in ${fn.name}`);
    }
  }
}
if (functionCount !== data.scan.functionOccurrences || blockCount !== data.scan.blockOccurrences) {
  throw new Error("scan totals diverge from rendered evidence");
}
const evaluation = data.modules.find((module) => module.id === "MOD-EVAL");
if (!evaluation || (evaluation.reviewState?.findings || []).length !== 0) {
  throw new Error("Evaluation Atlas must contain only current unresolved findings");
}
if (evaluation.functions.some((fn) => (fn.reviewRefs || []).length !== 0)) {
 throw new Error("Evaluation function tree retained references to removed findings");
}
if ((evaluation.evaluationChains || []).map((chain) => chain.id).join("") !== "ABC") {
 throw new Error("02 must project the three Evaluation capabilities from the 01 source registry");
}
const evaluationFunctions = new Map(
 evaluation.functions.map((fn) => [`${fn.sourcePath}::${fn.name}`, fn]),
);
for (const chain of evaluation.evaluationChains) {
 if (!Array.isArray(chain.ownedFiles) || !chain.ownedFiles.length) {
  throw new Error(`Evaluation chain ${chain.id} lost its owned-file projection`);
 }
 for (const ref of chain.functions) {
  const fn = evaluationFunctions.get(`${ref.sourcePath}::${ref.name}`);
  if (fn) {
   if (!(fn.chainIds || []).includes(chain.id)) {
    throw new Error(`Evaluation chain ${chain.id} lost function ${ref.sourcePath}::${ref.name}`);
   }
  } else {
   // Chain 可引用其他 module 的 owner; 只验证 source projection, 不伪造 Evaluation ownership.
   const sourcePath = path.resolve(repoRoot, ref.sourcePath);
   if (!fs.existsSync(sourcePath) || !(ref.sourceLine > 0) || !ref.sourceHref) {
    throw new Error(`Evaluation chain ${chain.id} lost external function ${ref.sourcePath}::${ref.name}`);
   }
   const declaration = fs.readFileSync(sourcePath, "utf8").split(/\r?\n/)[ref.sourceLine - 1] || "";
   const shortName = ref.name.split(".").at(-1);
   if (!declaration.includes(`def ${shortName}(`)) {
    throw new Error(`Evaluation chain ${chain.id} external source link drifted: ${ref.sourcePath}::${ref.name}`);
   }
  }
 }
 const assignedCount = evaluation.functions.filter((fn) => fn.chainIds.includes(chain.id)).length;
 if (chain.assignedFunctionCount !== assignedCount) {
  throw new Error(`Evaluation chain ${chain.id} assigned count drifted`);
 }
 for (const ownedFile of chain.ownedFiles) {
  const ownedFunctions = evaluation.functions.filter((fn) => fn.sourcePath === ownedFile);
  if (!ownedFunctions.length || ownedFunctions.some((fn) => !fn.chainIds.includes(chain.id))) {
   throw new Error(`Evaluation chain ${chain.id} did not classify every function in ${ownedFile}`);
  }
 }
}
if (evaluation.functions.some((fn) => !Array.isArray(fn.chainIds))) {
 throw new Error("every scanned Evaluation function must expose its chain-assignment fact");
}
if (!evaluation.functions.some((fn) => fn.chainIds.length > 1)) {
 throw new Error("shared Evaluation helpers must remain visibly multi-chain");
}
const exclusiveCount = evaluation.functions.filter((fn) => fn.chainIds.length === 1).length;
const sharedCount = evaluation.functions.filter((fn) => fn.chainIds.length > 1).length;
const unassignedCount = evaluation.functions.filter((fn) => fn.chainIds.length === 0).length;
if (exclusiveCount + sharedCount + unassignedCount !== evaluation.functions.length) {
 throw new Error("Evaluation chain/shared/unassigned sections do not conserve the full function inventory");
}
const evaluationClasses = Object.fromEntries(
  ["annotated", "trivial", "legacy", "candidate"].map((name) => [
    name,
    evaluation.functions.filter((fn) => fn.annotationClass === name).length,
  ]),
);
if (evaluationClasses.candidate !== 0) {
  throw new Error(`Evaluation still has ${evaluationClasses.candidate} unreviewed annotation candidates`);
}
const classifiedEvaluationCount = Object.values(evaluationClasses).reduce((sum, count) => sum + count, 0);
if (
  classifiedEvaluationCount !== evaluation.functions.length
  || evaluationClasses.annotated === 0
  || evaluationClasses.trivial === 0
) {
  throw new Error(`Evaluation annotation coverage drifted: ${JSON.stringify(evaluationClasses)}`);
}
console.log(
  `code_quality_evidence_atlas OK modules=${data.modules.length} `
  + `unique_functions=${data.scan.uniqueFunctions} occurrences=${functionCount} `
  + `unique_blocks=${data.scan.uniqueBlocks} block_occurrences=${blockCount}`,
);
