import fs from "node:fs";
import { renderModuleInspector } from "./layouts/module_inspector.js";

const repoMap = JSON.parse(fs.readFileSync("../../architecture/01_repo_architecture.data.json", "utf8"));
const indexHtml = fs.readFileSync("../../index.html", "utf8");
const wrapperHtml = fs.readFileSync("../../03_femr_module_inspector.html", "utf8");
const viewerSource = fs.readFileSync("architecture_atlas.html", "utf8");
const rendererSource = fs.readFileSync("layouts/module_inspector.js", "utf8");
const inspector = repoMap.moduleInspector;

if (typeof renderModuleInspector !== "function") {
  throw new Error("Module Inspector must export renderModuleInspector()");
}
if (!inspector || !Array.isArray(inspector.stages) || inspector.stages.length !== 7) {
  throw new Error("repository atlas must expose exactly seven familiar training stages");
}
const expectedTitles = [
  "Train Entrypoint & Config",
  "Data Loader & Scenario",
  "Environment & Observation",
  "Network Forward",
  "K-step Rollout & Evidence",
  "Loss, Backprop & Update",
  "Checkpoint & Evaluation",
];
if (inspector.stages.map((stage) => stage.title).join("|") !== expectedTitles.join("|")) {
  throw new Error("Module Inspector must preserve the agreed deep-learning vocabulary");
}

const modules = (repoMap.systems || []).flatMap((system) => system.modules || []);
const moduleIds = modules.map((module) => module.id);
const moduleIdSet = new Set(moduleIds);
if (moduleIdSet.size !== moduleIds.length) throw new Error("repository module registry contains duplicate ids");
if (!moduleIdSet.has(inspector.defaultModuleId)) throw new Error("defaultModuleId must select one registered module");

const stageIds = inspector.stages.map((stage) => stage.id);
if (new Set(stageIds).size !== stageIds.length) throw new Error("Module Inspector stage ids must be unique");
for (const stage of inspector.stages) {
  const allowedKeys = ["color", "id", "interfaceOut", "moduleIds", "responsibility", "title"];
  if (Object.keys(stage).sort().join(",") !== allowedKeys.sort().join(",")) {
    throw new Error(`${stage.id} must reference the owner registry instead of duplicating code details`);
  }
  if (!stage.moduleIds.length || stage.moduleIds.some((id) => !moduleIdSet.has(id))) {
    throw new Error(`${stage.id} references a missing module`);
  }
}
const assignedRuntimeIds = inspector.stages.flatMap((stage) => stage.moduleIds);
if (new Set(assignedRuntimeIds).size !== assignedRuntimeIds.length) {
  throw new Error("a runtime module is assigned to more than one Training Main Loop stage");
}
if (assignedRuntimeIds.slice().sort().join(",") !== repoMap.runtimeOrder.slice().sort().join(",")) {
  throw new Error("Module Inspector stages must cover every runtime module exactly once");
}
if ((inspector.supportModuleIds || []).slice().sort().join(",") !== repoMap.supportOrder.slice().sort().join(",")) {
  throw new Error("Module Inspector support tabs must cover every supporting module exactly once");
}
for (const module of modules) {
  if (!(module.mainRoute || []).length || module.mainRoute.length !== (module.mainRouteTitles || []).length) {
    throw new Error(`${module.id} needs an aligned function-chain route`);
  }
}

const policyModule = modules.find((module) => module.id === "MOD-POLICY");
const correctionModule = modules.find((module) => module.id === "MOD-CORRECTION");
const evaluationModule = modules.find((module) => module.id === "MOD-EVAL");
const correctionFunctions = correctionModule?.files
  ?.find((file) => file.path.endsWith("task_space_correction.py"))?.functions || [];
if (!policyModule?.mainRoute?.[0]?.includes("update_distribution()")) {
  throw new Error("Network Forward B1 must map distribution construction to update_distribution()");
}
for (const requiredFunction of [
  "apply_frontres_task_corrections()",
  "_frontres_contact_consistent_position_correction()",
  "_write_frontres_command_correction()",
]) {
  if (!correctionFunctions.includes(requiredFunction)) {
    throw new Error(`Task-Space Correction function chain is missing ${requiredFunction}`);
  }
}
const evaluationChains = evaluationModule?.evaluationChains || [];
if (evaluationChains.map((chain) => chain.id).join("") !== "ABC") {
 throw new Error("Evaluation must expose exactly the three accepted A-C capabilities");
}
const evaluationFiles = new Set((evaluationModule.files || []).map((file) => file.path));
for (const chain of evaluationChains) {
 if (!['current', 'validation', 'legacy'].includes(chain.status) || !chain.statusLabel || !chain.purpose) {
  throw new Error(`Evaluation chain ${chain.id} has an incomplete human-readable status`);
 }
 if (!chain.functions.length) throw new Error(`Evaluation chain ${chain.id} has no ordered functions`);
 if (!Array.isArray(chain.ownedFiles) || !chain.ownedFiles.length) {
  throw new Error(`Evaluation chain ${chain.id} must declare the files whose internal helpers belong to it`);
 }
 for (const ownedFile of chain.ownedFiles) {
  if (!evaluationFiles.has(ownedFile)) {
   throw new Error(`Evaluation chain ${chain.id} owns an undeclared file ${ownedFile}`);
  }
 }
 for (const ref of chain.functions) {
  if (!evaluationFiles.has(ref.sourcePath) || !ref.name) {
   throw new Error(`Evaluation chain ${chain.id} references an undeclared owner function`);
  }
 }
}

if (!indexHtml.includes("03 FEMR Module Inspector") || indexHtml.includes("03A FEMR Module Architecture")) {
 throw new Error("Architecture index must expose one consolidated 03 Module Inspector entrypoint");
}
if (
 !indexHtml.includes('./03_femr_module_inspector.html')
 || !wrapperHtml.includes("03 FEMR Module Inspector")
 || !wrapperHtml.includes("03_femr_module_inspector.html")
 || !wrapperHtml.includes("../../architecture/01_repo_architecture.data.json")
) {
 throw new Error("03 Module Inspector must expose a stable wrapper beside 04 Code Quality Evidence Atlas");
}
if (!viewerSource.includes("Boolean(data.moduleInspector) && viewMode !== \"repository_reading\"")) {
 throw new Error("03 must open Module Inspector by default and preserve an explicit wide-view escape hatch");
}
for (const requiredRendererFact of [
  "Module Index",
  "Training Main Loop",
  "Selected Module",
  "drawModuleIndex",
  "drawTrainingSpine",
 "drawModuleDetail",
 "drawEvaluationChainDetail",
 "moduleDetailHeight",
 "routeCodeDetails(module)",
  'file ? "file" : "owner"',
  'addText("fn"',
]) {
  if (!rendererSource.includes(requiredRendererFact)) {
    throw new Error(`Module Inspector renderer is missing ${requiredRendererFact}`);
  }
}

const trainingSpineCall = rendererSource.lastIndexOf("drawTrainingSpine(drawing");
const moduleIndexCall = rendererSource.lastIndexOf("drawModuleIndex(drawing");
if (trainingSpineCall < 0 || moduleIndexCall < 0 || trainingSpineCall > moduleIndexCall) {
  throw new Error("Training Main Loop must render above Module Index");
}
if (rendererSource.includes('addText("input"') || rendererSource.includes('addText("output"')) {
  throw new Error("selected module function cards must not render input/output fields");
}
if (!rendererSource.includes("Math.ceil(details.length / columns)")) {
 throw new Error("long function chains must wrap instead of shrinking the canvas typography");
}
if (!rendererSource.includes("module.evaluationChains") || !rendererSource.includes("evaluationChainRowHeight")) {
 throw new Error("Evaluation must render ordered file/function chains instead of the mixed generic B route");
}

console.log(`module_inspector OK stages=${inspector.stages.length} runtime_modules=${assignedRuntimeIds.length} support_modules=${inspector.supportModuleIds.length}`);
