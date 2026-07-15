import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const dataPath = path.resolve(here, "../../architecture/01_repo_architecture.data.json");
const data = JSON.parse(fs.readFileSync(dataPath, "utf8"));
const viewerPath = path.resolve(here, "architecture_atlas.html");
const viewer = fs.readFileSync(viewerPath, "utf8");
const scriptMatch = viewer.match(/<script type="module">([\s\S]*?)<\/script>/);
if (!scriptMatch) throw new Error("viewer module script not found");
const parseableScript = scriptMatch[1].replace(/^\s*import rough[^;]+;\s*$/m, "const rough = {};");
new vm.Script(parseableScript, { filename: "architecture_atlas.inline.mjs" });

if (data.layout !== "repository_reading_atlas") {
  throw new Error("repository atlas must use repository_reading_atlas layout");
}
if (!(data.defaultZoom >= 0.5 && data.defaultZoom <= 1.0)) {
  throw new Error("repository atlas defaultZoom must preserve readable text");
}
if (!Array.isArray(data.systems) || data.systems.length < 7) {
  throw new Error("repository atlas must expose the major repository systems");
}

const ids = new Set();
const moduleIds = new Set();
let moduleCount = 0;
for (const system of data.systems) {
  if (!system.id || !system.title || !system.summary || !system.color) {
    throw new Error(`incomplete system: ${JSON.stringify(system)}`);
  }
  if (ids.has(system.id)) throw new Error(`duplicate id: ${system.id}`);
  ids.add(system.id);
  if (!Array.isArray(system.modules) || system.modules.length === 0) {
    throw new Error(`system has no modules: ${system.id}`);
  }
  for (const module of system.modules) {
    moduleCount += 1;
    if (ids.has(module.id)) throw new Error(`duplicate id: ${module.id}`);
    ids.add(module.id);
    moduleIds.add(module.id);
  for (const key of ["title", "summary", "owns", "mustNot", "gap"]) {
   if (!module[key]) throw new Error(`${module.id} missing ${key}`);
}
  if (!Array.isArray(module.mainRoute) || module.mainRoute.length < 2) {
   throw new Error(`${module.id} must expose a multi-step formal mainRoute`);
  }
  if (!Array.isArray(module.mainRouteTitles) || module.mainRouteTitles.length !== module.mainRoute.length) {
   throw new Error(`${module.id} must provide one readable title per mainRoute step`);
  }
  for (const [index, step] of module.mainRoute.entries()) {
   if (!String(step).startsWith(`B${index + 1} `) || !String(step).includes("->")) {
    throw new Error(`${module.id} has invalid mainRoute step ${index + 1}: ${step}`);
   }
  }
  for (const retired of ["upstream", "downstream"]) {
   if (retired in module) throw new Error(`${module.id} retains retired field ${retired}`);
  }
    if (!Array.isArray(module.objects) || module.objects.length === 0) {
      throw new Error(`${module.id} has no semantic objects`);
    }
    if (!Array.isArray(module.files) || module.files.length === 0) {
      throw new Error(`${module.id} has no owner files`);
    }
    if (!Array.isArray(module.review) || module.review.length === 0) {
      throw new Error(`${module.id} has no review questions`);
    }
    if (!Array.isArray(module.tests) || module.tests.length === 0) {
      throw new Error(`${module.id} has no evidence boundary`);
    }
  }
}

if (moduleCount < 15) throw new Error(`expected broad repository coverage, got ${moduleCount} modules`);

const runtimeOrder = data.runtimeOrder || [];
const supportOrder = data.supportOrder || [];
if (new Set(runtimeOrder).size !== runtimeOrder.length) throw new Error("runtimeOrder contains duplicates");
if (new Set(supportOrder).size !== supportOrder.length) throw new Error("supportOrder contains duplicates");
const orderedIds = new Set([...runtimeOrder, ...supportOrder]);
for (const id of moduleIds) {
  if (!orderedIds.has(id)) throw new Error(`module is missing from runtime/support reading order: ${id}`);
}
for (const id of orderedIds) {
  if (!moduleIds.has(id)) throw new Error(`reading order references missing module: ${id}`);
}
if (runtimeOrder.includes("MOD-ENV-REWARD")) {
  throw new Error("GMT Environment Reward is a supporting boundary, not the Segment runtime main path");
}

console.log(`repository_reading_atlas OK systems=${data.systems.length} modules=${moduleCount} width=${data.canvasWidth}`);
