import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import rough from "./node_modules/roughjs/bundled/rough.esm.js";

const html = fs.readFileSync("architecture_atlas.html", "utf8");
const repoMap = JSON.parse(fs.readFileSync("../../architecture/01_repo_architecture.data.json", "utf8"));
const flowMap = JSON.parse(fs.readFileSync("../../runtime/02_frontres_flow.data.json", "utf8"));
const conceptTabs = JSON.parse(fs.readFileSync("../../concept/03_frontres_concept_tabs.data.json", "utf8"));
const formalAuditMap = JSON.parse(fs.readFileSync("../../runtime/04_stage3_formal_runtime_audit.data.json", "utf8"));

if (typeof rough.svg !== "function") {
  throw new Error("roughjs import succeeded but rough.svg is missing");
}

if (!html.includes('import rough from "./node_modules/roughjs/bundled/rough.esm.js";')) {
  throw new Error("architecture_atlas.html does not import local roughjs");
}

if (!html.includes('new EventSource("/events")')) {
  throw new Error("architecture_atlas.html is not wired to the auto-refresh event stream");
}

if (!html.includes('<main id="layout" class="editor-hidden">')) {
  throw new Error("architecture_atlas.html should hide the editor sidebar by default");
}

if (!html.includes('<button id="toggle-editor">Show Editor</button>')) {
  throw new Error("architecture_atlas.html default toggle label should be Show Editor");
}

if (!html.includes("let autoFitWidth = true;")) {
  throw new Error("architecture_atlas.html should auto-fit width by default");
}

const scriptMatch = html.match(/<script type="module">([\s\S]*?)<\/script>/);
if (!scriptMatch) {
  throw new Error("architecture_atlas.html should keep its module script");
}

new vm.Script(scriptMatch[1].replace(/^\s*import rough.*$/m, "const rough = {};"));

if (!html.includes("function measuredLineCount")) {
  throw new Error("architecture_atlas.html should measure wrapped text before drawing cards");
}

if (!html.includes("function flowIdBadgeWidth")) {
  throw new Error("architecture_atlas.html should size flow-tree id badges from id text");
}

if (!html.includes("function renderMethodFigure")) {
  throw new Error("architecture_atlas.html should provide the paper-style method_figure renderer");
}

if (!html.includes("rc.headerBottom")) {
  throw new Error("architecture_atlas.html should use dynamic header height for layout start");
}

if (!html.includes("data.flowAllowTruncate ? data.flowFieldMaxLines || null : null")) {
  throw new Error("architecture_atlas.html should expand flow cards unless truncation is explicitly allowed");
}

for (const requiredId of [
  'id="toggle-editor"',
  'id="zoom-out"',
  'id="zoom-in"',
  'id="zoom-fit"',
  'id="zoom-reset"',
  'id="stage"',
]) {
  if (!html.includes(requiredId)) {
    throw new Error(`architecture_atlas.html is missing viewer control ${requiredId}`);
  }
}

for (const requiredHandler of [
  'stage.addEventListener("wheel"',
  'stage.addEventListener("mousedown"',
  'window.addEventListener("resize"',
  'window.addEventListener("mousemove"',
  'window.addEventListener("mouseup"',
]) {
  if (!html.includes(requiredHandler)) {
    throw new Error(`architecture_atlas.html is missing interaction handler ${requiredHandler}`);
  }
}

if (!html.includes("../../concept/03_frontres_concept_tabs.data.json")) {
  throw new Error("architecture_atlas.html default data path must point to ../../concept/");
}

if (repoMap.layout !== "repository_reading_atlas") {
  throw new Error("architecture/01_repo_architecture.data.json must use layout=repository_reading_atlas");
}
if (!html.includes("function renderRepositoryReadingAtlas(data)")) {
  throw new Error("architecture_atlas.html is missing repository reading renderer");
}
if (!html.includes('layout === "repository_reading_atlas"')) {
  throw new Error("architecture_atlas.html does not route repository_reading_atlas");
}

const repoRoot = path.resolve(process.cwd(), "../../../..");
const checkedRepoPaths = [];
const missingRepoPaths = [];
for (const system of repoMap.systems || []) {
  for (const module of system.modules || []) {
    for (const block of module.files || []) {
    if (typeof block.path !== "string" || block.path.trim() === "") {
      throw new Error(`repo block ${block.id || "(unknown)"} must include a concrete path`);
    }
    const ownerPath = block.path.split("::")[0].trim().replace(/\/$/, "");
    if (ownerPath.includes("*")) {
      continue;
    }
    checkedRepoPaths.push(ownerPath);
    if (!fs.existsSync(path.resolve(repoRoot, ownerPath))) {
      missingRepoPaths.push(`${block.id || "(unknown)"} ${ownerPath}`);
    }
    }
  }
}
if (missingRepoPaths.length) {
  throw new Error(`repo owner paths missing:\n${missingRepoPaths.join("\n")}`);
}

if (flowMap.layout !== "repository_reading_atlas" || !Array.isArray(flowMap.runtimeOrder)) {
  throw new Error("runtime/02_frontres_flow.data.json must use layout=repository_reading_atlas with runtimeOrder[]");
}
for (const requiredId of ["DP-PERTURB", "DP-SEGMENT", "DP-KSTEP", "DP-HSL", "DP-WARMUP", "DP-REPAIR", "DP-GMT", "DP-PAIRED", "DP-GAIN"]) {
  if (!JSON.stringify(flowMap).includes(`\"${requiredId}\"`)) {
    throw new Error(`method-to-code reading atlas is missing design card ${requiredId}`);
  }
}
const flowModules = (flowMap.systems || []).flatMap((system) => system.modules || []);
const flowModuleIds = flowModules.map((module) => module.id);
if (new Set(flowModuleIds).size !== flowModuleIds.length) {
  throw new Error("method-to-code reading atlas contains duplicate module ids");
}
if (
  flowMap.runtimeOrder.length !== flowModuleIds.length
  || flowMap.runtimeOrder.some((id) => !flowModuleIds.includes(id))
) {
  throw new Error("method-to-code runtimeOrder must contain every reading card exactly once");
}
for (const module of flowModules) {
  if (!(module.files || []).length || !(module.objects || []).length) {
    throw new Error(`method-to-code card ${module.id} must include files and core objects`);
  }
  if ((module.mainRoute || []).length !== (module.mainRouteTitles || []).length) {
    throw new Error(`method-to-code card ${module.id} route/title counts differ`);
  }
  for (const [index, route] of module.mainRoute.entries()) {
    if (!route.startsWith(`B${index + 1} `)) {
      throw new Error(`method-to-code card ${module.id} route numbering is not sequential`);
    }
  }
  for (const block of module.files) {
    const ownerPath = block.path.split("::")[0].trim().replace(/\/$/, "");
    if (!fs.existsSync(path.resolve(repoRoot, ownerPath))) {
      throw new Error(`method-to-code owner path missing: ${block.id} ${ownerPath}`);
    }
  }
}

if (
  conceptTabs.layout !== "method_figure"
  || !Array.isArray(conceptTabs.nodes)
  || !Array.isArray(conceptTabs.edges)
  || !Array.isArray(conceptTabs.acceptance)
) {
  throw new Error(
    "concept/03_frontres_concept_tabs.data.json must use layout=method_figure with nodes[], edges[], and acceptance[]"
  );
}

for (const requiredId of ["M-02", "SR-01", "M-06", "M-04", "M-10", "Q-PAIR", "Q-01", "M-03", "M-05"]) {
  if (!JSON.stringify(conceptTabs).includes(`\"${requiredId}\"`)) {
    throw new Error(`method figure is missing active design block ${requiredId}`);
  }
}

if (
  formalAuditMap.layout !== "repository_reading_atlas"
  || !Array.isArray(formalAuditMap.runtimeOrder)
  || !Array.isArray(formalAuditMap.systems)
) {
  throw new Error("runtime/04_stage3_formal_runtime_audit.data.json must use the 01 repository_reading_atlas schema");
}
const formalAuditIds = [
  "AUDIT-ROUTE-01", "AUDIT-PERTURB-01", "AUDIT-PERTURB-02", "AUDIT-SEGDATA-01",
  "AUDIT-SAMPLER-01", "AUDIT-KPLAN-01", "AUDIT-KROLLOUT-01", "AUDIT-OBS-01",
  "AUDIT-ACTION-01", "AUDIT-APPLY-01", "AUDIT-GMT-01", "AUDIT-PAIR-01",
  "AUDIT-PAIR-EVIDENCE-01", "AUDIT-GAIN-01", "AUDIT-RETURN-01", "AUDIT-HSL-LOAD-01",
  "AUDIT-WARMUP-01", "AUDIT-PPO-01", "AUDIT-PERSIST-01", "AUDIT-DIAG-01",
];
for (const requiredId of formalAuditIds) {
  if (!JSON.stringify(formalAuditMap).includes(`\"${requiredId}\"`)) {
    throw new Error(`formal runtime audit atlas is missing ${requiredId}`);
  }
}
if (formalAuditMap.runtimeOrder.join(",") !== formalAuditIds.join(",")) {
  throw new Error("formal runtime audit atlas runtimeOrder must match the official Stage 3 probe order");
}
const formalAuditModules = formalAuditMap.systems.flatMap((system) => system.modules || []);
for (const module of formalAuditModules) {
  if (module.cardKind !== "runtime_probe" || !module.probe?.owner || !module.probe?.insertion) {
    throw new Error(`formal runtime audit card ${module.id} must expose its exact probe owner and insertion location`);
  }
  if (!(module.probe.capture || []).length || !(module.probe.failIf || []).length) {
    throw new Error(`formal runtime audit card ${module.id} must expose captured objects and failure criteria`);
  }
  if (!Array.isArray(module.probeSteps) || module.probeSteps.length !== module.mainRoute.length) {
    throw new Error(`formal runtime audit card ${module.id} must map every B-step to one probe boundary`);
  }
  if (module.probeSteps.some((step) => !step.location || !step.capture || !step.whyHere || !step.failureOwner || !step.sourceHref || !step.sourceLine)) {
    throw new Error(`formal runtime audit card ${module.id} has a B-step without location, capture, rationale, failure owner, or source link`);
  }
  if (module.probeSteps.some((step) => !step.sourceHref.startsWith("/open-source?path=") || !step.sourceHref.includes(`line=${step.sourceLine}`))) {
    throw new Error(`formal runtime audit card ${module.id} has a non-local source link`);
  }
}
const formalAuditWhyHere = formalAuditModules.flatMap((module) => module.probeSteps.map((step) => step.whyHere));
if (formalAuditWhyHere.length !== 60 || new Set(formalAuditWhyHere).size !== 60) {
  throw new Error("formal runtime audit requires 60 non-template whyHere decisions");
}
const atlasServerSource = fs.readFileSync("serve_architecture.mjs", "utf8");
if (
  !atlasServerSource.includes('requestUrl.pathname === "/open-source"')
  || !atlasServerSource.includes('"--goto"')
  || !atlasServerSource.includes("Visual Studio Code.app/Contents/Resources/app/bin/code")
  || atlasServerSource.includes('["-a", "Visual Studio Code", "--args", "--goto"')
) {
  throw new Error("atlas server must open validated source links through VS Code --goto");
}

console.log(`roughjs atlas import and data contracts ok; checked ${checkedRepoPaths.length} repo owner paths`);
