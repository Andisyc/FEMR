import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import rough from "./node_modules/roughjs/bundled/rough.esm.js";
import { renderModuleInspector } from "./layouts/module_inspector.js";

const html = fs.readFileSync("architecture_atlas.html", "utf8");
const repoMap = JSON.parse(fs.readFileSync("../../architecture/01_repo_architecture.data.json", "utf8"));
const conceptTabs = JSON.parse(fs.readFileSync("../../concept/03_frontres_concept_tabs.data.json", "utf8"));
const designContractReview = JSON.parse(fs.readFileSync("../../runtime/04_frontres_design_inspector.data.json", "utf8"));
const moduleTestAtlas = JSON.parse(fs.readFileSync("../../testing/05_frontres_module_test_atlas.data.json", "utf8"));

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

new vm.Script(scriptMatch[1].replace(/^\s*import .*$/gm, ""));

if (typeof renderModuleInspector !== "function") {
  throw new Error("module_inspector renderer import failed");
}

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
if (!html.includes("function renderInspector(data)")) {
 throw new Error("architecture_atlas.html is missing the shared Inspector renderer");
}
if (!html.includes("layout === \"design_transaction_inspector\"")
  || !html.includes("layout === \"module_test_inspector\"")) {
 throw new Error("architecture_atlas.html does not route design_transaction_inspector");
}
if (designContractReview.layout !== "design_transaction_inspector" || !Array.isArray(designContractReview.cards)
  || !Array.isArray(designContractReview.transaction?.steps)) {
 throw new Error("runtime/04_frontres_design_inspector.data.json must expose cards[] plus one Transaction spine");
}
if (moduleTestAtlas.layout !== "module_test_inspector" || !Array.isArray(moduleTestAtlas.cards)
  || !Array.isArray(moduleTestAtlas.transaction?.steps)) {
 throw new Error("testing/05_frontres_module_test_atlas.data.json must expose cards[] plus one testing spine");
}
if (!html.includes('layout === "repository_reading_atlas"')) {
  throw new Error("architecture_atlas.html does not route repository_reading_atlas");
}
if (
  !html.includes("Boolean(data.moduleInspector)")
  || !html.includes("renderModuleInspector(data")
) {
  throw new Error("architecture_atlas.html does not route 01 through Module Inspector by default");
}
if (
  !html.includes('window.location.protocol === "file:"')
  || !html.includes('http://127.0.0.1:8765/auxiliary/atlas_app/architecture_atlas.html')
) {
  throw new Error("architecture_atlas.html must redirect file opens to the HTTP source-link origin");
}
const serverPath = path.resolve(process.cwd(), "serve_architecture.mjs");
const serverSource = fs.readFileSync(serverPath, "utf8");
if (!serverSource.includes('["--reuse-window", "--goto", gotoTarget]')) {
  throw new Error("source-link server must focus the current VS Code window through --reuse-window --goto");
}
if (!serverSource.includes('if (code === 0)')) {
  throw new Error("source-link server must wait for a successful VS Code CLI exit before returning success");
}

const repoRoot = path.resolve(process.cwd(), "../../../..");
const checkedRepoPaths = [];
const missingRepoPaths = [];
const checkReadingCardSourceLinks = (atlas, atlasLabel) => {
  for (const module of (atlas.systems || []).flatMap((system) => system.modules || [])) {
    for (const block of module.files || []) {
      if (block.path.endsWith("/") || block.path.includes("*")) continue;
      if (!Number.isInteger(block.sourceLine) || block.sourceLine < 1) {
        throw new Error(`${atlasLabel} ${block.id} must include a positive sourceLine`);
      }
      const expectedHref = `/open-source?path=${encodeURIComponent(block.path)}&line=${block.sourceLine}`;
      if (block.sourceHref !== expectedHref) {
        throw new Error(`${atlasLabel} ${block.id} has stale sourceHref`);
      }
      const sourcePath = path.resolve(repoRoot, block.path);
      const sourceLineCount = fs.readFileSync(sourcePath, "utf8").split(/\r?\n/).length;
      if (block.sourceLine > sourceLineCount) {
        throw new Error(`${atlasLabel} ${block.id} sourceLine exceeds ${block.path}`);
      }
    }
  }
};

checkReadingCardSourceLinks(repoMap, "01 repository atlas");
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
