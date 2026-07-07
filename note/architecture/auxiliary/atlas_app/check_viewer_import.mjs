import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import rough from "./node_modules/roughjs/bundled/rough.esm.js";

const html = fs.readFileSync("architecture_atlas.html", "utf8");
const repoMap = JSON.parse(fs.readFileSync("../../architecture/01_repo_architecture.data.json", "utf8"));
const flowMap = JSON.parse(fs.readFileSync("../../runtime/02_frontres_flow.data.json", "utf8"));
const conceptTabs = JSON.parse(fs.readFileSync("../../concept/03_frontres_concept_tabs.data.json", "utf8"));
const statAuditMap = JSON.parse(fs.readFileSync("../../runtime/07_frontres_stat_audit.data.json", "utf8"));

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

if (repoMap.layout !== "repo_tree") {
  throw new Error("architecture/01_repo_architecture.data.json must use layout=repo_tree");
}

const repoRoot = path.resolve(process.cwd(), "../../../..");
const checkedRepoPaths = [];
const missingRepoPaths = [];
for (const file of repoMap.files || []) {
  for (const block of file.blocks || []) {
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
if (missingRepoPaths.length) {
  throw new Error(`repo owner paths missing:\n${missingRepoPaths.join("\n")}`);
}

if (flowMap.layout !== "repo_tree" || !Array.isArray(flowMap.files)) {
  throw new Error("runtime/02_frontres_flow.data.json must use layout=repo_tree with files[]");
}

if (conceptTabs.layout !== "flow_tree" || !Array.isArray(conceptTabs.nodes)) {
  throw new Error("concept/03_frontres_concept_tabs.data.json must use layout=flow_tree with nodes[]");
}

if (statAuditMap.layout !== "flow_tree" || !Array.isArray(statAuditMap.nodes)) {
  throw new Error("runtime/07_frontres_stat_audit.data.json must use layout=flow_tree with nodes[]");
}

if (!JSON.stringify(statAuditMap).includes("FRS3-STAT-001")) {
  throw new Error("runtime/07_frontres_stat_audit.data.json must include the semantic stat inventory node");
}

console.log(`roughjs atlas import and data contracts ok; checked ${checkedRepoPaths.length} repo owner paths`);
