import fs from "node:fs";
import vm from "node:vm";

const html = fs.readFileSync("architecture_atlas.html", "utf8");
for (const shape of ["encoder", "decoder", "context_encoder"]) {
  if (!html.includes(`node.shape === "${shape}"`)) {
    throw new Error(`method_figure renderer is missing shape=${shape}`);
  }
}
for (const marker of ["const centeredTrackerText", "anchor: \"middle\"", "data.layerDivider", "stroke-dasharray"]) {
  if (!html.includes(marker)) throw new Error(`method_figure renderer is missing ${marker}`);
}
const scriptMatch = html.match(/<script type="module">([\s\S]*?)<\/script>/);
if (!scriptMatch) throw new Error("viewer module script is missing");
new vm.Script(scriptMatch[1].replace(/^\s*import .*$/gm, ""));

const data = JSON.parse(
  fs.readFileSync("../../concept/08_trajectory_conditioned_execution_alignment.data.json", "utf8"),
);

if (data.layout !== "method_figure") {
  throw new Error(`expected layout=method_figure, got ${data.layout}`);
}

for (const field of ["zones", "callouts", "acceptance"]) {
  if ((data[field] || []).length !== 0) {
    throw new Error(`Concept Figure must not render ${field}`);
  }
}

for (const field of ["claim", "subtitle"]) {
  if (field in data) throw new Error(`Concept Figure must not include ${field}`);
}

const requiredNodes = [
  "ICA3-T-01",
  "ICA3-T-02",
  "ICA3-T-03",
  "ICA3-T-04",
  "ICA3-T-05",
  "ICA3-T-06",
  "ICA3-T-09",
  "ICA3-T-10",
  "ICA3-T-11",
  "ICA3-D-01",
  "ICA3-D-02",
  "ICA3-D-03",
  "ICA3-D-04",
  "ICA3-D-05",
];
const nodes = new Map((data.nodes || []).map((node) => [node.id, node]));

for (const id of requiredNodes) {
  if (!nodes.has(id)) throw new Error(`missing node ${id}`);
}
if (nodes.size !== requiredNodes.length) {
  throw new Error("Concept Figure contains unexpected or retired nodes");
}
if (nodes.has("ICA3-T-07")) {
  throw new Error("Support and Query must remain merged into one Rollout node");
}
if (
  nodes.get("ICA3-T-05")?.title !== "Rollout" ||
  !nodes.get("ICA3-T-05")?.summary?.includes("第一次轨迹作为 Context")
) {
  throw new Error("Rollout node must explain that the first trajectory becomes Context");
}

const dividers = data.layerDividers || (data.layerDivider ? [data.layerDivider] : []);
if (dividers.length !== 1) {
  throw new Error("Concept Figure must contain one labeled horizontal layer divider");
}
for (const divider of dividers) {
  if (
    !Number.isFinite(divider.x1) ||
    !Number.isFinite(divider.x2) ||
    !Number.isFinite(divider.y) ||
    divider.x1 >= divider.x2
  ) {
    throw new Error("Concept Figure contains an invalid horizontal layer divider");
  }
}
const [demoDivider] = dividers;
if (
  demoDivider.topLabel !== "Context-Conditioned Tracker Training" ||
  demoDivider.bottomLabel !== "Controlled Real-World Proof of Concept"
) {
  throw new Error("Concept Figure layer divider labels do not match the two figure layers");
}

const requiredShapes = new Map([
  ["ICA3-T-02", "encoder"],
  ["ICA3-T-03", "decoder"],
  ["ICA3-T-06", "context_encoder"],
]);
for (const [id, shape] of requiredShapes) {
  if (nodes.get(id)?.shape !== shape) throw new Error(`${id} must use shape=${shape}`);
}

for (const node of nodes.values()) {
  for (const key of ["x", "y", "w", "h"]) {
    if (!Number.isFinite(node[key])) throw new Error(`${node.id} has invalid ${key}`);
  }
  if (!node.summary) throw new Error(`${node.id} must include one concise design statement`);
  if (node.shape && !["encoder", "decoder", "context_encoder"].includes(node.shape)) {
    throw new Error(`${node.id} has unsupported shape ${node.shape}`);
  }
  for (const forbidden of ["owner", "status", "codeRefs"]) {
    if (forbidden in node) throw new Error(`${node.id} includes forbidden field ${forbidden}`);
  }
  if (/[。.!！?？；;]$/.test(String(node.summary).trim())) {
    throw new Error(`${node.id} summary ends with sentence punctuation`);
  }
}

const nodeList = [...nodes.values()];
for (let i = 0; i < nodeList.length; i += 1) {
  for (let j = i + 1; j < nodeList.length; j += 1) {
    const a = nodeList[i];
    const b = nodeList[j];
    const overlap =
      a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
    if (overlap) throw new Error(`nodes overlap: ${a.id} and ${b.id}`);
  }
}

function anchor(node, side, offset = 0) {
  const delta = Number.isFinite(Number(offset)) ? Number(offset) : 0;
  if (side === "left") return [node.x, node.y + node.h / 2 + delta];
  if (side === "right") return [node.x + node.w, node.y + node.h / 2 + delta];
  if (side === "top") return [node.x + node.w / 2 + delta, node.y];
  if (side === "bottom") return [node.x + node.w / 2 + delta, node.y + node.h];
  throw new Error(`unsupported anchor ${side}`);
}

function edgePoints(edge) {
  const from = nodes.get(edge.from);
  const to = nodes.get(edge.to);
  if (!from) throw new Error(`edge source missing: ${edge.from}`);
  if (!to) throw new Error(`edge target missing: ${edge.to}`);
  const dx = to.x + to.w / 2 - (from.x + from.w / 2);
  const dy = to.y + to.h / 2 - (from.y + from.h / 2);
  const fromAnchor =
    edge.fromAnchor || (Math.abs(dx) >= Math.abs(dy) ? (dx >= 0 ? "right" : "left") : dy >= 0 ? "bottom" : "top");
  const toAnchor =
    edge.toAnchor || (Math.abs(dx) >= Math.abs(dy) ? (dx >= 0 ? "left" : "right") : dy >= 0 ? "top" : "bottom");
  return [
    anchor(from, fromAnchor, edge.fromOffset),
    ...(edge.via || []),
    anchor(to, toAnchor, edge.toOffset),
  ];
}

function segmentHitsRect(a, b, node, inset = 2) {
  const minX = node.x + inset;
  const maxX = node.x + node.w - inset;
  const minY = node.y + inset;
  const maxY = node.y + node.h - inset;
  let t0 = 0;
  let t1 = 1;
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  for (const [p, q] of [
    [-dx, a[0] - minX],
    [dx, maxX - a[0]],
    [-dy, a[1] - minY],
    [dy, maxY - a[1]],
  ]) {
    if (p === 0 && q < 0) return false;
    if (p !== 0) {
      const ratio = q / p;
      if (p < 0) t0 = Math.max(t0, ratio);
      else t1 = Math.min(t1, ratio);
      if (t0 > t1) return false;
    }
  }
  return true;
}

const routes = (data.edges || []).map((edge) => ({ edge, points: edgePoints(edge) }));
for (const { edge, points } of routes) {
  if (edge.label && /[。.!！?？；;]$/.test(String(edge.label).trim())) {
    throw new Error(`${edge.from}->${edge.to} label ends with sentence punctuation`);
  }
  for (const node of nodeList) {
    if (node.id === edge.from || node.id === edge.to) continue;
    for (let index = 1; index < points.length; index += 1) {
      if (segmentHitsRect(points[index - 1], points[index], node)) {
        throw new Error(`${edge.from}->${edge.to} crosses block ${node.id}`);
      }
    }
  }
}

function collinearOverlapLength(a, b, c, d) {
  const ab = [b[0] - a[0], b[1] - a[1]];
  const cd = [d[0] - c[0], d[1] - c[1]];
  const cross = (u, v) => u[0] * v[1] - u[1] * v[0];
  const offset = [c[0] - a[0], c[1] - a[1]];
  if (Math.abs(cross(ab, cd)) > 1e-6 || Math.abs(cross(ab, offset)) > 1e-6) return 0;
  const axis = Math.abs(ab[0]) >= Math.abs(ab[1]) ? 0 : 1;
  const first = [Math.min(a[axis], b[axis]), Math.max(a[axis], b[axis])];
  const second = [Math.min(c[axis], d[axis]), Math.max(c[axis], d[axis])];
  return Math.max(0, Math.min(first[1], second[1]) - Math.max(first[0], second[0]));
}

for (let firstIndex = 0; firstIndex < routes.length; firstIndex += 1) {
  for (let secondIndex = firstIndex + 1; secondIndex < routes.length; secondIndex += 1) {
    const first = routes[firstIndex];
    const second = routes[secondIndex];
    for (let a = 1; a < first.points.length; a += 1) {
      for (let b = 1; b < second.points.length; b += 1) {
        if (
          collinearOverlapLength(
            first.points[a - 1],
            first.points[a],
            second.points[b - 1],
            second.points[b],
          ) > 2
        ) {
          throw new Error(
            `connectors overlap: ${first.edge.from}->${first.edge.to} and ${second.edge.from}->${second.edge.to}`,
          );
        }
      }
    }
  }
}

const requiredEdges = [
  "ICA3-T-01->ICA3-T-02",
  "ICA3-T-02->ICA3-T-11",
  "ICA3-T-11->ICA3-T-03",
  "ICA3-T-03->ICA3-T-04",
  "ICA3-T-04->ICA3-T-05",
  "ICA3-T-05->ICA3-T-06",
  "ICA3-T-06->ICA3-T-11",
  "ICA3-T-05->ICA3-T-09",
  "ICA3-T-10->ICA3-T-09",
  "ICA3-T-09->ICA3-T-06",
  "ICA3-T-09->ICA3-T-02",
  "ICA3-T-09->ICA3-T-03",
  "ICA3-D-01->ICA3-D-02",
  "ICA3-D-02->ICA3-D-03",
  "ICA3-D-03->ICA3-D-04",
  "ICA3-D-04->ICA3-D-05",
];
const edgePairs = new Set((data.edges || []).map((edge) => `${edge.from}->${edge.to}`));
for (const pair of requiredEdges) {
  if (!edgePairs.has(pair)) throw new Error(`missing interaction ${pair}`);
}
if (routes.length !== requiredEdges.length) {
  throw new Error(`Concept Figure contains unexpected interactions: ${routes.length}`);
}
if (edgePairs.has("ICA3-T-01->ICA3-T-07")) {
  throw new Error("Planner-to-Query connector is redundant and must remain removed");
}
if ([...edgePairs].some((pair) => pair.includes("ICA3-T-07"))) {
  throw new Error("retired Query Rollout must not retain connectors");
}

function route(from, to) {
  const match = routes.find(({ edge }) => edge.from === from && edge.to === to);
  if (!match) throw new Error(`missing route ${from}->${to}`);
  return match;
}

function isHorizontal(a, b) {
  return Math.abs(a[1] - b[1]) < 1e-6;
}

function isVertical(a, b) {
  return Math.abs(a[0] - b[0]) < 1e-6;
}

const contextRoute = route("ICA3-T-06", "ICA3-T-11");
if (contextRoute.points.length !== 2) {
  throw new Error("Context-to-Latent-Add must be a single straight connector");
}
if (contextRoute.edge.label !== "Δz") {
  throw new Error("Context Encoder must supply calibration latent Δz");
}
if (contextRoute.edge.labelSize < 14) {
  throw new Error("Delta-z label must remain visually prominent");
}

const nominalRoute = route("ICA3-T-02", "ICA3-T-11");
const calibratedRoute = route("ICA3-T-11", "ICA3-T-03");
if (nominalRoute.edge.label !== "z" || calibratedRoute.edge.label !== "z + Δz") {
  throw new Error("Tracker bottleneck must explicitly render z + Δz latent calibration");
}
if (nominalRoute.edge.labelSize < 14 || calibratedRoute.edge.labelSize < 14) {
  throw new Error("Tracker latent labels must remain visually prominent");
}
if (nodes.get("ICA3-T-11")?.summarySize < 14) {
  throw new Error("Latent Add summary must remain visually prominent");
}

const supportRoute = route("ICA3-T-05", "ICA3-T-06");
const supportEnd = supportRoute.points.at(-1);
const supportBeforeEnd = supportRoute.points.at(-2);
if (!isHorizontal(supportBeforeEnd, supportEnd) || supportEnd[0] >= supportBeforeEnd[0]) {
  throw new Error("Support-to-Context must point leftward into Context Encoder");
}

const rolloutRoute = route("ICA3-T-05", "ICA3-T-09");
if (rolloutRoute.points.length !== 2) {
  throw new Error("Rollout-to-Learning must be a single straight connector");
}

const executionRoute = route("ICA3-T-04", "ICA3-T-05");
const executionSource = nodes.get("ICA3-T-04");
const executionTarget = nodes.get("ICA3-T-05");
if (
  executionRoute.edge.label !== "执行" ||
  executionRoute.edge.labelX <= executionSource.x + executionSource.w ||
  executionRoute.edge.labelX >= executionTarget.x
) {
  throw new Error("Execution label must remain centered in the gap between its endpoint blocks");
}

const learning = nodes.get("ICA3-T-09");
const correctTrajectory = nodes.get("ICA3-T-10");
const mainAxisIds = [
  "ICA3-T-01",
  "ICA3-T-02",
  "ICA3-T-11",
  "ICA3-T-03",
  "ICA3-T-04",
  "ICA3-T-05",
  "ICA3-T-09",
];
const mainAxisGaps = mainAxisIds.slice(1).map((id, index) => {
  const previous = nodes.get(mainAxisIds[index]);
  const current = nodes.get(id);
  return current.x - (previous.x + previous.w);
});
if (Math.max(...mainAxisGaps) - Math.min(...mainAxisGaps) > 10) {
  throw new Error(`main-axis spacing must remain even: ${mainAxisGaps.join(",")}`);
}

const lowerIds = ["ICA3-T-01", "ICA3-T-02", "ICA3-T-11", "ICA3-T-03", "ICA3-T-04", "ICA3-T-05", "ICA3-T-06", "ICA3-T-09", "ICA3-T-10"];
const demoIds = ["ICA3-D-01", "ICA3-D-02", "ICA3-D-03", "ICA3-D-04", "ICA3-D-05"];
if (lowerIds.some((id) => nodes.get(id).y < 0 || nodes.get(id).y + nodes.get(id).h >= demoDivider.y)) {
  throw new Error("Tracker training nodes must remain above the dashed divider");
}
if (demoIds.some((id) => nodes.get(id).y <= demoDivider.y)) {
  throw new Error("Controlled proof-of-concept nodes must remain below the second dashed divider");
}

const demoAxisGaps = demoIds.slice(1).map((id, index) => {
  const previous = nodes.get(demoIds[index]);
  const current = nodes.get(id);
  return current.x - (previous.x + previous.w);
});
if (Math.max(...demoAxisGaps) - Math.min(...demoAxisGaps) > 10) {
  throw new Error(`demo-axis spacing must remain even: ${demoAxisGaps.join(",")}`);
}

if (correctTrajectory.title !== "Correct Trajectory") {
  throw new Error("Calibration supervision must come from Correct Trajectory");
}
if (correctTrajectory.x + correctTrajectory.w / 2 !== learning.x + learning.w / 2) {
  throw new Error("Correct Trajectory must sit directly above Calibration Learning");
}
const correctTrajectoryRoute = route("ICA3-T-10", "ICA3-T-09");
if (
  correctTrajectoryRoute.edge.label !== "无缺陷轨迹目标" ||
  correctTrajectoryRoute.points.length !== 2 ||
  !isVertical(correctTrajectoryRoute.points[0], correctTrajectoryRoute.points[1])
) {
  throw new Error("Correct Trajectory must connect straight down to Calibration Learning");
}

console.log(`in-context execution calibration figure ok; nodes=${nodes.size} edges=${routes.length}`);
