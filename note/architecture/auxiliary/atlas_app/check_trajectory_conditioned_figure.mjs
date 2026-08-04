import fs from "node:fs";

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
  "ICA-T-01",
  "ICA-T-02",
  "ICA-T-03",
  "ICA-T-04",
  "ICA-T-05",
  "ICA-T-06",
  "ICA-T-07",
  "ICA-T-08",
  "ICA-T-09",
  "ICA-D-01",
  "ICA-D-02",
  "ICA-D-03",
  "ICA-D-04",
  "ICA-D-05",
  "ICA-D-06",
];
const nodes = new Map((data.nodes || []).map((node) => [node.id, node]));

for (const id of requiredNodes) {
  if (!nodes.has(id)) throw new Error(`missing node ${id}`);
}

for (const node of nodes.values()) {
  for (const key of ["x", "y", "w", "h"]) {
    if (!Number.isFinite(node[key])) throw new Error(`${node.id} has invalid ${key}`);
  }
  if (!node.summary) throw new Error(`${node.id} must include one concise design statement`);
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

const requiredEdges = [
  "ICA-T-01->ICA-T-02",
  "ICA-T-02->ICA-T-03",
  "ICA-T-02->ICA-T-09",
  "ICA-T-03->ICA-T-04",
  "ICA-T-04->ICA-T-05",
  "ICA-T-05->ICA-T-06",
  "ICA-T-09->ICA-T-06",
  "ICA-T-06->ICA-T-07",
  "ICA-T-07->ICA-T-08",
  "ICA-T-08->ICA-T-04",
  "ICA-T-08->ICA-T-06",
  "ICA-T-08->ICA-D-02",
  "ICA-D-01->ICA-D-02",
  "ICA-D-02->ICA-D-03",
  "ICA-D-03->ICA-D-04",
  "ICA-D-06->ICA-D-04",
  "ICA-D-04->ICA-D-05",
];
const edgePairs = new Set((data.edges || []).map((edge) => `${edge.from}->${edge.to}`));
for (const pair of requiredEdges) {
  if (!edgePairs.has(pair)) throw new Error(`missing interaction ${pair}`);
}

console.log(`trajectory-conditioned figure ok; nodes=${nodes.size} edges=${routes.length}`);
