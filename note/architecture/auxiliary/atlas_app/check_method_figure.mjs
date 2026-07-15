import fs from "node:fs";
import vm from "node:vm";


const html = fs.readFileSync("architecture_atlas.html", "utf8");
const data = JSON.parse(fs.readFileSync("../../concept/03_frontres_concept_tabs.data.json", "utf8"));

if (data.layout !== "method_figure") {
  throw new Error(`expected layout=method_figure, got ${data.layout}`);
}
if (!html.includes("function renderMethodFigure")) {
  throw new Error("method_figure renderer is missing");
}
const scriptMatch = html.match(/<script type="module">([\s\S]*?)<\/script>/);
if (!scriptMatch) throw new Error("viewer module script is missing");
new vm.Script(scriptMatch[1].replace(/^\s*import rough.*$/m, "const rough = {};"));

const nodes = new Map((data.nodes || []).map((node) => [node.id, node]));

function textUnits(text) {
  return Array.from(String(text || "")).reduce(
    (sum, ch) => sum + (/[\u3400-\u9fff\uff00-\uffef]/.test(ch) ? 1.85 : 1),
    0,
  );
}

function splitLongWord(word, maxUnits) {
  if (textUnits(word) <= maxUnits) return [word];
  const chunks = [];
  let chunk = "";
  for (const ch of Array.from(word)) {
    if (chunk && textUnits(chunk + ch) > maxUnits) {
      chunks.push(chunk);
      chunk = ch;
    } else {
      chunk += ch;
    }
  }
  if (chunk) chunks.push(chunk);
  return chunks;
}

function wrappedLineCount(text, width, charWidth) {
  const maxUnits = Math.max(8, Math.floor(width / charWidth));
  const tokens = String(text || "")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .flatMap((word) => splitLongWord(word, maxUnits));
  const lines = [];
  let line = "";
  for (const token of tokens) {
    const needsSpace = line && /[A-Za-z0-9)]$/.test(line) && /^[A-Za-z0-9(]/.test(token);
    const probe = line ? `${line}${needsSpace ? " " : ""}${token}` : token;
    if (textUnits(probe) > maxUnits && line) {
      lines.push(line);
      line = token;
    } else {
      line = probe;
    }
  }
  if (line) lines.push(line);
  return Math.max(1, lines.length);
}
for (const id of [
  "M-02", "M-03", "M-04", "M-05", "M-06", "M-10",
  "Q-01", "Q-PAIR", "SR-01",
]) {
  if (!nodes.has(id)) throw new Error(`method figure missing node ${id}`);
}
for (const node of nodes.values()) {
  for (const key of ["x", "y", "w", "h"]) {
    if (!Number.isFinite(node[key])) throw new Error(`${node.id} has invalid ${key}`);
  }
  if (!node.summary && node.kind !== "external") {
    throw new Error(`${node.id} must include one concise design statement`);
  }
  for (const forbidden of ["owner", "status", "codeRefs"]) {
    if (forbidden in node) throw new Error(`${node.id} must not include Concept Figure field ${forbidden}`);
  }
  const compact = node.h < 140;
  if (String(node.title).trim().split(/\s+/).length > 4) {
    throw new Error(`${node.id} title exceeds the cognitive-load budget`);
  }
  if (textUnits(node.summary) > 48) {
    throw new Error(`${node.id} summary exceeds the cognitive-load budget`);
  }
  if (/[。.!！?？；;]$/.test(String(node.summary).trim())) {
    throw new Error(`${node.id} summary must not end with sentence punctuation`);
  }
  const titleLines = wrappedLineCount(node.title, node.w - 32, 7);
  if (titleLines > 2) throw new Error(`${node.id} title requires ${titleLines} lines`);
  const summaryLineHeight = compact ? 13 : 15;
  const summaryY = node.y + 60 + Math.max(0, titleLines - 1) * 16;
  const summaryBottom = node.y + node.h - 14;
  const summaryMaxLines = Math.max(1, Math.floor((summaryBottom - summaryY) / summaryLineHeight));
  const summaryLines = wrappedLineCount(node.summary, node.w - 32, compact ? 5.4 : 5.8);
  if (summaryLines > summaryMaxLines) {
    throw new Error(`${node.id} summary requires ${summaryLines} lines but only ${summaryMaxLines} fit`);
  }
}

for (const field of ["zones", "callouts", "acceptance"]) {
  if ((data[field] || []).length !== 0) {
    throw new Error(`Concept Figure must not render ${field}`);
  }
}
for (const field of ["claim", "subtitle"]) {
  if (field in data) throw new Error(`Concept Figure must not include ${field}`);
}

const nodeList = [...nodes.values()];
for (let i = 0; i < nodeList.length; i += 1) {
  for (let j = i + 1; j < nodeList.length; j += 1) {
    const a = nodeList[i];
    const b = nodeList[j];
    const overlap = a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
    if (overlap) throw new Error(`method nodes overlap: ${a.id} and ${b.id}`);
  }
}

function anchor(node, side) {
  if (side === "left") return [node.x, node.y + node.h / 2];
  if (side === "right") return [node.x + node.w, node.y + node.h / 2];
  if (side === "top") return [node.x + node.w / 2, node.y];
  if (side === "bottom") return [node.x + node.w / 2, node.y + node.h];
  return [node.x + node.w / 2, node.y + node.h / 2];
}

function edgeAnchors(from, to, edge) {
  if (edge.fromAnchor || edge.toAnchor) {
    return [anchor(from, edge.fromAnchor || "right"), anchor(to, edge.toAnchor || "left")];
  }
  const dx = to.x + to.w / 2 - (from.x + from.w / 2);
  const dy = to.y + to.h / 2 - (from.y + from.h / 2);
  if (Math.abs(dx) >= Math.abs(dy)) {
    return [anchor(from, dx >= 0 ? "right" : "left"), anchor(to, dx >= 0 ? "left" : "right")];
  }
  return [anchor(from, dy >= 0 ? "bottom" : "top"), anchor(to, dy >= 0 ? "top" : "bottom")];
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
  for (const [p, q] of [[-dx, a[0] - minX], [dx, maxX - a[0]], [-dy, a[1] - minY], [dy, maxY - a[1]]]) {
    if (p === 0 && q < 0) return false;
    if (p !== 0) {
      const r = q / p;
      if (p < 0) t0 = Math.max(t0, r);
      else t1 = Math.min(t1, r);
      if (t0 > t1) return false;
    }
  }
  return true;
}

for (const edge of data.edges || []) {
  if (!nodes.has(edge.from)) throw new Error(`edge source missing: ${edge.from}`);
  if (!nodes.has(edge.to)) throw new Error(`edge target missing: ${edge.to}`);
  if (edge.label && textUnits(edge.label) > 18) {
    throw new Error(`${edge.from}->${edge.to} label exceeds the cognitive-load budget`);
  }
  const from = nodes.get(edge.from);
  const to = nodes.get(edge.to);
  const [start, end] = edgeAnchors(from, to, edge);
  const points = [start, ...(edge.via || []), end];
  for (const node of nodeList) {
    if (node.id === edge.from || node.id === edge.to) continue;
    for (let index = 1; index < points.length; index += 1) {
      if (segmentHitsRect(points[index - 1], points[index], node)) {
        throw new Error(`${edge.from}->${edge.to} crosses block ${node.id}`);
      }
    }
  }
}

const edgePairs = new Set((data.edges || []).map((edge) => `${edge.from}->${edge.to}`));
for (const pair of [
  "M-02->SR-01",
  "M-06->SR-01",
  "SR-01->M-04",
  "M-04->M-10",
  "M-10->Q-PAIR",
  "SR-01->Q-PAIR",
  "Q-PAIR->Q-01",
  "M-03->M-05",
  "M-05->M-04",
  "Q-01->M-04",
  "Q-01->SR-01",
]) {
  if (!edgePairs.has(pair)) throw new Error(`method interaction missing: ${pair}`);
}

console.log(
  `method figure contract ok; nodes=${nodes.size} edges=${(data.edges || []).length}`
);
