import fs from "node:fs";
import rough from "./node_modules/roughjs/bundled/rough.esm.js";

const html = fs.readFileSync("frontres_concept_tabs.html", "utf8");

if (typeof rough.svg !== "function") {
  throw new Error("roughjs import succeeded but rough.svg is missing");
}

if (!html.includes('import rough from "./node_modules/roughjs/bundled/rough.esm.js";')) {
  throw new Error("frontres_concept_tabs.html does not import local roughjs");
}

if (!html.includes('new EventSource("./events")')) {
  throw new Error("frontres_concept_tabs.html is not wired to the auto-refresh event stream");
}

console.log("roughjs viewer import contract ok");
