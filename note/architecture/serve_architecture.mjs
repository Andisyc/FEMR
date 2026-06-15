import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const port = Number(process.env.PORT || 8765);
const watchedFile = path.join(__dirname, "03_frontres_concept_tabs.data.json");
const clients = new Set();

const mimeTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".svg", "image/svg+xml; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
]);

function sendEvent(eventName) {
  for (const client of clients) {
    client.write(`event: ${eventName}\n`);
    client.write(`data: ${Date.now()}\n\n`);
  }
}

function safeResolve(urlPath) {
  const cleanPath = decodeURIComponent(urlPath.split("?")[0]);
  const relativePath = cleanPath === "/" ? "frontres_concept_tabs.html" : cleanPath.slice(1);
  const resolved = path.resolve(__dirname, relativePath);
  if (!resolved.startsWith(__dirname)) return null;
  return resolved;
}

const server = http.createServer((req, res) => {
  if (req.url === "/events") {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    res.write("\n");
    clients.add(res);
    req.on("close", () => clients.delete(res));
    return;
  }

  const filePath = safeResolve(req.url || "/");
  if (!filePath) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.readFile(filePath, (error, data) => {
    if (error) {
      res.writeHead(error.code === "ENOENT" ? 404 : 500);
      res.end(error.code === "ENOENT" ? "Not found" : String(error));
      return;
    }
    res.writeHead(200, {
      "Content-Type": mimeTypes.get(path.extname(filePath)) || "application/octet-stream",
      "Cache-Control": "no-cache",
    });
    res.end(data);
  });
});

fs.watch(watchedFile, { persistent: true }, () => {
  sendEvent("architecture-data");
});

server.listen(port, "127.0.0.1", () => {
  console.log(`FrontRES architecture viewer: http://127.0.0.1:${port}/frontres_concept_tabs.html`);
  console.log(`Watching: ${watchedFile}`);
});
