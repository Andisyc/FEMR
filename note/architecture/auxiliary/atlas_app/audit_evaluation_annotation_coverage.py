"""List Evaluation functions that still require white-box annotation review."""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
ARCHITECTURE_PATH = REPO_ROOT / "note/architecture/architecture/01_repo_architecture.data.json"
BLOCK_RE = re.compile(r"^\s*#\s*B\d+\s*:", re.MULTILINE)


@dataclass(frozen=True)
class FunctionFact:
    path: str
    name: str
    line: int
    span: int
    branches: int
    blocks: int
    legacy: bool

    @property
    def annotation_class(self) -> str:
        if self.legacy:
            return "legacy"
        if self.blocks:
            return "annotated"
        if self.span <= 8 and self.branches <= 1:
            return "trivial"
        return "candidate"


class FunctionCollector(ast.NodeVisitor):
    def __init__(self, *, path: str, text: str, legacy: bool) -> None:
        self.path = path
        self.text = text
        self.lines = text.splitlines()
        self.legacy = legacy
        self.scope: list[str] = []
        self.functions: list[FunctionFact] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        name = ".".join((*self.scope, node.name))
        end = node.end_lineno or node.lineno
        source = "\n".join(self.lines[node.lineno - 1 : end])
        branches = sum(
            isinstance(item, (ast.If, ast.For, ast.While, ast.Try, ast.Match, ast.With))
            for item in ast.walk(node)
        )
        self.functions.append(
            FunctionFact(
                path=self.path,
                name=name,
                line=node.lineno,
                span=end - node.lineno + 1,
                branches=branches,
                blocks=len(BLOCK_RE.findall(source)),
                legacy=self.legacy,
            )
        )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)


def _evaluation_file_entries() -> list[dict[str, object]]:
    architecture = json.loads(ARCHITECTURE_PATH.read_text(encoding="utf-8"))
    for system in architecture.get("systems", []):
        for module in system.get("modules", []):
            if module.get("id") == "MOD-EVAL":
                return list(module.get("files", []))
    raise RuntimeError("MOD-EVAL is missing from the repository architecture")


def _matches_declared(name: str, declared: set[str]) -> bool:
    return any(
        name == item or name.startswith(f"{item}.") or name.endswith(f".{item}")
        for item in declared
    )


def collect() -> list[FunctionFact]:
    facts: list[FunctionFact] = []
    for entry in _evaluation_file_entries():
        relative_path = str(entry["path"])
        path = REPO_ROOT / relative_path
        if path.suffix != ".py" or not path.is_file():
            continue
        role = str(entry.get("role", "")).lower()
        legacy = (
            "historical" in role
            or role.startswith("explicit legacy")
            or "legacy" in Path(relative_path).stem
        )
        collector = FunctionCollector(
            path=relative_path,
            text=path.read_text(encoding="utf-8"),
            legacy=legacy,
        )
        collector.visit(ast.parse(collector.text, filename=str(path)))
        selected = collector.functions
        if entry.get("atlasFunctionScope") == "declared":
            declared = {str(item).removesuffix("()") for item in entry.get("functions", [])}
            selected = [item for item in selected if _matches_declared(item.name, declared)]
        facts.extend(selected)
    return facts


if __name__ == "__main__":
    facts = collect()
    if len(sys.argv) > 1:
        facts = [fact for fact in facts if sys.argv[1] in fact.path]
    counts = {name: 0 for name in ("annotated", "trivial", "legacy", "candidate")}
    for fact in facts:
        counts[fact.annotation_class] += 1
        if fact.annotation_class == "candidate":
            print(
                f"{fact.path}:{fact.line}\t{fact.name}\t"
                f"span={fact.span}\tbranches={fact.branches}\tblocks={fact.blocks}"
            )
    print(
        "SUMMARY "
        + " ".join(f"{name}={counts[name]}" for name in ("annotated", "trivial", "legacy", "candidate"))
    )
