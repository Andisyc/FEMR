#!/usr/bin/env python3
"""Build the minimal code-quality evidence atlas from current Python source."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
ARCH_ROOT = APP_DIR.parents[1]
REPO_ROOT = ARCH_ROOT.parents[1]
REGISTRY_PATH = ARCH_ROOT / "architecture" / "01_repo_architecture.data.json"
OUTPUT_PATH = ARCH_ROOT / "architecture" / "02_code_quality_evidence.data.json"
REVIEW_STATE_PATH = ARCH_ROOT / "architecture" / "02_code_quality_review_state.json"
BLOCK_RE = re.compile(r"^\s*#\s*(B\d+)\s*:\s*(.*?)\s*$")


@dataclass(frozen=True)
class SourceFunction:
    name: str
    purpose: str
    source_line: int
    span: int
    branches: int
    blocks: tuple[dict[str, object], ...]
    simple_kind: str


def _short_contract(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    doc = ast.get_docstring(node, clean=True)
    if not doc:
        return "无函数说明"
    first = next((line.strip() for line in doc.splitlines() if line.strip()), "")
    first = re.split(r"(?<=[。.!?])\s+", first, maxsplit=1)[0].strip()
    return first or "无函数说明"


def _blocks(lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    for line_number in range(node.lineno, node.end_lineno or node.lineno):
        match = BLOCK_RE.match(lines[line_number - 1])
        if not match:
            continue
        purpose = match.group(2).strip()
        result.append(
            {
                "id": match.group(1),
                "purpose": purpose or "无代码块说明",
                "sourceLine": line_number,
            }
        )
    return tuple(result)


def _simple_kind(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Classify a reviewed small function by its visible responsibility."""

    statements = list(node.body)
    if statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant):
        statements = statements[1:]
    is_property = any(isinstance(decorator, ast.Name) and decorator.id == "property" for decorator in node.decorator_list)
    if is_property:
        if (
            len(statements) == 1
            and isinstance(statements[0], ast.Return)
            and isinstance(statements[0].value, (ast.Attribute, ast.Subscript, ast.Name, ast.Constant))
        ):
            return "字段代理"
        return "派生指标"
    if len(statements) == 1 and (
        isinstance(statements[0], ast.Return)
        and isinstance(statements[0].value, (ast.Call, ast.Await))
        or isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, (ast.Call, ast.Await))
    ):
        return "薄接口"
    if node.args.args and node.args.args[0].arg in {"self", "cls"}:
        return "薄接口"
    return "纯工具"


class FunctionCollector(ast.NodeVisitor):
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.scope: list[str] = []
        self.functions: list[SourceFunction] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified_name = ".".join([*self.scope, node.name])
        self.functions.append(
            SourceFunction(
                name=qualified_name,
                purpose=_short_contract(node),
                source_line=node.lineno,
                span=(node.end_lineno or node.lineno) - node.lineno + 1,
                branches=sum(
                    isinstance(item, (ast.If, ast.For, ast.While, ast.Try, ast.Match, ast.With))
                    for item in ast.walk(node)
                ),
                blocks=_blocks(self.lines, node),
                simple_kind=_simple_kind(node),
            )
        )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)


def _scan_file(relative_path: str) -> list[SourceFunction]:
    source_path = REPO_ROOT / relative_path
    if source_path.suffix != ".py" or not source_path.is_file():
        return []
    text = source_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(source_path))
    collector = FunctionCollector(lines)
    collector.visit(tree)
    return sorted(collector.functions, key=lambda item: item.source_line)


def _source_href(path: str, line: int) -> str:
    from urllib.parse import quote

    return f"/open-source?path={quote(path, safe='')}&line={line}"


def _git_blob_hash(relative_path: str) -> str:
    payload = (REPO_ROOT / relative_path).read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324 - Git object identity.


def _review_state() -> dict[str, object]:
    if not REVIEW_STATE_PATH.is_file():
        return {"schemaVersion": 1, "modules": {}}
    state = json.loads(REVIEW_STATE_PATH.read_text(encoding="utf-8"))
    if state.get("schemaVersion") != 1 or not isinstance(state.get("modules"), dict):
        raise ValueError("code-quality review state requires schemaVersion=1 and modules")
    return state


def _project_function_chains(
    source_module: dict[str, object],
    functions: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Resolve one architecture-owned chain registry against scanned functions."""

    raw_chains = source_module.get("evaluationChains", [])
    if not raw_chains:
        return []
    by_identity: dict[tuple[str, str], list[dict[str, object]]] = {}
    by_path: dict[str, list[dict[str, object]]] = {}
    for function in functions:
        identity = (str(function["sourcePath"]), str(function["name"]))
        by_identity.setdefault(identity, []).append(function)
        by_path.setdefault(identity[0], []).append(function)
        function["chainIds"] = []

    chain_ids: set[str] = set()
    projected: list[dict[str, object]] = []
    for raw_chain in raw_chains:
        chain_id = str(raw_chain.get("id", ""))
        if not chain_id or chain_id in chain_ids:
            raise ValueError(f"{source_module['id']} requires unique nonempty evaluation chain ids")
        chain_ids.add(chain_id)
        owned_files = [str(path) for path in raw_chain.get("ownedFiles", [])]
        for owned_file in owned_files:
            owned_functions = by_path.get(owned_file, [])
            if not owned_functions:
                raise ValueError(
                    f"{source_module['id']} chain {chain_id} owns an unscanned file: {owned_file}"
                )
            for function in owned_functions:
                if chain_id not in function["chainIds"]:
                    function["chainIds"].append(chain_id)
        refs: list[dict[str, object]] = []
        for raw_ref in raw_chain.get("functions", []):
            identity = (str(raw_ref.get("sourcePath", "")), str(raw_ref.get("name", "")))
            matches = by_identity.get(identity, [])
            if len(matches) != 1:
                raise ValueError(
                    f"{source_module['id']} chain {chain_id} must resolve exactly one scanned function: "
                    f"{identity[0]}::{identity[1]}"
                )
            function = matches[0]
            if chain_id not in function["chainIds"]:
                function["chainIds"].append(chain_id)
            refs.append(
                {
                    "sourcePath": identity[0],
                    "name": identity[1],
                    "sourceLine": function["sourceLine"],
                    "sourceHref": function["sourceHref"],
                }
            )
        assigned_function_count = sum(
            chain_id in function["chainIds"]
            for function in functions
        )
        projected.append(
            {
                **raw_chain,
                "ownedFiles": owned_files,
                "functions": refs,
                "assignedFunctionCount": assigned_function_count,
            }
        )
    return projected


def build() -> dict[str, object]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    review_state = _review_state()
    inspector = registry["moduleInspector"]
    stage_by_module = {
        module_id: stage
        for stage in inspector["stages"]
        for module_id in stage["moduleIds"]
    }
    source_modules = {
        module["id"]: module
        for system in registry["systems"]
        for module in system["modules"]
    }
    modules: list[dict[str, object]] = []
    unresolved_paths: list[str] = []
    for module_id in registry["runtimeOrder"]:
        source_module = source_modules[module_id]
        stage = stage_by_module[module_id]
        functions: list[dict[str, object]] = []
        module_review = review_state["modules"].get(module_id, {})
        findings = module_review.get("findings", []) if isinstance(module_review, dict) else []
        review_refs: dict[tuple[str, str], list[str]] = {}
        normalized_findings: list[dict[str, object]] = []
        for finding in findings:
            if not isinstance(finding, dict):
                raise ValueError(f"{module_id} review finding must be an object")
            source_path = str(finding.get("sourcePath", ""))
            source_blob = str(finding.get("sourceBlob", ""))
            current_blob = _git_blob_hash(source_path) if source_path else ""
            normalized_findings.append(
                {
                    **finding,
                    "currentStatus": (
                        "stale"
                        if source_blob and current_blob and source_blob != current_blob
                        else str(finding.get("status", "open"))
                    ),
                    "sourceHref": _source_href(source_path, int(finding.get("sourceLine", 1))),
                }
            )
            for function_name in finding.get("functionNames", []):
                review_refs.setdefault((source_path, str(function_name)), []).append(str(finding["id"]))
        seen_paths: set[str] = set()
        for file_entry in source_module.get("files", []):
            relative_path = file_entry["path"]
            if relative_path in seen_paths:
                continue
            seen_paths.add(relative_path)
            scanned = _scan_file(relative_path)
            if not scanned:
                unresolved_paths.append(relative_path)
                continue
            if file_entry.get("atlasFunctionScope") == "declared":
                declared_names = {
                    str(name).removesuffix("()")
                    for name in file_entry.get("functions", [])
                }
                scanned = [
                    item
                    for item in scanned
                    if any(
                        item.name == name
                        or item.name.startswith(f"{name}.")
                        or item.name.endswith(f".{name}")
                        for name in declared_names
                    )
                ]
            role = str(file_entry.get("role", "")).lower()
            legacy = (
                "historical" in role
                or role.startswith("explicit legacy")
                or "legacy" in Path(relative_path).stem
            )
            for item in scanned:
                declared_eval_dispatch = (
                    module_id == "MOD-EVAL"
                    and relative_path == "source/rsl_rl/rsl_rl/runners/on_policy_runner.py"
                    and not item.blocks
                )
                annotation_class = (
                    "legacy"
                    if legacy
                    else "annotated"
                    if item.blocks
                    else "trivial"
                    if declared_eval_dispatch or (item.span <= 8 and item.branches <= 1)
                    else "candidate"
                )
                function = {
                    "name": item.name,
                    "purpose": item.purpose,
                    "annotationClass": annotation_class,
                    "simpleKind": (
                        "薄接口"
                        if declared_eval_dispatch
                        else item.simple_kind
                        if annotation_class == "trivial"
                        else None
                    ),
                    "sourcePath": relative_path,
                    "sourceLine": item.source_line,
                    "sourceHref": _source_href(relative_path, item.source_line),
                    "blocks": [],
                    "reviewRefs": review_refs.get((relative_path, item.name), []),
                }
                for block in item.blocks:
                    block_line = int(block["sourceLine"])
                    function["blocks"].append(
                        {
                            **block,
                            "sourceHref": _source_href(relative_path, block_line),
                        }
                    )
                functions.append(function)
        evaluation_chains = _project_function_chains(source_module, functions)
        modules.append(
            {
                "id": module_id,
                "title": source_module["title"],
                "stageId": stage["id"],
                "color": stage["color"],
                "functions": functions,
                "evaluationChains": evaluation_chains,
                "reviewState": {
                    "assessment": module_review.get("assessment", "unreviewed"),
                    "reviewedAt": module_review.get("reviewedAt", ""),
                    "reportPath": module_review.get("reportPath", ""),
                    "findings": normalized_findings,
                },
            }
        )

    unique_functions = {
        (function["sourcePath"], function["sourceLine"], function["name"])
        for module in modules
        for function in module["functions"]
    }
    unique_blocks = {
        (function["sourcePath"], block["sourceLine"], block["id"])
        for module in modules
        for function in module["functions"]
        for block in function["blocks"]
    }
    return {
        "title": "FEMR Code Quality Evidence Atlas",
        "subtitle": "查看全部函数与 B 代码块; Evaluation 可在按链路和按文件两种完整投影间切换",
        "layout": "code_quality_evidence_atlas",
        "defaultStageId": inspector["stages"][0]["id"],
        "defaultModuleId": inspector["stages"][0]["moduleIds"][0],
        "sourceRegistry": str(REGISTRY_PATH.relative_to(REPO_ROOT)),
        "reviewStateRegistry": str(REVIEW_STATE_PATH.relative_to(REPO_ROOT)),
        "stages": [
            {
                "id": stage["id"],
                "title": stage["title"],
                "color": stage["color"],
                "moduleIds": stage["moduleIds"],
            }
            for stage in inspector["stages"]
        ],
        "modules": modules,
        "scan": {
            "pythonFiles": len(
                {
                    function["sourcePath"]
                    for module in modules
                    for function in module["functions"]
                }
            ),
            "uniqueFunctions": len(unique_functions),
            "functionOccurrences": sum(len(module["functions"]) for module in modules),
            "uniqueBlocks": len(unique_blocks),
            "blockOccurrences": sum(
                len(function["blocks"])
                for module in modules
                for function in module["functions"]
            ),
            "unresolvedPaths": sorted(set(unresolved_paths)),
        },
    }


if __name__ == "__main__":
    OUTPUT_PATH.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_PATH.relative_to(REPO_ROOT))
