"""
从 AMASS 风格的 npz 数据集中筛选出适合验证实验的动作序列。

两种模式：
  --discover   先扫描所有文件名，输出高频词汇表，帮助确定关键词
  （默认）     按 CATEGORIES 关键词筛选，去除镜像(_reflect)后去重

数据集结构（本项目实测）：
  DATA_ROOT/<子数据集>/<subject>/amass_g1_<类别>-<编号>-<描述>-<subject>_poses.npz
  镜像文件：同名加 _reflect 后缀

用法：
  # Step 1：先看数据集里都有哪些词
  python scripts/filter_motions.py --data_root /hdd0/.../AMASS_G1NPZ_Final --discover

  # Step 2：根据词汇表修改下方 CATEGORIES，再筛选
  python scripts/filter_motions.py --data_root /hdd0/.../AMASS_G1NPZ_Final --save results/motions.txt
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  配置区：根据 --discover 输出的词汇表修改这里
# ─────────────────────────────────────────────────────────────────────────────

CATEGORIES = {
    "行走 (Walking)": [
        r"\bwalk",                   # walk, walking, walked, ...
        r"\bjog", r"\brun",
        r"\bstroll", r"\bmarch",
    ],
    "侧移 (Lateral)": [
        r"\bside",                   # sidestep, sideways, side step, ...
        r"\blateral", r"\bstrafe",
    ],
    "转身 (Turning)": [
        r"\bturn",                   # turn, turning, turned, ...
        r"\bspin", r"\brotate", r"\bpivot",
    ],
    "上肢大幅运动 (Upper-body)": [
        r"\bdance", r"\bwave",
        r"\bgesture", r"\bswing",
        r"\bpunch", r"\bkick", r"\bthrow",
        r"\breach", r"\bclap",
    ],
}

TARGET_PER_CATEGORY = 4   # 每类最多保留几条（不含镜像）
MIN_SEC_DEFAULT     = 3.0
FPS_DEFAULT         = 30.0

# ─────────────────────────────────────────────────────────────────────────────


def _read_num_frames(path: Path) -> int | None:
    try:
        data = np.load(path, mmap_mode="r")
        if "joint_pos" in data:
            return int(data["joint_pos"].shape[0])
        for key in data.files:
            arr = data[key]
            if arr.ndim >= 1:
                return int(arr.shape[0])
    except Exception:
        pass
    return None


def _is_reflect(stem: str) -> bool:
    return stem.endswith("_reflect")


def _base_stem(stem: str) -> str:
    """去掉 _reflect 后缀，得到原始序列名。"""
    return stem[: -len("_reflect")] if _is_reflect(stem) else stem


def _tokenize(stem: str) -> list[str]:
    """
    将文件名分词：按空格、连字符、下划线分割，过滤纯数字和长度 <= 2 的词。
    例：amass_g1_accident-14-dodge turn down-aita_poses
        → ['amass', 'g1', 'accident', 'dodge', 'turn', 'down', 'aita', 'poses']
    """
    tokens = re.split(r"[\s\-_]+", stem.lower())
    return [t for t in tokens if len(t) > 2 and not t.isdigit()]


def _match(stem: str, patterns: list[str]) -> bool:
    name = stem.lower()
    for pat in patterns:
        if re.search(pat, name):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  模式 1：词汇发现
# ─────────────────────────────────────────────────────────────────────────────

def discover(data_root: str, top_n: int = 60) -> None:
    """
    遍历所有文件名，输出最高频词汇，帮助用户确定关键词。
    同时按子数据集分组，显示每个子集的示例文件名。
    """
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"[ERROR] 目录不存在: {root}")

    counter: Counter = Counter()
    subdataset_examples: dict[str, list[str]] = {}
    total = 0

    for path in sorted(root.rglob("*.npz")):
        # 跳过镜像，避免词汇计数翻倍
        if _is_reflect(path.stem):
            continue
        total += 1

        # 子数据集 = data_root 的直接子目录名
        try:
            sub = path.relative_to(root).parts[0]
        except IndexError:
            sub = "."
        if sub not in subdataset_examples:
            subdataset_examples[sub] = []
        if len(subdataset_examples[sub]) < 3:
            subdataset_examples[sub].append(path.name)

        counter.update(_tokenize(path.stem))

    print(f"\n共扫描 {total} 条序列（已排除镜像）\n")

    print("═" * 60)
    print(f"  高频词汇 TOP {top_n}（词 : 出现次数）")
    print("═" * 60)
    for word, cnt in counter.most_common(top_n):
        bar = "█" * min(cnt // max(1, total // 40), 30)
        print(f"  {word:<20} {cnt:>5}  {bar}")

    print("\n═" * 60 + "\n  各子数据集名称及示例文件名\n" + "═" * 60)
    for sub, examples in sorted(subdataset_examples.items()):
        print(f"\n  [{sub}]")
        for ex in examples:
            print(f"    {ex}")

    print("\n[提示] 根据上方词汇表修改脚本中的 CATEGORIES 字典，再运行筛选。")


# ─────────────────────────────────────────────────────────────────────────────
#  模式 2：按关键词筛选
# ─────────────────────────────────────────────────────────────────────────────

def scan(data_root: str, min_sec: float = MIN_SEC_DEFAULT,
         fps: float = FPS_DEFAULT,
         target: int = TARGET_PER_CATEGORY) -> dict[str, list[Path]]:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"[ERROR] 目录不存在: {root}")

    min_frames = int(min_sec * fps)
    results: dict[str, list[Path]] = {cat: [] for cat in CATEGORIES}
    # 用 base_stem 去重，防止同一序列的镜像版本也被选入
    seen: dict[str, set[str]] = {cat: set() for cat in CATEGORIES}
    skipped_short = 0
    total = 0

    for path in sorted(root.rglob("*.npz")):
        total += 1
        stem = path.stem

        # 匹配类别（用 base_stem 做关键词匹配，镜像和原版都能匹配）
        base = _base_stem(stem)
        matched_cat = None
        for cat, patterns in CATEGORIES.items():
            if _match(base, patterns):
                matched_cat = cat
                break
        if matched_cat is None:
            continue

        # 去重：同一 base_stem 只取一条（优先选原版，跳过镜像）
        if base in seen[matched_cat]:
            continue
        if len(results[matched_cat]) >= target:
            continue

        # 时长过滤
        n_frames = _read_num_frames(path)
        if n_frames is not None and n_frames < min_frames:
            skipped_short += 1
            continue

        # 只收录原版（非镜像），镜像留给用户自行决定是否追加
        if _is_reflect(stem):
            continue

        results[matched_cat].append(path)
        seen[matched_cat].add(base)

    print(f"\n扫描完成：共 {total} 个 .npz，"
          f"跳过 {skipped_short} 个过短（< {min_sec:.1f}s @ {fps:.0f}fps）\n")
    return results


def print_results(results: dict[str, list[Path]], data_root: str,
                  fps: float = FPS_DEFAULT) -> None:
    root = Path(data_root).expanduser().resolve()
    any_empty = False
    for cat, paths in results.items():
        print(f"── {cat}  ({len(paths)} 条，已去除镜像) ──")
        if not paths:
            print("  （未找到，请先运行 --discover 查看词汇表）")
            any_empty = True
        for p in paths:
            try:
                rel = p.relative_to(root)
            except ValueError:
                rel = p
            n = _read_num_frames(p)
            dur = f"{n/fps:.1f}s" if n else "?s"
            print(f"  {rel}  [{dur}]")
        print()

    if any_empty:
        print("[提示] 运行 --discover 查看数据集实际词汇，再修改 CATEGORIES。\n")


def save_results(results: dict[str, list[Path]], out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for cat, paths in results.items():
            f.write(f"# {cat}\n")
            for p in paths:
                f.write(f"{p}\n")
            f.write("\n")
    print(f"[已保存] {out}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="筛选验证实验用动作序列")
    parser.add_argument("--data_root", type=str, required=True,
                        help="npz 数据集根目录")
    parser.add_argument("--discover",  action="store_true",
                        help="词汇发现模式：输出高频词表，帮助确定关键词")
    parser.add_argument("--top_n",     type=int,   default=60,
                        help="--discover 模式显示前 N 个高频词，默认 60")
    parser.add_argument("--min_sec",   type=float, default=MIN_SEC_DEFAULT,
                        help=f"最短时长（秒），默认 {MIN_SEC_DEFAULT}")
    parser.add_argument("--fps",       type=float, default=FPS_DEFAULT,
                        help=f"动作帧率，默认 {FPS_DEFAULT}")
    parser.add_argument("--target",    type=int,   default=TARGET_PER_CATEGORY,
                        help=f"每类最多保留几条（不含镜像），默认 {TARGET_PER_CATEGORY}")
    parser.add_argument("--save",      type=str,   default=None,
                        help="将路径列表保存到文件（可选）")
    args = parser.parse_args()

    if args.discover:
        discover(args.data_root, top_n=args.top_n)
        return

    results = scan(args.data_root, min_sec=args.min_sec,
                   fps=args.fps, target=args.target)
    print_results(results, args.data_root, fps=args.fps)

    if args.save:
        save_results(results, args.save)

    all_paths = [str(p) for paths in results.values() for p in paths]
    if all_paths:
        print("── 示例：逐条运行验证实验 ──")
        for p in all_paths:
            print(f"  python scripts/robustness_validation/run_validation.py "
                  f"\\\n      --motion {p} \\\n      --checkpoint /path/to/model.pt --headless")


if __name__ == "__main__":
    main()
