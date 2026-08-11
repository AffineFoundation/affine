"""Build per-taskset catalogs: one jsonl line per usable task.

Panel-excluded and metadata-less rows are dropped here, so batch selection
never even proposes them (the converter re-checks — double exclusion).
Catalogs are deterministic snapshots; rebuild by deleting the file.

  PYTHONPATH=/root/affine:/root/prime-lane /root/venv/bin/python \
      -m primelane.catalog [--only NAME]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tomllib
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from primelane.lanespec import (
    NL2REPO_IMAGE_PREFIX,
    SPECS,
    SWESMITH_LANG_DATASETS,
    SWESMITH_REPO_CAP,
    TERMINAL_LEGO_DATASET,
    LaneSpec,
    _swesmith_meta,
    panel_keys,
)

CATALOG_DIR = Path("/root/prime-lane/data/catalogs")
REQUIRED_LEGO_FILES = (
    "task.toml", "instruction.md", "tests/test.sh", "tests/test_outputs.py",
)


def catalog_path(name: str) -> Path:
    return CATALOG_DIR / f"{name}.jsonl"


def _write_catalog(name: str, kept: list[dict], summary: dict) -> dict:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = catalog_path(name).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for meta in kept:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    tmp.replace(catalog_path(name))
    print(json.dumps(summary))
    return summary


def _panel_drop(meta: dict, panel_ids: set[str], panel_repos: set[str],
                panel_bare: set[str]) -> bool:
    repo = meta["repo"]
    return (repo in panel_repos or repo in panel_bare
            or meta["uid"] in panel_ids)


def build_hf_catalog(spec: LaneSpec) -> dict:
    assert spec.row_meta is not None
    panel_ids, panel_repos, panel_bare = panel_keys()
    rows = load_dataset(spec.dataset, split=spec.split)
    kept: list[dict] = []
    n_panel = n_unusable = 0
    seen: set[str] = set()
    for row in rows:
        meta = spec.row_meta(dict(row))
        if meta is None or meta["uid"] in seen:
            n_unusable += 1
            continue
        if _panel_drop(meta, panel_ids, panel_repos, panel_bare):
            n_panel += 1
            continue
        seen.add(meta["uid"])
        kept.append(meta)
    return _write_catalog(spec.name, kept, {
        "taskset": spec.name, "dataset": spec.dataset,
        "total": len(rows), "kept": len(kept),
        "panel_excluded": n_panel, "unusable": n_unusable,
    })


def build_swesmith_catalog(spec: LaneSpec) -> dict:
    panel_ids, panel_repos, panel_bare = panel_keys()
    kept: list[dict] = []
    n_panel = n_unusable = n_capped = 0
    total = 0
    seen: set[str] = set()
    per_repo: Counter[str] = Counter()
    for lang_key, dataset, language in SWESMITH_LANG_DATASETS:
        rows = load_dataset(dataset, split=spec.split)
        total += len(rows)
        for row in rows:
            meta = _swesmith_meta(dict(row), lang_key, language)
            if meta is None or meta["uid"] in seen:
                n_unusable += 1
                continue
            if _panel_drop(meta, panel_ids, panel_repos, panel_bare):
                n_panel += 1
                continue
            repo = meta["repo"]
            if per_repo[repo] >= SWESMITH_REPO_CAP:
                n_capped += 1
                continue
            per_repo[repo] += 1
            seen.add(meta["uid"])
            kept.append(meta)
    return _write_catalog(spec.name, kept, {
        "taskset": spec.name, "dataset": "SWE-bench/SWE-smith-*",
        "total": total, "kept": len(kept),
        "panel_excluded": n_panel, "unusable": n_unusable,
        "repo_capped": n_capped, "repo_cap": SWESMITH_REPO_CAP,
        "n_repos": len(per_repo),
    })


def _terminal_lego_root() -> Path:
    """Match terminal_lego_v1.taskset.git_cache_root layout."""
    home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    name = TERMINAL_LEGO_DATASET.replace("/", "--")
    return home / "terminal-lego-git" / name


def ensure_terminal_lego_checkout() -> Path:
    root = _terminal_lego_root()
    if (root / ".git").is_dir() and any(root.glob("task_*")):
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    tmp = root.parent / f".tmp-{root.name}"
    if tmp.exists():
        subprocess.run(["rm", "-rf", str(tmp)], check=False)
    env = dict(os.environ)
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    url = f"https://huggingface.co/datasets/{TERMINAL_LEGO_DATASET}"
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(tmp)],
        check=True, env=env,
    )
    if root.exists():
        subprocess.run(["rm", "-rf", str(root)], check=False)
    tmp.rename(root)
    return root


def build_terminal_lego_catalog(spec: LaneSpec) -> dict:
    root = ensure_terminal_lego_checkout()
    kept: list[dict] = []
    n_unusable = 0
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        if not all((task_dir / f).is_file() for f in REQUIRED_LEGO_FILES):
            n_unusable += 1
            continue
        if not (task_dir / "environment" / "Dockerfile").is_file():
            n_unusable += 1
            continue
        try:
            env = tomllib.loads(
                (task_dir / "task.toml").read_text()).get("environment", {})
        except Exception:
            n_unusable += 1
            continue
        image = env.get("docker_image") or ""
        if not image:
            n_unusable += 1
            continue
        uid = task_dir.name
        kept.append({
            "uid": uid,
            "sid": f"terminal_lego__{_dotless_task(uid)}-0",
            "repo": f"terminal-lego/{uid}",
            "language": "shell",
            "image": image,
            "task_dir": str(task_dir),
        })
    return _write_catalog(spec.name, kept, {
        "taskset": spec.name, "dataset": TERMINAL_LEGO_DATASET,
        "total": len(kept) + n_unusable, "kept": len(kept),
        "panel_excluded": 0, "unusable": n_unusable,
    })


def _dotless_task(uid: str) -> str:
    return uid.replace(".", "_").replace("-", "_")


def build_terminal_bench_2_catalog(spec: LaneSpec) -> dict:
    """Load the Harbor taskset once to enumerate task names + images."""
    from terminal_bench_2_v1.taskset import TerminalBench2Config, TerminalBench2Taskset

    tasks = list(TerminalBench2Taskset(TerminalBench2Config()).load())
    kept: list[dict] = []
    n_unusable = 0
    for task in tasks:
        data = task.data
        uid = data.name
        image = data.image or ""
        if not uid or not image:
            n_unusable += 1
            continue
        short = uid.rsplit("/", 1)[-1]
        kept.append({
            "uid": uid,
            "sid": f"terminal_bench_2__{_dotless_task(short)}-0",
            "repo": f"terminal-bench-2/{short}",
            "language": "shell",
            "image": image,
        })
    return _write_catalog(spec.name, kept, {
        "taskset": spec.name, "dataset": "terminal-bench/terminal-bench-2",
        "total": len(tasks), "kept": len(kept),
        "panel_excluded": 0, "unusable": n_unusable,
    })


def build_nl2repobench_catalog(spec: LaneSpec) -> dict:
    root = Path(
        "/root/prime-pilot/research-environments/environments/"
        "code/nl2repobench_v1/nl2repobench_v1/test_files")
    kept: list[dict] = []
    n_unusable = 0
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        needed = ("start.md", "test_commands.json",
                  "test_files.json", "test_case_count.txt")
        if not all((task_dir / f).is_file() for f in needed):
            n_unusable += 1
            continue
        uid = task_dir.name
        kept.append({
            "uid": uid,
            "sid": f"nl2repobench__{_dotless_task(uid)}-0",
            "repo": f"nl2repobench/{uid}",
            "language": "python",
            "image": f"{NL2REPO_IMAGE_PREFIX}/{uid.lower()}:1.0",
        })
    return _write_catalog(spec.name, kept, {
        "taskset": spec.name, "dataset": str(root),
        "total": len(kept) + n_unusable, "kept": len(kept),
        "panel_excluded": 0, "unusable": n_unusable,
    })


def build_catalog(spec: LaneSpec) -> dict:
    kind = spec.catalog_kind
    if kind == "hf":
        return build_hf_catalog(spec)
    if kind == "swesmith":
        return build_swesmith_catalog(spec)
    if kind == "terminal_lego":
        return build_terminal_lego_catalog(spec)
    if kind == "terminal_bench_2":
        return build_terminal_bench_2_catalog(spec)
    if kind == "nl2repobench":
        return build_nl2repobench_catalog(spec)
    raise ValueError(f"unknown catalog_kind {kind!r}")


def load_catalog(name: str) -> list[dict]:
    path = catalog_path(name)
    if not path.exists():
        build_catalog(SPECS[name])
    return [json.loads(line) for line in open(path, encoding="utf-8")
            if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    for name, spec in SPECS.items():
        if args.only and name != args.only:
            continue
        build_catalog(spec)


if __name__ == "__main__":
    main()
