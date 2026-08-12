"""Build per-source catalogs: one jsonl line per usable task.

Catalog rows are the task identity contract (schema.REQUIRED_TASK_KEYS):

  uid       key used in the eval filter/tasks list AND as the trace's task
            name (mini_swe: the swebench instance id)
  sid       stratification id fed to datagen.slicer.make_traj_id — shaped
            "stem-N" with a dot-free stem so the corpus repo-stratum
            (segment before the first ".") groups by repository
  repo      owner/name where the dataset provides it (bare name for r2e)
  language  lowercase language tag ("python", "go", "shell", ...)
  image     docker image ref the task pulls (per-batch prune target)

Panel-excluded and metadata-less rows are dropped here, so batch selection
never even proposes them (the view re-checks — double exclusion).
Catalogs are deterministic snapshots; rebuild by deleting the file or
`python -m rollouts.catalog [--only NAME]`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import tomllib
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from rollouts.config import RolloutsConfig, load_config
from rollouts.panel import panel_drop, panel_keys
from rollouts.registry import Registry, Source, load_registry

log = logging.getLogger("rollouts.catalog")

_SCALESWE_PR_RE = re.compile(r"_pr(\d+)$")
_TRAILING_NUM_RE = re.compile(r"-(\d+)$")

SWESMITH_REPO_CAP = 100
SWESMITH_LANG_DATASETS: tuple[tuple[str, str, str], ...] = (
    # (short lang key used in task name, HF dataset, catalog language tag)
    ("py", "SWE-bench/SWE-smith-py", "python"),
    ("go", "SWE-bench/SWE-smith-go", "go"),
    ("java", "SWE-bench/SWE-smith-java", "java"),
    ("js", "SWE-bench/SWE-smith-js", "js"),
    ("ts", "SWE-bench/SWE-smith-ts", "ts"),
    ("rs", "SWE-bench/SWE-smith-rs", "rs"),
    ("cpp", "SWE-bench/SWE-smith-cpp", "cpp"),
    ("php", "SWE-bench/SWE-smith-php", "php"),
)

TERMINAL_LEGO_DATASET = "PrimeIntellect/Terminal-Lego-15k"
REQUIRED_LEGO_FILES = (
    "task.toml", "instruction.md", "tests/test.sh", "tests/test_outputs.py",
)
NL2REPO_IMAGE_PREFIX = "ghcr.io/multimodal-art-projection/nl2repobench"
NL2REPO_DEFAULT_DIR = ("/root/prime-pilot/research-environments/environments/"
                       "code/nl2repobench_v1/nl2repobench_v1/test_files")

# Image namespaces owned exclusively by verifiers-runner sources; used for
# leftover-container reaping (mini_swe containers never match these).
VERIFIERS_IMAGE_PREFIXES = (
    "aweaiteam/scaleswe",
    "docker.io/swerebenchv2/", "swerebenchv2/",
    "namanjain12/",
    "mswebench/",
    "swebench/swesmith",
    "jierun/sweb.eval",
    "terminal-lego/",
    "alexgshaw/",
    "ghcr.io/multimodal-art-projection/nl2repobench",
)


def _dotless(stem: str) -> str:
    return stem.replace(".", "_")


def _dotless_task(uid: str) -> str:
    return uid.replace(".", "_").replace("-", "_")


def instance_repo(instance_id: str) -> str:
    """owner__name-123 -> owner/name (swebench instance id convention)."""
    return instance_id.rsplit("-", 1)[0].replace("__", "/", 1)


# -- per-dataset row-meta derivations (catalog kind "hf") ------------------------

def _scaleswe_meta(row: dict) -> dict | None:
    iid = row["instance_id"]
    m = _SCALESWE_PR_RE.search(iid)
    num = m.group(1) if m else "0"
    user, repo = row["user"], row["repo"]
    return {
        "uid": iid,
        "sid": f"{_dotless(user)}__{_dotless(repo)}-{num}",
        "repo": f"{user}/{repo}".lower(),
        "language": (row.get("language") or "python").lower(),
        "image": row["image_url"],
    }


def _swerebench_v2_meta(row: dict) -> dict | None:
    if not row["image_name"].startswith("docker.io/"):
        return None
    iid = row["instance_id"]
    m = _TRAILING_NUM_RE.search(iid)
    stem = iid[: m.start()] if m else iid
    num = m.group(1) if m else "0"
    return {
        "uid": iid,
        "sid": f"{_dotless(stem)}-{num}",
        "repo": row["repo"].lower(),
        "language": (row.get("language") or "").lower(),
        "image": row["image_name"],
    }


def _r2e_meta(row: dict) -> dict | None:
    sha = row["commit_hash"]
    repo = row["repo_name"]
    return {
        "uid": sha,
        "sid": f"{_dotless(repo)}-{int(sha[:6], 16)}",
        "repo": repo.lower(),  # bare name; matched against panel bare set
        "language": "python",
        "image": row["docker_image"],
    }


def _multiswe_meta(row: dict) -> dict | None:
    iid = row.get("instance_id") or f"{row['org']}__{row['repo']}-{row['number']}"
    m = _TRAILING_NUM_RE.search(iid)
    stem = iid[: m.start()] if m else iid
    num = m.group(1) if m else str(row.get("number") or 0)
    image = row.get("docker_image") \
        or f"mswebench/{row['org']}_m_{row['repo']}:pr-{row['number']}".lower()
    return {
        "uid": iid,
        "sid": f"{_dotless(stem)}-{num}",
        "repo": f"{row['org']}/{row['repo']}".lower(),
        "language": (row.get("lang") or "").lower(),
        "image": image,
    }


def _swelego_meta(row: dict) -> dict | None:
    iid = row.get("instance_id")
    if not iid or not row.get("image_name"):
        return None
    m = _TRAILING_NUM_RE.search(iid)
    stem = iid[: m.start()] if m else iid
    num = m.group(1) if m else "0"
    return {
        "uid": iid,
        "sid": f"{_dotless(stem)}-{num}",
        "repo": (row.get("repo") or "").lower(),
        "language": "python",
        "image": row["image_name"],
    }


ROW_META = {
    "scaleswe": _scaleswe_meta,
    "swerebench_v2": _swerebench_v2_meta,
    "r2e": _r2e_meta,
    "multiswe": _multiswe_meta,
    "swelego": _swelego_meta,
}


def _swesmith_meta(row: dict, lang_key: str, language: str) -> dict | None:
    iid = row.get("instance_id")
    image = row.get("image_name")
    if not iid or not image:
        return None
    repo = (row.get("repo") or iid.rsplit(".", 1)[0]).lower()
    # sid: language-qualified stem so strata don't collide across langs
    m = _TRAILING_NUM_RE.search(iid)
    stem = iid[: m.start()] if m else iid
    num = m.group(1) if m else "0"
    return {
        "uid": f"{lang_key}:{iid}",
        "sid": f"{lang_key}__{_dotless(stem)}-{num}",
        "repo": repo,
        "language": language,
        "image": image,
        "instance_id": iid,
    }


# -- builders --------------------------------------------------------------------

def catalog_path(cfg: RolloutsConfig, name: str) -> Path:
    return cfg.catalog_dir / f"{name}.jsonl"


def _write_catalog(cfg: RolloutsConfig, name: str, kept: list[dict],
                   summary: dict) -> dict:
    cfg.catalog_dir.mkdir(parents=True, exist_ok=True)
    tmp = catalog_path(cfg, name).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for meta in kept:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    tmp.replace(catalog_path(cfg, name))
    log.info("catalog %s: %s", name, json.dumps(summary))
    return summary


def build_hf_catalog(cfg: RolloutsConfig, src: Source) -> dict:
    row_meta = ROW_META[src.row_meta]
    panel = panel_keys()
    rows = load_dataset(src.dataset, split=src.split)
    kept: list[dict] = []
    n_panel = n_unusable = 0
    seen: set[str] = set()
    for row in rows:
        meta = row_meta(dict(row))
        if meta is None or meta["uid"] in seen:
            n_unusable += 1
            continue
        if panel_drop(meta["uid"], meta["repo"], panel):
            n_panel += 1
            continue
        seen.add(meta["uid"])
        kept.append(meta)
    return _write_catalog(cfg, src.name, kept, {
        "source": src.name, "dataset": src.dataset,
        "total": len(rows), "kept": len(kept),
        "panel_excluded": n_panel, "unusable": n_unusable,
    })


def build_hf_swebench_catalog(cfg: RolloutsConfig, src: Source) -> dict:
    """Nebius-style swebench pool for the mini_swe runner. Meta-only rows;
    the runner re-loads the (disk-cached) dataset to materialize subsets."""
    panel = panel_keys()
    rows = load_dataset(src.dataset, split=src.split)
    kept: list[dict] = []
    n_panel = n_no_image = 0
    seen: set[str] = set()
    for row in rows:
        image = row.get("docker_image") or row.get("image_name") or ""
        if not image:
            n_no_image += 1
            continue
        iid = row["instance_id"]
        if iid in seen:
            continue
        repo = instance_repo(iid).lower()
        if panel_drop(iid, repo, panel):
            n_panel += 1
            continue
        seen.add(iid)
        kept.append({
            "uid": iid,
            "sid": iid,   # legacy slicer identity: make_traj_id(instance_id)
            "repo": repo,
            "language": (row.get("language") or "python").lower(),
            "image": image,
            "dataset": src.dataset,
            "split": src.split,
        })
    return _write_catalog(cfg, src.name, kept, {
        "source": src.name, "dataset": src.dataset,
        "total": len(rows), "kept": len(kept),
        "panel_excluded": n_panel, "no_image": n_no_image,
    })


def build_swesmith_catalog(cfg: RolloutsConfig, src: Source) -> dict:
    panel = panel_keys()
    kept: list[dict] = []
    n_panel = n_unusable = n_capped = 0
    total = 0
    seen: set[str] = set()
    per_repo: Counter[str] = Counter()
    for lang_key, dataset, language in SWESMITH_LANG_DATASETS:
        rows = load_dataset(dataset, split=src.split)
        total += len(rows)
        for row in rows:
            meta = _swesmith_meta(dict(row), lang_key, language)
            if meta is None or meta["uid"] in seen:
                n_unusable += 1
                continue
            if panel_drop(meta["uid"], meta["repo"], panel):
                n_panel += 1
                continue
            repo = meta["repo"]
            if per_repo[repo] >= SWESMITH_REPO_CAP:
                n_capped += 1
                continue
            per_repo[repo] += 1
            seen.add(meta["uid"])
            kept.append(meta)
    return _write_catalog(cfg, src.name, kept, {
        "source": src.name, "dataset": "SWE-bench/SWE-smith-*",
        "total": total, "kept": len(kept),
        "panel_excluded": n_panel, "unusable": n_unusable,
        "repo_capped": n_capped, "repo_cap": SWESMITH_REPO_CAP,
        "n_repos": len(per_repo),
    })


_DOCKER_COPY_RE = re.compile(r"^\s*(?:COPY|ADD)\s+(.+)$",
                             re.IGNORECASE | re.MULTILINE)


def _dockerfile_missing_sources(context: Path, dockerfile: Path) -> list[str]:
    """Relative COPY/ADD sources absent from the build context.

    ~51% of Terminal-Lego tasks ship a Dockerfile that copies ./task_file
    without the checkout carrying it. Probing the official prebuilt images
    (published on the Prime registry) showed /app/task_file is an *empty*
    scratch dir the agent works in, so those tasks are recovered by stubbing
    the dir locally; anything else missing stays unbuildable.
    """
    missing: list[str] = []
    text = dockerfile.read_text(errors="ignore")
    for args in _DOCKER_COPY_RE.findall(text):
        if args.lstrip().startswith("["):        # JSON form: ["src", "dst"]
            try:
                parts = [str(p) for p in json.loads(args)]
            except json.JSONDecodeError:
                continue
        else:
            parts = args.split()
        if any(p.startswith("--from=") for p in parts):
            continue                             # copies from a build stage
        parts = [p for p in parts if not p.startswith("--")]
        if len(parts) < 2:
            continue
        for src in parts[:-1]:
            if src.startswith(("http://", "https://")):
                continue                         # ADD from URL
            rel = src.strip('"').lstrip("./").rstrip("/")
            if not rel:
                continue                         # "COPY . /x" = whole context
            if any(ch in rel for ch in "*?["):
                if not any(context.glob(rel)):
                    missing.append(rel)
            elif not (context / rel).exists():
                missing.append(rel)
    return missing


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


def build_terminal_lego_catalog(cfg: RolloutsConfig, src: Source) -> dict:
    root = ensure_terminal_lego_checkout()
    kept: list[dict] = []
    n_unusable = 0
    n_unbuildable = 0
    n_stubbed = 0
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        if not all((task_dir / f).is_file() for f in REQUIRED_LEGO_FILES):
            n_unusable += 1
            continue
        dockerfile = task_dir / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            n_unusable += 1
            continue
        missing = _dockerfile_missing_sources(dockerfile.parent, dockerfile)
        if missing:
            # task_file is the agent's empty working dir in the official
            # images — safe to materialize; any other gap is a real one.
            if all(m == "task_file" for m in missing):
                (dockerfile.parent / "task_file").mkdir(exist_ok=True)
                n_stubbed += 1
            else:
                n_unbuildable += 1
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
    return _write_catalog(cfg, src.name, kept, {
        "source": src.name, "dataset": TERMINAL_LEGO_DATASET,
        "total": len(kept) + n_unusable + n_unbuildable, "kept": len(kept),
        "panel_excluded": 0, "unusable": n_unusable,
        "unbuildable": n_unbuildable, "task_file_stubbed": n_stubbed,
    })


def build_terminal_bench_2_catalog(cfg: RolloutsConfig, src: Source) -> dict:
    """Load the Harbor taskset once to enumerate task names + images.
    Import is runtime-optional by design: the taskset package only exists
    inside the verifiers environment on the pod."""
    from terminal_bench_2_v1.taskset import (  # noqa: PLC0415
        TerminalBench2Config,
        TerminalBench2Taskset,
    )

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
    return _write_catalog(cfg, src.name, kept, {
        "source": src.name, "dataset": "terminal-bench/terminal-bench-2",
        "total": len(tasks), "kept": len(kept),
        "panel_excluded": 0, "unusable": n_unusable,
    })


def build_nl2repobench_catalog(cfg: RolloutsConfig, src: Source) -> dict:
    root = Path(os.environ.get("ROLLOUTS_NL2REPO_DIR", NL2REPO_DEFAULT_DIR))
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
    return _write_catalog(cfg, src.name, kept, {
        "source": src.name, "dataset": str(root),
        "total": len(kept) + n_unusable, "kept": len(kept),
        "panel_excluded": 0, "unusable": n_unusable,
    })


BUILDERS = {
    "hf": build_hf_catalog,
    "hf_swebench": build_hf_swebench_catalog,
    "swesmith": build_swesmith_catalog,
    "terminal_lego": build_terminal_lego_catalog,
    "terminal_bench_2": build_terminal_bench_2_catalog,
    "nl2repobench": build_nl2repobench_catalog,
}


def build_catalog(cfg: RolloutsConfig, src: Source) -> dict:
    return BUILDERS[src.catalog](cfg, src)


def load_catalog(cfg: RolloutsConfig, src: Source) -> list[dict]:
    path = catalog_path(cfg, src.name)
    if not path.exists():
        build_catalog(cfg, src)
    return [json.loads(line) for line in open(path, encoding="utf-8")
            if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    registry: Registry = load_registry()
    for name, src in registry.sources.items():
        if args.only and name != args.only:
            continue
        build_catalog(cfg, src)


if __name__ == "__main__":
    main()
