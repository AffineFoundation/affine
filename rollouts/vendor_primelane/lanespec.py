"""Per-taskset lane specs: dataset identity, uid/repo/language derivation,
and bench-panel exclusion keys shared by the catalog builder and converter.

Row-meta contract per taskset (catalog line):
  uid       dataset key used in the eval filter/tasks list AND as the
            trace's task name
  sid       stratification id fed to datagen.slicer.make_traj_id — shaped
            "stem-N" with a dot-free stem so the corpus repo-stratum
            (segment before the first ".") groups by repository
  repo      owner/name where the dataset provides it (bare name for r2e)
  language  lowercase language tag ("python", "go", "shell", ...)
  image     docker image ref the task pulls (per-batch prune target)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

PANEL_PATH = Path("/root/affine/evalsrv/data/swe_rebench_lite_ids.json")

_SCALESWE_PR_RE = re.compile(r"_pr(\d+)$")
_TRAILING_NUM_RE = re.compile(r"-(\d+)$")

SelectMode = Literal["filter_fn", "tasks"]
CatalogKind = Literal["hf", "swesmith", "terminal_lego", "terminal_bench_2", "nl2repobench"]

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
NL2REPO_IMAGE_PREFIX = "ghcr.io/multimodal-art-projection/nl2repobench"


def panel_keys() -> tuple[set[str], set[str], set[str]]:
    """(instance ids, owner/name repos, bare repo names) — all lowercase
    except ids. Bare names cover tasksets whose rows carry no owner (r2e);
    over-exclusion on a bare-name collision is accepted as conservative."""
    ids = set(json.loads(PANEL_PATH.read_text())["instance_ids"])
    repos = {i.rsplit("-", 1)[0].replace("__", "/").lower() for i in ids}
    bare = {r.split("/", 1)[1] for r in repos}
    return ids, repos, bare


def _dotless(stem: str) -> str:
    return stem.replace(".", "_")


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


@dataclass(frozen=True)
class LaneSpec:
    name: str                     # short source tag on turn records
    taskset_id: str               # verifiers taskset package id
    select: SelectMode = "filter_fn"
    catalog_kind: CatalogKind = "hf"
    dataset: str = ""
    split: str = "train"
    uid_field: str = "instance_id"  # HF row field for filter_fn mode
    pass_dataset: bool = True       # whether eval_cmd sets dataset-name/split
    row_meta: Callable[[dict], dict | None] | None = None
    extra_flags: tuple[str, ...] = field(default_factory=tuple)
    local_docker_build: bool = False


SPECS: dict[str, LaneSpec] = {
    "scaleswe": LaneSpec(
        name="scaleswe", taskset_id="scaleswe-v1",
        dataset="PrimeIntellect/Scale-SWE-Verified", split="train",
        uid_field="instance_id", row_meta=_scaleswe_meta,
        # Catalog + pull failures already cover availability; the taskset's
        # own docker-hub tag enumeration costs ~17 registry calls per batch.
        extra_flags=("--env.taskset.filter-unavailable-images", "False"),
    ),
    "swerebench_v2": LaneSpec(
        name="swerebench_v2", taskset_id="swerebench-v2-full",
        dataset="PrimeIntellect/SWE-rebench-V2", split="train",
        uid_field="instance_id", row_meta=_swerebench_v2_meta,
    ),
    "r2e_gym": LaneSpec(
        name="r2e_gym", taskset_id="r2e-gym-v1",
        dataset="PrimeIntellect/R2E-Gym-Subset-Verified", split="train",
        uid_field="commit_hash", row_meta=_r2e_meta,
    ),
    "multiswe": LaneSpec(
        name="multiswe", taskset_id="multiswe-v1",
        dataset="PrimeIntellect/Multi-SWE-RL-Verified", split="train",
        uid_field="instance_id", row_meta=_multiswe_meta,
    ),
    "swesmith": LaneSpec(
        name="swesmith", taskset_id="swesmith-v1",
        select="filter_fn", catalog_kind="swesmith",
        dataset="", split="train", uid_field="instance_id",
        pass_dataset=False, row_meta=None,
    ),
    "swelego": LaneSpec(
        name="swelego", taskset_id="swelego-v1",
        dataset="PrimeIntellect/SWE-Lego-Real-Data-Verified", split="resolved",
        uid_field="instance_id", row_meta=_swelego_meta,
    ),
    "terminal_lego": LaneSpec(
        name="terminal_lego", taskset_id="terminal-lego-v1",
        select="tasks", catalog_kind="terminal_lego",
        pass_dataset=False, local_docker_build=True,
    ),
    "terminal_bench_2": LaneSpec(
        name="terminal_bench_2", taskset_id="terminal-bench-2-v1",
        select="tasks", catalog_kind="terminal_bench_2",
        pass_dataset=False,
    ),
    "nl2repobench": LaneSpec(
        name="nl2repobench", taskset_id="nl2repobench-v1",
        select="tasks", catalog_kind="nl2repobench",
        pass_dataset=False,
    ),
}

# Image namespaces owned exclusively by this lane's tasksets. Used for
# leftover-container reaping — the main datagen loop's swerebench/sweb.eval
# images never match these prefixes.
LANE_IMAGE_PREFIXES = (
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
