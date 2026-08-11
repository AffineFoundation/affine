"""Bench-panel exclusion keys, shared by catalogs, views, and validation.

The 25-instance advisory bench panel must stay disjoint from the training
corpus: pinned instance ids AND every instance from the same repos are
excluded. Bare repo names cover tasksets whose rows carry no owner (r2e);
over-exclusion on a bare-name collision is accepted as conservative.
"""

from __future__ import annotations

import importlib.util
import json
import os
from functools import lru_cache
from pathlib import Path

PanelKeys = tuple[set[str], set[str], set[str]]   # (ids, repos, bare names)


def panel_path() -> Path:
    override = os.environ.get("ROLLOUTS_PANEL_PATH")
    if override:
        return Path(override)
    spec = importlib.util.find_spec("evalsrv")
    if spec is None or not spec.origin:
        raise RuntimeError("evalsrv package not importable and "
                           "ROLLOUTS_PANEL_PATH not set")
    return Path(spec.origin).parent / "data" / "swe_rebench_lite_ids.json"


@lru_cache(maxsize=1)
def panel_keys() -> PanelKeys:
    ids = set(json.loads(panel_path().read_text())["instance_ids"])
    repos = {i.rsplit("-", 1)[0].replace("__", "/").lower() for i in ids}
    bare = {r.split("/", 1)[1] for r in repos}
    return ids, repos, bare


def panel_drop(uid: str, repo: str, panel: PanelKeys) -> bool:
    ids, repos, bare = panel
    repo = (repo or "").lower()
    return repo in repos or repo in bare or uid in ids
