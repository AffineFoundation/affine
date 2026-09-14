"""Docker disk GC for the datagen pods, run before every docker batch.

Why (2026-09-13): `affine_tmax` builds one image per task (`local_docker_build`,
~2 GB each, 14,600 tasks); the per-batch `prune_images` only untags a batch's
images when a stopped container still references them, and the build cache
grows without bound. affine-datagen-2 filled its disk twice on 2026-09-12
(322 GB reclaimed by hand); datagen-3/-4 died on other hosts, possibly the
same way. A full disk kills every running rollout on the pod.

Rule (conservative on purpose):
  * nothing happens while free space on the docker root is >= MIN_FREE_PCT;
  * stage 1: dangling images (`docker image prune -f`);
  * stage 2: images in GC_IMAGE_PREFIXES (the per-task local builds and the
    recoverable pipeline's shadow tags) that no container -- running or
    stopped -- references and that are older than IMAGE_AGE_H hours;
  * stage 3, only if still under the threshold: build cache older than
    IMAGE_AGE_H (`docker builder prune`; costs rebuild time, never data).
Never touched: any image referenced by a container, the verifiers taskset
namespaces (catalog.VERIFIERS_IMAGE_PREFIXES: swerebench, namanjain12,
mswebench, terminal-lego, ...), base images (python, swipl, ...).

Every removal is logged (`rollouts.diskgc`) with the space before / after.

Knobs (env): ROLLOUTS_GC_MIN_FREE_PCT (25), ROLLOUTS_GC_IMAGE_AGE_H (6),
ROLLOUTS_GC_DISABLE=1 turns the GC off.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone

from rollouts.catalog import VERIFIERS_IMAGE_PREFIXES

log = logging.getLogger("rollouts.diskgc")

GC_IMAGE_PREFIXES = ("prime/primeintellect/tmax", "recoverable.local/")
MIN_FREE_PCT = float(os.environ.get("ROLLOUTS_GC_MIN_FREE_PCT", 25))
IMAGE_AGE_H = float(os.environ.get("ROLLOUTS_GC_IMAGE_AGE_H", 6))
DISABLED = os.environ.get("ROLLOUTS_GC_DISABLE", "") == "1"
_DOCKER_TIMEOUT_S = 120
_PRUNE_TIMEOUT_S = 900


def _docker(*args: str, timeout: int = _DOCKER_TIMEOUT_S) -> str:
    proc = subprocess.run(["docker", *args], capture_output=True, text=True,
                          timeout=timeout)
    return proc.stdout


def docker_root() -> str:
    try:
        root = _docker("info", "--format", "{{.DockerRootDir}}").strip()
    except (subprocess.SubprocessError, OSError):
        root = ""
    return root or "/var/lib/docker"


def free_pct(path: str) -> float:
    usage = shutil.disk_usage(path)
    return 100.0 * usage.free / usage.total if usage.total else 100.0


def _gb(n_bytes: int) -> float:
    return round(n_bytes / 1e9, 1)


def _parse_created(text: str) -> float | None:
    """`docker images --format {{.CreatedAt}}` -> epoch seconds
    (e.g. `2026-09-13 03:41:06 +0000 UTC`)."""
    parts = text.strip().split()
    if len(parts) < 3:
        return None
    try:
        dt = datetime.strptime(" ".join(parts[:3]), "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        return None
    return dt.astimezone(timezone.utc).timestamp()


def _referenced_images() -> set[str]:
    """Image ids and image refs of every container, running or stopped
    (`.Image` = the image id, `.Config.Image` = the ref it was created
    with; both spellings kept)."""
    refs: set[str] = set()
    ids = _docker("ps", "-aq").split()
    if not ids:
        return refs
    out = _docker("inspect", "--format", "{{.Image}}\t{{.Config.Image}}", *ids)
    for line in out.splitlines():
        for part in line.split("\t"):
            part = part.strip()
            if part:
                refs.add(part)
                if part.startswith("sha256:"):
                    refs.add(part[len("sha256:"):])
    return refs


def _gc_candidates(now: float) -> list[tuple[str, str]]:
    """(repo:tag, image id) of GC-namespace images older than IMAGE_AGE_H
    that no container references."""
    refs = _referenced_images()
    out = _docker("images", "--no-trunc", "--format",
                  "{{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}")
    picked: list[tuple[str, str]] = []
    for line in out.splitlines():
        cols = line.split("\t")
        if len(cols) != 4:
            continue
        repo, tag, image_id, created_at = (c.strip() for c in cols)
        ref = f"{repo}:{tag}"
        if not repo.startswith(GC_IMAGE_PREFIXES):
            continue
        if repo.startswith(VERIFIERS_IMAGE_PREFIXES):
            continue
        short_id = image_id[len("sha256:"):] if image_id.startswith("sha256:") else image_id
        if (ref in refs or image_id in refs or short_id in refs
                or any(r and short_id.startswith(r) for r in refs if len(r) >= 12)):
            continue
        created = _parse_created(created_at)
        if created is None or now - created < IMAGE_AGE_H * 3600:
            continue
        picked.append((ref, image_id))
    return picked


def gc_if_low(reason: str = "batch") -> None:
    """The per-batch hook. Cheap when the disk is fine (one statvfs)."""
    if DISABLED:
        return
    try:
        root = docker_root()
        free0 = free_pct(root)
        if free0 >= MIN_FREE_PCT:
            return
        used0 = shutil.disk_usage(root).used
        log.warning("disk GC (%s): %s has %.1f%% free (< %.0f%%)",
                    reason, root, free0, MIN_FREE_PCT)
        now = time.time()

        pruned = _docker("image", "prune", "-f", timeout=_PRUNE_TIMEOUT_S)
        reclaimed_line = next((l for l in pruned.splitlines()
                               if l.startswith("Total reclaimed space")), "")
        log.info("disk GC stage 1: dangling images pruned (%s)",
                 reclaimed_line or "nothing")

        removed = []
        for ref, _image_id in _gc_candidates(now):
            proc = subprocess.run(["docker", "rmi", ref], capture_output=True,
                                  text=True, timeout=_DOCKER_TIMEOUT_S)
            if proc.returncode == 0:
                removed.append(ref)
            else:
                log.info("disk GC: kept %s (%s)", ref,
                         (proc.stderr or "").strip().splitlines()[-1:] or "rmi failed")
        if removed:
            log.info("disk GC stage 2: removed %d unreferenced image(s) older than "
                     "%.0f h: %s", len(removed), IMAGE_AGE_H, ", ".join(removed))
        else:
            log.info("disk GC stage 2: no unreferenced %s image older than %.0f h",
                     "/".join(GC_IMAGE_PREFIXES), IMAGE_AGE_H)

        if free_pct(root) < MIN_FREE_PCT:
            out = _docker("builder", "prune", "-f", "--filter",
                          f"until={int(IMAGE_AGE_H)}h", timeout=_PRUNE_TIMEOUT_S)
            reclaimed_line = next((l for l in out.splitlines()
                                   if l.startswith("Total")), "")
            log.info("disk GC stage 3: build cache older than %.0f h pruned (%s)",
                     IMAGE_AGE_H, reclaimed_line or "nothing")

        free1 = free_pct(root)
        used1 = shutil.disk_usage(root).used
        log.warning("disk GC done: %.1f%% -> %.1f%% free, %.1f GB reclaimed",
                    free0, free1, _gb(max(0, used0 - used1)))
        if free1 < MIN_FREE_PCT:
            log.error("disk GC could not reach %.0f%% free (%.1f%%); images in use "
                      "or non-GC namespaces hold the space", MIN_FREE_PCT, free1)
    except Exception:  # noqa: BLE001 - GC must never fail a batch
        log.warning("disk GC failed", exc_info=True)
