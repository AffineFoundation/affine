"""Shared paths, knobs and helpers for ops/curriculum.

Everything here is deterministic: no clocks in the hashed outputs, sorted
keys, canonical JSON. Secrets are never read here (see publish.py).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import sys
import tomllib
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
AFFINE_DIR = REPO / "affine"
if str(AFFINE_DIR) not in sys.path:
    sys.path.insert(0, str(AFFINE_DIR))

STATE_DIR = REPO / "affine" / "state" / "curriculum"
INDEX_CACHE = STATE_DIR / "index_cache"
SNAPSHOT_DIR = STATE_DIR / "snapshots"
LATEST_PATH = STATE_DIR / "latest.json"
CRITERION_HISTORY = STATE_DIR / "criterion_history.jsonl"
HISTORY_PATH = REPO / "affine" / "state" / "history.jsonl"
EVALS_DIR = REPO / "affine" / "state" / "evals"
LOCAL_CHUNKS = REPO / "affine" / "state" / "corpus_cache" / "chunks"
SOURCES_TOML = REPO / "rollouts" / "rollouts" / "sources.toml"
PROBES_PATH = REPO / "affine" / "state" / "teacher_probe" / "probes.jsonl"
DATA_BASE = "https://data.affine.io"
PUBLIC_EVALS_BASE = "https://affine.io"
SLICE_N = 1300
FIRST_WVK13_CHALLENGE = "chal-00367"
DEFAULT_GROUP = "coding"

# The verdict artifact does not stamp weight_version_key; the fork times are
# public (llms.txt "Fork history"). A verdict's wvk is the last fork at or
# before its `at`.
WVK_FORKS: list[tuple[str, int]] = [
    ("2026-08-27T20:00:00+00:00", 10),
    ("2026-09-05T12:12:00+00:00", 11),
    ("2026-09-05T16:30:00+00:00", 12),
    ("2026-09-09T02:00:00+00:00", 13),
    ("2026-09-10T21:00:00+00:00", 14),
    ("2026-09-12T17:01:00+00:00", 15),
    ("2026-09-13T13:01:00+00:00", 16),
    ("2026-09-14T10:41:00+00:00", 17),
]

DEPTH_BINS: list[tuple[int, str]] = [
    (5_000, "d0_lt5k"), (20_000, "d1_5-20k"), (50_000, "d2_20-50k"),
    (100_000, "d3_50-100k"), (10 ** 12, "d4_gt100k"),
]

CURRICULUM_DEFAULTS: dict = {
    "mode": "shadow",
    "weights_path": "ops/curriculum/out/groups.json",
    "rule_version": 1,
    "share_unit": "slice_keys",
    "counted_rule": "v1",          # v1 | v1.1 | v1.2 | v2 -- which rule's weights drive the published shares
    "auto_apply_on_pass": False,   # coordinator 2026-09-16 17:04 UTC: the job flips mode -> apply when the decision table says APPLY
    "v2_eps": 0.20,                # uniform floor mass over ALL strata (no-forgetting guarantee)
    "v2_gamma": 1.0,
    "v2_component_weights": {"action": 0.25, "forfeit": 0.25, "score": 0.25, "gap": 0.25},
    "v11_s_gate": 0.5,
    "half_life_verdicts": 60,
    "n_0": 8,
    "gamma": 1.0,
    "eps": 0.02,
    "theta_pct": 0.25,
    "m_max": 3,
    "floor_coding_terminal": 0.40,
    "floor_frac_of_static": 0.5,
    "group_cap": 0.60,
    "max_share_shift": 0.05,
    "min_new_verdicts": 10,
    "recurrence_window_verdicts": 50,
    "recurrence_group_cap": 0.18,
    "recurrence_turn_cap": 0.20,
    "counterfactual_verdicts": 20,
    "near_king_z": 5.0,
    "first_challenge": FIRST_WVK13_CHALLENGE,
}

USER_AGENT = "affine-curriculum/1"
_WS = re.compile(r"\s+")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def norm_text(s: str | None) -> str:
    return _WS.sub(" ", s or "").strip()


def canonical_json(obj) -> bytes:
    """Byte-stable JSON: sorted keys, compact separators, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def clean_float(x) -> float | None:
    """Floats for canonical JSON: None for NaN/inf, rounded to 12 significant
    digits so two builds with different summation order agree."""
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    if x == 0.0:
        return 0.0
    return float(f"{x:.12g}")


def wvk_of(at: str | None) -> int | None:
    if not at:
        return None
    out = None
    for t, w in WVK_FORKS:
        if at >= t:
            out = w
    return out


def depth_bin(n_prefix_chars) -> str:
    n = int(n_prefix_chars or 0)
    for hi, label in DEPTH_BINS:
        if n < hi:
            return label
    return DEPTH_BINS[-1][1]


def load_sources_toml() -> dict:
    return tomllib.loads(SOURCES_TOML.read_text())


def load_curriculum_cfg(raw: dict | None = None) -> dict:
    raw = raw if raw is not None else load_sources_toml()
    cfg = dict(CURRICULUM_DEFAULTS)
    cfg.update({k: v for k, v in (raw.get("curriculum") or {}).items()})
    if cfg["mode"] not in ("off", "shadow", "apply"):
        raise ValueError(f"[curriculum].mode must be off|shadow|apply, got {cfg['mode']!r}")
    return cfg


def load_static_mix(raw: dict | None = None) -> dict[str, float]:
    raw = raw if raw is not None else load_sources_toml()
    return {g: float(v) for g, v in (raw.get("mix") or {}).items()}


def load_src2grp(raw: dict | None = None) -> dict[str, str]:
    raw = raw if raw is not None else load_sources_toml()
    return {name: cfg.get("group", DEFAULT_GROUP)
            for name, cfg in (raw.get("source") or {}).items()}


def group_of_stratum(stratum_src: str | None, source: str | None,
                     src2grp: dict[str, str], groups: set[str]) -> str:
    """Fold routing class of a turn. A namespaced stratum (`math:0643`,
    `king_fail:0010`, `coding:b00123`) names its group; a `repo|phase`
    stratum belongs to its source's group."""
    s = str(stratum_src or "")
    if ":" in s:
        ns = s.split(":", 1)[0]
        if ns in groups:
            return ns
    return src2grp.get(str(source or ""), DEFAULT_GROUP)


_SUB_SUFFIX = re.compile(r"#\d+$")


def base_stratum(stratum: str | None, stratum_src: str | None) -> str:
    """The stratum a turn belongs to before any sub-strata split. The fold
    keeps the original key in `stratum_src`; without it, strip the plan's
    `#k` suffix."""
    if stratum_src:
        return str(stratum_src)
    return _SUB_SUFFIX.sub("", str(stratum or ""))


# -- public fetch (anonymous, cached, sha-verified where the caller knows it) --
_http = httpx.Client(timeout=300, follow_redirects=True,
                     headers={"User-Agent": USER_AGENT})


def fetch_bytes(url: str) -> bytes:
    r = _http.get(url)
    r.raise_for_status()
    return r.content


def cached_public(key: str, *, sha256: str | None = None, gz_payload_sha: bool = False,
                  base: str = DATA_BASE, cache_dir: Path = INDEX_CACHE) -> Path:
    """`base/key` on disk under cache_dir/key. When `sha256` is given the
    object is verified (over the gunzipped payload if gz_payload_sha)."""
    dst = cache_dir / key
    if dst.exists():
        return dst
    blob = fetch_bytes(f"{base}/{key}")
    if sha256:
        got = sha256_bytes(gzip.decompress(blob) if gz_payload_sha else blob)
        if got != sha256:
            raise RuntimeError(f"{key}: sha mismatch {got} != {sha256}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, dst)
    return dst


def manifest_by_sha(sha: str, base: str = DATA_BASE) -> dict:
    path = cached_public(f"corpus/manifests/{sha}.json", sha256=sha, base=base)
    return json.loads(path.read_text())


def index_table_for_manifest(manifest: dict, base: str = DATA_BASE) -> pa.Table:
    idx = manifest["index"]
    path = cached_public(idx["key"], sha256=idx["sha256"], base=base)
    cols = ["turn_id", "rollout_id", "traj_id", "stratum", "source", "action_kind",
            "n_prefix_chars"]
    schema_names = pq.read_schema(path).names
    if "stratum_src" in schema_names:
        cols.append("stratum_src")
    return pq.read_table(path, columns=cols)


def write_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def write_json(obj, path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(body)
    os.replace(tmp, path)
    return body


def env_value(name: str) -> str:
    """Secret lookup: process env, then the repo .env (KEY=VALUE lines), then
    the validator's frozen env snapshot. Never logged."""
    if os.environ.get(name):
        return os.environ[name]
    for p in (REPO / ".env", Path(os.path.expanduser("~/.affine-validator.env"))):
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""
