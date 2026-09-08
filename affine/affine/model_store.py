"""HF model pinning, repo hygiene checks, and copy detection.

Miners submit HF repos pinned to a 40-hex git revision. Everything here runs
metadata-only on the root machine (no weight download — the eval server pulls
the full snapshot). Defenses, in intake order:

  1. Revision pinning (TOCTOU): all reads use the committed revision.
  2. Repo-name policy: pattern match + coldkey token (anti-impersonation).
  3. File hygiene: safetensors present in a canonical layout, no *.py, no
     auto_map in config.json (no remote-code execution on the eval box),
     total size under the cap (disk-DoS).
  4. Copy detection vs the king: per-file safetensors blob sha256 from the HF
     API. Identical weights ⇒ reject, unless the challenger's commit date
     provably precedes the king's — then the original author displaces the
     thief without an eval (crown_earlier).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

from . import r2, r2protocol as proto

# HF raises these for a repo/revision that no longer exists or is gated. The
# import path moved across huggingface_hub versions; support both.
try:
    from huggingface_hub.errors import (GatedRepoError,
                                         RepositoryNotFoundError,
                                         RevisionNotFoundError)
except ImportError:  # older huggingface_hub
    from huggingface_hub.utils import (GatedRepoError,
                                        RepositoryNotFoundError,
                                        RevisionNotFoundError)

log = logging.getLogger("affine.model_store")

_SAFETENSORS_SHARD_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")


@dataclass(frozen=True)
class ModelRef:
    repo: str
    revision: str  # 40-hex git commit sha

    @property
    def immutable_ref(self) -> str:
        return f"{self.repo}@{self.revision}"


@dataclass
class RepoInfo:
    files: list[str]
    config: dict
    safetensors_blobs: dict[str, str]  # filename -> sha256
    total_safetensors_bytes: int
    total_repo_bytes: int
    committed_at: datetime | None


def _api(hf_token: str) -> HfApi:
    return HfApi(token=hf_token or None)


def resolve_head_revision(repo: str, hf_token: str = "") -> str:
    """Current HEAD commit sha of a repo (used only to pin the seed king)."""
    return _api(hf_token).model_info(repo).sha


MAX_TREE_ENTRIES = 20000  # hard stop while listing an attacker-controlled repo
MAX_CONFIG_BYTES_HARD = 16 * 1024 * 1024  # refuse to download bigger configs
MAX_MANIFEST_BYTES = 2 * 1024 * 1024


class R2Reader:
    """Metadata reader for `r2://bucket/prefix/` refs: the signed manifest
    is the file tree, config.json is fetched from the prefix. Shape-compatible
    with the HF path so hygiene / arch / copy checks run unchanged."""

    def __init__(self, s3):
        self.s3 = s3

    def fetch_manifest(self, bucket: str, prefix: str) -> dict:
        raw = r2.get_bytes(self.s3, bucket, prefix + "manifest.json",
                           MAX_MANIFEST_BYTES)
        manifest = json.loads(raw)
        proto.validate_manifest_shape(manifest)
        return manifest

    def repo_info_from_manifest(self, bucket: str, prefix: str, manifest: dict,
                                uploaded_at: datetime | None = None) -> RepoInfo:
        files = [f["path"] for f in manifest["files"]] + ["manifest.json"]
        by_path = {f["path"]: f for f in manifest["files"]}
        if "config.json" not in by_path:
            raise ValueError("manifest lists no config.json")
        if int(by_path["config.json"]["size"]) > MAX_CONFIG_BYTES_HARD:
            raise ValueError("config.json exceeds the hard size cap")
        raw = r2.get_bytes(self.s3, bucket, prefix + "config.json",
                           MAX_CONFIG_BYTES_HARD)
        if proto.sha256_hex(raw) != by_path["config.json"]["sha256"]:
            raise ValueError("config.json sha256 differs from the manifest")
        try:
            config = json.loads(raw)
        except ValueError as e:
            raise ValueError(f"config.json is not JSON: {e}") from e
        blobs = {f["path"]: f["sha256"] for f in manifest["files"]
                 if f["path"].endswith(".safetensors")}
        total_st = sum(int(f["size"]) for f in manifest["files"]
                       if f["path"].endswith(".safetensors"))
        total_all = sum(int(f["size"]) for f in manifest["files"])
        if uploaded_at is not None and uploaded_at.tzinfo is None:
            uploaded_at = uploaded_at.replace(tzinfo=timezone.utc)
        return RepoInfo(files=files, config=config, safetensors_blobs=blobs,
                        total_safetensors_bytes=total_st,
                        total_repo_bytes=total_all, committed_at=uploaded_at)

    def repo_info(self, ref: ModelRef) -> RepoInfo:
        """Raises when the prefix/manifest is missing or the manifest's
        model_digest is not the pinned revision (content moved under us)."""
        bucket, prefix = proto.parse_r2_ref(ref.repo)
        manifest = self.fetch_manifest(bucket, prefix)
        if manifest["model_digest"] != ref.revision:
            raise ValueError(f"manifest digest {manifest['model_digest'][:12]} != "
                             f"pinned {ref.revision[:12]}")
        head = self.s3.head_object(Bucket=bucket, Key=prefix + "manifest.json")
        return self.repo_info_from_manifest(bucket, prefix, manifest,
                                            uploaded_at=head.get("LastModified"))

    def status(self, ref: ModelRef) -> tuple[str, RepoInfo | None]:
        bucket, prefix = proto.parse_r2_ref(ref.repo)
        exists = r2.object_exists(self.s3, bucket, prefix + "manifest.json")
        if exists is None:
            return "unknown", None
        if not exists:
            return "gone", None
        try:
            return "ok", self.repo_info(ref)
        except ValueError as e:
            # Manifest present but not the pinned content: permanently wrong.
            log.warning("r2 ref %s is not the pinned content: %s", ref.repo, e)
            return "gone", None
        except Exception:
            log.warning("availability probe inconclusive for %s", ref.repo,
                        exc_info=True)
            return "unknown", None


def fetch_repo_info(ref: ModelRef, hf_token: str = "",
                    r2_reader: R2Reader | None = None) -> RepoInfo:
    """Metadata-only snapshot of the pinned revision: file list, config.json,
    per-file blob digests, sizes, commit timestamp. Raises on missing repo or
    revision (callers record `revision_not_found`)."""
    if proto.is_r2_ref(ref.repo):
        if r2_reader is None:
            raise ValueError(f"{ref.repo}: R2 submissions are not configured")
        return r2_reader.repo_info(ref)
    api = _api(hf_token)
    tree = []
    for i, t in enumerate(api.list_repo_tree(
            ref.repo, revision=ref.revision, recursive=True, expand=True)):
        if i >= MAX_TREE_ENTRIES:
            raise ValueError(f"repo tree exceeds {MAX_TREE_ENTRIES} entries")
        tree.append(t)
    files = [getattr(t, "path", "") for t in tree]

    blobs: dict[str, str] = {}
    total_st = 0
    total_all = 0
    config_size = 0
    for t in tree:
        path = getattr(t, "path", "")
        size = int(getattr(t, "size", 0) or 0)
        total_all += size
        if path == "config.json":
            config_size = size
        if not path.endswith(".safetensors"):
            continue
        total_st += size
        lfs = getattr(t, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs else None
        if sha:
            blobs[path] = str(sha).lower().removeprefix("sha256:")

    # Guard the download of an attacker-controlled config.json by its listed
    # size before fetching it (the contract-level cap is checked in hygiene).
    if config_size > MAX_CONFIG_BYTES_HARD:
        raise ValueError(f"config.json is {config_size} bytes (> hard cap)")
    raw = api.hf_hub_download(
        ref.repo, "config.json", revision=ref.revision)
    with open(raw) as f:
        config = json.load(f)

    committed_at = None
    try:
        commits = api.list_repo_commits(ref.repo, revision=ref.revision)
        for c in commits:
            if getattr(c, "commit_id", "") == ref.revision:
                committed_at = getattr(c, "created_at", None)
                break
        if committed_at is None and commits:
            committed_at = getattr(commits[0], "created_at", None)
    except Exception:
        log.debug("commit timestamp unavailable for %s", ref.immutable_ref,
                  exc_info=True)
    if committed_at is not None and committed_at.tzinfo is None:
        committed_at = committed_at.replace(tzinfo=timezone.utc)
    return RepoInfo(files=files, config=config, safetensors_blobs=blobs,
                    total_safetensors_bytes=total_st,
                    total_repo_bytes=total_all, committed_at=committed_at)


def fetch_repo_info_or_status(ref: ModelRef, hf_token: str = "",
                              r2_reader: R2Reader | None = None,
                              ) -> tuple[str, RepoInfo | None]:
    """Probe a repo's availability while fetching its metadata.

    Returns one of:
      ("ok", RepoInfo)  — repo+revision exist AND files are downloadable.
      ("gone", None)    — repo/revision missing or gated (a *permanent* signal;
                          the caller may act on it, e.g. dethrone a dead king).
      ("unknown", None) — inconclusive (network/HF hiccup); the caller must NOT
                          treat this as gone (never dethrone on uncertainty).

    "ok" requires an uncached file HEAD, not just readable metadata: a repo
    gated *after* we first saw it keeps a public tree and a locally cached
    config.json, so fetch_repo_info alone reports it healthy while every
    weight download 403s (observed: reign #1 gated post-crown, kept earning).
    """
    if proto.is_r2_ref(ref.repo):
        if r2_reader is None:
            log.warning("cannot probe %s: R2 not configured", ref.repo)
            return "unknown", None
        return r2_reader.status(ref)
    try:
        info = fetch_repo_info(ref, hf_token)
        # HEAD the config blob at the pinned revision. Never served from the
        # local HF cache, so this is the actual download-permission check.
        get_hf_file_metadata(
            hf_hub_url(ref.repo, "config.json", revision=ref.revision),
            token=hf_token or None)
        return "ok", info
    except (RepositoryNotFoundError, RevisionNotFoundError, GatedRepoError):
        return "gone", None
    except Exception:
        log.warning("availability probe inconclusive for %s",
                    ref.immutable_ref, exc_info=True)
        return "unknown", None


def validate_repo_name(repo: str, pattern: str,
                       token_pairs: list[tuple[str, str]]) -> str | None:
    """Return a rejection reason or None. `token_pairs` are (prefix, suffix)
    anti-impersonation pairs from the submitter's hotkey/coldkey ss58; the
    repo id must contain both halves of at least one pair."""
    if not re.match(pattern, repo):
        return f"repo {repo!r} does not match required pattern {pattern!r}"
    r = repo.lower()
    if token_pairs and not any(p in r and s in r for p, s in token_pairs):
        wanted = " or ".join(f"{p}…{s}" for p, s in token_pairs)
        return (f"repo {repo!r} must embed the submitter's coldkey or hotkey "
                f"(first+last chars of the ss58, e.g. {wanted})")
    return None


def validate_repo_hygiene(info: RepoInfo, *, max_size_gb: float,
                          max_total_repo_gb: float,
                          allow_python_files: bool,
                          allow_auto_map: bool,
                          max_repo_files: int = 5000,
                          max_config_bytes: int = 1 << 20) -> str | None:
    """Return a rejection reason or None. Metadata-only; no weights touched."""
    if len(info.files) > max_repo_files:
        return f"repo lists {len(info.files)} files > {max_repo_files} cap"
    py_files = sorted(f for f in info.files if f.endswith(".py"))
    if py_files and not allow_python_files:
        return f"repo ships *.py files (not allowed): {py_files[:3]}"
    if "auto_map" in info.config and not allow_auto_map:
        return "auto_map present in config.json (custom modeling code is not allowed)"
    config_bytes = len(json.dumps(info.config))
    if config_bytes > max_config_bytes:
        return f"config.json is {config_bytes} bytes > {max_config_bytes} cap"

    st_files = [f for f in info.files if f.endswith(".safetensors")]
    if not st_files:
        return "no .safetensors files in repo"
    has_single = "model.safetensors" in info.files
    has_index = "model.safetensors.index.json" in info.files
    has_shards = any(_SAFETENSORS_SHARD_RE.match(f) for f in st_files)
    if not (has_single or (has_index and has_shards)):
        if has_shards and not has_index:
            return "missing model.safetensors.index.json for sharded layout"
        return f"safetensors present but not in canonical layout: {st_files[:3]}"

    size_gb = info.total_safetensors_bytes / 1e9
    if size_gb > max_size_gb:
        return (f"oversized: {size_gb:.1f} GB of .safetensors > {max_size_gb:.0f} GB "
                f"cap (check for fp32 weights, duplicated shards, optimizer state)")
    # The safetensors cap alone lets a miner attach hundreds of GB of junk
    # (datasets, gguf, fp32 duplicates under other extensions) that the eval
    # pod would still have to download: cap the WHOLE repo too.
    total_gb = info.total_repo_bytes / 1e9
    if total_gb > max_total_repo_gb:
        return (f"oversized: {total_gb:.1f} GB total repo > {max_total_repo_gb:.0f} GB "
                f"cap (non-weight junk files count against the pod's disk)")
    return None


def validate_repo_arch(info: RepoInfo, pinned: dict,
                       alternatives: list[dict] | tuple[dict, ...] = ()) -> str | None:
    """Return a rejection reason or None. `pinned` is a nested dict of
    config.json keys that must match exactly (subset match: keys absent from
    `pinned` are unconstrained). Pinning the compute-graph shape to the genesis
    family keeps every crown a fine-tune of the seed model — in particular it
    excludes uploading the frozen teacher itself, whose thoughts would land
    in-band on G and top R by construction (it IS the distillation target).
    `alternatives` are further profiles, any one of which also admits
    (2026-09-04: the text-only Qwen3_5MoeForCausalLM extraction, whose
    config.json is the genesis text_config flattened to the root).
    Metadata-only; empty `pinned` disables the check."""

    def walk(want: dict, have: object, path: str) -> str | None:
        if not isinstance(have, dict):
            return f"config.json {path or '<root>'} is not a table"
        for key, expect in want.items():
            where = f"{path}.{key}" if path else key
            if key not in have:
                return f"config.json missing pinned key {where}"
            got = have[key]
            if isinstance(expect, dict):
                fault = walk(expect, got, where)
                if fault:
                    return fault
            elif got != expect:
                return (f"architecture mismatch at {where}: "
                        f"{got!r} != pinned {expect!r}")
        return None

    fault = walk(pinned, info.config, "")
    if fault is None:
        return None
    if any(walk(alt, info.config, "") is None for alt in alternatives):
        return None
    return f"arch not pinned to the genesis family: {fault}"


@dataclass
class CopyVerdict:
    action: str  # "reject" | "crown_earlier"
    reason: str
    challenger_committed_at: str | None
    king_committed_at: str | None


def check_model_copy(chall_ref: ModelRef, chall_info: RepoInfo,
                     king_ref: ModelRef, king_info: RepoInfo | None) -> CopyVerdict | None:
    """Detect a weight-for-weight copy of the king. None = not a copy (or
    undeterminable — fail open, the duel handles ties via the k·SE floor)."""
    if chall_ref.repo == king_ref.repo and chall_ref.revision == king_ref.revision:
        return CopyVerdict(
            action="reject",
            reason="challenger is identical to the current king (same repo and revision)",
            challenger_committed_at=None, king_committed_at=None)
    if king_info is None:
        return None
    cb, kb = chall_info.safetensors_blobs, king_info.safetensors_blobs
    if not cb or not kb:
        return None
    # Compare digest MULTISETS, not filename→digest maps: renaming shards or
    # padding the repo with an extra junk .safetensors must not evade the
    # check. It is a copy when every king blob appears in the challenger.
    c_counts, k_counts = Counter(cb.values()), Counter(kb.values())
    if any(c_counts[d] < n for d, n in k_counts.items()):
        return None

    c_ts, k_ts = chall_info.committed_at, king_info.committed_at
    meta = {
        "challenger_committed_at": c_ts.isoformat() if c_ts else None,
        "king_committed_at": k_ts.isoformat() if k_ts else None,
    }
    base = (f"all {len(kb)} king safetensors blobs present in challenger; "
            f"challenger committed {meta['challenger_committed_at']}, "
            f"king committed {meta['king_committed_at']}")
    # Fail-safe: never crown an "earlier original" without both timestamps.
    if c_ts is None or k_ts is None:
        return CopyVerdict(action="reject",
                           reason=f"copy of king (timestamps unavailable): {base}", **meta)
    if c_ts < k_ts:
        return CopyVerdict(
            action="crown_earlier",
            reason=f"identical weights with earlier commit time — original author: {base}",
            **meta)
    return CopyVerdict(action="reject", reason=f"copy of king: {base}", **meta)
