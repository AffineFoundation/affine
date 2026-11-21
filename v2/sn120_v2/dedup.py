"""Duplicate-batch evidence, pure clustering, and low-rank update cosine.

Stage A compares exact batch hashes and normalized (prompt, completion) sets.
Stage B compares scaled LoRA updates without materializing delta-W. Controller
I/O stays at the boundary: malformed or missing evidence never merges an entry.
"""
from __future__ import annotations

import json
import math
import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .canonical import canonical_json, digest, is_sha256, sha256_hex
from .errors import ContractError, VerifyError
from .params import Params
from .types import DUPLICATE, Entry, EntryDecision, FrozenBatch, LoRAArtifact, Registration, TrainReport

STAGE_EXACT = "exact_batch"
STAGE_ROLLOUT = "rollout_overlap"
STAGE_KING_EXACT = "past_king_exact"
STAGE_KING_ROLLOUT = "past_king_rollout"
STAGE_LORA = "lora_cosine"
STAGES = frozenset((STAGE_EXACT, STAGE_ROLLOUT, STAGE_KING_EXACT, STAGE_KING_ROLLOUT, STAGE_LORA))

def normalized_rollouts(raw: bytes) -> frozenset[str] | None:
    """Hashes of stripped, whitespace-collapsed (prompt, output), rewards excluded.

    All accepted affine batch versions share this group shape. An empty or
    unparsable set is unknown evidence and fails open.
    """
    try:
        body = json.loads(raw)
        if not isinstance(body, dict) or not isinstance(body.get("groups"), list):
            return None
        hashes: set[str] = set()
        for group in body["groups"]:
            if not isinstance(group, dict) or not isinstance(group.get("prompt"), str) or not isinstance(group.get("rollouts"), list):
                return None
            prompt = " ".join(group["prompt"].split())
            for rollout in group["rollouts"]:
                if not isinstance(rollout, dict) or not isinstance(rollout.get("output"), str):
                    return None
                text = canonical_json([prompt, " ".join(rollout["output"].split())])
                hashes.add(sha256_hex(text))
        return frozenset(hashes) or None
    except (ValueError, TypeError, AttributeError):
        return None


def rollout_overlap(left: frozenset[str] | None, right: frozenset[str] | None) -> float | None:
    if not left or not right:
        return None
    return len(left & right) / min(len(left), len(right))


@dataclass(frozen=True)
class PriorityEntry:
    entry: Entry
    first_seen_unix: int


def _member(entry: Entry, first_seen: int, stage: str = "", similarity: float = 1.0) -> dict[str, Any]:
    return {
        "queue_pos": entry.queue_pos, "uid": entry.reg.uid, "hotkey": entry.reg.hotkey,
        "registration_block": entry.reg.registration_block, "batch_sha256": entry.batch.batch_sha256,
        "first_seen_unix": first_seen if math.isfinite(first_seen) else None, "stage": stage, "similarity": similarity,
    }


def _decision(member: Mapping[str, Any], reason: str) -> EntryDecision:
    return EntryDecision(member["queue_pos"], member["uid"], member["hotkey"], member["registration_block"],
                         member["batch_sha256"], DUPLICATE, reason=reason)


def _finish(kept: list[dict], dropped: list[dict]) -> tuple[list[EntryDecision], list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in dropped:
        groups[row["keeper_queue_pos"]].append(row)
    decisions, clusters = [], []
    for keeper in kept:
        copies = groups.get(keeper["queue_pos"], [])
        for row in copies:
            reason = (f"duplicate of queue_pos {keeper['queue_pos']} (UID {keeper['uid']}, {keeper['hotkey']}) "
                      f"at {row['stage']}: similarity {row['similarity']:.6g}")
            decisions.append(_decision(row, reason))
        if copies:
            clusters.append({"keeper": keeper, "dropped": copies})
    return sorted(decisions, key=lambda row: row.queue_pos), clusters


def cluster_entries(
    entries: Sequence[Entry],
    first_seen: Mapping[str, int],
    rollout_sets: Mapping[str, frozenset[str] | None],
    params: Params,
) -> tuple[list[Entry], list[EntryDecision], list[dict]]:
    """Greedy stage-A clustering. Dropped entries never become comparison anchors."""
    if not params.dedup_enabled or not entries:
        return list(entries), [], []
    ordered = sorted(entries, key=lambda entry: (first_seen.get(entry.batch.batch_sha256, math.inf), entry.queue_pos))
    kept: list[dict] = []
    kept_hashes: list[frozenset[str] | None] = []
    dropped: list[dict] = []
    for entry in ordered:
        seen = first_seen.get(entry.batch.batch_sha256, math.inf)
        current_hashes = rollout_sets.get(entry.batch.batch_sha256)
        candidate = _member(entry, seen)
        copy = None
        for index, keeper in enumerate(kept):
            if entry.batch.batch_sha256 == keeper["batch_sha256"]:
                copy = (STAGE_EXACT, 1.0)
                break
            similarity = rollout_overlap(current_hashes, kept_hashes[index])
            if similarity is not None and similarity >= params.dedup_rollout_overlap:
                copy = (STAGE_ROLLOUT, similarity)
                break
        if copy is None:
            kept.append(candidate)
            kept_hashes.append(current_hashes)
        else:
            dropped.append(candidate | {"stage": copy[0], "similarity": copy[1], "keeper_queue_pos": keeper["queue_pos"]})
    decisions, clusters = _finish(kept, dropped)
    positions = {row.queue_pos for row in decisions}
    # Restore original queue order after first-seen ordering selected keepers.
    return [entry for entry in entries if entry.queue_pos not in positions], decisions, clusters


def cluster_past_kings(
    entries: Sequence[Entry],
    first_seen: Mapping[str, int],
    rollout_sets: Mapping[str, frozenset[str] | None],
    kings: Sequence[Mapping[str, Any]],
    params: Params,
) -> tuple[list[Entry], list[EntryDecision], list[dict]]:
    """Copies of payout-window winners lose to the winner, regardless of first_seen."""
    if not params.dedup_enabled:
        return list(entries), [], []
    keep, decisions, clusters = [], [], []
    for entry in entries:
        matched = None
        for king in kings:
            if entry.batch.batch_sha256 == king["batch_sha256"]:
                matched = king, STAGE_KING_EXACT, 1.0
                break
            similarity = rollout_overlap(rollout_sets.get(entry.batch.batch_sha256), king.get("rollouts"))
            if similarity is not None and similarity >= params.dedup_rollout_overlap:
                matched = king, STAGE_KING_ROLLOUT, similarity
                break
        if matched is None:
            keep.append(entry)
            continue
        king, stage, similarity = matched
        keeper = {"queue_pos": -1, "uid": king["uid"], "hotkey": king["hotkey"],
                  "registration_block": 0, "batch_sha256": king["batch_sha256"],
                  "first_seen_unix": king.get("first_seen_unix", 0), "stage": "", "similarity": 1.0}
        row = _member(entry, first_seen.get(entry.batch.batch_sha256, math.inf), stage, similarity) | {"keeper_queue_pos": -1}
        reason = f"duplicate of past king UID {king['uid']} ({king['hotkey']}) at {stage}: similarity {similarity:.6g}"
        decisions.append(_decision(row, reason))
        clusters.append({"keeper": keeper, "dropped": [row]})
    return keep, decisions, clusters


# ---- safetensors and low-rank cosine ----

def safetensors(raw: bytes) -> dict[str, tuple[tuple[int, ...], str, memoryview]]:
    """Read float32 tensors from one safetensors file, without third-party imports."""
    if len(raw) < 8:
        raise VerifyError("short safetensors header")
    size = struct.unpack("<Q", raw[:8])[0]
    if size > len(raw) - 8:
        raise VerifyError("truncated safetensors header")
    try:
        header = json.loads(raw[8:8 + size])
    except ValueError as exc:
        raise VerifyError(f"bad safetensors JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise VerifyError("safetensors header is not an object")
    tensors = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, dict) or spec.get("dtype") != "F32" or len(spec.get("shape", ())) != 2:
            raise VerifyError(f"LoRA tensor {name} is not a float32 matrix")
        begin, end = spec.get("data_offsets", (None, None))
        shape = spec["shape"]
        if type(begin) is not int or type(end) is not int or not all(type(n) is int and n >= 0 for n in shape):
            raise VerifyError(f"LoRA tensor {name} has invalid bounds")
        start, stop = 8 + size + begin, 8 + size + end
        if not 0 <= start <= stop <= len(raw) or end - begin != 4 * math.prod(shape):
            raise VerifyError(f"LoRA tensor {name} has inconsistent size")
        tensors[name] = (tuple(shape), "<f", memoryview(raw)[start:stop])
    return tensors


def load_safetensors(raw: bytes) -> dict[str, tuple[tuple[int, ...], str, memoryview]]:
    return safetensors(raw)


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


def gram(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    """Rows of two small matrices as one ordinary float64 Python Gram matrix."""
    return [[dot(a, b) for b in right] for a in left]


def matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    if not left or not right:
        return []
    cols = list(zip(*right))
    return [[math.fsum(a * b for a, b in zip(row, col)) for col in cols] for row in left]


def trace_product(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    product = matmul(left, right)
    return math.fsum(product[i][i] for i in range(min(len(product), len(product[0]) if product else 0)))


@dataclass
class LoraMatrices:
    """Per-module A rows, B columns and scale. Gram pieces are cached per LoRA."""
    modules: dict[str, tuple[list[list[float]], list[list[float]], float]]

    @classmethod
    def read(cls, artifact: LoRAArtifact, get_file: Callable[[str], bytes]) -> LoraMatrices:
        raw = get_file(f"{artifact.prefix}/adapter_model.safetensors")
        if sha256_hex(raw) != artifact.files.get("adapter_model.safetensors"):
            raise VerifyError("LoRA weights do not match their recorded sha256")
        tensors = safetensors(raw)
        a_names = {name[:-len(".lora_A.weight")]: name for name in tensors if name.endswith(".lora_A.weight")}
        b_names = {name[:-len(".lora_B.weight")]: name for name in tensors if name.endswith(".lora_B.weight")}
        if not a_names or set(a_names) != set(b_names):
            raise VerifyError("LoRA does not contain matching A and B matrices")
        modules = {}
        try:
            import numpy as np
        except ImportError:
            np = None
        for base in sorted(a_names):
            (ra, inputs), _, a_raw = tensors[a_names[base]]
            (outputs, rb), _, b_raw = tensors[b_names[base]]
            if ra != rb or ra != artifact.rank:
                raise VerifyError(f"LoRA rank mismatch at {base}")
            if np is not None:
                a = np.frombuffer(a_raw, dtype="<f4").reshape(ra, inputs)
                b = np.frombuffer(b_raw, dtype="<f4").reshape(outputs, rb)
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise VerifyError(f"LoRA has non-finite values at {base}")
            else:
                a_values = list(struct.unpack(f"<{ra * inputs}f", a_raw))
                b_values = list(struct.unpack(f"<{outputs * rb}f", b_raw))
                a = [a_values[i * inputs:(i + 1) * inputs] for i in range(ra)]
                b = [b_values[i * rb:(i + 1) * rb] for i in range(outputs)]
                if not all(math.isfinite(x) for row in (*a, *b) for x in row):
                    raise VerifyError(f"LoRA has non-finite values at {base}")
            modules[base] = (a, b, artifact.alpha / ra)
        if not modules:
            raise VerifyError("LoRA has no matrices")
        return cls(modules)

    def norms(self) -> dict[str, float]:
        try:
            import numpy as np
        except ImportError:
            np = None
        out = {}
        for name, (a, b, scale) in self.modules.items():
            if np is not None:
                # The rank x rank Gram products are float64, not float32 products cast afterward.
                a64, b64 = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
                aa_t = a64 @ a64.T
                bb = b64.T @ b64
                value = float(np.sum(bb * aa_t.T, dtype=np.float64) * scale * scale)
            else:
                # B^T B is rank x rank; A A^T is the matching rank x rank Gram.
                value = trace_product(gram(list(zip(*b)), list(zip(*b))), gram(a, a)) * scale * scale
            if not math.isfinite(value) or value < 0:
                raise VerifyError(f"negative or non-finite update norm at {name}")
            # A zero target module contributes nothing; only a zero global norm is unusable.
            out[name] = value
        return out

    def norm(self) -> float:
        return math.sqrt(math.fsum(self.norms().values()))


def cosine(left: LoraMatrices, right: LoraMatrices, left_norms=None, right_norms=None) -> float | None:
    if not left.modules or set(left.modules) != set(right.modules):
        return None
    try:
        import numpy as np
    except ImportError:
        np = None
    terms = []
    for name in sorted(left.modules):
        a1, b1, s1 = left.modules[name]
        a2, b2, s2 = right.modules[name]
        if np is not None:
            if a1.shape[1] != a2.shape[1] or b1.shape[0] != b2.shape[0]:
                return None
            # Only these two modules are cast: B1^T B2 and A2 A1^T are float64, with no delta-W allocation.
            a1_64, b1_64 = np.asarray(a1, dtype=np.float64), np.asarray(b1, dtype=np.float64)
            a2_64, b2_64 = np.asarray(a2, dtype=np.float64), np.asarray(b2, dtype=np.float64)
            bt_b = b1_64.T @ b2_64
            aa_t = a2_64 @ a1_64.T
            value = float(np.sum(bt_b * aa_t.T, dtype=np.float64) * s1 * s2)
        else:
            if len(a1[0]) != len(a2[0]) or len(b1) != len(b2):
                return None
            # B1^T B2 and A2 A1^T: both rank x rank, as in the NumPy path.
            b1_cols, b2_cols = list(zip(*b1)), list(zip(*b2))
            bt_b = [[dot(col1, col2) for col2 in b2_cols] for col1 in b1_cols]
            aa_t = gram(a2, a1)
            value = trace_product(bt_b, aa_t) * s1 * s2
        if not math.isfinite(value):
            return None
        terms.append(value)
    inner = math.fsum(terms)
    n1 = math.sqrt(math.fsum((left_norms or left.norms()).values()))
    n2 = math.sqrt(math.fsum((right_norms or right.norms()).values()))
    if not n1 or not n2 or not math.isfinite(n1) or not math.isfinite(n2) or not math.isfinite(inner):
        return None
    value = inner / (n1 * n2)
    return value if math.isfinite(value) else None


def _cluster_loaded(ordered, matrices, artifacts, first_seen, params, say):
    kept, dropped = [], []
    norms = {}
    for entry in ordered:
        if entry.queue_pos not in matrices:
            kept.append(_member(entry, first_seen.get(entry.batch.batch_sha256, math.inf)))
            continue
        try:
            norms[entry.queue_pos] = matrices[entry.queue_pos].norms()
            if not math.fsum(norms[entry.queue_pos].values()) > 0:
                raise VerifyError("zero global update norm")
        except Exception as exc:
            say(f"dedup stage B: queue_pos {entry.queue_pos} has an unusable norm: {exc}")
            kept.append(_member(entry, first_seen.get(entry.batch.batch_sha256, math.inf)))
            continue
        match = None
        for keeper in kept:
            other = matrices.get(keeper["queue_pos"])
            if other is None or keeper["queue_pos"] not in norms:
                continue
            left, right = matrices[entry.queue_pos], other
            same = artifacts[entry.queue_pos].report.seed == artifacts[keeper["queue_pos"]].report.seed
            floor = params.dedup_cos_shared_seed if same else params.dedup_cos_own_seed
            similarity = cosine(left, right, norms[entry.queue_pos], norms[keeper["queue_pos"]])
            if similarity is not None and similarity >= floor:
                match = keeper, similarity
                break
        if match is None:
            kept.append(_member(entry, first_seen.get(entry.batch.batch_sha256, math.inf)))
        else:
            keeper, similarity = match
            dropped.append(_member(entry, first_seen.get(entry.batch.batch_sha256, math.inf), STAGE_LORA, similarity)
                           | {"keeper_queue_pos": keeper["queue_pos"]})
    return _finish(kept, dropped)


def stage_b(
    entries: Sequence[Entry], artifacts: Mapping[int, LoRAArtifact], first_seen: Mapping[str, int], params: Params,
    get_file: Callable[[str], bytes], expected_seeds: Mapping[int, int], excluded: Iterable[int] = (),
    log: Callable[[str], None] | None = None,
) -> tuple[list[EntryDecision], list[dict]]:
    """Whole-stage fail open: bad evidence is logged; infrastructure errors return no drops."""
    if not params.dedup_enabled:
        return [], []
    say = log or (lambda text: None)
    null = set(excluded)
    ordered = sorted((entry for entry in entries if entry.queue_pos in artifacts and entry.queue_pos not in null),
                     key=lambda entry: (first_seen.get(entry.batch.batch_sha256, math.inf), entry.queue_pos))
    matrices = {}
    for entry in ordered:
        artifact = artifacts[entry.queue_pos]
        try:
            if artifact.batch_sha256 != entry.batch.batch_sha256 or artifact.report.seed != expected_seeds.get(entry.queue_pos):
                raise VerifyError("batch or training seed verification failed")
            matrices[entry.queue_pos] = LoraMatrices.read(artifact, get_file)
        except Exception as exc:  # per-entry fail open; never merge on unknown evidence
            say(f"dedup stage B: queue_pos {entry.queue_pos} cannot be compared: {type(exc).__name__}: {exc}")
    try:
        return _cluster_loaded(ordered, matrices, artifacts, first_seen, params, say)
    except Exception as exc:
        say(f"dedup stage B failed open with no drops: {type(exc).__name__}: {exc}")
        return [], []


def dryrun(paths: Sequence[Path], params: Params, first_seen_unix: Sequence[int] | None = None,
           lora_dirs: Sequence[Path] = ()) -> dict[str, Any]:
    """Offline analysis of explicit local files. No writes, cache, state directory, store or network.

    ``lora_dirs`` contain ``adapter_model.safetensors``, ``adapter_config.json`` and ``train_report.json``.
    It is deliberately an explicit-file tool: only the operator can choose which frozen batch and result belong
    together. Missing or invalid evidence is unknown and never creates a duplicate.
    """
    from .admission import Batch, Invalid, check_batch, parse_groups, signal_counts

    if first_seen_unix is not None and len(first_seen_unix) != len(paths):
        raise ContractError("--first-seen must occur once per --batch")
    if lora_dirs and len(lora_dirs) != len(paths):
        raise ContractError("--lora must occur once per --batch, or not at all")
    entries, reports, raws = [], [], []
    for pos, path in enumerate(paths):
        raw = path.read_bytes()
        body = json.loads(raw)
        hotkey = body.get("hotkey") if isinstance(body, dict) else None
        if not isinstance(hotkey, str) or not hotkey:
            raise ContractError(f"{path}: a frozen batch needs its hotkey to check its own rules")
        try:
            batch: Batch | None = check_batch(raw, hotkey, params)
            error = None
        except Invalid as exc:
            batch, error = None, {"rule": exc.rule, "reason": str(exc)}
        train = body.get("train", {}) if isinstance(body, dict) else {}
        recipe = params.recipe.for_miner(train)
        floor = max(params.min_signal_rollouts, recipe.minibatches)
        # Read-only diagnostics: even a rejected file gets its actual activity count.
        # This is not a second admission path; ``admission`` retains the real rejection.
        active = signal_counts(batch.groups, recipe)[0] if batch is not None else None
        if active is None:
            try:
                kept = parse_groups(body, hotkey, params.max_prompts, params.max_rollouts)
                active = signal_counts(kept, recipe)[0]
            except (Invalid, ContractError, OverflowError, TypeError, ValueError):
                active = None
        groups = body.get("groups") if isinstance(body, dict) else None
        rollouts = sum(len(group.get("rollouts", ())) for group in groups) if isinstance(groups, list) else 0
        reg = Registration(pos, hotkey, pos)
        frozen = FrozenBatch(hotkey, sha256_hex(raw), str(path), body.get("version", ""), len(groups or ()), rollouts, train)
        entries.append(Entry(pos, reg, frozen))
        cap = recipe.global_norm_cap
        reports.append({"queue_pos": pos, "batch_sha256": frozen.batch_sha256, "rollouts": rollouts,
                        "nonzero_advantage_rollouts": active, "min_signal_rollouts": floor,
                        "would_fail_min_rollouts": active is None or active < floor,
                        "static_null_update": cap is not None and cap < params.min_step_norm,
                        "step_norm": None, "step_norm_nonfinite": False, "would_be_null_update": None,
                        "admission": error})
        raws.append(raw)
    seen = {entry.batch.batch_sha256: (first_seen_unix[entry.queue_pos] if first_seen_unix is not None else entry.queue_pos)
            for entry in entries}
    hashes = {entry.batch.batch_sha256: normalized_rollouts(raw) for entry, raw in zip(entries, raws)}
    _, stage_a_rows, stage_a_clusters = cluster_entries(entries, seen, hashes, params)
    artifacts, expected, excluded = {}, {}, set()
    for pos, directory in enumerate(lora_dirs):
        try:
            report = TrainReport.from_json((directory / "train_report.json").read_bytes())
            config = json.loads((directory / "adapter_config.json").read_bytes())
            if not isinstance(config, dict):
                raise ContractError("adapter_config.json must be an object")
            recipe = params.recipe.for_miner(entries[pos].batch.train)
            rank, alpha = int(config.get("r", recipe.rank)), float(config.get("lora_alpha", recipe.alpha))
            files = {"adapter_model.safetensors": sha256_hex((directory / "adapter_model.safetensors").read_bytes())}
            artifact = LoRAArtifact(digest(files), files, {name: (directory / name).stat().st_size for name in files},
                                    str(directory), "", entries[pos].batch.batch_sha256, params.recipe_id,
                                    rank, alpha, tuple(config.get("target_modules", recipe.target_modules)), report)
            artifacts[pos] = artifact
            expected[pos] = report.seed
            norm = report.step_norm
            reports[pos] |= {"step_norm": norm if math.isfinite(norm) else None,
                             "step_norm_nonfinite": not math.isfinite(norm),
                             "would_be_null_update": not math.isfinite(norm) or norm < params.min_step_norm}
            if reports[pos]["would_be_null_update"]:
                excluded.add(pos)
        except (OSError, ValueError, TypeError, KeyError, ContractError) as exc:
            reports[pos]["lora_error"] = f"{type(exc).__name__}: {exc}"
    stage_a_dropped = {row.queue_pos for row in stage_a_rows}
    # The controller never lets a Stage-A duplicate become a Stage-B keeper.
    survivors = [entry for entry in entries if entry.queue_pos not in stage_a_dropped]
    stage_b_rows, stage_b_clusters = stage_b(survivors, artifacts, seen, params,
                                               lambda key: Path(key).read_bytes(), expected, excluded)
    return {"read_only": True, "entries": reports,
            "exact_duplicate_clusters": [
                {"keeper": cluster["keeper"],
                 "dropped": [row for row in cluster["dropped"] if row["stage"] == STAGE_EXACT]}
                for cluster in stage_a_clusters
                if any(row["stage"] == STAGE_EXACT for row in cluster["dropped"])
            ],
            "stage_a": {"decisions": [row.to_dict() for row in stage_a_rows], "clusters": stage_a_clusters},
            "stage_b": {"decisions": [row.to_dict() for row in stage_b_rows], "clusters": stage_b_clusters,
                        "available": bool(lora_dirs)}}
