"""Tooling for the published sn120-math-prior-v1 release.

Imported only by ``ingest-prior``: the controller never imports pyarrow.
"""
from __future__ import annotations

import collections
import datetime
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from ...canonical import canonical_json, file_sha256, sha256_hex
from ...errors import V2Error
from . import (
    MANIFEST, MANDATE, MAX_ANSWER_CHARS, MAX_PROBLEM_CHARS, MIN_ANSWER_CHARS,
    UNSCREENED_METHOD, load_reserve, normalise, read_raw, reserve_dir,
)

PRIOR_NAME, PRIOR_VERSION = "sn120-math-prior", "v1"
BAND = (0.15, 0.85)
NO_SIGNAL_SHARE = 0.10
COLUMNS = ("id", "problem", "answer", "answer_type", "source", "source_id", "license",
           "subsource", "topic", "difficulty_meta", "also_in", "in_sn120_pool")
PASS_FIELDS = ("pass_rate_72b_tir", "pass_72b_tir", "llama8b_solve", "medium_pass", "pass_rate", "accuracy")
PREFIXES = {
    "AI-MO/NuminaMath-1.5": "numinaprior",
    "BytedTsinghua-SIA/DAPO-Math-17k": "dapomathprior",
    "MathArena/arxivmath-training": "arxivprior",
    "POLARIS-Project/Polaris-Dataset-53K": "polarisprior",
    "SynthLabsAI/Big-Math-RL-Verified": "bigmath",
    "agentica-org/DeepScaleR-Preview-Dataset": "deepscaleprior",
    "nvidia/AceReason-Math": "acereasonprior",
    "nvidia/Nemotron-Math-v2": "nemoprior",
    "nvidia/OpenMathReasoning": "openmathprior",
    "open-r1/OpenR1-Math-220k": "openr1prior",
    "ycchen/Crystal-Math-Preview": "crystalprior",
    "zwhe99/DeepMath-103K": "deepmathprior",
}
# k/8 is the published difficulty label, not a pass rate. This deterministic
# direction-reversing map is an estimate, not a newly fitted difficulty model.
DIFFICULTY_RATES = {f"{number}/8": 1.0 - number / 8 for number in range(9)} | {
    "easy": 0.85, "medium": 0.5, "hard": 0.15,
}


def _unit(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 else None


def prior_p(meta: Any) -> float | None:
    """Published rate, documented aggregate or difficulty map; otherwise None.

    Direct rates take precedence. Nemotron uses the unweighted mean of its
    published gpt-oss rates. OpenR1 r1_correct is a count out of eight. Polaris
    k/8 maps to 1-k/8. Out-of-domain rates/counts are not silently accepted.
    """
    if not isinstance(meta, dict):
        return None
    for name in PASS_FIELDS:
        value = _unit(meta.get(name))
        if value is not None:
            return value
    rates = meta.get("gptoss_acc")
    if isinstance(rates, dict) and rates:
        values = [_unit(value) for value in rates.values()]
        if all(value is not None for value in values):
            return sum(values) / len(values)
    correct = meta.get("r1_correct")
    if type(correct) in (int, float) and math.isfinite(correct) and 0 <= correct <= 8:
        return float(correct / 8)
    return DIFFICULTY_RATES.get(meta.get("polaris_diff"))


def metadata(raw: Any) -> dict[str, Any]:
    if raw in (None, ""):
        return {}
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def verify_release(root: Path) -> dict[str, Any]:
    """Check every published file's SHA-256 before decoding any parquet row."""
    try:
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        shards = manifest["shards"]
        if manifest["name"] != PRIOR_NAME or manifest["version"] != PRIOR_VERSION or not isinstance(shards, list) or not shards:
            raise V2Error("prior release must be sn120-math-prior v1 with at least one shard")
        expected_files, rows = set(), 0
        for shard in shards:
            if set(shard) != {"file", "rows", "sha256"} or not re.fullmatch(r"prior-[0-9]{2}-of-[0-9]{2}\.parquet", shard["file"]):
                raise V2Error("invalid prior shard manifest row")
            if type(shard["rows"]) is not int or shard["rows"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", shard["sha256"]):
                raise V2Error("invalid prior shard rows or sha256")
            if shard["file"] in expected_files:
                raise V2Error("prior MANIFEST repeats a shard")
            path = root / shard["file"]
            if not path.is_file() or file_sha256(path) != shard["sha256"]:
                raise V2Error(f"prior shard {shard['file']} is missing or does not match MANIFEST sha256")
            expected_files.add(shard["file"])
            rows += shard["rows"]
        if manifest.get("rows") != rows or manifest.get("columns") != list(COLUMNS):
            raise V2Error("prior MANIFEST row count or column list is inconsistent")
        unexpected = {path.name for path in root.glob("*.parquet")} - expected_files
        if unexpected:
            raise V2Error(f"unlisted prior parquet file: {sorted(unexpected)[0]}")
        return manifest
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise V2Error(f"cannot read prior release: {exc}") from exc


def read_rows(root: Path, manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # optional tooling dependency
    except ImportError as exc:
        raise V2Error("ingest-prior needs pyarrow; install v2's 'prior' extra") from exc
    try:
        for shard in manifest["shards"]:
            parquet = pq.ParquetFile(root / shard["file"])
            if parquet.schema_arrow.names != list(COLUMNS) or parquet.metadata.num_rows != shard["rows"]:
                raise V2Error(f"prior shard {shard['file']} has an unexpected schema or row count")
            for batch in parquet.iter_batches(batch_size=4096):
                for row in batch.to_pylist():
                    yield row
    except V2Error:
        raise
    except Exception as exc:
        raise V2Error(f"cannot decode prior parquet: {type(exc).__name__}: {exc}") from exc


def _slug(source: str, license_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (source + "-" + license_name).lower()).strip("-")[:64]


def _existing(root: Path) -> tuple[set[str], set[str], list[dict[str, Any]], dict[str, int]]:
    reserve, catalog = load_reserve(root, with_catalog=True, allow_retired=True)
    ids = set(catalog.task_ids) | reserve.withdrawn_ids
    problems = {normalise(catalog.task(task_id).problem) for task_id in catalog.task_ids}
    shards = []
    next_numbers: dict[str, int] = {}
    for shard in reserve.shards:
        rows = [json.loads(line) for line in (root / "shards" / shard.shard).read_bytes().splitlines()]
        shards.append(shard)
        for row in rows:
            ids.add(row["id"])
            if row["audit"]["method"] == UNSCREENED_METHOD:
                # The source's original prior id is retained after allocating
                # the normal <prefix>:<next number> reserve id.
                ids.add(row["source"].rsplit(":", 1)[-1])
            problems.add(normalise(row["problem"]))  # withdrawn rows remain dedupe evidence
            prefix, number = row["id"].rsplit(":", 1)
            if number.isdigit():
                next_numbers[prefix] = max(next_numbers.get(prefix, -1), int(number))
    return ids, problems, shards, next_numbers


def _valid(row: dict[str, Any]) -> bool:
    from ..math_grader import grade_answer
    return grade_answer(row["answer"], row["problem"], "<think>x</think> \\boxed{" + row["answer"] + "}", False).code == "P"


def _select(rows: Iterable[dict[str, Any]], n: int, seed: str, low: float, high: float,
            no_signal_share: float, ids: set[str], problems: set[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    filtered: dict[str, int] = collections.Counter()
    sources: dict[str, int] = collections.Counter()
    no_signal = 0
    no_signal_cap = int(n * no_signal_share)
    seen: set[str] = set()
    for row in rows:
        source = str(row["source"]).split("@", 1)[0]
        prefix = PREFIXES.get(source)
        if prefix is None:
            filtered["unknown_source"] += 1
            continue
        original_id = str(row["id"])
        if original_id in seen:
            filtered["duplicate_input_id"] += 1
            continue
        seen.add(original_id)
        meta = metadata(row.get("difficulty_meta"))
        p = prior_p(meta)
        if row.get("in_sn120_pool"):
            filtered["in_sn120_pool"] += 1
        elif meta.get("key_from_majority") in (True, "True"):
            filtered["key_from_majority"] += 1
        elif p is not None and not low <= p <= high:
            filtered["outside_prior_band"] += 1
        elif original_id in ids or normalise(str(row["problem"])) in problems:
            filtered["already_in_base_reserve_or_retired"] += 1
        elif not isinstance(row["problem"], str) or not isinstance(row["answer"], str) or not row["problem"].strip() or len(row["problem"]) > MAX_PROBLEM_CHARS:
            filtered["problem_length"] += 1
        elif not MIN_ANSWER_CHARS <= len(row["answer"].strip()) <= MAX_ANSWER_CHARS or any(char in row["answer"] for char in "\r\n\x00"):
            filtered["answer_format"] += 1
        elif MANDATE in row["problem"]:
            filtered["mandate"] += 1
        else:
            candidates.append((sha256_hex(f"{seed}|{original_id}".encode()), row | {"prior_p": p, "_source_key": source}))
    kept: list[dict[str, Any]] = []
    for _, row in sorted(candidates, key=lambda pair: pair[0]):
        if len(kept) == n:
            break
        if row["prior_p"] is None and no_signal >= no_signal_cap:
            filtered["no_signal_over_share"] += 1
            continue
        source = row["_source_key"]
        if sources[source] >= n * 0.40:
            filtered["source_cap"] += 1
            continue
        if not _valid(row):
            # The release self-check never bypasses this checkout's grader.
            # A rejected row is replaced in the same deterministic order.
            filtered["grader_self_check"] += 1
            continue
        kept.append(row)
        sources[source] += 1
        no_signal += int(row["prior_p"] is None)
    if len(kept) != n:
        raise V2Error(f"only {len(kept)} eligible rows remain after filters and caps; requested {n}")
    return kept, dict(sorted(filtered.items()))


def _validate(row: dict[str, Any]) -> None:
    problem, answer = row["problem"], row["answer"]
    if not isinstance(problem, str) or not isinstance(answer, str) or not problem.strip() or len(problem) > MAX_PROBLEM_CHARS:
        raise V2Error(f"prior task {row['id']} has an invalid problem length")
    if not MIN_ANSWER_CHARS <= len(answer.strip()) <= MAX_ANSWER_CHARS or any(char in answer for char in "\r\n\x00"):
        raise V2Error(f"prior task {row['id']} has an invalid answer format")
    if MANDATE in problem:
        raise V2Error(f"prior task {row['id']} embeds the grader mandate")
    from ..math_grader import grade_answer
    grade = grade_answer(answer, problem, "<think>x</think> \\boxed{" + answer + "}", False)
    if grade.code != "P":
        raise V2Error(f"prior task {row['id']} failed the live grader answer-format self-check")


def ingest_prior(args: Any) -> dict[str, Any]:
    if type(args.n) is not int or args.n <= 0 or not args.seed:
        raise V2Error("ingest-prior needs a positive integer --n and a nonempty --seed")
    if not 0 <= args.min_p <= args.max_p <= 1 or not 0 <= args.no_signal_share <= 1:
        raise V2Error("prior band must be within [0,1]; no-signal share must be in [0,1]")
    root = reserve_dir()
    prior_root = Path(args.prior_dir)
    manifest = verify_release(prior_root)
    ids, problems, shards, next_numbers = _existing(root)
    selected, filtered = _select(read_rows(prior_root, manifest), args.n, args.seed,
                                 args.min_p, args.max_p, args.no_signal_share, ids, problems)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in selected:
        source = row["_source_key"]
        prefix = PREFIXES[source]
        # Original IDs are nonnumeric, so allocation has the same stable shape
        # as ordinary ingest. A previously ingested same-prefix row is excluded
        # by its source_id below before a new numeric suffix is assigned.
        task_id = f"{prefix}:{next_numbers.get(prefix, -1) + 1}"
        next_numbers[prefix] = next_numbers.get(prefix, -1) + 1
        record = {
            "id": task_id, "env": "affine_math", "problem": row["problem"], "answer": row["answer"],
            "source": f"{row['source']}:{row['source_id']}", "license": row["license"],
            "added_at": args.added_at or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
            "audit": {"method": UNSCREENED_METHOD, "verdict": "unscreened"},
        }
        if row["prior_p"] is not None:
            record["prior_p"] = row["prior_p"]
        _validate(record)
        groups[(source, row["license"])].append(record)
        problems.add(normalise(record["problem"]))
    stamp = args.added_at or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    number = max((int(shard.shard[:4]) for shard in shards), default=0)
    written: list[dict[str, Any]] = []
    history = (root / MANIFEST).read_bytes().splitlines()
    appended: list[bytes] = []
    for (source, license_name), rows in sorted(groups.items()):
        for row in rows:
            row["added_at"] = stamp
        number += 1
        name = f"{number:04d}-{_slug(source, license_name)}.jsonl"
        raw = b"".join(canonical_json(row) + b"\n" for row in rows)
        line = {"shard": name, "sha256": sha256_hex(raw), "count": len(rows), "added_at": stamp}
        with (root / "shards" / name).open("xb") as stream:
            stream.write(raw)
        sidecar = {"input": f"{PRIOR_NAME}-{PRIOR_VERSION}", "rejected": [], "unscreened": True}
        with (root / "shards" / (name + ".rejected.json")).open("xb") as stream:
            stream.write(canonical_json(sidecar) + b"\n")
        appended.append(canonical_json(line))
        written.append(line | {"source": source, "license": license_name})
    history[-1:-1] = appended
    (root / MANIFEST).write_bytes(b"\n".join(history) + b"\n")
    final, _ = load_reserve(root)
    histogram = collections.Counter()
    actual_sources = collections.Counter()
    for shard in written:
        for line in (root / "shards" / shard["shard"]).read_bytes().splitlines():
            row = json.loads(line)
            p = row.get("prior_p")
            bucket = "no_signal" if p is None else f"{int(p * 10) / 10:.1f}-{int(p * 10) / 10 + .1:.1f}"
            histogram[bucket] += 1
            actual_sources[row["source"].split("@", 1)[0]] += 1
    return {"ingested": sum(row["count"] for row in written), "requested": args.n, "seed": args.seed,
            "band": [args.min_p, args.max_p], "no_signal_share": args.no_signal_share,
            "by_source": dict(sorted(actual_sources.items())), "prior_p_histogram": dict(sorted(histogram.items())),
            "filtered": filtered, "shards": written, "reserve_manifest_sha256": final.manifest_sha,
            "tasks": sum(shard.count for shard in final.shards), "audit": {"method": UNSCREENED_METHOD, "verdict": "unscreened"}}
