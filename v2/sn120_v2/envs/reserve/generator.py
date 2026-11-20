"""Consume one public Generator accepted-task epoch into the live reserve.

Only rows from the checksum-bound global math prior may enter. The original
license, source revision and the Generator's frozen-king prior are public.
"""
from __future__ import annotations

import collections
import datetime
import json
import math
import re
import shutil
import urllib.request
import time
from pathlib import Path
from typing import Any

from ...canonical import canonical_json, sha256_hex
from ...errors import V2Error
from ..math_grader import _options, grade_answer
from ..math_pool import MANDATE, POOL_V2_0_FILE, POOL_V2_0_SHA256, load_pool
from . import (
    MANIFEST, MAX_ANSWER_CHARS, MAX_PROBLEM_CHARS, atomic_write, ingest_lock, load_reserve, normalise, reserve_dir, task_row,
)
from ..math_pool import Task
from .__main__ import _write_new, utc_now

LEDGER = "generator-ingested.jsonl"
EPOCH = re.compile(r"[0-9]{8}T[0-9]{4}Z\Z")
POOL_SHA256 = "45b793b4d00d1f7225d34fd35e58018612a0ded00d96a9c23b750cfb18994115"
POOL = f"sn120-math-prior-v1:{POOL_SHA256}"
ALLOWED_PRIOR_P = {2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8}
PROOF = re.compile(r"\b(?:prove|show(?:\s+that)?|demonstrate)\b|证明", re.IGNORECASE)
VALUE = re.compile(
    r"\b(?:find|determine|calculate|compute|evaluate|value|solve\s+for|how\s+(?:many|much)|"
    r"what\s+is|求|计算|多少|值)\b", re.IGNORECASE)
# The prior's documented redistribution set; NC/ND and unrecognized strings
# are never guessed into an allowed license.
LICENSES = {
    "apache-2.0": "apache-2-0", "apache 2.0": "apache-2-0",
    "mit": "mit", "cc-by-4.0": "cc-by-4-0", "cc by 4.0": "cc-by-4-0",
    "cc-by-sa-4.0": "cc-by-sa-4-0", "cc-by-3.0": "cc-by-3-0",
    "mixed (cc by 4.0 / cc by-sa 4.0)": "cc-by-4-0-cc-by-sa-4-0",
    "http://creativecommons.org/licenses/by/4.0/": "cc-by-4-0",
    "http://creativecommons.org/licenses/by-sa/4.0/": "cc-by-sa-4-0",
    "http://creativecommons.org/licenses/by/3.0/": "cc-by-3-0",
    "http://creativecommons.org/publicdomain/zero/1.0/": "cc0-1-0",
    "http://creativecommons.org/licenses/publicdomain/": "public-domain",
    "cc0 1.0": "cc0-1-0", "cc0-1.0": "cc0-1-0",
}


class Incomplete(V2Error):
    pass


class InputChanged(V2Error):
    pass


def _read_input(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        raw = path.read_bytes()
        if not raw.endswith(b"\n"):
            raise ValueError("last row is not newline-terminated")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("each input row must be an object")
        return raw, rows
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise Incomplete(f"incomplete: cannot read complete accepted JSONL: {exc}") from exc


def _done(path: Path, raw: bytes, rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8").strip()
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            count, sha = text.split()
            document = {"rows": int(count), "sha256": sha}
        if document.get("format") == "generator-intake-v2" or "lines" in document:
            if "rows" not in document and type(document.get("lines")) is int:
                document = dict(document, rows=document["lines"])
        if type(document.get("rows")) is not int or document["rows"] != len(rows):
            raise ValueError(f"done rows={document.get('rows')} but input rows={len(rows)}")
        if document.get("sha256") != sha256_hex(raw):
            raise ValueError("done sha256 does not match accepted JSONL")
        return document
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise Incomplete(f"incomplete: invalid completion marker: {exc}") from exc


def _results(path: Path | None, epoch: str, count: int) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        row = json.loads(path.read_bytes())
        if row.get("version", row.get("schema")) != "affine-generator-results-v1" or row.get("epoch") != epoch:
            raise ValueError("results.epoch does not equal --epoch")
        if type(row.get("counts", {}).get("accepted")) is not int or row["counts"]["accepted"] != count:
            raise ValueError(f"counts.accepted={row.get('counts', {}).get('accepted')} but input rows={count}")
        if row.get("pool") != POOL:
            raise ValueError("results.pool is not the pinned sn120-math-prior-v1 asset")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("king", ""))):
            raise ValueError("results.king is not a sha256")
        return row
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise Incomplete(f"incomplete: {exc}") from exc


def _prior(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    try:
        document = json.loads(path.read_bytes())
        if document.get("schema") != "sn120-math-prior-index-v1" or document.get("pool") != POOL:
            raise V2Error("prior index is not bound to the expected pool asset sha256")
        index = document["problems"]
        if not isinstance(index, dict) or document.get("rows") != len(index):
            raise V2Error("prior index row count does not match its entries")
        for key, row in index.items():
            if not re.fullmatch(r"[0-9a-f]{64}", key) or set(row) != {"source", "license"} or any(not isinstance(value, str) for value in row.values()):
                raise V2Error("invalid prior index entry")
        print(f"prior index: {len(index):,} problems; pool {POOL_SHA256}")
        return index
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise V2Error(f"cannot read prior index: {exc}") from exc


def _ledger(root: Path) -> dict[str, dict[str, Any]]:
    path = root / LEDGER
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    result = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("epoch") in result:
            raise V2Error("invalid or repeated generator ingestion ledger epoch")
        result[row["epoch"]] = row
    return result


def _active_ids(args: Any, catalog: Any) -> tuple[set[str], dict[str, str]]:
    if getattr(args, "active", None):
        try:
            raw = Path(args.active).read_bytes()
            ids = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise V2Error(f"cannot read active id list: {exc}") from exc
    else:
        def get(key: str) -> bytes:
            error = None
            for attempt in range(3):
                try:
                    url = args.public_url.rstrip("/") + "/" + key + f"?reserve={time.time_ns()}"
                    request = urllib.request.Request(url, headers={"Cache-Control": "no-cache", "Accept-Encoding": "identity", "User-Agent": "curl/8.5.0"})
                    with urllib.request.urlopen(request, timeout=30) as reply:
                        return reply.read()
                except Exception as exc:
                    error = exc
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            raise V2Error(f"cannot fetch the public active pool: {type(error).__name__}: {error}") from error
        summary = json.loads(get("public/pool/summary.json"))
        sha = summary["active_sha"]
        raw = get(f"public/pool/active/{sha}.json")
        if sha256_hex(raw) != sha:
            raise V2Error("active id list does not match summary.active_sha")
        ids = json.loads(raw)
    if not isinstance(ids, list) or any(not isinstance(task, str) for task in ids) or len(ids) != len(set(ids)):
        raise V2Error("active pool must be a unique JSON list of task ids")
    return set(ids), {normalise(catalog.task(task).problem): task for task in ids if task in catalog.task_ids}


def _problems(root: Path, reserve: Any, catalog: Any) -> tuple[dict[str, str], dict[str, str]]:
    base, archive = load_pool(), load_pool(POOL_V2_0_FILE, POOL_V2_0_SHA256)
    known = {normalise(base.task(task).problem): "base" for task in base.task_ids}
    for task in archive.task_ids:
        known.setdefault(normalise(archive.task(task).problem), "base")
    retired_path = root / "retired.json"
    retired_ids = {row["id"] for row in json.loads(retired_path.read_bytes())} if retired_path.exists() else set()
    reserve_problems = {}
    earlier = {}
    for shard in reserve.shards:
        for line in (root / "shards" / shard.shard).read_bytes().splitlines():
            row = json.loads(line)
            norm = normalise(row["problem"])
            earlier[norm] = "earlier_epoch" if row["id"].startswith("gen:") else "reserve"
            reserve_problems[norm] = earlier[norm]
    for task in retired_ids:
        if task in catalog.task_ids:
            known[normalise(catalog.task(task).problem)] = "retired"
    for task in catalog.task_ids:
        if task not in retired_ids:
            known.setdefault(normalise(catalog.task(task).problem), "base")
    known.update(reserve_problems)
    return known, earlier


def _license(row: dict[str, Any]) -> tuple[str, str] | None:
    value = row.get("license", "")
    if not isinstance(value, str):
        return None
    slug = LICENSES.get(value.strip().casefold())
    return (value, slug) if slug else None


def _reject(counts: collections.Counter, drops: list[dict[str, Any]], number: int, row: dict[str, Any], reason: str) -> None:
    counts[reason] += 1
    drops.append({"row": number, "source_id": row.get("source_id"), "reason": reason})


def _recover_epoch(root: Path, rows: list[dict[str, Any]], listed: list[Any], results: dict[str, Any], input_sha: str, epoch: str, prior: dict[str, dict[str, str]], active: dict[str, str], known: dict[str, str], earlier: dict[str, str]) -> dict[str, Any] | None:
    """Rebuild epoch rows and recover only an exactly matching manifest history."""
    if not listed:
        return None
    originals = [shard for shard in listed if "-b1-" not in shard.shard]
    backfills = [shard for shard in listed if "-b1-" in shard.shard]

    def shard_rows(shard: Any) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (root / "shards" / shard.shard).read_bytes().splitlines()]

    def replay(history: list[Any], next_gen: int, next_shard: int, backfill: bool) -> tuple[dict[str, bytes], collections.Counter] | None:
        existing = {}
        for shard in history:
            for row in shard_rows(shard):
                if not row["source"].startswith(f"sn120-generator:{epoch}:"):
                    raise V2Error(f"epoch {epoch} has a listed shard with invalid provenance")
                existing[normalise(row["problem"])] = row
        groups: dict[str, list[dict[str, Any]]] = {}
        counts: collections.Counter = collections.Counter()
        seen: set[str] = set()
        for row in rows:
            problem, answer = row.get("problem", ""), row.get("answer", "")
            norm = normalise(str(problem))
            original = existing.get(norm)
            reason = ""
            if norm in seen:
                reason = "duplicate_same_file"
            elif not isinstance(problem, str) or not isinstance(answer, str) or not isinstance(row.get("task_id", row.get("source_id")), str):
                reason = "schema"
            elif original is None and norm in active:
                reason = "duplicate_active"
            elif original is None and norm in earlier:
                reason = f"duplicate_{earlier[norm]}"
            elif original is None and norm in known:
                reason = f"duplicate_{known[norm]}"
            else:
                prior_row = prior.get(sha256_hex(norm.encode()), {})
                license_value = row.get("license", original.get("license") if original else prior_row.get("license"))
                licensed = _license({"license": license_value})
                if licensed is None:
                    reason = "not_in_prior" if not license_value and not row.get("source", original.get("source") if original else prior_row.get("source")) else "license"
                elif type(row.get("prior_p")) not in (int, float) or not math.isfinite(row["prior_p"]) or row["prior_p"] not in ALLOWED_PRIOR_P:
                    reason = "prior_p"
                elif PROOF.search(problem) and not VALUE.search(problem):
                    reason = "proof"
                elif not problem.strip() or len(problem) > MAX_PROBLEM_CHARS:
                    reason = "problem_length"
                elif not answer.strip() or len(answer.strip()) > MAX_ANSWER_CHARS or any(c in answer for c in "\r\n\x00"):
                    reason = "answer_format"
                elif MANDATE in problem:
                    reason = "mandate"
                elif _options(problem):
                    reason = "mc"
                elif grade_answer(answer, problem, "<think>x</think> \\boxed{" + answer + "}", False).code != "P":
                    reason = "grader"
            if reason:
                counts[reason] += 1
                continue
            source = original["source"].split(";prior_source=", 1)[1] if original and ";prior_source=" in original["source"] else row.get("source", original.get("source") if original else prior_row.get("source")) or "generator-row"
            source_id = row.get("task_id", row.get("source_id"))
            if original is not None and (original["answer"], original["source"].removeprefix(f"sn120-generator:{epoch}:").split(";", 1)[0], original["prior_p"]) != (answer, source_id, row["prior_p"]):
                raise InputChanged(f"epoch {epoch} has a different accepted row")
            if original is not None:
                audit, added_at, task_id = original["audit"], original["added_at"], original["id"]
            else:
                revision = row.get("prior_king", row.get("king_revision", results.get("king", "unpublished")))
                if not isinstance(revision, str) or not revision:
                    raise V2Error("Generator row king revision must be a nonempty string")
                audit = {"verdict": "keep",
                         "method": f"SN120 Generator mech 1 accepted: frozen king {revision[:12]} s/8 in 2..6 at n=8, prior_p=s/8; ingest filters: dedupe, license, MC, proof, grader self-check",
                         "ref": f"https://affine.io/generator/epochs/{epoch}/results.json" + (f"#king={revision}" if "prior_king" in row or "king_revision" in row else "")}
                added_at, task_id = history[0].added_at, f"gen:{next_gen}"
                next_gen += 1
            groups.setdefault(licensed[1], []).append(task_row(
                Task(task_id, "affine_math", problem, answer),
                source=f"sn120-generator:{epoch}:{source_id};prior_source={source}",
                license=licensed[0], added_at=added_at, audit=audit, prior_p=row["prior_p"]))
            seen.add(norm)
        expected = {}
        for license_slug, kept in sorted(groups.items()):
            name = f"{next_shard:04d}-sn120-generator-{epoch.lower()}{'-b1' if backfill else ''}-{license_slug}.jsonl"
            next_shard += 1
            expected[name] = b"".join(canonical_json(row) + b"\n" for row in kept)
        return expected, counts

    if not originals:
        return None
    first_gen = min(int(row["id"].split(":", 1)[1]) for shard in originals for row in shard_rows(shard))
    first_shard = min(int(shard.shard[:4]) for shard in originals)
    replayed = replay(originals, first_gen, first_shard, False)
    if replayed is None:
        return None
    expected, original_counts = replayed
    actual = {shard.shard: sha256_hex((root / "shards" / shard.shard).read_bytes()) for shard in originals}
    if {name: sha256_hex(raw) for name, raw in expected.items()} != actual:
        return None
    next_gen = max(int(row["id"].split(":", 1)[1]) for shard in originals for row in shard_rows(shard)) + 1
    next_shard = max(int(shard.shard[:4]) for shard in originals) + 1
    counts = original_counts
    if backfills:
        replayed = replay(backfills, next_gen, next_shard, True)
        if replayed is None:
            return None
        backfill_expected, backfill_counts = replayed
        counts += backfill_counts
        expected.update(backfill_expected)
        actual.update({shard.shard: sha256_hex((root / "shards" / shard.shard).read_bytes()) for shard in backfills})
    if {name: sha256_hex(raw) for name, raw in expected.items()} != actual:
        return None
    names = [shard.shard for shard in originals + backfills]
    ledger = {"epoch": epoch, "input_sha256": input_sha, "rows": len(rows), "ingested": 0,
              "rejected": dict(sorted(counts.items())), "rejected_rows": len(rows),
              "shards": names, "at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
              "reserve_manifest_sha": sha256_hex((root / MANIFEST).read_bytes()),
              "problems": sorted({sha256_hex(normalise(str(row.get("problem", ""))).encode()) for row in rows})}
    previous = (root / LEDGER).read_bytes() if (root / LEDGER).exists() else b""
    atomic_write(root / (LEDGER + ".new"), previous + canonical_json(ledger) + b"\n")
    (root / (LEDGER + ".new")).replace(root / LEDGER)
    return ledger | {"message": "already present in manifest; ledger recovered", "already_ingested": True}


def ingest_generator(args: Any) -> dict[str, Any]:
    epoch = str(args.epoch)
    if not EPOCH.fullmatch(epoch):
        raise V2Error("epoch must be YYYYMMDDTHHMMZ")
    input_path = Path(args.input)
    raw, rows = _read_input(input_path)
    input_sha = sha256_hex(raw)
    done_path = Path(args.done) if getattr(args, "done", None) else input_path.with_suffix(".done")
    done = None
    if done_path.is_file():
        try:
            done = _done(done_path, raw, rows)
        except Incomplete as exc:
            # An existing epoch distinguishes a bad new file from an initially
            # incomplete one. The CLI maps this to exit 4, with no writes.
            if (ledger := _ledger(reserve_dir()).get(epoch)) and ledger.get("input_sha256") != input_sha:
                raise InputChanged(str(exc)) from exc
            raise
    results = _results(Path(args.results), epoch, len(rows)) if args.results and Path(args.results).is_file() else None
    if done is None and results is None:
        raise Incomplete("incomplete: accepted JSONL needs a .done marker or matching results.json")
    root = reserve_dir()
    if not args.prior_index and any(not (isinstance(row.get("source"), str) and row.get("license")) for row in rows):
        raise V2Error("--prior-index is required when accepted rows lack source/license")
    prior = _prior(Path(args.prior_index) if args.prior_index else None)
    if not (root / MANIFEST).is_file():
        raise V2Error("live reserve has no MANIFEST.jsonl; start PoolRuntime to seed it")
    with ingest_lock(root, exclusive=True):
        previous = _ledger(root).get(epoch)
        if previous is not None and previous.get("input_sha256") == input_sha:
            previous.setdefault("original_input_sha256", previous.get("original_input_sha256", input_sha))
            if results and not previous.get("king_revisions"):
                previous["king_revisions"] = [results["king"]]
                previous["king"] = results["king"]
            if not previous.get("problems"):
                previous["problems"] = sorted({sha256_hex(normalise(str(row.get("problem", ""))).encode()) for row in rows})
                history = [previous if row.get("epoch") == epoch else row for row in map(json.loads, (root / LEDGER).read_bytes().splitlines())]
                atomic_write(root / LEDGER, b"".join(canonical_json(row) + b"\n" for row in history))
            return previous | {"message": "already ingested", "already_ingested": True}
        reserve, catalog = load_reserve(root, with_catalog=True, allow_retired=True)
        listed = [shard for shard in reserve.shards if f"-sn120-generator-{epoch.lower()}-" in shard.shard and "-b1-" not in shard.shard]
        backfill = previous is not None and previous.get("input_sha256") != input_sha
        _, active_problems = _active_ids(args, catalog)
        known, earlier = _problems(root, reserve, catalog)
        if backfill:
            incoming = {sha256_hex(normalise(str(row.get("problem", ""))).encode()) for row in rows}
            known_problems = set(previous.get("problems", ()))
            if done is None:
                raise InputChanged(f"epoch {epoch} was already ingested with other bytes; backfill needs .done")
            if previous.get("backfill_seen", {}).get(input_sha):
                return previous | {"message": "backfill_seen", "already_ingested": True}
            if incoming <= known_problems:
                previous.setdefault("backfill_seen", {})[input_sha] = {"rows": len(rows), "note": "backfill_seen", "at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")}
                history = [previous if row.get("epoch") == epoch else row for row in map(json.loads, (root / LEDGER).read_bytes().splitlines())]
                atomic_write(root / LEDGER, b"".join(canonical_json(row) + b"\n" for row in history))
                return previous | {"message": "backfill_seen", "already_ingested": True}
            # The original input is bound by its original input SHA; do not
            # silently treat a different complete file as its re-ingestion.
            # New problems go through ordinary dedupe below.
        if not backfill:
            if listed:
                recovered = _recover_epoch(root, rows, listed, results or {}, input_sha, epoch, prior, active_problems, known, earlier)
                if recovered is not None:
                    return recovered
                raise InputChanged(f"epoch {epoch} has different reserve history")
            if previous is not None:
                raise InputChanged(f"epoch {epoch} was already ingested with a different input sha256")
        ids = set(catalog.task_ids) | set(reserve.withdrawn_ids)
        next_gen = 1 + max((int(task.split(":", 1)[1]) for task in ids if task.startswith("gen:") and task.split(":", 1)[1].isdigit()), default=-1)
        next_shard = 1 + max((int(shard.shard[:4]) for shard in reserve.shards), default=0)
        stamp = utc_now()
        groups: dict[str, list[dict[str, Any]]] = {}
        sidecars: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        counts: collections.Counter = collections.Counter()
        kings = sorted({str(row.get("prior_king", row.get("king_revision", (results or {}).get("king", "unpublished")))) for row in rows})
        def audit_for(row: dict[str, Any]) -> dict[str, Any]:
            revision = row.get("prior_king", row.get("king_revision", (results or {}).get("king", "unpublished")))
            if not isinstance(revision, str) or not revision:
                raise V2Error("Generator row king revision must be a nonempty string")
            return {
                "verdict": "keep",
                "method": f"SN120 Generator mech 1 accepted: frozen king {revision[:12]} s/8 in 2..6 at n=8, prior_p=s/8; ingest filters: dedupe, license, MC, proof, grader self-check",
                "ref": f"https://affine.io/generator/epochs/{epoch}/results.json" + (f"#king={revision}" if "prior_king" in row or "king_revision" in row else ""),
            }
        for number, row in enumerate(rows):
            norm = normalise(str(row.get("problem", "")))
            extra = set(row) - {"problem", "answer", "source_id", "prior_p", "source", "license", "task_id", "prior_king", "king_revision"}
            reason = ""
            if {"problem", "answer"} - set(row) or extra or any(not isinstance(row.get(key), str) for key in ("problem", "answer")) or not isinstance(row.get("task_id", row.get("source_id")), str) or not row.get("task_id", row.get("source_id")):
                reason = "schema"
            elif norm in active_problems:
                reason = "duplicate_active"
            elif norm in earlier:
                reason = f"duplicate_{earlier[norm]}"
            elif norm in known:
                reason = f"duplicate_{known[norm]}"
            if reason:
                _reject(counts, sidecars["unsharded"], number, row, reason)
                continue
            original = prior.get(sha256_hex(norm.encode()), {})
            license_name = row.get("license", original.get("license"))
            source = row.get("source", original.get("source"))
            if not license_name and not source:
                _reject(counts, sidecars["unsharded"], number, row, "not_in_prior")
                continue
            licensed = _license({"license": license_name})
            if licensed is None:
                _reject(counts, sidecars["unsharded"], number, row, "license")
                continue
            license_name, license_slug = licensed
            source = str(source or "generator-row")
            problem, answer = row["problem"], row["answer"]
            if type(row.get("prior_p")) not in (int, float) or not math.isfinite(row["prior_p"]) or row["prior_p"] not in ALLOWED_PRIOR_P:
                reason = "prior_p"
            elif PROOF.search(problem) and not VALUE.search(problem):
                reason = "proof"
            elif not problem.strip() or len(problem) > MAX_PROBLEM_CHARS:
                reason = "problem_length"
            elif not answer.strip() or len(answer.strip()) > MAX_ANSWER_CHARS or any(c in answer for c in "\r\n\x00"):
                reason = "answer_format"
            elif MANDATE in problem:
                reason = "mandate"
            elif _options(problem):
                reason = "mc"
            elif grade_answer(answer, problem, "<think>x</think> \\boxed{" + answer + "}", False).code != "P":
                reason = "grader"
            if reason:
                _reject(counts, sidecars[license_slug], number, row, reason)
                continue
            task = Task(f"gen:{next_gen}", "affine_math", problem, answer)
            next_gen += 1
            groups.setdefault(license_slug, []).append(task_row(
                task, source=f"sn120-generator:{epoch}:{row.get('task_id', row.get('source_id'))};prior_source={source}",
                license=license_name, added_at=stamp, audit=audit_for(row), prior_p=row["prior_p"]))
            known[norm] = earlier[norm] = "same_file"
        lines = (root / MANIFEST).read_bytes().splitlines()
        staged_root = None
        shard_rows, shard_names = [], []
        if groups:
            staged_root = root / f".generator-stage-{input_sha[:16]}"
            if staged_root.exists():
                shutil.rmtree(staged_root)
            shutil.copytree(root, staged_root, ignore=shutil.ignore_patterns(".generator-*", ".ingest.lock"))
            for license_slug, kept in sorted(groups.items()):
                name = f"{next_shard:04d}-sn120-generator-{epoch.lower()}{'-b1' if backfill else ''}-{license_slug}.jsonl"
                next_shard += 1
                content = b"".join(canonical_json(row) + b"\n" for row in kept)
                line = {"shard": name, "sha256": sha256_hex(content), "count": len(kept), "added_at": stamp}
                _write_new(staged_root / "shards" / name, content)
                drops = sidecars.get(license_slug, [])
                if drops:
                    _write_new(staged_root / "shards" / (name + ".rejected.json"), canonical_json({"input": input_path.name, "epoch": epoch, "rejected": drops}) + b"\n")
                lines[-1:-1] = [canonical_json(line)]
                shard_names.append(name)
                shard_rows.append({"shard": name, "sha256": line["sha256"], "ingested": len(kept), "rejected": len(drops)})
            atomic_write(staged_root / MANIFEST, b"\n".join(lines) + b"\n")
            verified, _ = load_reserve(staged_root)
        else:
            verified = reserve
        ingested = sum(len(group) for group in groups.values())
        ledger = {"epoch": epoch, "input_sha256": input_sha, "rows": len(rows), "ingested": ingested,
                  "rejected": dict(sorted(counts.items())), "rejected_rows": len(rows) - ingested,
                  "shards": shard_names, "at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                  "reserve_manifest_sha": verified.manifest_sha,
                  "king_revisions": kings,
                  "king": (results or {}).get("king", previous.get("king") if previous else None),
                  **({"original_input_sha256": previous.get("original_input_sha256", previous.get("input_sha256"))} if backfill else {}),
                  "problems": sorted({sha256_hex(normalise(str(row.get("problem", ""))).encode()) for row in rows} | (set(previous.get("problems", ())) if backfill else set())),
                  **({"backfill": True, "backfill_parent_sha256": previous.get("input_sha256"),
                       "backfill_seen": dict(previous.get("backfill_seen", {}), **{input_sha: {"note": "backfill_seen", "rows": len(rows)}})} if backfill else {}),
                  **({"done_sha256": done.get("sha256")} if done else {})}
        if staged_root is not None:
            for detail in shard_rows:
                name = detail["shard"]
                _write_new(root / "shards" / name, (staged_root / "shards" / name).read_bytes())
                rejected = staged_root / "shards" / (name + ".rejected.json")
                if rejected.exists():
                    _write_new(root / "shards" / rejected.name, rejected.read_bytes())
            atomic_write(root / MANIFEST, (staged_root / MANIFEST).read_bytes())
            load_reserve(root)
            shutil.rmtree(staged_root)
        if backfill:
            ledger["input_sha256"] = previous.get("original_input_sha256", previous.get("input_sha256"))
            ledger["shards"] = sorted(set(previous.get("shards", ())) | set(shard_names))
            ledger["backfill_shards"] = shard_names
            ledger["backfill_ingested"] = ingested
            ledger["ingested"] = int(previous.get("ingested", 0)) + ingested
            history = [ledger if row.get("epoch") == epoch else row for row in map(json.loads, (root / LEDGER).read_bytes().splitlines())]
            atomic_write(root / LEDGER, b"".join(canonical_json(row) + b"\n" for row in history))
        else:
            previous_ledger = (root / LEDGER).read_bytes() if (root / LEDGER).exists() else b""
            atomic_write(root / (LEDGER + ".new"), previous_ledger + canonical_json(ledger) + b"\n")
            (root / (LEDGER + ".new")).replace(root / LEDGER)
        return ledger | {"shard_details": shard_rows, "counts": dict(sorted(counts.items())), "ingested": ingested, "epoch_ingested": ledger["ingested"], "message": "backfill_ingested" if backfill and ingested else "backfill_seen" if backfill else "ingested" if ingested else "nothing kept; epoch recorded"}
