"""Read-only, allowlisted publication of durable fixed32/heldout128 evidence.

This never launches an evaluation, decodes private artifacts, or changes an
evaluation contract. Missing historical output IDs and stop metadata stay missing.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import re
import tarfile

from dashboard.learner_projection import AUTHORITY, authenticated, canonical

MAX_JSON = 64 * 1024**2
MAX_ARCHIVE = 512 * 1024**2
HEX = re.compile(r"[0-9a-f]{64}\Z")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path, expected=None, limit=MAX_JSON):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise ValueError("canonical evidence file required")
    if path.stat().st_size > limit:
        raise ValueError("bounded evaluation evidence")
    raw = path.read_bytes()
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("evaluation evidence digest")
    return raw


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _token_count(value):
    if type(value) is int and 0 <= value <= 8192:
        return value, None
    if isinstance(value, list) and len(value) <= 8192 and all(
        type(v) is int and 0 <= v < 2**31 for v in value
    ):
        return len(value), value
    raise ValueError("bounded token count or token IDs required")


def project_turn(turn, cap, *, eos_only_early_stop=False):
    """Do not call a cap-length output truncated: EOS can be its final token."""
    if type(cap) is not int or not 1 <= cap <= 8192:
        raise ValueError("signed output budget")
    count, tokens = _token_count(turn.get("output", turn.get("output_tokens")))
    if count > cap:
        raise ValueError("output exceeds signed budget")
    result = {
        "output_length": count,
        "output_token_ids": tokens,
        "output_token_ids_status": "available" if tokens is not None else "not_retained",
        "max_output_tokens": cap,
        "reached_token_budget": count == cap,
        "stop_reason": None,
        "stop_reason_basis": "not_retained",
    }
    for key in ("output_sha256", "prompt_sha256"):
        if key in turn:
            if not isinstance(turn[key], str) or not HEX.fullmatch(turn[key]):
                raise ValueError("token digest")
            result[key] = turn[key]
    if "prompt_tokens" in turn:
        result["prompt_length"] = _token_count(turn["prompt_tokens"])[0]
    explicit = turn.get("stop_reason", turn.get("finish_reason"))
    if explicit in ("eos", "eos_token", "length", "budget", "max_tokens", "token_budget"):
        result["stop_reason"] = "eos" if explicit in ("eos", "eos_token") else "budget"
        result["stop_reason_basis"] = "recorded"
    elif eos_only_early_stop and count < cap:
        result["stop_reason"] = "eos"
        result["stop_reason_basis"] = "inferred_from_authenticated_sampler_and_length"
    elif eos_only_early_stop and count == cap:
        result["stop_reason"] = "eos_or_budget"
        result["stop_reason_basis"] = "final_token_not_retained"
    return result


def project_ack(envelope, expected_source, *, known_early_stop_source=None):
    """Authenticate original job/report/terminal before allowlisting any fields."""
    ack = authenticated(envelope, AUTHORITY)
    job = authenticated(ack["original_job"], AUTHORITY)
    manifest = authenticated(job["manifest"], AUTHORITY)
    report, terminal = ack["original_report"], ack.get("original_terminal")
    if (ack.get("version") != "owned-cached-evaluation-durable-ack-v1"
            or ack.get("durable_report_full_readback") is not True
            or ack.get("job_sha256") != digest(job)
            or ack.get("report_sha256") != digest(report)
            or job.get("role") != "evaluate"
            or manifest["source_bundle"]["sha256"] != expected_source
            or ack.get("checkpoint") != manifest["checkpoint"]):
        raise ValueError("durable original evaluation binding")
    # Early fixed32 ACKs retained the original signed job and report, but did
    # not retain a terminal record or per-object readback table. Do not invent
    # those fields or discard the ROOT-authenticated original report.
    if terminal is not None:
        if (ack.get("original_terminal_sha256") != digest(terminal)
                or terminal.get("phase") != "complete" or type(terminal.get("exit_code")) is not int
                or terminal["exit_code"] != 0 or terminal["job_id"] != job["job_id"]
                or not job["created_at"] <= terminal["started_at"] <= terminal["finished_at"] < job["expires_at"]):
            raise ValueError("original evaluation terminal")
        for name, obj in (("original-job", ack["original_job"]), ("original-report", report),
                          ("original-terminal", terminal)):
            binding = ack["full_readback_objects"][name]
            if binding["sha256"] != digest(obj) or binding["bytes"] != len(canonical(obj)):
                raise ValueError("original full readback binding")
    if (report.get("job_id") != job["job_id"] or report.get("job_sha256") != digest(job)
            or report.get("checkpoint") != manifest["checkpoint"]["id"]
            or report.get("epoch") != manifest["epoch"]
            or report.get("source_files") != job["source_files"]
            or report.get("runtime_versions") != job["runtime_versions"]
            or report.get("chain_transactions") is not False or report.get("role") != "evaluate"
            or not _finite(report.get("completed_at"))
            or not job["created_at"] <= report["completed_at"] < job["expires_at"]):
        raise ValueError("completed evaluation identity")
    if len(job["heldout"]) != 1:
        raise ValueError("one scoped evaluation chunk")
    plan = job["heldout"][0]
    indices, seeds = plan["indices"], plan["seeds"]
    if len(indices) != 32 or len(seeds) != 32 or len(set(indices)) != 32:
        raise ValueError("fixed32 or heldout128 chunk only")
    wanted = set(zip(indices, seeds))
    observed = set()
    cap = plan["harness"]["max_output_tokens"]
    can_infer = bool(known_early_stop_source) and all(
        job["source_files"].get(k) == v for k, v in known_early_stop_source.items()
    )
    tasks = []
    for row in report.get("heldout", []):
        key = (row["index"], row["seed"])
        if key not in wanted or key in observed or row.get("checkpoint") != manifest["checkpoint"]["id"]:
            raise ValueError("exact task identity")
        observed.add(key)
        verdict = row.get("classification")
        if verdict not in ("positive", "negative", "neutral", "unresolved") or not _finite(row.get("reward")):
            raise ValueError("explicit native verdict")
        if ((verdict == "positive" and row["reward"] != 1)
                or (verdict in ("negative", "neutral", "unresolved") and row["reward"] != 0)
                or not isinstance(row.get("task_hash"), str) or not HEX.fullmatch(row["task_hash"])):
            raise ValueError("native verdict and task consistency")
        turns = [project_turn(t, cap, eos_only_early_stop=can_infer) for t in row["turns"]]
        tasks.append({"index": key[0], "seed": key[1], "task_hash": row["task_hash"],
                      "verdict": verdict, "reward": row["reward"], "turns": turns,
                      "output_length": sum(t["output_length"] for t in turns),
                      "native_graded": row.get("native_graded") is True,
                      "proof_verification_performed": row.get("proof_verification_performed") is True})
    for failure in report.get("heldout_failures", []):
        key = (failure["index"], failure["seed"])
        if key not in wanted or key in observed:
            raise ValueError("evaluation failure task identity")
        observed.add(key)
        tasks.append({"index": key[0], "seed": key[1], "verdict": "evaluation_error",
                      "output_length": None, "turns": [], "error_message": "not_published"})
    if observed != wanted:
        raise ValueError("retain complete evaluation denominator")
    tasks.sort(key=lambda r: indices.index(r["index"]))
    return {"checkpoint": manifest["checkpoint"]["id"], "original_epoch": manifest["epoch"],
            "job_id": job["job_id"], "completed_at": report["completed_at"],
            "job_sha256": digest(job), "report_sha256": digest(report),
            "ack_sha256": digest(envelope), "source_sha256": expected_source,
            "source_files_sha256": digest(job["source_files"]),
            "requested_count": 32, "tasks": tasks, "original_terminal_retained": terminal is not None,
            "harness": {k: plan["harness"].get(k) for k in
                        ("version", "policy", "max_output_tokens", "temperature", "top_p")},
            "output_text_status": "not_retained_in_original_report"}


def collect_evaluation_evidence(source_root, finalized):
    """Return safe epoch documents; caller handles signing and publication."""
    root = Path(source_root).resolve()
    known = {}
    for name in ("cached_sampling", "owned_cached_evaluation"):
        p = Path(__file__).resolve().parents[1] / "subnet" / (name + ".py")
        known["subnet/" + name + ".py"] = hashlib.sha256(p.read_bytes()).hexdigest()
    collected = []
    issues = []
    cached = root / "state/dashboard/cached-evaluator-sources.ROOT-SIGNED.json"
    if cached.exists():
        scope = authenticated(json.loads(read(cached)), AUTHORITY)
        for state in scope["states"]:
            for path in sorted((Path(state) / "durable-evaluation-acks").glob("*.json")):
                try:
                    envelope = json.loads(read(path))
                    ack = authenticated(envelope, AUTHORITY)
                    job = authenticated(ack["original_job"], AUTHORITY)
                    manifest = authenticated(job["manifest"], AUTHORITY)
                    if len(job.get("heldout", [])) != 1:
                        raise ValueError("one scoped fixed32 suite")
                    suite = job["heldout"][0]
                    # The same evaluator queue also serves a separate research
                    # experiment. Queue co-location is not publication scope.
                    if (manifest["epoch"].startswith("nonpayable-fixed32-cap2048-20261009-")
                            and suite["harness"].get("max_output_tokens") == 2048):
                        continue
                    if (suite["indices"] != scope["indices"]
                            or suite["seeds"] != [20261002 + i * 1000 for i in scope["indices"]]
                            or suite["harness"].get("max_output_tokens") != 1024):
                        raise ValueError("exact public fixed32 scope")
                    row = project_ack(envelope, scope["source_sha256"],
                                      known_early_stop_source=known)
                    row["suite"] = "fixed32"
                    collected.append(row)
                except (ValueError, KeyError, TypeError, OSError):
                    issues.append({"kind": "fixed32", "record_id": hashlib.sha256(str(path).encode()).hexdigest(),
                                   "status": "unavailable_or_failed_authentication"})
    heldout = root / "state/dashboard/heldout128-sources.ROOT-SIGNED.json"
    if heldout.exists():
        scope = authenticated(json.loads(read(heldout)), AUTHORITY)
        for entry in scope["evaluations"]:
            try:
                summary = authenticated(json.loads(read(entry["summary_path"], entry["summary_sha256"])), AUTHORITY)
                groups = summary.get("groups") or {next(iter(entry["groups"])): summary["group"]}
                for label, paths in entry["groups"].items():
                    archive_sha = groups[label]["archive_sha256"]
                    raw = read(paths["archive_path"], archive_sha, MAX_ARCHIVE)
                    archive_ack = json.loads(read(paths["archive_ack_path"], paths["archive_ack_sha256"]))
                    if (archive_ack.get("R2_full_GET_verified") is not True
                            or archive_ack.get("archive_sha256") != archive_sha
                            or archive_ack.get("archive_bytes") != len(raw)):
                        raise ValueError("durable archive binding")
                    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
                        members = archive.getmembers()
                        if (len(members) > 4096 or len({m.name for m in members}) != len(members)
                                or sum(m.size for m in members) > MAX_ARCHIVE
                                or any(m.issym() or m.islnk() or m.name.startswith("/")
                                       or ".." in Path(m.name).parts for m in members)):
                            raise ValueError("bounded ordinary archive members")
                        acks = [m for m in members if m.isfile() and m.name.startswith("durable-evaluation-acks/")
                                and m.name.endswith(".json")]
                        if len(acks) != 4 or any(m.size > MAX_JSON for m in acks):
                            raise ValueError("four original heldout chunks")
                        envelopes = [json.load(archive.extractfile(m)) for m in acks]
                    # Apply the existing full heldout128 provenance rules before
                    # attaching its cohort label to these projected outputs.
                    from dashboard.heldout128_projection import original, comparison_binding
                    originals = [original(a, scope["source_sha256"], scope["source_files_sha256"])
                                 for a in envelopes]
                    originals.sort(key=lambda o: o[0]["group"])
                    plans = [dict(group=i, **o[3]) for i, o in enumerate(originals)]
                    if ([o[0]["group"] for o in originals] != list(range(4))
                            or digest(plans) != scope["cohort_sha256"]
                            or len({comparison_binding(o) for o in originals}) != 1):
                        raise ValueError("exact original heldout128 cohort and runtime")
                    batch = [project_ack(a, scope["source_sha256"], known_early_stop_source=known)
                             for a in envelopes]
                    if (len({r["checkpoint"] for r in batch}) != 1
                            or len({(v["index"], v["seed"]) for r in batch for v in r["tasks"]}) != 128
                            or {v["index"] for r in batch for v in r["tasks"]} & set(scope["excluded_indices"])):
                        raise ValueError("one full heldout128 checkpoint cohort")
                    successes = sum(v["reward"] == 1 for r in batch for v in r["tasks"])
                    result = groups[label]["result"]["result"]
                    score = result["score"]
                    if (score["count"] != 128 or score["cohort_sha256"] != scope["cohort_sha256"]
                            or score["successes"] != successes or score["mean_reward"] != successes / 128
                            or result["retirement"]["status"] != "complete"):
                        raise ValueError("full heldout128 summary score")
                    if summary.get("version") == "owned-cached-heldout128-checkpoint-actual-v1":
                        checkpoint = summary["checkpoint"]
                        checkpoint = checkpoint["id"] if isinstance(checkpoint, dict) else checkpoint
                        if checkpoint != batch[0]["checkpoint"] or summary["successes"] != successes:
                            raise ValueError("summary checkpoint binding")
                    elif summary.get("version") == "owned-cached-heldout128-paired-actual-v1":
                        if summary[label + "_successes"] != successes:
                            raise ValueError("paired summary outcome binding")
                    else:
                        raise ValueError("known heldout128 summary")
                    for row in batch:
                        row.update(suite="heldout128", archive_sha256=archive_sha,
                                   summary_sha256=entry["summary_sha256"], cohort_sha256=scope["cohort_sha256"])
                    collected.extend(batch)
            except (ValueError, KeyError, TypeError, OSError, tarfile.TarError):
                issues.append({"kind": "heldout128", "record_id": entry.get("summary_sha256"),
                               "status": "unavailable_or_failed_authentication"})
    from dashboard.endpoint_evaluation_evidence_projection import collect_endpoint_evidence
    endpoint_rows, endpoint_issues = collect_endpoint_evidence(root, finalized)
    from dashboard.legacy_evaluation_evidence_projection import collect_legacy_evaluation_evidence
    legacy_rows, legacy_issues = collect_legacy_evaluation_evidence(root, finalized)
    result = {}
    for epoch, final in finalized.items():
        checkpoints = {final["input_checkpoint"], final["output_checkpoint"]}
        matched = []
        seen = set()
        for row in collected:
            if row["checkpoint"] not in checkpoints or row["job_id"] in seen:
                continue
            seen.add(row["job_id"])
            stage = "post_update" if row["checkpoint"] == final["output_checkpoint"] else "input_checkpoint"
            matched.append(dict(row, checkpoint_association=stage))
        matched.extend(endpoint_rows.get(epoch, []))
        matched.extend(legacy_rows.get(epoch, []))
        result[epoch] = {"version": "finalized-epoch-evaluation-evidence-v1", "epoch_id": epoch,
                         "status": "available" if matched else "unavailable",
                         "evaluations": matched, "availability": "available" if matched else "not_retained_or_not_evaluated",
                         "collection_issues": issues,
                         "endpoint_availability": endpoint_issues,
                         "legacy_availability": legacy_issues,
                         "limits": ["Historical reports can retain counts and hashes without output token IDs or text.",
                                    "At the output cap, EOS and budget exhaustion are indistinguishable without the final token.",
                                    "Checkpoint association does not imply this evaluation used the training harness or output budget."]}
    return result
