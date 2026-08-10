"""Dashboard publishing: state → data/*.json on Hippius (+ the static site).

Published objects (all world-readable, no secrets):
  data/dashboard.json   king, reign chain, queue, live eval, stats, machine
  data/history.json     recent verdicts / failures (tail of history.jsonl)
  data/benchmarks.json  advisory tau2 results per model (albedo-style)
  data/contract.json    the public chain contract (subset of affine.toml)
  data/validator_log.txt            recent validator log tail (redacted)
  data/history_full.jsonl.gz        complete verdict/failure audit log
  data/bench_history_full.jsonl.gz  complete bench result log
  evals/index.jsonl     manifest of published full duel records
  evals/{cid}.json.gz   full duel record: rollouts, teacher refs, logprobs

The website only fetches the small data/*.json at load; the heavy evals/*
objects are lazy training data for miners, never on the page path.

Writes are rate-limited and fail-open: presentation must never block or
break the validator loop.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import logging
import re
import time
from pathlib import Path

from . import __version__
from .config import Config
from .hippius import Hippius
from .state import State, now_iso

log = logging.getLogger("affine.dashboard")

HISTORY_TAIL = 100
BENCH_TAIL = 200

# Validator log publishing (transparency: anyone can watch the control plane
# operate). Tail only — the full files grow forever on disk.
LOG_TAIL_BYTES = 256 * 1024
LOG_PUSH_INTERVAL_S = 60
# Secrets never reach the log (provisioner redacts them at the source), but
# pod SSH endpoints do (tunnel commands). Scrub network coordinates before
# anything leaves this box — the public record shows WHAT happened, not WHERE
# the rented GPUs live.
_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_SSH_PORT_RE = re.compile(r"(-p\s+)\d{2,5}\b")


def redact_log_text(text: str) -> str:
    text = _IP_RE.sub("[ip]", text)
    return _SSH_PORT_RE.sub(r"\1[port]", text)


# Per-side scalars the site charts: the score (reason) + telemetry. Legacy
# S* v2 fields (valid/S/mean_lambda2/...) are kept so pre-fork history rows
# still render. Rollout-level detail belongs in the Hippius artifact, not in
# the history file every page load fetches.
_SIDE_FIELDS = ("reason", "mean_l1lift", "mean_eta", "mean_len_z", "mean_len_y",
                "len_z_delta", "len_y_delta",
                "gate_pass_rate", "bank_frac", "calib_ratio", "baseline_abs",
                "n_turns", "n_pairs",
                # legacy (pre-fork verdicts)
                "valid", "S", "mean_lambda2", "baseline_band_exceeded")


def _slim_side(side: dict | None) -> dict | None:
    if not side:
        return None
    return {k: side[k] for k in _SIDE_FIELDS if k in side}


def _side_score(side: dict | None) -> float | None:
    """Score of one side: Reason (v3) with legacy S* fallback."""
    if not side:
        return None
    r = side.get("reason")
    return r if r is not None else side.get("S")


class Dashboard:
    def __init__(self, cfg: Config, state: State, hippius: Hippius):
        self.cfg = cfg
        self.state = state
        self.hippius = hippius
        self._last_flush = 0.0
        self._min_interval = cfg.validator.dashboard_flush_min_interval_s
        self._last_log_push = 0.0
        # pm2 writes logs relative to its cwd (affine/), see ecosystem.config.js.
        self._logs_dir = Path(__file__).resolve().parents[1] / "logs"
        # Hotkey→uid from the validator metagraph (updated each tick).
        self.uid_of: dict[str, int] = {}

    # -- website -----------------------------------------------------------------
    def push_website(self) -> None:
        site = Path(__file__).resolve().parents[1] / "website"
        if site.exists():
            self._rebuild_llms_txt()
            n = self.hippius.push_directory(site)
            log.info("pushed %d website files to hippius", n)
        self.flush(force=True)
        self._push_contract()
        self.publish_evals()

    # -- training data -------------------------------------------------------------
    def publish_evals(self) -> None:
        """Upload full duel records + manifest + complete audit logs.

        Artifacts are immutable (one per challenge_id); uploads already
        acknowledged by Hippius are skipped via a local marker file, so a
        cooldown or crash just retries on the next publish.
        """
        d = self.state.evals_dir
        if d.exists():
            marker = d / ".pushed.json"
            pushed: set[str] = set()
            if marker.exists():
                try:
                    pushed = set(json.loads(marker.read_text()))
                except json.JSONDecodeError:
                    log.warning("corrupt %s; re-uploading all artifacts", marker)
            changed = False
            for path in sorted(d.glob("*.json.gz")):
                if path.name in pushed:
                    continue
                ok = self.hippius.put_bytes(
                    f"evals/{path.name}", path.read_bytes(),
                    "application/gzip",
                    cache_control="public, max-age=31536000, immutable")
                if not ok:
                    break  # Hippius cooling down; retry next publish.
                pushed.add(path.name)
                changed = True
            if changed:
                marker.write_text(json.dumps(sorted(pushed)))
            idx = self.state.evals_index_path
            if idx.exists():
                self.hippius.put_bytes("evals/index.jsonl", idx.read_bytes(),
                                       "application/x-ndjson",
                                       cache_control="no-cache")
        self._push_full_logs()

    def _push_full_logs(self) -> None:
        """Complete history/bench logs, gzipped. Only called on verdicts and
        website pushes — never on the 5s flush cadence."""
        for path, key in ((self.state.history_path,
                           "data/history_full.jsonl.gz"),
                          (self.state.bench_history_path,
                           "data/bench_history_full.jsonl.gz")):
            if path.exists():
                self.hippius.put_bytes(key, gzip.compress(path.read_bytes()),
                                       "application/gzip",
                                       cache_control="no-cache")

    def _rebuild_llms_txt(self) -> None:
        """Regenerate website/llms.txt from live sources before upload."""
        script = Path(__file__).resolve().parents[1] / "scripts" / "build_llms_txt.py"
        if not script.is_file():
            log.warning("build_llms_txt.py missing; skipping llms.txt regenerate")
            return
        # Load by path so the validator does not need scripts/ on PYTHONPATH.
        spec = importlib.util.spec_from_file_location("affine_build_llms_txt", script)
        if spec is None or spec.loader is None:
            log.warning("could not load %s; skipping llms.txt regenerate", script)
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        log.info("regenerated website/llms.txt")

    def _push_contract(self) -> None:
        """The public rules miners build against. No secrets in affine.toml."""
        ms = self.cfg.miner_serving
        contract = {
            "subnet": self.cfg.raw["subnet"],
            "submission": self.cfg.raw["submission"],
            "teacher": {"repo": self.cfg.teacher.repo},
            "dataset": self.cfg.raw["dataset"],
            "duel": self.cfg.raw["duel"],
            "bench": {k: v for k, v in self.cfg.raw["bench"].items()},
            # Serving-stack disclosure: how challenger checkpoints are loaded.
            # Stock vLLM, never --trust-remote-code; dtype follows the
            # checkpoint's own config.json (vLLM "auto"). Exact invocation:
            # code/evalsrv/engine.py _vllm_cmd. Live installed versions:
            # api/v1/snapshot eval_machine.versions.
            "serving": {
                "engine": "vllm",
                "trust_remote_code": False,
                "dtype": "auto",
                "tensor_parallel_size": int(ms["tp"]),
                "max_model_len": int(ms["max_model_len"]),
                "gpu_memory_utilization": float(ms["gpu_memory_utilization"]),
                "max_num_batched_tokens": int(ms["max_num_batched_tokens"]),
                "gpu_types": list(self.cfg.eval_machine.gpu_types),
            },
            "version": __version__,
        }
        self.hippius.put_json("data/contract.json", contract)
        self._write_public_json("contract.json", contract)

    # -- periodic state ---------------------------------------------------------------
    def flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_flush < self._min_interval:
            return
        self._last_flush = now
        s = self.state
        king = s.king
        payload = {
            "generated_at": now_iso(),
            "version": __version__,
            "phase": s.phase,
            "king": ({
                "repo": king.repo, "revision": king.revision,
                "hotkey": king.hotkey, "reign_number": king.reign_number,
                "crowned_at": king.crowned_at, "block": king.block,
                "score": king.score,
            } if king else None),
            # Full lineage for the UI; `size` is the equal-share payout window.
            "reign": {
                "size": self.cfg.king_chain_size,
                "members": self._annotate_uids(
                    s.king_lineage_members(self.cfg.king_chain_size)),
            },
            # Hotkey list kept for older readers / miners (payout window only).
            "reign_chain": s.king_chain_hotkeys(self.cfg.king_chain_size),
            "queue": [{
                "challenge_id": e.challenge_id, "repo": e.repo,
                "hotkey": e.hotkey, "queued_at": e.queued_at,
                "retry_count": e.retry_count,
            } for e in s.queue],
            "current_eval": s.current_eval,
            # Newest-last reveal→intake decisions (commit ≠ duel queue row).
            "intake": list(s.intake),
            "bench_jobs": s.bench_jobs,
            # stats.queued = lifetime enqueues (not "waiting now").
            "stats": {
                **s.stats,
                "enqueued_total": s.stats.get("queued", 0),
                "duel_queue_len": len(s.queue) + (1 if s.current_eval else 0),
            },
            "eval_machine": {
                "provider": s.eval_machine.get("provider"),
                "created_at": s.eval_machine.get("created_at"),
                # Live serving-stack disclosure (vllm/transformers/torch as
                # reported by the pod's /health) — miners pre-flight against it.
                "versions": s.eval_machine.get("versions"),
            } if s.eval_machine else None,
            "bench_machine": {
                "provider": s.bench_machine.get("provider"),
                "created_at": s.bench_machine.get("created_at"),
            } if s.bench_machine else None,
        }
        self.hippius.put_json("data/dashboard.json", payload)
        # Local hot-path snapshot for affine-dash (live current_eval lives in
        # memory; Hippius alone is too slow / cold for the interactive UI).
        self._write_public_json("snapshot.json", payload)
        self._flush_history()
        self._flush_benchmarks()
        self._push_validator_log()

    def _push_validator_log(self) -> None:
        """Publish a redacted tail of the validator log. Rate-limited well
        below the flush cadence — logs are for humans and monitors, not for
        the page load path."""
        now = time.monotonic()
        if now - self._last_log_push < LOG_PUSH_INTERVAL_S:
            return
        self._last_log_push = now
        sections = []
        for name in ("validator.err.log", "validator.out.log"):
            path = self._logs_dir / name
            if not path.exists() or path.stat().st_size == 0:
                continue
            with open(path, "rb") as f:
                f.seek(max(0, path.stat().st_size - LOG_TAIL_BYTES))
                raw = f.read()
            text = raw.decode("utf-8", errors="replace")
            # Drop the first line: likely partial after the seek.
            if len(raw) == LOG_TAIL_BYTES:
                text = text.split("\n", 1)[-1]
            sections.append(f"===== {name} (tail) =====\n{redact_log_text(text)}")
        if not sections:
            return
        body = (f"# Affine validator log tail — generated {now_iso()}\n"
                f"# Pod network coordinates are redacted; everything else is "
                f"the live control-plane log.\n\n" + "\n\n".join(sections))
        self.hippius.put_bytes("data/validator_log.txt", body.encode(),
                               "text/plain; charset=utf-8",
                               cache_control="no-cache")

    def _tail_jsonl(self, path: Path, n: int) -> list[dict]:
        """Last n records without reading the whole file: history files grow
        forever, and this runs on every dashboard flush."""
        if not path.exists():
            return []
        # Read fixed-size chunks from the end until we have n+1 newlines.
        chunk = 1 << 16
        buf = b""
        with open(path, "rb") as f:
            f.seek(0, 2)
            pos = f.tell()
            while pos > 0 and buf.count(b"\n") <= n:
                step = min(chunk, pos)
                pos -= step
                f.seek(pos)
                buf = f.read(step) + buf
        lines = buf.split(b"\n")
        if pos > 0 and lines:
            lines = lines[1:]  # first line is likely partial
        rows = []
        for line in lines:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows[-n:]

    def _annotate_uids(self, members: list[dict]) -> list[dict]:
        out = []
        for m in members:
            row = dict(m)
            uid = self.uid_of.get(str(row.get("hotkey") or ""))
            if uid is not None:
                row["uid"] = int(uid)
            out.append(row)
        return out

    def _flush_history(self) -> None:
        rows = self._tail_jsonl(self.state.history_path, HISTORY_TAIL)
        # Strip bulky per-side summaries down to what the site renders.
        slim = []
        for r in reversed(rows):
            v = r.get("verdict") or {}
            hk = r.get("hotkey")
            uid = r.get("uid")
            if uid is None and hk:
                uid = self.uid_of.get(str(hk))
            slim.append({
                "event": r.get("event"), "at": r.get("at"),
                "challenge_id": r.get("challenge_id"),
                "repo": r.get("repo"), "hotkey": hk,
                "uid": uid,
                "duration_s": r.get("duration_s"),
                "accepted": r.get("accepted"),
                "error_code": r.get("error_code"),
                "error_detail": (r.get("error_detail") or "")[:2000],
                "z": v.get("z"), "margin": v.get("margin"),
                "se": v.get("se"), "k_sigma": v.get("k_sigma"),
                "n_paired_turns": v.get("n_paired_turns"),
                "rejection_reason": v.get("rejection_reason"),
                "reign_number": r.get("reign_number"),
                # Absolute score (Reason) for both sides; falls back to the
                # legacy S* field for pre-fork rows.
                "score": r.get("score", _side_score(v.get("challenger"))),
                "score_king": _side_score(v.get("king")),
                # Pre-fork rows stamp `gates`; Reason v3 rows stamp
                # `duel_params` + teacher length telemetry + duel_seconds.
                "gates": v.get("gates"),
                "duel_params": v.get("duel_params"),
                "teacher": v.get("teacher"),
                "duel_seconds": v.get("duel_seconds"),
                "challenger": _slim_side(v.get("challenger")),
                "king": _slim_side(v.get("king")),
            })
        self.hippius.put_json("data/history.json", slim)
        self._write_public_json("history.json", slim)

    def _flush_benchmarks(self) -> None:
        done = self._tail_jsonl(self.state.bench_history_path, BENCH_TAIL)
        # Presentation contract: one value per model (keyed by repo, so a
        # re-benched king keeps a single column) — its latest successful score.
        # Failures never reach the panel; they live in bench history and logs.
        models: dict[str, dict] = {}
        for row in done:
            result = row.get("result") or {}
            if not result.get("ok") or result.get("score") is None:
                continue
            m = models.setdefault(row["repo"], {
                "model_repo": row["repo"], "revision": row["revision"],
                "label": row.get("label", ""), "hotkey": row.get("hotkey", ""),
                "suites": {},
            })
            m["revision"] = row["revision"]  # rows are chronological: last ok wins
            m["suites"][row["suite"]] = {
                "score": result.get("score"),
                "ok": True,
                "n_sims": result.get("n_sims"),
                "finished_at": row.get("finished_at"),
            }
        active = [{
            "job_id": j["job_id"], "model_repo": j["repo"], "suite": j["suite"],
            "state": j["state"], "queued_at": j["queued_at"],
        } for j in self.state.bench_jobs]
        bench_payload = {
            "generated_at": now_iso(),
            "suites": list(self.cfg.bench.suites),
            "models": list(models.values()),
            "active": active,
        }
        self.hippius.put_json("data/benchmarks.json", bench_payload)
        self._write_public_json("benchmarks.json", bench_payload)

    def _write_public_json(self, name: str, data) -> None:
        """Atomically mirror a presentation object under state/public/."""
        public = self.cfg.state_dir / "public"
        try:
            public.mkdir(parents=True, exist_ok=True)
            path = public / name
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, default=str))
            tmp.replace(path)
        except OSError as exc:
            log.warning("local public write %s failed: %s", name, exc)
