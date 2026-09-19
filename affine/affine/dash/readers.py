"""Filesystem readers for validator state_dir (read-only)."""

from __future__ import annotations

import gzip
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__, payout
from ..config import Config
from ..state import now_iso

log = logging.getLogger("affine.dash.readers")

# Same scrub as dashboard.redact_log_text (duplicated so the dash process
# never imports the validator-side module): pod SSH coordinates stay private.
_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_SSH_PORT_RE = re.compile(r"(-p\s+)\d{2,5}\b")

# pm2 writes logs relative to its cwd (affine/), see ecosystem.config.js.
_LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_TAIL_BYTES = 8 << 20
_LOG_MAX_LINES = 400


def duel_log_lines(challenge_id: str) -> list[str]:
    """Validator log lines mentioning this duel, redacted, oldest first.

    Greps a bounded tail of the validator logs so the endpoint stays cheap
    even as the files grow without rotation.
    """
    needle = challenge_id
    out: list[str] = []
    for name in ("validator.out.log", "validator.err.log"):
        path = _LOGS_DIR / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            with open(path, "rb") as f:
                f.seek(max(0, path.stat().st_size - _LOG_TAIL_BYTES))
                raw = f.read()
        except OSError as exc:
            log.warning("log read failed %s: %s", path, exc)
            continue
        text = raw.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if needle in line:
                line = _IP_RE.sub("[ip]", line)
                out.append(_SSH_PORT_RE.sub(r"\1[port]", line))
    return out[-_LOG_MAX_LINES:]


def public_dir(cfg: Config) -> Path:
    return cfg.state_dir / "public"


def read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("bad json %s: %s", path, exc)
        return None


def load_public_snapshot(cfg: Config) -> dict | None:
    data = read_json(public_dir(cfg) / "snapshot.json")
    return data if isinstance(data, dict) else None


def load_public_history(cfg: Config) -> list | None:
    data = read_json(public_dir(cfg) / "history.json")
    return data if isinstance(data, list) else None


def load_public_benchmarks(cfg: Config) -> dict | None:
    data = read_json(public_dir(cfg) / "benchmarks.json")
    return data if isinstance(data, dict) else None


def load_public_audits(cfg: Config) -> list | None:
    """Post-crown exploit-audit verdict list (newest first)."""
    data = read_json(public_dir(cfg) / "audits.json")
    return data if isinstance(data, list) else None


def load_audit_detail(cfg: Config, reign: int) -> dict | None:
    """One audit's replayable workspace: the verdict entry, the pinned
    manifest (input sha256s), the machine verdict, and the analysis prose.
    Everything here is also on the public bucket under audits/reign_NNNN/."""
    entries = load_public_audits(cfg) or []
    entry = next((e for e in entries
                  if int(e.get("reign_number", -1)) == int(reign)), None)
    ws = public_dir(cfg) / "audits" / f"reign_{int(reign):04d}"
    manifest = read_json(ws / "manifest.json")
    verdict = read_json(ws / "verdict.json")
    analysis_path = ws / "analysis.md"
    analysis = None
    if analysis_path.exists():
        try:
            analysis = analysis_path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("bad analysis %s: %s", analysis_path, exc)
    if entry is None and manifest is None:
        return None
    return {
        "reign_number": int(reign),
        "entry": entry,
        "manifest": manifest,
        "verdict": verdict,
        "analysis_md": analysis,
    }


def _reign_members(king: dict | None, window_s: float) -> list[dict]:
    """Mirror State.king_lineage_members without loading/mutating State
    (same pure computation: affine.payout). The accessibility sweep result
    is memory-only in the validator, so this fallback treats every crown
    inside its window as paid."""
    return payout.annotate_lineage(
        payout.lineage_rows(king), window_s=window_s,
        now=datetime.now(timezone.utc))


def reconstruct_snapshot(cfg: Config) -> dict:
    """Build a snapshot from state.json when public/snapshot.json is missing.

    Read-only: never instantiates State (load() can reconcile/write).
    `current_eval` / `phase` are memory-only unless dashboard has flushed a
    public snapshot, so this fallback omits live progress.
    """
    raw = read_json(cfg.state_dir / "state.json")
    if not isinstance(raw, dict):
        raw = {}
    king = raw.get("king")
    members = _reign_members(king if isinstance(king, dict) else None,
                             cfg.king_payout_window_s)
    paid = payout.paid_crowns(members)
    queue = raw.get("queue") or []
    return {
        "generated_at": now_iso(),
        "version": __version__,
        "phase": {"name": "unknown", "since": raw.get("flushed_at")},
        "king": ({
            "repo": king.get("repo"), "revision": king.get("revision"),
            "hotkey": king.get("hotkey"),
            "reign_number": king.get("reign_number"),
            "crowned_at": king.get("crowned_at"), "block": king.get("block"),
            "score": king.get("score"),
        } if isinstance(king, dict) else None),
        "reign": {"size": cfg.king_chain_size,
                  "payout_window_hours": cfg.king_payout_window_s / 3600,
                  "members": members},
        "payout": {
            "rule": payout.rule_text(cfg.king_payout_window_s / 3600),
            "window_hours": cfg.king_payout_window_s / 3600,
            "effective_at": cfg.king_payout_rule_effective_at or None,
            "burn": not paid, "n_paid": len(paid), "paid": paid,
            "shares_by_hotkey": payout.shares_by_hotkey(members),
        },
        "reign_chain": list(dict.fromkeys(m["hotkey"] for m in paid)),
        "queue": [{
            "challenge_id": e.get("challenge_id"), "repo": e.get("repo"),
            "hotkey": e.get("hotkey"), "queued_at": e.get("queued_at"),
            "retry_count": e.get("retry_count", 0),
        } for e in queue if isinstance(e, dict)],
        "current_eval": None,
        "intake": list(raw.get("intake") or []),
        "bench_jobs": raw.get("bench_jobs") or [],
        "stats": {
            **(raw.get("stats") or {}),
            "enqueued_total": (raw.get("stats") or {}).get("queued", 0),
            "duel_queue_len": len(queue),
        },
        "eval_machine": {
            "provider": (raw.get("eval_machine") or {}).get("provider"),
            "created_at": (raw.get("eval_machine") or {}).get("created_at"),
        } if raw.get("eval_machine") else None,
        "bench_machine": {
            "provider": (raw.get("bench_machine") or {}).get("provider"),
            "created_at": (raw.get("bench_machine") or {}).get("created_at"),
        } if raw.get("bench_machine") else None,
    }


def snapshot(cfg: Config) -> dict:
    pub = load_public_snapshot(cfg)
    if pub is not None:
        return pub
    return reconstruct_snapshot(cfg)


def contract_payload(cfg: Config) -> dict:
    # The validator publishes submission_r2 (its mailbox-signing identity)
    # into public/contract.json; the dash has no secrets, so it relays that
    # block instead of deriving it. submit.py auth reads it from here.
    published = read_json(public_dir(cfg) / "contract.json")
    r2_block = ((published or {}).get("submission_r2")
                if isinstance(published, dict) else None)
    return {
        "submission_r2": r2_block or {"enabled": False},
        "subnet": cfg.raw["subnet"],
        "payout": payout.contract_block(
            cfg.king_payout_window_s / 3600,
            cfg.king_payout_rule_effective_at, cfg.burn_uid),
        "submission": cfg.raw["submission"],
        "teacher": {"repo": cfg.teacher.repo},
        "dataset": cfg.raw["dataset"],
        "duel": cfg.raw["duel"],
        "bench": {k: v for k, v in cfg.raw["bench"].items()},
        "dashboard": {
            "public_base_url": cfg.dashboard.get("public_base_url", ""),
        },
        "version": __version__,
    }


def eval_artifact_path(cfg: Config, challenge_id: str) -> Path:
    return cfg.state_dir / "evals" / f"{challenge_id}.json.gz"


def load_eval_artifact(cfg: Config, challenge_id: str) -> dict | None:
    path = eval_artifact_path(cfg, challenge_id)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, gzip.BadGzipFile) as exc:
        log.warning("bad eval artifact %s: %s", path, exc)
        return None


def bench_artifact_path(cfg: Config, job_id: str) -> Path:
    return cfg.state_dir / "benches" / f"{job_id}.json.gz"


def load_bench_artifact(cfg: Config, job_id: str) -> dict | None:
    path = bench_artifact_path(cfg, job_id)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, gzip.BadGzipFile) as exc:
        log.warning("bad bench artifact %s: %s", path, exc)
        return None


def bench_artifact_index(cfg: Config, limit: int = 200) -> list[dict]:
    """Newest-first manifest rows of published bench rollout records."""
    path = cfg.state_dir / "benches" / "index.jsonl"
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        log.warning("bench index read failed: %s", exc)
        return []
    return rows[::-1][:limit]


def history_row_from_raw(r: dict) -> dict:
    """Normalize a history.jsonl record into the slim dashboard shape."""
    v = r.get("verdict") or {}
    return {
        "event": r.get("event"),
        "at": r.get("at"),
        "challenge_id": r.get("challenge_id"),
        "repo": r.get("repo"),
        "hotkey": r.get("hotkey"),
        "uid": r.get("uid"),
        "duration_s": r.get("duration_s"),
        "revision": r.get("revision"),
        "accepted": r.get("accepted"),
        "error_code": r.get("error_code"),
        # Keep the validator's full failure text (state caps at 2000).
        "error_detail": (r.get("error_detail") or "")[:2000],
        "z": v.get("z"),
        "margin": v.get("margin"),
        "se": v.get("se"),
        "n_paired_turns": v.get("n_paired_turns"),
        "n_forfeit_turns": v.get("n_forfeit_turns"),
        "near_miss": v.get("near_miss"),         # sequential near-miss stamp
        "protocol_probe": v.get("protocol_probe"),  # admission probe result
        # wvk 22 sd-meter (teacher-sd units); shadow read on wvk-21 rows.
        "sd_meter": slim_sd_meter(v),
        "rejection_reason": v.get("rejection_reason"),
        "reign_number": r.get("reign_number"),
        "score": r.get("score", _side_score(v.get("challenger"))),
        "score_king": _side_score(v.get("king")),
        "gates": v.get("gates"),                 # pre-fork verdicts
        "duel_params": v.get("duel_params"),     # Reason v3 verdicts
        "teacher": v.get("teacher"),
        "duel_seconds": v.get("duel_seconds"),
        "challenger": v.get("challenger"),
        "king": v.get("king"),
        "challenger_wins": v.get("challenger_wins"),
        # wvk-15 era (2026-09-12 17:01 -> 2026-09-13 13:01 UTC, retired):
        # window stamps on verdicts, and the window_close / crown_revoked
        # rows themselves. The site labels these as the retired rule.
        "crown_mode": v.get("crown_mode") or r.get("crown_mode"),
        "window_id": v.get("window_id", r.get("window_id")),
        "outcome": r.get("outcome"),
        "via": r.get("via") or v.get("via"),
        "revoked_reason": r.get("revoked_reason"),
        "revoked_code": r.get("revoked_code"),
        "revoked_by": r.get("revoked_by"),
        # wvk 19: the confirmation slice of a first-slice crown pass.
        "confirmation": v.get("confirmation"),
    }


def _side_score(side: dict | None) -> float | None:
    """Score of one side: Reason (v3) with legacy S* fallback."""
    if not side:
        return None
    r = side.get("reason")
    return r if r is not None else side.get("S")


# The sd-meter block a verdict carries under `shadow.sd_meter` (shadow read
# from 2026-09-18 10:41 UTC, the rule since wvk 22). Everything the site
# plots per side, without the formula prose and the cost ledger.
_SD_TOP_KEYS = ("role", "anchor", "margin", "se", "z", "sd_diff",
                "n_paired_turns", "n_forfeit_turns", "would_crown",
                "live_gates_pass", "knobs", "sigma_by_dialect")
_SD_SIDE_KEYS = ("mean", "mean_valid", "sd_valid", "n_turns", "n_valid",
                 "n_forfeits", "n_unscorable", "mean_z_R", "mean_typ_c",
                 "mean_z_A", "bind_frac", "n_leg_dropped",
                 "mean_content_share", "mean_R", "mean_A", "mean_mc")


def slim_sd_meter(verdict: dict | None) -> dict | None:
    """Per-side sd-meter telemetry (teacher-sd units) for history rows.

    `role = "rule"` (wvk >= 22) means margin / se / z here equal the
    verdict's own; `role = "shadow"` (wvk 21 shadow read) means the
    verdict was decided by min(R, G) and this block is what the sd-meter
    would have said."""
    sd = ((verdict or {}).get("shadow") or {}).get("sd_meter")
    if not isinstance(sd, dict):
        return None
    out = {k: sd.get(k) for k in _SD_TOP_KEYS if k in sd}
    for side in ("challenger", "king"):
        s = sd.get(side)
        if isinstance(s, dict):
            out[side] = {k: s.get(k) for k in _SD_SIDE_KEYS if k in s}
    return out
