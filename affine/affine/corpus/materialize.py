"""Shared stratum key + turn materialization from trajectory records."""

from __future__ import annotations

import re

# Same stratification pattern as evalsrv.dueling / corpus_push.
PHASE_RE = re.compile(r"(func_basic|func_pm|lm_rewrite|lm_modify|combine_|pr_\d+)")


def stratum_key(rec: dict) -> str:
    """Repo×phase stratum used by sample_slice.

    Prefers an explicit ``stratum`` field (Parquet index); otherwise derives
    from ``traj_id`` exactly as the v1 duel path did.
    """
    if rec.get("stratum"):
        return str(rec["stratum"])
    tid = rec.get("traj_id", "") or ""
    repo = tid.split(".", 1)[0] if tid else "unknown"
    m = PHASE_RE.search(tid)
    phase = m.group(1) if m else "other"
    return f"{repo}|{phase}"


def materialize_turn(traj: dict, turn_meta: dict) -> dict:
    """Expand one scorable turn from a trajectory record.

    ``turn_meta`` is an entry from ``traj["turns"]`` (must include
    ``turn_idx`` and ``msg_pos``). Returns a v1-shaped turn dict with
    ``prefix`` and ``reference_turn`` so scoring code is unchanged.
    """
    messages = traj["messages"]
    msg_pos = int(turn_meta["msg_pos"])
    if msg_pos < 0 or msg_pos >= len(messages):
        raise ValueError(
            f"msg_pos {msg_pos} out of range for traj {traj.get('traj_id')}")
    asst = messages[msg_pos]
    if asst.get("role") != "assistant":
        raise ValueError(
            f"messages[{msg_pos}] is not assistant in {traj.get('traj_id')}")
    prefix = [{"role": m["role"], "content": m["content"]}
              for m in messages[:msg_pos]]
    if not prefix or prefix[-1]["role"] != "user":
        raise ValueError(
            f"prefix must end on user for {traj.get('traj_id')}:{turn_meta.get('turn_idx')}")
    out = {
        "traj_id": traj["traj_id"],
        "turn_idx": int(turn_meta["turn_idx"]),
        "prefix": prefix,
        "reference_turn": asst["content"],
        "suffix_len": int(turn_meta.get("suffix_len", len(asst["content"]))),
        "n_prefix_chars": int(turn_meta.get(
            "n_prefix_chars", sum(len(m["content"]) for m in prefix))),
        "instance_id": traj.get("instance_id", ""),
        "repo": traj.get("repo", ""),
        "model": traj.get("model", ""),
        "phase": turn_meta.get("phase") or traj.get("phase", ""),
        "action_kind": traj.get("action_kind", "bash"),
        "generated_at": traj.get("generated_at", ""),
    }
    if traj.get("source") is not None:
        out["source"] = traj["source"]
    if traj.get("language") is not None:
        out["language"] = traj["language"]
    return out
