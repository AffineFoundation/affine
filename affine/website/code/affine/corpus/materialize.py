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


def node_path(nodes: list[dict], node_id: int) -> list[dict]:
    """Root-to-node message path through a view record's node graph
    (`nodes[i] = {parent, role, content}`), the node itself last."""
    chain: list[dict] = []
    seen = 0
    j: int | None = node_id
    while j is not None:
        if not 0 <= j < len(nodes) or seen > len(nodes):
            raise ValueError(f"node graph broken at node {j}")
        chain.append(nodes[j])
        j = nodes[j].get("parent")
        seen += 1
    chain.reverse()
    return chain


def materialize_turn(traj: dict, turn_meta: dict) -> dict:
    """Expand one scorable turn from a trajectory / view record.

    ``turn_meta`` is an entry from ``traj["turns"]``. Two layouts:
      * v2 chunks (`traj["messages"]`, meta `msg_pos`): the prefix is the
        linear message list before the reply;
      * v4 view records (`traj["nodes"]`, meta `node_id`): the prefix is the
        root-to-parent path of the reply node in the message graph — the
        exact prompt the model saw, however the harness shaped its history.
    Returns a v1-shaped turn dict with ``prefix`` and ``reference_turn`` so
    scoring code is unchanged.
    """
    if "node_id" in turn_meta:
        chain = node_path(traj["nodes"], int(turn_meta["node_id"]))
        asst = chain[-1]
        before = chain[:-1]
        where = f"node {turn_meta['node_id']}"
    else:
        messages = traj["messages"]
        msg_pos = int(turn_meta["msg_pos"])
        if msg_pos < 0 or msg_pos >= len(messages):
            raise ValueError(
                f"msg_pos {msg_pos} out of range for traj {traj.get('traj_id')}")
        asst = messages[msg_pos]
        before = messages[:msg_pos]
        where = f"messages[{msg_pos}]"
    if asst.get("role") != "assistant":
        raise ValueError(f"{where} is not assistant in {traj.get('traj_id')}")
    prefix = [{"role": m["role"], "content": m["content"]} for m in before]
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
        "action_kind": (turn_meta.get("action_kind")
                        or traj.get("action_kind", "bash")),
        "generated_at": traj.get("generated_at", ""),
    }
    if traj.get("rollout_id"):
        out["rollout_id"] = traj["rollout_id"]
    if traj.get("source") is not None:
        out["source"] = traj["source"]
    if traj.get("language") is not None:
        out["language"] = traj["language"]
    if traj.get("stratum"):
        out["stratum"] = traj["stratum"]
    return out
