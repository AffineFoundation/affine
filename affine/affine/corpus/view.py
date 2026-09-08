"""duel_turns@v4 — D as a view over rollout traces.

One view record per rollout. It holds the plain-text message *graph* the
model actually saw (baked through the teacher's chat template when the
harness used native tools) and one turn meta per scorable reply:

    {
      "view": "duel_turns@v4", "rollout_id", "traj_id", "instance_id",
      "repo", "model", "policy": {id, harness, action_kind}, "source",
      "language", "stratum"?, "action_kind", "generated_at",
      "nodes": [{"parent": int|None, "role", "content"}, ...],
      "turns": [{"turn_idx", "node_id", "phase", "suffix_len",
                 "n_prefix_chars", "action_kind"}, ...],
    }

A turn's prefix is the root->parent path of its node
(affine.corpus.materialize.node_path); its reference is the node's content.
Paths are built from `sampled_paths` — one per model reply, exactly the
prompt the model was sent — so a harness that drops, rewrites, compacts
or forks history is represented faithfully with no per-harness rule.
Nodes are shared between replies whenever their (parent, role, content)
agree, so a linear rollout is a chain and a forked one is a tree.

The slicer (datagen.slicer.slice_messages, single-reply mode) decides which
replies are scorable: prefix ends on user, exactly one action in the
policy's dialect, no verbatim leakage of that action into the prefix, and
the 120k-char prefix cap. Those rules are unchanged from duel_turns@v3;
v3 records and v4 turns materialize to identical prefix/reference pairs
(derivation-parity gate, 2026-09-02).

`reasoning_content` is not in this view; it stays in the trace. A view that
keeps prior reasoning in tool-loop prefixes is a new spec over the same
traces, never new rollouts.
"""

from __future__ import annotations

import hashlib
import json
import re

from affine import dialects
from affine.corpus.materialize import materialize_turn
from affine.corpus.trace import trace_conversations, trace_error_type
from datagen.slicer import MAX_PREFIX_CHARS, slice_messages

VIEW_SPEC = "duel_turns@v4"
WS_RE = re.compile(r"\s+")
# Bench-panel keys: (instance ids, owner/repo names, bare repo names).
PanelKeys = tuple[set[str], set[str], set[str]]


def _norm(s: str) -> str:
    return WS_RE.sub(" ", s).strip().lower()


def run_tag(trace: dict) -> str:
    """Identity tag in traj_id: the adapter's stamp when present (mini_swe:
    sha256 of the raw traj file), else sha256 of the sorted trace dump."""
    stamped = (trace.get("info") or {}).get("run_tag")
    if stamped:
        return stamped
    return hashlib.sha256(
        json.dumps(trace, sort_keys=True).encode()).hexdigest()[:8]


def deliberate_final_reply(trace: dict) -> bool:
    """Did the model END this rollout on purpose, with an untruncated reply?

    True iff the harness stopped because the agent said it was done
    (`stop_condition == "agent_completed"`, not max_turns / timeout) and the
    last sampled reply's model call finished with `stop` (not `length`).
    Only such a final reply may become a `text` turn — the report the model
    chose to close on, not a cap it ran into. Traces without call records
    (older dumps) are treated as untruncated: the stop condition alone
    decides."""
    if trace.get("stop_condition") != "agent_completed":
        return False
    nodes = trace.get("nodes") or []
    sampled = [i for i, nd in enumerate(nodes)
               if nd.get("sampled")
               and (nd.get("message") or {}).get("role") == "assistant"]
    if not sampled:
        return False
    last = sampled[-1]
    finishes = [c.get("finish_reason") for c in (trace.get("calls") or [])
                if c.get("node") == last]
    return not finishes or finishes[-1] != "length"


def build_view_record(envelope: dict, *, baker=None,
                      generated_at: str | None = None) -> dict | None:
    """View record for one envelope, or None when nothing is scorable
    (errored rollout, no reply passes the slicer). Raises ToolParityError /
    TraceShapeError for traces that cannot be represented plain."""
    trace = envelope["trace"]
    task = envelope["task"]
    policy = envelope["policy"]
    if trace_error_type(trace) is not None:
        return None
    convs = trace_conversations(trace, baker)
    stamped_at = (trace.get("info") or {}).get("generated_at")
    common = dict(
        instance_id=task["sid"],
        repo=task.get("repo") or "",
        model=policy["model"],
        run_tag=run_tag(trace),
        generated_at=stamped_at or generated_at or "",
        # Pre-dialect envelopes carry no action_kind: they were all bash.
        action_kind=policy.get("action_kind") or dialects.DEFAULT_KIND)

    final_is_text = deliberate_final_reply(trace)

    nodes: list[dict] = []
    by_key: dict[tuple[int | None, str, str], int] = {}
    turns: list[dict] = []
    traj_id = ""
    for i, conv in enumerate(convs):
        recs = slice_messages(conv, turn=(i, len(convs)),
                              text_final=(final_is_text and i == len(convs) - 1),
                              **common)
        if not recs:
            continue
        rec = recs[0]
        traj_id = rec["traj_id"]
        parent: int | None = None
        for m in [*rec["prefix"],
                  {"role": "assistant", "content": rec["reference_turn"]}]:
            key = (parent, m["role"], m["content"])
            nid = by_key.get(key)
            if nid is None:
                nid = len(nodes)
                nodes.append({"parent": parent, "role": m["role"],
                              "content": m["content"]})
                by_key[key] = nid
            parent = nid
        turns.append({
            "turn_idx": rec["turn_idx"],
            "node_id": parent,
            "phase": rec["phase"],
            "suffix_len": rec["suffix_len"],
            "n_prefix_chars": rec["n_prefix_chars"],
            "action_kind": rec["action_kind"],
        })
    if not turns:
        return None
    record = {
        "view": VIEW_SPEC,
        "rollout_id": envelope["rollout_id"],
        "traj_id": traj_id,
        "instance_id": common["instance_id"],
        "repo": common["repo"],
        "model": common["model"],
        "policy": {"id": policy.get("id", ""),
                   "harness": policy.get("harness", ""),
                   "action_kind": common["action_kind"]},
        "source": envelope["source"],
        "language": task.get("language") or "",
        "action_kind": common["action_kind"],
        "generated_at": common["generated_at"],
        "nodes": nodes,
        "turns": turns,
    }
    if task.get("stratum"):
        record["stratum"] = str(task["stratum"])
    return record


def legacy_view_record(traj: dict, *, legacy_epoch: int) -> dict:
    """v2 trajectory chunk record (linear `messages` + `msg_pos` metas) ->
    v4 record with a chain graph. v2 records were cut from mini-swe's own
    message list, so they never had a phantom turn: the chain is exact."""
    messages = traj["messages"]
    nodes = [{"parent": (i - 1 if i else None), "role": m["role"],
              "content": m["content"]} for i, m in enumerate(messages)]
    turns = [{
        "turn_idx": int(t["turn_idx"]),
        "node_id": int(t["msg_pos"]),
        "phase": t.get("phase", ""),
        "suffix_len": int(t.get("suffix_len",
                                len(messages[int(t["msg_pos"])]["content"]))),
        "n_prefix_chars": int(t.get("n_prefix_chars", 0)),
        "action_kind": traj.get("action_kind", dialects.DEFAULT_KIND),
    } for t in traj["turns"]]
    record = {
        "view": VIEW_SPEC,
        "rollout_id": "",
        "traj_id": traj["traj_id"],
        "instance_id": traj.get("instance_id", ""),
        "repo": traj.get("repo", ""),
        "model": traj.get("model", ""),
        "policy": {"id": "", "harness": "mini_swe",
                   "action_kind": traj.get("action_kind", dialects.DEFAULT_KIND)},
        "source": traj.get("source", "swe"),
        "language": traj.get("language", "python"),
        "action_kind": traj.get("action_kind", dialects.DEFAULT_KIND),
        "generated_at": traj.get("generated_at", ""),
        "legacy_epoch": int(legacy_epoch),
        "nodes": nodes,
        "turns": turns,
    }
    if traj.get("stratum"):
        record["stratum"] = str(traj["stratum"])
    return record


def view_turns(record: dict) -> list[dict]:
    """Every scorable turn of a view record as a v1-shaped turn dict
    (prefix + reference_turn + tags) — what validate_turns and the duel
    consume."""
    out = []
    for meta in record["turns"]:
        t = materialize_turn(record, meta)
        t["source"] = record.get("source", "")
        t["language"] = record.get("language", "")
        if record.get("stratum"):
            t["stratum"] = record["stratum"]
        out.append(t)
    return out


def validate_turns(records: list[dict], *, panel: PanelKeys | None = None,
                   allowed_kinds: tuple[str, ...] | list[str] | None = None,
                   ) -> tuple[list[dict], dict[str, int]]:
    """The fold's per-turn admission contract (was ops/datagen_refresh.py's
    prefilter + rollouts validate_records). Returns (kept, drop counts).

    `allowed_kinds`: [dataset].allowed_action_kinds for the fold — a
    dialect outside it is refused; None admits every REGISTERED dialect
    (staging semantics: a not-yet-admitted dialect builds its backlog)."""
    panel_ids, panel_repos, panel_bare = panel or (set(), set(), set())
    kinds = tuple(allowed_kinds) if allowed_kinds is not None \
        else tuple(dialects.DIALECTS)
    drops: dict[str, int] = {}
    kept: list[dict] = []
    seen: set[str] = set()

    def drop(reason: str) -> None:
        drops[reason] = drops.get(reason, 0) + 1

    for rec in records:
        tid, tix = rec.get("traj_id"), rec.get("turn_idx")
        if not (isinstance(tid, str) and tid and isinstance(tix, int)):
            drop("bad_ids")
            continue
        turn_id = f"{tid}:{tix}"
        if turn_id in seen:
            drop("dup_turn_id")
            continue
        repo = str(rec.get("repo", "")).lower()
        if (repo in panel_repos or repo in panel_bare
                or str(rec.get("instance_id", "")) in panel_ids):
            drop("bench_panel_overlap")
            continue
        kind = rec.get("action_kind")
        reason = dialects.admission_reason(kind, kinds)
        if reason:
            drop(reason)
            continue
        prefix = rec.get("prefix")
        if (not isinstance(prefix, list) or not prefix
                or not all(isinstance(m, dict)
                           and isinstance(m.get("role"), str)
                           and isinstance(m.get("content"), str)
                           for m in prefix)
                or prefix[-1]["role"] != "user"):
            drop("prefix_shape")
            continue
        if sum(len(m["content"]) for m in prefix) > MAX_PREFIX_CHARS:
            drop("prefix_too_long")
            continue
        reason, action = dialects.reference_check(
            prefix, rec.get("reference_turn") or "", kind)
        if reason:
            drop(reason)
            continue
        body = _norm(action)
        if len(body) > 40 and any(body in _norm(m["content"]) for m in prefix):
            drop("reference_leaked_into_prefix")
            continue
        seen.add(turn_id)
        kept.append(rec)
    return kept, drops
