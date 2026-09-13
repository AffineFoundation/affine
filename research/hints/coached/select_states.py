#!/usr/bin/env python
"""Pick the states for the coached-teacher recovery run and write, per state,
the continuation state file (ops/recoverable format) and the coach context.

Runs on the validator box (trace chunk cache, loop labeler, king-review pivot
tables and the recoverable side-tables live there). Nothing it reads is
modified.

Selection (Jacob, 2026-09-13: "pick important states where the hint matters
and will help us learn; more filtering, better signal"):

  reign 12 (`--king d76150805915`), failed king rollouts on the agent
  harnesses (mini-swe textbased, verifiers bash, Terminus 2; ACP harnesses
  through the same-task proxy), king-group sources only:
    tier 0  states the recoverable side-table already tried 3x and the plain
            teacher FAILED (majority) -- the plain control is free and a
            coached solve there is hint-decisive by construction;
    tier 1  admitted king-review pivots (confidence >= 0.7) not in the table;
    tier 2  first loop onsets not in the table (the fold's labeler);
    tier 3  same-task proxy tasks for ACP-harness failures (whole task);
    skipped states the plain teacher already solves in a majority (a hint
            cannot be decisive there) and later onsets (rank >= 2: inside the
            wreckage, half the recovery rate).
  reign 11 top-up (`--topup-king 0ce59769300c --topup-max N`), to cover the
  SWE / terminal harnesses reign 12 has no failures on in the trace cache:
  side-table rows with >= 3 plain continuations and 0 solves on the SWE /
  terminal sources, pivots and first onsets only.

Output (`--out`): states.jsonl + states/<id>.json (run_coached.py input),
coach_ctx/<unit_stem>.json (coach_proxy.py input), selection.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "affine"))
sys.path.insert(0, str(REPO / "ops" / "recoverable"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import candidates as CAND  # noqa: E402  (ops/recoverable/candidates.py)
import common as C  # noqa: E402
import states as ST  # noqa: E402  (ops/recoverable/states.py)
import turnset as TS  # noqa: E402  (research/hints/turnset.py: hindsight)
from affine.corpus.view import rollout_outcome  # noqa: E402
from affine.toolbake import ToolBaker  # noqa: E402

SWE_TERMINAL_SOURCES = {"multiswe", "swesmith", "scaleswe", "r2e_gym", "swerebench_v2",
                        "swelego", "terminal_bench_2", "terminal_lego", "nl2repobench",
                        "affine_nl2lib"}
EXCLUDE_SOURCES = {"affine_wiki", "affine_agent", "affine_math"}


def load_table(path: Path) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        out[(r["rollout_id"], int(r["turn_idx"]), r["state_kind"])] = r
    return out


def plain_control(row: dict | None) -> dict | None:
    """The side-table's plain continuations as a control block (None when
    fewer than 2 OK continuations exist)."""
    if not row or row.get("teacher_status") != "ok":
        return None
    conts = [c for c in row.get("continuations") or [] if c.get("status", "ok") == "ok"]
    n = int(row.get("n_continuations") or len(conts) or 0)
    if n < 2:
        return None
    return {"source": "side_table", "n": n, "n_solved": int(row.get("n_solved") or 0),
            "trace_ids": [c.get("trace_id") for c in conts if c.get("trace_id")],
            "model": row.get("model"), "timestamp": row.get("timestamp")}


def king_ctx(env: dict, turn_idx: int | None, st: dict, pivot: dict | None) -> dict:
    trace = env["trace"]
    hs = TS.hindsight(trace, rollout_outcome(trace))
    transcript = hs["transcript"]
    if len(transcript) > C.KING_TRANSCRIPT_CAP:
        transcript = (transcript[: C.KING_TRANSCRIPT_CAP // 2]
                      + "\n[... middle of the transcript elided ...]\n"
                      + transcript[-C.KING_TRANSCRIPT_CAP // 2:])
    acts = hs.get("actions") or []
    fut = acts[turn_idx:] if turn_idx is not None else list(acts)
    if st.get("king_action"):
        fut.append(st["king_action"])
    prefix = st.get("messages") or []
    task_prompt = st.get("task_prompt") or next(
        (C.message_text(m.get("content")) for m in prefix if m.get("role") == "user"), "")
    return {
        "unit": st.get("proxy_key") or st["state_id"],
        "state_id": st["state_id"],
        "king_digest": CAND.king_digest12(env.get("policy") or {}),
        "rollout_id": env["rollout_id"],
        "turn_idx": turn_idx,
        "resume_kind": st["resume_kind"],
        "harness": st["harness"],
        "source": env["source"],
        "task_name": st.get("task_name"),
        "task_prompt": task_prompt,
        "king_outcome": rollout_outcome(trace),
        "king_stop_condition": trace.get("stop_condition"),
        "king_n_replies": hs.get("n_replies"),
        "king_transcript": transcript,
        "king_future_actions": [a for a in fut if a],
        "pivot": ({k: pivot.get(k) for k in ("failure_category", "rationale", "should_have",
                                            "confidence")} if pivot else None),
        "n_prefix_messages": len(prefix),
        "n_prefix_assistant": sum(1 for m in prefix if m.get("role") == "assistant"),
        "prefix_last_obs_sha": (C.sha(C.message_text(prefix[-1].get("content")))
                                if prefix and prefix[-1].get("role") in ("user", "tool") else None),
        "prompt_version": C.PROMPT_VERSION,
    }


def emit(out: Path, env: dict, turn: int, kind: str, label: dict, tier: int, tag: str,
         control: dict | None, pivot: dict | None, rows: list, counts) -> None:
    st = ST.build_state(env, turn, kind, label)
    if st is None:
        counts["bad_turn_idx"] += 1
        return
    path = out / "states" / (C.unit_stem(st["state_id"]) + ".json")
    path.write_text(json.dumps(st, ensure_ascii=False))
    ctx = king_ctx(env, None if st["resume_kind"] == ST.SAME_TASK else turn, st, pivot)
    (out / "coach_ctx" / (C.unit_stem(ctx["unit"]) + ".json")).write_text(
        json.dumps(ctx, ensure_ascii=False))
    meta = {k: v for k, v in st.items()
            if k not in ("messages", "king_reply", "task_system_prompt", "task_prompt", "label")}
    meta["path"] = str(path)
    meta["tier"] = tier
    meta["select_tag"] = tag
    meta["king_digest"] = ctx["king_digest"]
    meta["plain_control"] = control
    meta["pivot_category"] = (pivot or {}).get("failure_category")
    rows.append(meta)
    counts[f"state:{tag}:{kind}:{st['harness']}:{env['source']}"] += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chunks", required=True, type=Path)
    ap.add_argument("--state-dir", required=True, type=Path,
                    help="affine/state (king_pivots/, recoverable/)")
    ap.add_argument("--king", default="d76150805915")
    ap.add_argument("--topup-king", default="0ce59769300c")
    ap.add_argument("--topup-max", type=int, default=60)
    ap.add_argument("--max-states", type=int, default=200)
    ap.add_argument("--cap-per-source", type=int, default=40)
    ap.add_argument("--cap-source", default="nl2repobench=12,affine_nl2lib=4",
                    help="per-source overrides `src=N,...` (the write-a-repo-from-spec "
                         "environments recover at ~3 %% for anyone; a few suffice)")
    ap.add_argument("--min-confidence", type=float, default=0.7)
    ap.add_argument("--kinds", default="textbased,bash,terminus,same_task")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    kinds = set(args.kinds.split(","))
    (args.out / "states").mkdir(parents=True, exist_ok=True)
    (args.out / "coach_ctx").mkdir(parents=True, exist_ok=True)
    baker = ToolBaker.from_pretrained()
    counts: collections.Counter = collections.Counter()
    rows: list[dict] = []
    per_source: collections.Counter = collections.Counter()

    # ---- reign 12 ------------------------------------------------------------
    table = load_table(args.state_dir / "recoverable" / f"{args.king}.jsonl")
    pivots = CAND.load_pivots(args.state_dir / "king_pivots" / f"{args.king}.jsonl",
                              args.min_confidence, {"format_error"})
    cands: list[tuple[tuple, dict, int, str, dict, dict | None, dict | None, str]] = []
    for env in CAND.iter_king_envelopes(args.chunks, args.king, kinds, EXCLUDE_SOURCES):
        counts["r12_failed_rollouts"] += 1
        rid = env["rollout_id"]
        harness = (env.get("policy") or {}).get("harness", "")
        is_proxy = ST.resume_kind_of(harness) == ST.SAME_TASK
        onsets: dict[int, int] = {}
        if not is_proxy:
            try:
                onsets = CAND.onset_turns(env, baker)
            except Exception as e:  # noqa: BLE001 - labeler shape errors: no onset states
                counts[f"label_error:{type(e).__name__}"] += 1
        piv_rows = pivots.get(rid, {})
        seen_turns: set[int] = set()
        # table rows first (tier 0 / skipped)
        for (r_rid, turn, kind), row in table.items():
            if r_rid != rid:
                continue
            ctrl = plain_control(row)
            if ctrl is None:
                continue
            seen_turns.add(turn)
            if ctrl["n_solved"] * 2 > ctrl["n"]:
                counts["r12_skipped_plain_solved"] += 1
                continue
            label = piv_rows.get(turn) if kind == "pivot" else {
                "loop_onset_of": onsets.get(turn), "onset_rank": row.get("onset_rank"),
                "labeler": "affine.corpus.loops"}
            cands.append(((0, 0 if kind == "pivot" else 1, turn), env, turn, kind,
                          label or {"node_id": row.get("node_id")}, ctrl, piv_rows.get(turn),
                          "r12_t0_plainfail"))
        if is_proxy:
            turn0 = min(piv_rows) if piv_rows else (min(onsets) if onsets else None)
            if turn0 is None:
                counts["r12_proxy_no_label"] += 1
                continue
            kind = "pivot" if turn0 in piv_rows else "loop_onset"
            label = piv_rows.get(turn0) or {"loop_onset_of": onsets.get(turn0), "onset_rank": 1,
                                            "labeler": "affine.corpus.loops"}
            cands.append(((3, 0, turn0), env, turn0, kind, label, None, piv_rows.get(turn0),
                          "r12_t3_proxy"))
            continue
        for turn, prow in piv_rows.items():
            if turn in seen_turns:
                continue
            cands.append(((1, 0, turn), env, turn, "pivot", prow, None, prow, "r12_t1_pivot"))
            seen_turns.add(turn)
        if onsets:
            first = min(onsets)
            if first not in seen_turns:
                cands.append(((2, 1, first), env, first, "loop_onset",
                              {"loop_onset_of": onsets[first], "onset_rank": 1,
                               "labeler": "affine.corpus.loops"}, None, piv_rows.get(first),
                              "r12_t2_first_onset"))
            counts["r12_later_onsets_skipped"] += len(onsets) - 1
    cands.sort(key=lambda c: c[0])
    caps = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.cap_source.split(",") if "=" in kv}
    for key, env, turn, kind, label, ctrl, piv, tag in cands:
        if len(rows) >= args.max_states:
            counts["r12_deferred_by_max"] += 1
            continue
        if per_source[env["source"]] >= caps.get(env["source"], args.cap_per_source):
            counts[f"r12_capped:{env['source']}"] += 1
            continue
        per_source[env["source"]] += 1
        emit(args.out, env, turn, kind, label, key[0], tag, ctrl, piv, rows, counts)
    n_r12 = len(rows)

    # ---- reign 11 top-up ----------------------------------------------------------
    if args.topup_king and args.topup_max > 0 and len(rows) < args.max_states:
        t11 = load_table(args.state_dir / "recoverable" / f"{args.topup_king}.jsonl")
        piv11 = CAND.load_pivots(args.state_dir / "king_pivots" / f"{args.topup_king}.jsonl",
                                 args.min_confidence, {"format_error"})
        want: dict[str, list[tuple]] = collections.defaultdict(list)
        for (rid, turn, kind), row in t11.items():
            if row.get("proxy") or row.get("harness") not in ("mini_swe_textbased", "bash", "terminus_2"):
                continue
            if row.get("source") not in SWE_TERMINAL_SOURCES:
                continue
            ctrl = plain_control(row)
            if ctrl is None or ctrl["n"] < 3 or ctrl["n_solved"] > 0:
                continue
            if kind == "loop_onset" and int(row.get("onset_rank") or 1) > 1:
                continue
            prio = (0 if kind == "pivot" else 1, 0 if row.get("harness") != "mini_swe_textbased" else 1,
                    int(turn))
            want[rid].append((prio, turn, kind, row, ctrl))
        counts["r11_candidate_rollouts"] = len(want)
        idx = ST.index_chunks(args.chunks)
        cache = ST.ChunkCache(args.chunks, idx)
        picked: list[tuple] = []
        for rid, items in want.items():
            items.sort()
            picked.append((items[0][0], rid, items[0]))
        picked.sort(key=lambda p: p[0])
        n11 = 0
        harness_cap = collections.Counter()
        for _, rid, (prio, turn, kind, row, ctrl) in picked:
            if n11 >= args.topup_max or len(rows) >= args.max_states:
                break
            # keep the top-up spread over the three harnesses
            if harness_cap[row["harness"]] >= max(args.topup_max // 2, 8) and row["harness"] == "mini_swe_textbased":
                continue
            env = cache.load(rid)
            if env is None:
                counts["r11_missing_trace"] += 1
                continue
            piv = piv11.get(rid, {}).get(turn)
            label = piv or {"loop_onset_of": None, "onset_rank": row.get("onset_rank"),
                            "labeler": "affine.corpus.loops", "node_id": row.get("node_id")}
            emit(args.out, env, turn, kind, label, 5, "r11_topup_plainfail", ctrl, piv, rows, counts)
            harness_cap[row["harness"]] += 1
            n11 += 1

    # Round-robin over selection blocks so any prefix of the file (the first
    # 50 states, a deadline cut) is a mix of tiers, harnesses and kings.
    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        buckets[r["select_tag"]].append(r)
    ordered: list[dict] = []
    while any(buckets.values()):
        for tag in sorted(buckets):
            if buckets[tag]:
                ordered.append(buckets[tag].pop(0))
    rows = ordered
    with open(args.out / "states.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = [f"# state selection: {len(rows)} states ({n_r12} reign 12, {len(rows) - n_r12} top-up)", ""]
    for k, v in sorted(counts.items()):
        lines.append(f"{v:6d}  {k}")
    by = collections.Counter((r["select_tag"], r["harness"]) for r in rows)
    lines.append("")
    for k, v in sorted(by.items()):
        lines.append(f"{v:6d}  {k[0]} / {k[1]}")
    lines.append("")
    lines.append(f"with side-table plain control: {sum(1 for r in rows if r.get('plain_control'))}")
    (args.out / "selection.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
