"""Step 3 -- the per-reign report from the cached judgments.

Reads `<out-dir>/cache/judgments.jsonl` (filtered to the sampled rollouts
and the current prompt version) and writes `<out-dir>/report.md` +
`<out-dir>/report.json`:

  - failure categories by env group and by harness (counts, %)
  - pivotal-turn depth distribution (absolute and relative to rollout length)
  - the top recurring pivot patterns (clustered on the judge's generic
    pattern label + the repeated action's head token), with examples
  - agreement with the deterministic labels (loop_onset / in_loop /
    no_action / completion / last turn) computed on the same rollouts
  - recoverability estimates, teacher-difference quotes
  - "what D should contain": each frequent pattern -> a data intervention

  python aggregate.py --out-dir <dir> --sample <dir>/sample.jsonl [--reign 11]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from judge import CATEGORIES, PROMPT_VERSION
from krlib import (TraceStore, clip, load_env_groups, norm_ws, parse_rollout,
                   read_jsonl, short_action)

STOP = set("""a an the of to in on at for with and or by from into after before
without instead then than that this it its is are was were be been being as
same again re when while until does did do not no never always already
agent model command result output""".split())
TOOL_HEAD_RE = re.compile(r"^([A-Za-z_][\w.-]*)")

# Frequent pattern -> what D must contain so the teacher's references at
# duel time correct it (data-only; scoring untouched).
INTERVENTIONS = {
    "loop_after_ok": (
        "The onset state of every loop (the turn where the king first repeats "
        "a command after an OK result), teacher-labelled: group `king_loop_onset` "
        "(shipping) + the judged pivots of this review (`king_pivot`). Add the "
        "teacher's COMPLETION states (final `text` reply / submit turn / last tool "
        "call before a successful stop) as their own group so 'I am finished' is a "
        "state D contains."),
    "never_submits": (
        "Teacher completion states: the submit / final-answer / task_complete turn "
        "of successful teacher rollouts in EVERY harness (mini-swe `submit`, Claude "
        "Code final text, Terminus `task_complete: true`, wiki final answer). Upsample "
        "deep teacher turns where the diff is done and the teacher stops."),
    "ignores_error": (
        "Teacher turns that follow an error / failing-test observation and CHANGE "
        "course (the 98.5 % recovery turns): sample them by `prev_obs_kind == error` "
        "so D over-represents 'read the traceback, then do something different'."),
    "wrong_plan": (
        "Early-phase teacher turns on the same tasks (the first 1/3 of successful "
        "teacher rollouts): plan-forming states where the teacher reads the failing "
        "test / issue before editing. Pair king pivots with the teacher's rollout on "
        "the same task so the reference at that state comes from a solved trajectory."),
    "premature_finish": (
        "Teacher pre-completion verification turns: the last 2-3 turns before a "
        "successful teacher stop (run tests, check diff, THEN finish). For math: "
        "teacher `boxed` turns on the same problems, plus answer-form checks."),
    "tool_misuse": (
        "Teacher turns in the SAME harness (Terminus JSON batches, Claude Code / Kimi "
        "tool schemas, Hermes) -- coverage-matrix fill; teacher rollouts do not exist "
        "yet for kimi_code / hermes_agent / terminus_2 and must be generated first."),
    "format_error": (
        "Teacher turns in the same harness right after a harness nudge (format-error "
        "observation -> well-formed reply). Terminus needs the most: 9 % of king "
        "replies there parse to no action."),
    "context_loss": (
        "Deep teacher turns (depth > median) with long prefixes where the teacher "
        "still acts on what it learned earlier; keep prefixes under the 110k-token "
        "cap so refs survive."),
    "timeout_no_progress": (
        "Long successful teacher trajectories (top quartile by length) so D shows "
        "exploration that converges; plus teacher turns at the king's "
        "`first_useless_turn` states."),
    "misreads_task": (
        "Turn-0/1 teacher states on the same tasks (the teacher's first reading of "
        "the issue) -- pair by task id so the reference shows the correct reading."),
    "other": "Case by case; read the category_note quotes below.",
}


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.0f} %" if d else "-"


def primary_pivot(rec: dict) -> tuple[dict | None, str]:
    """(pivot dict, source) -- stage-1 (blind) pivots first, the stage-2
    revised list only when the blind pass gave none."""
    n = rec.get("n_turns") or 0

    def valid(p) -> bool:
        return (isinstance(p, dict) and isinstance(p.get("turn"), int)
                and 0 <= p["turn"] < n)
    s1 = [p for p in (rec.get("stage1") or {}).get("pivotal_turns") or [] if valid(p)]
    if s1:
        return s1[0], "blind"
    s2 = [p for p in (rec.get("stage2") or {}).get("revised_pivotal_turns") or [] if valid(p)]
    if s2:
        return s2[0], "revealed"
    return None, "none"


def all_pivots(rec: dict) -> list[tuple[dict, str]]:
    n = rec.get("n_turns") or 0
    out = []
    for src, key, stage in (("blind", "pivotal_turns", "stage1"),
                            ("revealed", "revised_pivotal_turns", "stage2")):
        for p in (rec.get(stage) or {}).get(key) or []:
            if isinstance(p, dict) and isinstance(p.get("turn"), int) and 0 <= p["turn"] < n:
                out.append((p, src))
    return out


def category_of(rec: dict) -> str:
    c = (rec.get("stage2") or {}).get("failure_category")
    return c if c in CATEGORIES else "other"


def pattern_tokens(rec: dict) -> set[str]:
    s2 = rec.get("stage2") or {}
    s1 = rec.get("stage1") or {}
    text = str(s2.get("pivot_pattern") or "")
    toks = {t for t in re.findall(r"[a-z][a-z_\-]+", text.lower()) if t not in STOP}
    toks = {re.sub(r"(ing|ed|es|s)$", "", t) if len(t) > 4 else t for t in toks}
    rep = s1.get("repeated_action")
    if isinstance(rep, str) and rep.strip():
        m = TOOL_HEAD_RE.match(rep.strip().lstrip("`").split("(")[0].split("{")[0])
        if m:
            toks.add("act:" + m.group(1).lower())
            words = re.findall(r"[a-z][a-z_\-]+", rep.lower())
            if len(words) > 1 and m.group(1).lower() in ("bash", "cd", "git", "npx", "go", "python"):
                toks.add("act:" + " ".join(words[:3]))
    return toks


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def cluster_patterns(recs: list[dict], threshold: float = 0.4) -> list[dict]:
    """Greedy clustering inside each category: a judgment joins the first
    cluster whose token set it overlaps by >= threshold (Jaccard), else it
    starts a new one. Deterministic given the input order."""
    clusters: list[dict] = []
    for rec in sorted(recs, key=lambda r: r["rollout_id"]):
        cat = category_of(rec)
        toks = pattern_tokens(rec)
        best, best_j = None, 0.0
        for c in clusters:
            if c["category"] != cat:
                continue
            j = jaccard(toks, c["tokens"])
            if j > best_j:
                best, best_j = c, j
        if best is not None and best_j >= threshold:
            best["members"].append(rec)
            best["tokens"] |= toks
        else:
            clusters.append({"category": cat, "tokens": set(toks), "members": [rec]})
    for c in clusters:
        labels = Counter(str((m.get("stage2") or {}).get("pivot_pattern") or "").strip()
                         for m in c["members"])
        labels.pop("", None)
        c["label"] = labels.most_common(1)[0][0] if labels else clip(norm_ws(str(
            (c["members"][0].get("stage2") or {}).get("category_note") or "(no label)")), 80)
        c["n"] = len(c["members"])
        c["harnesses"] = Counter(m["harness"] for m in c["members"])
        c["envs"] = Counter(m["env_group"] for m in c["members"])
        c["tokens"] = sorted(c["tokens"])
    clusters.sort(key=lambda c: (-c["n"], c["label"]))
    return clusters


def det_agreement(recs: list[dict]) -> dict:
    """How the judged primary pivot relates to the deterministic labels of
    the same rollout (computed in krlib.parse_rollout with the
    king-loop-labels definitions)."""
    out = Counter()
    multi = [r for r in recs if (r.get("n_turns") or 0) > 1]
    for r in multi:
        p, src = primary_pivot(r)
        if p is None:
            out["no_pivot"] += 1
            continue
        t = p["turn"]
        d = r["det"]
        onsets = d["loop_onsets"]
        out["judged"] += 1
        if t in onsets:
            out["pivot_is_loop_onset"] += 1
        if any(abs(t - o) <= 1 for o in onsets):
            out["pivot_within_1_of_onset"] += 1
        if t in d["in_loop"]:
            out["pivot_in_loop"] += 1
        if t in d["escape"]:
            out["pivot_is_escape"] += 1
        if t in d["no_action"]:
            out["pivot_is_no_action"] += 1
        if t in d["completion"]:
            out["pivot_is_completion"] += 1
        if t == d["last_turn"]:
            out["pivot_is_last_turn"] += 1
        if onsets:
            out["rollouts_with_onset"] += 1
            first = onsets[0]
            if t < first:
                out["pivot_before_first_onset"] += 1
            elif t == first:
                out["pivot_at_first_onset"] += 1
            else:
                out["pivot_after_first_onset"] += 1
            if any(o <= t <= o + 2 for o in onsets):
                out["pivot_at_or_within_2_after_onset"] += 1
        else:
            out["rollouts_without_onset"] += 1
        if not onsets and not d["no_action"]:
            out["rollouts_no_det_label"] += 1
    # any pivot (not just primary) hitting an onset
    for r in multi:
        onsets = set(r["det"]["loop_onsets"])
        if onsets and any(p["turn"] in onsets for p, _ in all_pivots(r)):
            out["rollouts_any_pivot_is_onset"] += 1
    out["n_multi_turn"] = len(multi)
    return dict(out)


def depth_stats(recs: list[dict]) -> dict:
    abs_d, rel_d, first_useless = [], [], []
    for r in recs:
        if (r.get("n_turns") or 0) <= 1:
            continue
        p, _ = primary_pivot(r)
        if p is None:
            continue
        abs_d.append(p["turn"])
        rel_d.append(p["turn"] / max(1, r["n_turns"] - 1))
        fu = (r.get("stage1") or {}).get("first_useless_turn")
        if isinstance(fu, int):
            first_useless.append(fu / max(1, r["n_turns"] - 1))

    def q(xs, p):
        if not xs:
            return None
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))]
    buckets = Counter()
    for x in rel_d:
        buckets["0-20 %" if x < .2 else "20-40 %" if x < .4 else "40-60 %" if x < .6
                else "60-80 %" if x < .8 else "80-100 %"] += 1
    return {"n": len(abs_d), "abs_p10": q(abs_d, .1), "abs_p50": q(abs_d, .5),
            "abs_p90": q(abs_d, .9), "rel_p50": round(q(rel_d, .5), 2) if rel_d else None,
            "rel_buckets": dict(buckets),
            "first_useless_rel_p50": round(q(first_useless, .5), 2) if first_useless else None,
            "mean_turns": round(statistics.mean(r["n_turns"] for r in recs), 1) if recs else None}


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def cat_table(recs: list[dict], key: str) -> str:
    groups = sorted({r[key] for r in recs})
    cats = [c for c, _ in Counter(category_of(r) for r in recs).most_common()]
    rows = []
    for g in groups:
        sub = [r for r in recs if r[key] == g]
        cnt = Counter(category_of(r) for r in sub)
        rows.append([g, len(sub)] + [f"{cnt[c]} ({pct(cnt[c], len(sub))})" if cnt[c] else "" for c in cats])
    total = Counter(category_of(r) for r in recs)
    rows.append(["**all**", len(recs)] + [f"**{total[c]} ({pct(total[c], len(recs))})**" for c in cats])
    return table([key, "n"] + cats, rows)


def example_block(rec: dict, ts: TraceStore, env_groups: dict[str, str],
                  item_by_id: dict[str, dict]) -> str:
    p, src = primary_pivot(rec)
    s2 = rec.get("stage2") or {}
    lines = [f"- **{rec['harness']} / {rec['source']}**, task `{rec['sid']}`, "
             f"{rec['n_turns']} turns, stop `{rec['stop_condition']}`, rollout `{rec['rollout_id'][:12]}`"]
    if p is not None:
        lines.append(f"  - pivot: **turn {p['turn']}** (conf {p.get('confidence', '?')}, {src}) -- "
                     f"{norm_ws(str(p.get('rationale', '')))}")
        if p.get("should_have"):
            lines.append(f"  - should have: {norm_ws(str(p['should_have']))}")
        item = item_by_id.get(rec["rollout_id"])
        if item is not None:
            try:
                ro = parse_rollout(ts.load_envelope(item["chunk"], item["line"]), env_groups)
                t = ro.turns[p["turn"]]
                act = short_action(t, 300).replace("\n", " ⏎ ")
                lines.append(f"  - action at the pivot: `{clip(act, 300)}`  "
                             f"(observation: {t.obs_kind}; det label: {', '.join(ro.det_labels(p['turn']))})")
            except (KeyError, IndexError, OSError):
                pass
    if s2.get("category_note"):
        lines.append(f"  - judge: {norm_ws(str(s2['category_note']))}")
    if s2.get("what_teacher_did_differently"):
        lines.append(f"  - teacher: {clip(norm_ws(str(s2['what_teacher_did_differently'])), 500)}")
    rp = s2.get("recoverable_from_pivot") or {}
    if isinstance(rp, dict) and rp:
        lines.append(f"  - recoverable from the pivot: {rp.get('estimate')} "
                     f"(conf {rp.get('confidence', '?')}) -- {norm_ws(str(rp.get('why', '')))}")
    return "\n".join(lines)


def build_report(*, recs: list[dict], sample: list[dict], cells: list[dict], cost: dict,
                 king: str, reign: str | None, ts: TraceStore, env_groups: dict[str, str],
                 n_examples: int = 2, top_patterns: int = 15) -> tuple[str, dict]:
    item_by_id = {s["rollout_id"]: s for s in sample}
    model = recs[0]["judge_model"] if recs else "?"
    agent = [r for r in recs if (r.get("n_turns") or 0) > 1]
    single = [r for r in recs if (r.get("n_turns") or 0) <= 1]
    cats = Counter(category_of(r) for r in recs)
    clusters = cluster_patterns(agent)
    agree = det_agreement(recs)
    depth = depth_stats(recs)
    total_cost = sum(r["usage"]["cost_usd"] for r in recs)
    recov = defaultdict(list)
    for r in recs:
        rp = (r.get("stage2") or {}).get("recoverable_from_pivot")
        if isinstance(rp, dict) and isinstance(rp.get("estimate"), bool):
            recov[category_of(r)].append((rp["estimate"], float(rp.get("confidence") or 0.5)))
    teacher_n = sum(1 for r in recs if r.get("teacher"))
    teacher_solved = sum(1 for r in recs if (r.get("teacher") or {}).get("outcome") == "solved")
    believed_done = sum(1 for r in agent if (r.get("stage1") or {}).get("agent_believed_done") is True)
    looks_solved = Counter(str((r.get("stage1") or {}).get("task_looks_solved_to_you")) for r in recs)

    md: list[str] = []
    md.append(f"# King review -- `king-{king}`" + (f" (reign {reign})" if reign else ""))
    md.append("")
    md.append(f"Judge: `{model}` via OpenRouter, prompt `{PROMPT_VERSION}`, two stages per rollout "
              f"(pivot first, blind to the grade; then category with the grade and the teacher's "
              f"rollout revealed). {len(recs)} of {len(sample)} sampled failed king rollouts judged "
              f"({len(agent)} multi-turn agent rollouts, {len(single)} single-reply). "
              f"{teacher_n} had a teacher rollout on the same task ({teacher_solved} solved by the teacher). "
              f"Cost of the judgments in this report: **${total_cost:.2f}** "
              f"({sum(r['usage']['stage1'].get('prompt_tokens', 0) + r['usage']['stage2'].get('prompt_tokens', 0) for r in recs):,} prompt tokens, "
              f"{sum(r['usage']['stage1'].get('completion_tokens', 0) + r['usage']['stage2'].get('completion_tokens', 0) for r in recs):,} completion tokens).")
    md.append("")
    md.append("## Terms")
    md.append("")
    md.append("- **Rollout**: one full agent run on one task. **Turn**: one reply of the agent inside it (0-based index). "
              "**Observation**: what the harness returned after the turn's action.")
    md.append("- **Pivotal turn (pivot)**: the judge's pick for the turn where the run could still have been saved by a "
              "different decision. Chosen BEFORE the judge saw the grade. **Confidence** is the judge's own 0-1 estimate.")
    md.append("- **Failure category**: one label per rollout from a fixed list (definitions in the table below).")
    md.append("- **Deterministic labels**: string rules over the trace, no LLM -- `loop_onset` (first turn that re-issues an "
              "earlier action and gets the same observation), `in_loop`, `escape`, `no_action` (no parseable action), "
              "`completion` (last reply of a rollout the agent ended itself).")
    md.append("- **Recoverable from pivot**: the judge's estimate whether a strong model, continuing from the state just "
              "before the pivot, could still have solved the task -- the 'teacher-recoverable' question of the data plan.")
    md.append("")
    md.append("## Sample")
    md.append("")
    md.append(table(["env group", "harness", "failed available", "sampled", "judged", "with teacher rollout"],
                    [[c["env_group"], c["harness"], c["failed_available"], c["sampled"],
                      sum(1 for r in recs if r["env_group"] == c["env_group"] and r["harness"] == c["harness"]),
                      c["with_teacher"]] for c in cells]))
    md.append("")
    md.append("Stratified: up to N per (env group, harness) cell, deterministic seed, tasks the teacher also played "
              "preferred inside a cell. Errored rollouts (harness / API failures) excluded, as in the fold.")
    md.append("")
    md.append("## 1. Failure categories")
    md.append("")
    md.append(table(["category", "meaning"], [[f"`{k}`", v] for k, v in CATEGORIES.items()]))
    md.append("")
    md.append("### Overall")
    md.append("")
    md.append(table(["category", "rollouts", "share"],
                    [[f"`{c}`", n, pct(n, len(recs))] for c, n in cats.most_common()]))
    md.append("")
    md.append("### By env group")
    md.append("")
    md.append(cat_table(recs, "env_group"))
    md.append("")
    md.append("### By harness")
    md.append("")
    md.append(cat_table(recs, "harness"))
    md.append("")
    md.append(f"Judge's blind read (before the grade): the agent believed it was done in "
              f"{believed_done} of {len(agent)} multi-turn rollouts ({pct(believed_done, len(agent))}); "
              f"the task looked solved to the judge in {looks_solved.get('True', 0)} of {len(recs)} "
              f"(unsure {looks_solved.get('unsure', 0)}).")
    md.append("")
    md.append("## 2. Where the pivot sits")
    md.append("")
    md.append(f"Multi-turn rollouts with a pivot: {depth['n']}. Mean length {depth['mean_turns']} turns. "
              f"Primary pivot index p10 / p50 / p90 = **{depth['abs_p10']} / {depth['abs_p50']} / {depth['abs_p90']}**; "
              f"relative position (pivot / last turn) median **{depth['rel_p50']}**; "
              f"judge's `first_useless_turn` median relative position {depth['first_useless_rel_p50']}.")
    md.append("")
    md.append(table(["relative position of the pivot", "rollouts"],
                    [[k, depth["rel_buckets"].get(k, 0)] for k in ("0-20 %", "20-40 %", "40-60 %", "60-80 %", "80-100 %")]))
    md.append("")
    md.append(f"## 3. Top {min(top_patterns, len(clusters))} recurring pivot patterns")
    md.append("")
    md.append("Clustered on the judge's generic pattern label plus the head of the repeated action "
              "(greedy Jaccard >= 0.4 inside one category). One example per cluster; the quote is the "
              "judge's rationale for the pivot.")
    md.append("")
    for i, c in enumerate(clusters[:top_patterns], 1):
        md.append(f"### 3.{i} {c['label']}  --  {c['n']} rollouts, `{c['category']}`")
        md.append("")
        md.append(f"Harnesses: {', '.join(f'{k} {v}' for k, v in c['harnesses'].most_common())}. "
                  f"Env: {', '.join(f'{k} {v}' for k, v in c['envs'].most_common())}.")
        variants = Counter(str((m.get("stage2") or {}).get("pivot_pattern") or "") for m in c["members"])
        if len(variants) > 1:
            md.append("Labels in the cluster: " + "; ".join(f"_{k}_ ({v})" for k, v in variants.most_common(5)))
        md.append("")
        for m in c["members"][:n_examples]:
            md.append(example_block(m, ts, env_groups, item_by_id))
        md.append("")
    md.append("## 4. Agreement with the deterministic labels")
    md.append("")
    n = agree.get("judged", 0)
    rows = [
        ["multi-turn rollouts judged", agree.get("n_multi_turn", 0), ""],
        ["... with a usable primary pivot", n, ""],
        ["primary pivot IS a `loop_onset` turn", agree.get("pivot_is_loop_onset", 0), pct(agree.get("pivot_is_loop_onset", 0), n)],
        ["primary pivot within +-1 turn of a `loop_onset`", agree.get("pivot_within_1_of_onset", 0), pct(agree.get("pivot_within_1_of_onset", 0), n)],
        ["primary pivot is an `in_loop` turn", agree.get("pivot_in_loop", 0), pct(agree.get("pivot_in_loop", 0), n)],
        ["primary pivot is an `escape` turn", agree.get("pivot_is_escape", 0), pct(agree.get("pivot_is_escape", 0), n)],
        ["primary pivot is a `no_action` turn", agree.get("pivot_is_no_action", 0), pct(agree.get("pivot_is_no_action", 0), n)],
        ["primary pivot is the `completion` turn", agree.get("pivot_is_completion", 0), pct(agree.get("pivot_is_completion", 0), n)],
        ["primary pivot is the last turn", agree.get("pivot_is_last_turn", 0), pct(agree.get("pivot_is_last_turn", 0), n)],
        ["rollouts with >= 1 `loop_onset`", agree.get("rollouts_with_onset", 0), pct(agree.get("rollouts_with_onset", 0), n)],
        ["... pivot BEFORE the first onset", agree.get("pivot_before_first_onset", 0), pct(agree.get("pivot_before_first_onset", 0), agree.get("rollouts_with_onset", 0))],
        ["... pivot AT the first onset", agree.get("pivot_at_first_onset", 0), pct(agree.get("pivot_at_first_onset", 0), agree.get("rollouts_with_onset", 0))],
        ["... pivot AFTER the first onset", agree.get("pivot_after_first_onset", 0), pct(agree.get("pivot_after_first_onset", 0), agree.get("rollouts_with_onset", 0))],
        ["... pivot at an onset or within 2 turns after one", agree.get("pivot_at_or_within_2_after_onset", 0), pct(agree.get("pivot_at_or_within_2_after_onset", 0), agree.get("rollouts_with_onset", 0))],
        ["... ANY judged pivot (1-3) is an onset", agree.get("rollouts_any_pivot_is_onset", 0), pct(agree.get("rollouts_any_pivot_is_onset", 0), agree.get("rollouts_with_onset", 0))],
        ["rollouts with no deterministic label at all (no loop, no no_action)", agree.get("rollouts_no_det_label", 0), pct(agree.get("rollouts_no_det_label", 0), n)],
    ]
    md.append(table(["statistic", "count", "share"], rows))
    md.append("")
    md.append("Reading: the deterministic labeler marks the wreckage (the repeat); the judge is asked for the "
              "decision point. When the pivot lands BEFORE the first onset, the judged turn is new material the "
              "loop labeler cannot see. When it lands AT the onset, the two agree. AFTER means the judge blamed "
              "a turn inside or after the loop.")
    md.append("")
    md.append("## 5. Recoverable from the pivot?")
    md.append("")
    md.append(table(["category", "judged", "recoverable = true", "mean confidence"],
                    [[f"`{c}`", len(v), pct(sum(1 for e, _ in v if e), len(v)),
                      f"{statistics.mean(cf for _, cf in v):.2f}"] for c, v in
                     sorted(recov.items(), key=lambda kv: -len(kv[1]))]))
    md.append("")
    md.append("## 6. What the teacher did differently (judge quotes)")
    md.append("")
    for c, _ in cats.most_common(5):
        quotes = [norm_ws(str((r.get("stage2") or {}).get("what_teacher_did_differently")))
                  for r in recs if category_of(r) == c
                  and (r.get("stage2") or {}).get("what_teacher_did_differently")]
        if not quotes:
            continue
        md.append(f"**`{c}`** ({len(quotes)} with a teacher rollout):")
        md.append("")
        for q in quotes[:3]:
            md.append(f"- {clip(q, 450)}")
        md.append("")
    md.append("## 7. What D should contain")
    md.append("")
    md.append("Each frequent category mapped to a data intervention. Scoring, `[duel]`, the teacher and "
              "`weight_version_key` stay untouched; every row is a fold-routing or datagen change.")
    md.append("")
    md.append(table(["category", "rollouts", "top pattern", "what D should contain"],
                    [[f"`{c}`", n, next((cl["label"] for cl in clusters if cl["category"] == c), "-"),
                      INTERVENTIONS.get(c, "")] for c, n in cats.most_common() if n]))
    md.append("")
    md.append("The judged pivotal turns themselves are the side-table `king_pivots/<digest>.jsonl` "
              "(labels_out.py): each row is one (rollout, turn) the fold can route into a `king_pivot` group "
              "at confidence >= 0.7 with a leak-rule exemption, like `king_loop_onset`.")
    md.append("")
    md.append("## 8. Limits")
    md.append("")
    md.append("- One judge, one pass, temperature 0: no inter-judge agreement measured. Confidence is self-reported.")
    md.append("- Observations are truncated (900 head + 300 tail chars) and repeated stretches collapsed; the judge "
              "sees turn indices exactly but not every byte. Rollouts over the transcript cap have their middle elided "
              "(flagged per judgment in `transcript_stats.elided_middle`).")
    md.append("- The teacher rollout is shown compressed (action sequence + outcome), and only in stage 2, after the pivot.")
    md.append("- Single-reply rollouts (math) have a trivial pivot (turn 0); the value is the rationale, not the index.")
    md.append("- Sampling prefers tasks the teacher also played, so cells with teacher coverage are over-represented "
              "relative to the raw failure pool.")
    summary = {
        "king": king, "reign": reign, "judge_model": model, "prompt_version": PROMPT_VERSION,
        "n_sampled": len(sample), "n_judged": len(recs), "n_multi_turn": len(agent),
        "n_with_teacher": teacher_n, "cost_usd": round(total_cost, 4),
        "categories": dict(cats), "depth": depth, "agreement": agree,
        "recoverable": {c: {"n": len(v), "true": sum(1 for e, _ in v if e)} for c, v in recov.items()},
        "top_patterns": [{"label": c["label"], "category": c["category"], "n": c["n"],
                          "harnesses": dict(c["harnesses"]),
                          "rollouts": [m["rollout_id"] for m in c["members"]]}
                         for c in clusters[:top_patterns]],
        "cost_run": cost,
    }
    return "\n".join(md), summary


def load_judgments(out_dir: Path, sample: list[dict]) -> list[dict]:
    ids = {s["rollout_id"] for s in sample}
    seen: dict[str, dict] = {}
    for r in read_jsonl(out_dir / "cache" / "judgments.jsonl"):
        if r["rollout_id"] in ids and r.get("prompt_version") == PROMPT_VERSION:
            seen[r["rollout_id"]] = r  # last write wins
    return [seen[i] for i in sorted(seen)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--sample", type=Path, default=None)
    ap.add_argument("--reign", default=None)
    ap.add_argument("--examples", type=int, default=2)
    args = ap.parse_args()
    sample_path = args.sample or (args.out_dir / "sample.jsonl")
    sample = read_jsonl(sample_path)
    cells = json.loads((sample_path.parent / "sample_cells.json").read_text()) \
        if (sample_path.parent / "sample_cells.json").exists() else []
    cost = json.loads((args.out_dir / "cost.json").read_text()) \
        if (args.out_dir / "cost.json").exists() else {}
    recs = load_judgments(args.out_dir, sample)
    if not recs:
        raise SystemExit("no judgments for this sample yet")
    king = sample[0]["king"].replace("king-", "")
    md, summary = build_report(recs=recs, sample=sample, cells=cells, cost=cost, king=king,
                               reign=args.reign, ts=TraceStore(), env_groups=load_env_groups(),
                               n_examples=args.examples)
    (args.out_dir / "report.md").write_text(md)
    (args.out_dir / "report.json").write_text(json.dumps(summary, indent=1))
    print(f"report: {args.out_dir / 'report.md'} ({len(recs)} judgments, "
          f"${summary['cost_usd']:.2f})")


if __name__ == "__main__":
    main()
