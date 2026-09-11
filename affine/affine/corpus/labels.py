"""Deterministic failure labels for rollout traces (no LLM judge).

Input: one trace envelope (the R2 `traces/chunks/*.jsonl.gz` line shape,
rollouts/schema.py). Output: one JSON-serializable record per rollout with
rollout-level labels and a `turns` list of per-reply labels. Every label is
a string rule over the stored trace, so the same envelope always yields the
same record (`labels_version` bumps when a rule changes).

Vocabulary
  turn         one sampled assistant reply; `depth` = its 0-based index
  action       the command the reply issues in the policy's dialect
               (```bash block, native tool call(s), Terminus `commands`
               keystrokes, \\boxed{}); the final prose reply of a tool loop
               counts as a `text` action (affine.dialects TEXT_KIND)
  observation  everything the harness sent back before the next reply
               (tool results / user messages on the next reply's path)
  obs_kind     ok | error | empty | nudge (harness format complaint) | none

Rollout labels
  outcome            affine.corpus.view.rollout_outcome: solved / failed /
                     errored / unscored
  stop_condition     raw harness value (agent_completed, max_turns, error,
                     or mini-swe's Submitted / LimitsExceeded / ...)
  stop_kind          agent_completed | max_turns | timeout | loop_guard |
                     error | length_cap (agent_completed whose final reply
                     hit the token cap: the harness took a truncated reply
                     as final)
  n_turns            sampled replies (0 when the harness died before one)
  n_loops / n_loops_norm          onset..escape stretches, strict / normalized
  n_loop_turns / n_loop_turns_norm   onset + in_loop turns
  loop_lengths       strict view, one entry per loop (onset + in_loop turns)
  ends_in_loop / ends_in_loop_norm   the last labelled stretch never escaped
  no_action_count / no_action_rate   replies with no parseable action
  cap_hit_count      replies whose model call finished with `length`
  format_error_count replies the harness rejected (see turn `format_error`)
  nudge_count        observations of kind `nudge`
  completion_turn_idx / completion_turn_id   the reply that ended a
                     deliberately completed rollout (per-harness rule
                     below), None otherwise
  completion_kind    submit | text | finish_tool | task_complete | boxed
  completion_visible_chars  visible (non-reasoning) chars of that reply —
                     0 = the model "finished" with reasoning only
  premature_finish   completed, `outcome == failed`, at least one
                     observation seen, and none of the three observations
                     before the completion turn was an error (a one-reply
                     math rollout with a wrong box is a wrong answer, not a
                     premature finish)
  format_exit        stop_kind agent_completed with NO completion turn and
                     the last visible observation a `nudge`: mini-swe's
                     RepeatedFormatError exit (three rejected replies), which
                     verifiers records as a plain agent_completed
  harness / policy / model / action_kind / source / env_id / env_category
  king_digest        `king-<digest12>` parsed from the policy model tag;
                     None for non-king policies
  reign              from the caller's digest -> reign map, else None (the
                     trace does not carry it)

Turn labels
  loop               normal | loop_onset | in_loop | escape — STRICT view:
                     onset = the action is byte-identical to an earlier
                     action of the rollout AND its observation equals (first
                     200 normalized chars) the one that earlier action got;
                     in_loop = following turns until action AND observation
                     both change; escape = the first turn where both changed
  loop_norm          the same machine on normalized actions (lower-case,
                     whitespace collapsed, digits -> 0, hex runs -> H,
                     directory prefixes collapsed) — the view the king-loop
                     report used; merges `TaskUpdate #1/#2/#3` sequences
  repeat_exact / rep_exact_count / rep_exact_dist   earlier identical action
  repeat_norm / rep_norm_count / rep_norm_dist      earlier normalized match
  repeat_same_obs    the normalized repeat also got the same observation
  no_action          no single parseable action (tool_calls count for tool
                     harnesses); `n_actions` is the raw count (0 or 2+)
  cap_hit            finish_reason == length
  format_error       the harness rejected the reply: the next observation
                     is a `nudge`, OR the dialect parser found no action but
                     the reply carries an action-looking span (an open or
                     mislabelled fence, `<tool_call>`, `\\boxed{`, a JSON
                     object). A ```bash fence under mini-swe parses in the
                     dialect but the harness wants ```mswea_bash_command; the
                     nudge is what catches it, `fence_lang` says which fence
  nudged             the next observation is a nudge
  fence_lang         first ``` fence language of a bash-dialect reply ("" if
                     none)
  obs_kind / prev_obs_kind
  after_error_obs    previous observation error / empty / nudge
  after_ok_obs       previous observation ok
  recovery / persist previous observation was bad and the action changed /
                     repeated (normalized view)
  finish_turn        this reply is the rollout's completion turn (any
                     outcome)
  completion         finish_turn AND outcome == solved — the "I am done"
                     state of a successful rollout
  depth, prefix_chars (plain text of the prefix path: content + tool-call
  JSON, +-1 % vs the baked view), thought_chars (reasoning + visible text
  before the action), reasoning_chars, visible_chars, action_chars,
  action_kind (effective: the policy dialect, or `text` for the final prose
  reply of a completed tool loop), action_norm (normalized, 200 chars)

Completion rule per harness (the reply that ends a completed rollout;
applies only when stop_kind == agent_completed and only to the FINAL reply)
  mini_swe_textbased (bash)   the single command is `submit` or contains
                              COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT   -> submit
                              else a prose reply with no action     -> text
  bash / pi / claude_code / kimi_code / hermes_agent / null tool loops
  (tool_call)                 a tool call named like finish / submit /
                              done / attempt_completion / final_answer
                                                                     -> finish_tool
                              else the final reply with NO tool call  -> text
                              (visible_chars may be 0: reasoning-only)
  terminus_2 (terminus_json)  the JSON batch carries task_complete: true
                                                                     -> task_complete
  null math (boxed)           the reply contains \\boxed{...}         -> boxed
  text dialect                the final visible reply                -> text
A final reply that matches no rule leaves completion_turn_idx None: mini-swe
ended on RepeatedFormatError (format_exit), a Terminus batch without
task_complete, a boxed reply with no box.

Lives next to view.py so the fold and the dashboard import one rule set;
stdlib + affine.dialects + affine.corpus.trace/view only. No GPU, no
tokenizer: prefix size is chars.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from affine import dialects
from affine.corpus.trace import message_text, real_errors
from affine.corpus.view import rollout_outcome

LABELS_VERSION = 1

OBS_HEAD = 200
ACTION_NORM_CHARS = 200
PREMATURE_LOOKBACK = 3
BAD_OBS = frozenset({"error", "empty", "nudge"})
LOOP_TURN = frozenset({"loop_onset", "in_loop"})

WS_RE = re.compile(r"\s+")
NUM_RE = re.compile(r"\d+")
HEX_RE = re.compile(r"\b[0-9a-f]{8,}\b")
PATH_RE = re.compile(r"(?:/[\w.\-@+]+){2,}/")
FENCE_RE = re.compile(r"```([\w\-]*)[ \t]*\n")
FOREIGN_FENCE_RE = re.compile(r"```mswea_bash_command[ \t]*\n")
KING_DIGEST_RE = re.compile(r"king-([0-9a-f]{6,})")
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
FINISH_TOOL_RE = re.compile(
    r"^(?:submit|finish|done|complete|task_complete|complete_task|"
    r"attempt_completion|final_answer|submit_answer|finish_task)$", re.IGNORECASE)
TIMEOUT_ERR_RE = re.compile(r"timeout|timed out|exceeded its .* budget", re.IGNORECASE)
ERROR_RE = re.compile(
    r"traceback|error|exception|no such file|not found|command not found|"
    r"permission denied|<returncode>[1-9]|exit code [1-9]|exit status [1-9]|"
    r"failed|fatal:|cannot |syntax|wasted call|is not a terminal|timed out|"
    r"killed|segmentation fault", re.IGNORECASE)
# Harness format complaints are templates, so they are matched as such:
# anchored at the start of the observation (mini-swe "Format error:",
# Terminus "Previous response had parsing errors", Claude Code's proxy
# "Wasted call", verifiers "error: unknown tool") or as one of a few whole
# phrases. Bare words ("malformed", "invalid json", "exactly one") were
# matching tool output and system reminders in 30-60 % of tool-harness
# cases and are deliberately out.
NUDGE_HEAD_RE = re.compile(
    r"^(?:format error|previous response had parsing errors|wasted call|"
    r"error: unknown tool|unknown tool)", re.IGNORECASE)
NUDGE_RE = re.compile(
    r"expected exactly \d+ action|please always provide exactly one|"
    r"did not include an? (?:action|tool call|code block)|your response must|"
    r"no valid json object found|missing required fields:|"
    r"parser warnings from your truncated response|"
    r"refer to that earlier tool_result|"
    r"<system-reminder>[^<]{0,200}?same tool call has now been issued", re.IGNORECASE)
EMPTY_OBS_RE = re.compile(
    r"^(?:<returncode>\d+</returncode>\s*)?<output>\s*</output>$|"
    r"^\(?(?:no output|empty)\)?$|^current terminal screen:\s*$", re.IGNORECASE)
ACTION_LIKE_RE = re.compile(r"```|<tool_call>|\\boxed\{|\{\s*\"", re.DOTALL)

STOP_AGENT = frozenset({"agent_completed", "Submitted", "RepeatedFormatError"})
STOP_TURN_CAP = frozenset({"max_turns", "LimitsExceeded"})
STOP_TIMEOUT = frozenset({"TimeExceeded"})


# -- text helpers --------------------------------------------------------------

def norm_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()


def norm_action(s: str) -> str:
    s = s.lower()
    s = PATH_RE.sub("/", s)
    s = HEX_RE.sub("H", s)
    s = NUM_RE.sub("0", s)
    return norm_ws(s)


def canon_args(args) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return args.strip()
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(args)


def tool_call_action(tool_calls: list[dict]) -> str:
    parts = []
    for c in tool_calls:
        fn = c.get("function") or c
        parts.append(f"{fn.get('name')}({canon_args(fn.get('arguments'))})")
    return "\n".join(parts)


def tool_call_names(tool_calls: list[dict]) -> list[str]:
    return [str((c.get("function") or c).get("name") or "") for c in tool_calls]


def terminus_parse(action_json: str) -> dict | None:
    try:
        obj = json.loads(action_json)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def terminus_commands(action_json: str) -> str:
    """Repeat key for a Terminus batch: the keystrokes; an empty batch is
    still an action (wait / done)."""
    obj = terminus_parse(action_json)
    if obj is None:
        return action_json
    cmds = obj.get("commands") or []
    keys = "\n".join(str(c.get("keystrokes", c)) if isinstance(c, dict)
                     else str(c) for c in cmds).strip()
    if keys:
        return keys
    return "<task_complete>" if obj.get("task_complete") else "<wait>"


def bash_body(block: str) -> str:
    """Command text inside one closed ```bash block."""
    inner = block.split("\n", 1)[1] if "\n" in block else ""
    return inner.rsplit("```", 1)[0].strip()


def is_submit_command(body: str) -> bool:
    return body.strip() == "submit" or SUBMIT_MARKER in body


def obs_kind(obs: str | None) -> str:
    if obs is None:
        return "none"
    s = obs.strip()
    if not s or EMPTY_OBS_RE.match(norm_ws(s)):
        return "empty"
    head = s[:1500]
    if NUDGE_HEAD_RE.match(head) or NUDGE_RE.search(head):
        return "nudge"
    if ERROR_RE.search(head):
        return "error"
    return "ok"


def node_plain(m: dict) -> str:
    """Approximate plain text the teacher template renders for a message."""
    text = message_text(m.get("content"))
    if m.get("tool_calls"):
        text += "\n" + tool_call_action(m["tool_calls"])
    return text


def reply_chains(nodes: list[dict]) -> list[list[int]]:
    """Root->node id chain for every sampled assistant node (the node-id
    form of affine.corpus.trace.sampled_paths)."""
    if not nodes:
        return []
    linear = "parent" not in nodes[-1]
    out: list[list[int]] = []
    for i, nd in enumerate(nodes):
        m = nd.get("message") or {}
        if m.get("role") != "assistant" or not nd.get("sampled"):
            continue
        if linear:
            out.append(list(range(i + 1)))
            continue
        chain: list[int] = []
        j: int | None = i
        while j is not None and 0 <= j < len(nodes) and j not in chain:
            chain.append(j)
            j = nodes[j].get("parent")
        chain.reverse()
        out.append(chain)
    return out


# -- actions --------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    key: str            # what repeat detection compares ("" = no action)
    kind: str           # effective dialect (policy kind, or "text")
    n_actions: int      # raw parser count
    raw: str            # the action span as written
    fence_lang: str     # bash dialect: language of the first fence
    action_like: bool   # the reply looks like it tried to act
    tool_names: tuple[str, ...] = ()
    terminus_done: bool = False


def extract_action(m: dict, kind: str, *, is_final: bool,
                   agent_completed: bool) -> Action:
    content = message_text(m.get("content"))
    fence = FENCE_RE.search(content)
    fence_lang = fence.group(1) if (fence and kind == dialects.DEFAULT_KIND) else ""
    action_like = bool(ACTION_LIKE_RE.search(content))
    # mini-swe's own scaffold fence: the slicer rewrites it the same way.
    content = FOREIGN_FENCE_RE.sub("```bash\n", content)
    if m.get("tool_calls"):
        act = tool_call_action(m["tool_calls"])
        return Action(act, "tool_call", len(m["tool_calls"]), act, fence_lang,
                      True, tuple(tool_call_names(m["tool_calls"])))
    d = dialects.get(kind)
    acts = d.actions(content)
    if len(acts) == 1:
        raw = acts[0]
        if kind == "terminus_json":
            obj = terminus_parse(raw) or {}
            return Action(terminus_commands(raw), kind, 1, raw, fence_lang, True,
                          terminus_done=bool(obj.get("task_complete")))
        return Action(raw, kind, 1, raw, fence_lang, True)
    if (not acts and is_final and agent_completed and d.ends_in_text
            and content.strip()):
        stripped = content.strip()
        return Action(stripped, dialects.TEXT_KIND, 1, stripped, fence_lang, action_like)
    return Action("", kind, len(acts), "", fence_lang, action_like)


def completion_kind(kind: str, act: Action, m: dict) -> str | None:
    """Which completion rule the FINAL reply of an agent_completed rollout
    satisfies (module docstring), or None."""
    if act.kind == dialects.TEXT_KIND:
        return "text"
    if m.get("tool_calls"):
        if any(FINISH_TOOL_RE.match(n) for n in act.tool_names):
            return "finish_tool"
        return None
    if kind == "terminus_json":
        return "task_complete" if act.terminus_done else None
    if kind == "boxed":
        return "boxed" if act.n_actions >= 1 else None
    if kind == dialects.DEFAULT_KIND:
        if act.n_actions == 1 and is_submit_command(bash_body(act.raw)):
            return "submit"
        return None
    if kind == "tool_call" and not act.key:
        # A tool loop that stopped on a reply with neither a tool call nor
        # visible text: the model ended with reasoning only.
        return "text"
    if kind == dialects.TEXT_KIND and act.n_actions >= 1:
        return "text"
    return None


# -- loop machine ---------------------------------------------------------------

@dataclass
class _LoopMachine:
    """label_loops.py's onset / in_loop / escape automaton over one repeat
    key (exact or normalized action)."""
    seen: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    state: str = "normal"
    prev_key: str | None = None
    prev_ohead: str | None = None
    n_loops: int = 0
    lengths: list[int] = field(default_factory=list)
    cur: int = 0

    def step(self, key: str, k: int, ohead: str | None,
             obs_heads: list[str | None]) -> tuple[str, int, int | None, bool]:
        """-> (loop label, earlier occurrences, turns since last, same obs)."""
        hist = self.seen[key]
        rep_count = len(hist)
        rep_dist = k - hist[-1] if hist else None
        same_obs = bool(hist) and ohead is not None and ohead == obs_heads[hist[-1]]
        label = "normal"
        if hist and same_obs:
            if self.state == "normal":
                label = "loop_onset"
                self.state = "loop"
                self.n_loops += 1
                self.cur = 1
            else:
                label = "in_loop"
                self.cur += 1
        elif self.state == "loop":
            changed_act = self.prev_key is not None and key != self.prev_key
            changed_obs = (self.prev_key is not None and ohead is not None
                           and ohead != self.prev_ohead)
            if changed_act and changed_obs:
                label = "escape"
                self.state = "normal"
                self.lengths.append(self.cur)
                self.cur = 0
            elif changed_act and ohead is None:
                label = "normal"      # final turn: escape is undecidable
            else:
                label = "in_loop"
                self.cur += 1
        hist.append(k)
        self.prev_key = key
        self.prev_ohead = ohead
        return label, rep_count, rep_dist, same_obs

    def close(self) -> None:
        if self.cur:
            self.lengths.append(self.cur)

    @property
    def ends_in_loop(self) -> bool:
        return self.state == "loop"


# -- per-rollout labelling -------------------------------------------------------

def _observations(nodes: list[dict], chains: list[list[int]],
                  plain) -> list[str | None]:
    """Observation after reply k: the non-assistant messages of chain k+1
    past its common prefix with chain k. None for the last reply."""
    obs: list[str | None] = []
    for k, chain in enumerate(chains):
        if k + 1 >= len(chains):
            obs.append(None)
            continue
        nxt = chains[k + 1]
        n = 0
        while n < len(chain) and n < len(nxt) and chain[n] == nxt[n]:
            n += 1
        parts = [plain(i) for i in nxt[n:-1]
                 if (nodes[i].get("message") or {}).get("role") in ("user", "tool")]
        obs.append("\n".join(parts))
    return obs


def label_turns(nodes: list[dict], chains: list[list[int]], *, kind: str,
                finish: dict[int, str | None], agent_completed: bool
                ) -> tuple[list[dict], _LoopMachine, _LoopMachine, list[Action]]:
    n_rep = len(chains)
    plain_cache: dict[int, str] = {}

    def plain(i: int) -> str:
        if i not in plain_cache:
            plain_cache[i] = node_plain(nodes[i].get("message") or {})
        return plain_cache[i]

    obs = _observations(nodes, chains, plain)
    exact = _LoopMachine()
    norm = _LoopMachine()
    obs_heads: list[str | None] = []
    turns: list[dict] = []
    actions: list[Action] = []
    prev_act: tuple[str, str] | None = None   # (norm key, obs_kind) of last action turn
    for k, chain in enumerate(chains):
        node_id = chain[-1]
        m = nodes[node_id].get("message") or {}
        act = extract_action(m, kind, is_final=(k == n_rep - 1),
                             agent_completed=agent_completed)
        actions.append(act)
        content = message_text(m.get("content"))
        reasoning = m.get("reasoning_content") or ""
        visible_before = content
        if act.raw and act.kind != dialects.TEXT_KIND and act.raw in content:
            visible_before = content[:content.index(act.raw)]
        thought = (reasoning + visible_before).strip()
        o = obs[k]
        ohead = norm_ws(o)[:OBS_HEAD] if o is not None else None
        obs_heads.append(ohead)
        okind = obs_kind(o)
        prev_okind = turns[-1]["obs_kind"] if turns else "none"
        row = {
            "turn_idx": k, "node_id": node_id, "depth": k,
            "finish": finish.get(node_id),
            "action_kind": act.kind, "n_actions": act.n_actions,
            "no_action": not act.key, "cap_hit": finish.get(node_id) == "length",
            "nudged": okind == "nudge",
            "format_error": okind == "nudge" or (not act.key and act.action_like),
            "fence_lang": act.fence_lang,
            "loop": "normal", "loop_norm": "normal",
            "repeat_exact": False, "rep_exact_count": 0, "rep_exact_dist": None,
            "repeat_norm": False, "rep_norm_count": 0, "rep_norm_dist": None,
            "repeat_same_obs": False,
            "obs_kind": okind, "prev_obs_kind": prev_okind,
            "after_error_obs": prev_okind in BAD_OBS,
            "after_ok_obs": prev_okind == "ok",
            "recovery": False, "persist": False,
            "finish_turn": False, "completion": False,
            "prefix_chars": sum(len(plain(i)) for i in chain[:-1]),
            "thought_chars": len(thought), "reasoning_chars": len(reasoning),
            "visible_chars": len(content.strip()), "action_chars": len(act.raw),
            "action_norm": "",
        }
        if act.key:
            a_norm = norm_action(act.key)
            row["action_norm"] = a_norm[:ACTION_NORM_CHARS]
            label, cnt, dist, _ = exact.step(act.key.strip(), k, ohead, obs_heads)
            row.update(loop=label, repeat_exact=cnt > 0, rep_exact_count=cnt,
                       rep_exact_dist=dist)
            label, cnt, dist, same = norm.step(a_norm, k, ohead, obs_heads)
            row.update(loop_norm=label, repeat_norm=cnt > 0, rep_norm_count=cnt,
                       rep_norm_dist=dist, repeat_same_obs=same)
            if prev_act is not None and prev_act[1] in BAD_OBS:
                changed = a_norm != prev_act[0]
                row["recovery"] = changed
                row["persist"] = not changed
            prev_act = (a_norm, okind)
        turns.append(row)
    exact.close()
    norm.close()
    return turns, exact, norm, actions


def stop_kind(trace: dict, last_finish: str | None) -> str:
    sc = str(trace.get("stop_condition") or "")
    if sc in STOP_AGENT:
        return "length_cap" if last_finish == "length" else "agent_completed"
    if sc in STOP_TURN_CAP:
        return "max_turns"
    if sc in STOP_TIMEOUT:
        return "timeout"
    if "loop" in sc.lower():
        return "loop_guard"
    msgs = " ".join(str(e.get("message") or e.get("type") or "")
                    for e in real_errors(trace))
    if TIMEOUT_ERR_RE.search(msgs):
        return "timeout"
    return "error"


def king_digest(policy: dict) -> str | None:
    for v in (policy.get("model"), policy.get("id")):
        m = KING_DIGEST_RE.search(str(v or ""))
        if m:
            return m.group(1)
    return None


def source_groups_from_toml(path: str | Path) -> dict[str, str]:
    """source -> fold group (`env_category`) from rollouts' sources.toml."""
    raw = tomllib.loads(Path(path).read_text())
    return {name: str(cfg.get("group", "")) for name, cfg in
            (raw.get("source") or {}).items() if isinstance(cfg, dict)}


def label_envelope(e: dict, *, source_groups: dict[str, str] | None = None,
                   reigns: dict[str, int | str] | None = None) -> dict:
    """Rollout record with `turns` for one envelope. Never raises on a
    trace with no replies: the record then has n_turns 0."""
    t = e.get("trace") or {}
    policy = e.get("policy") or {}
    task = e.get("task") or {}
    nodes = t.get("nodes") or []
    chains = reply_chains(nodes)
    finish = {c["node"]: c.get("finish_reason") for c in (t.get("calls") or [])
              if isinstance(c, dict) and "node" in c}
    kind = policy.get("action_kind") or dialects.DEFAULT_KIND
    harness = policy.get("harness") or ""
    sc = str(t.get("stop_condition") or "")
    completed = sc in STOP_AGENT
    turns, exact, norm, actions = label_turns(
        nodes, chains, kind=kind, finish=finish, agent_completed=completed)
    outcome = rollout_outcome(t) if nodes or t else "errored"
    last_finish = finish.get(chains[-1][-1]) if chains else None
    skind = stop_kind(t, last_finish)

    comp_idx: int | None = None
    comp_kind: str | None = None
    comp_visible = 0
    if turns and skind == "agent_completed":
        last = turns[-1]
        m = nodes[last["node_id"]].get("message") or {}
        visible = message_text(m.get("content"))
        comp_kind = completion_kind(kind, actions[-1], m)
        # A prose reply right after a format nudge is the harness's third
        # strike (mini-swe RepeatedFormatError), not a report.
        if (comp_kind == "text" and len(turns) >= 2
                and turns[-2]["obs_kind"] == "nudge"):
            comp_kind = None
        if comp_kind is not None:
            comp_idx = last["turn_idx"]
            comp_visible = len(visible.strip())
            last["finish_turn"] = True
            last["completion"] = outcome == "solved"

    premature = False
    if comp_idx is not None and comp_idx >= 1 and outcome == "failed":
        window = turns[max(0, comp_idx - PREMATURE_LOOKBACK):comp_idx]
        premature = not any(tr["obs_kind"] == "error" for tr in window)
    format_exit = (skind == "agent_completed" and comp_idx is None
                   and len(turns) >= 2 and turns[-2]["obs_kind"] == "nudge")

    digest = king_digest(policy) if str(policy.get("id", "")).startswith("king_") \
        or policy.get("endpoint") == "king" else None
    n = len(turns)
    n_no_action = sum(tr["no_action"] for tr in turns)
    return {
        "labels_version": LABELS_VERSION,
        "rollout_id": e.get("rollout_id"),
        "source": e.get("source"), "env_id": e.get("env_id"),
        "env_category": (source_groups or {}).get(e.get("source") or "", ""),
        "harness": harness, "policy": policy.get("id", ""),
        "model": policy.get("model", ""), "action_kind": kind,
        "task_sid": task.get("sid"), "task_uid": task.get("uid"),
        "repo": task.get("repo"), "language": task.get("language"),
        "king_digest": digest,
        "reign": (reigns or {}).get(digest) if digest else None,
        "stored_at": e.get("stored_at"),
        "outcome": outcome, "stop_condition": sc or None, "stop_kind": skind,
        "n_turns": n,
        "n_loops": exact.n_loops, "n_loops_norm": norm.n_loops,
        "n_loop_turns": sum(tr["loop"] in LOOP_TURN for tr in turns),
        "n_loop_turns_norm": sum(tr["loop_norm"] in LOOP_TURN for tr in turns),
        "loop_lengths": exact.lengths,
        "ends_in_loop": exact.ends_in_loop, "ends_in_loop_norm": norm.ends_in_loop,
        "no_action_count": n_no_action,
        "no_action_rate": round(n_no_action / n, 4) if n else None,
        "cap_hit_count": sum(tr["cap_hit"] for tr in turns),
        "format_error_count": sum(tr["format_error"] for tr in turns),
        "nudge_count": sum(tr["nudged"] for tr in turns),
        "completion_turn_idx": comp_idx,
        "completion_turn_id": turns[comp_idx]["node_id"] if comp_idx is not None else None,
        "completion_kind": comp_kind,
        "completion_visible_chars": comp_visible if comp_idx is not None else None,
        "premature_finish": premature,
        "format_exit": format_exit,
        "turns": turns,
    }


def iter_turn_rows(record: dict) -> list[dict]:
    """Flatten one rollout record into per-turn rows carrying the rollout
    keys a dashboard groups by."""
    head = {k: record[k] for k in ("rollout_id", "source", "env_category",
                                   "harness", "policy", "outcome", "stop_kind",
                                   "king_digest", "reign", "task_sid")}
    return [{**head, **tr} for tr in record["turns"]]


# -- summary ---------------------------------------------------------------------

SUMMARY_COLUMNS = ("rollouts", "solved", "failed", "errored", "unscored",
                   "loop_rate_all", "loop_rate_failed", "onsets", "onsets_norm",
                   "ends_in_loop_failed", "no_action_rate", "cap_hits",
                   "format_errors", "format_exits", "completions",
                   "premature", "reasoning_only_finish")


def summarize(records, key=lambda r: (r["source"], r["harness"])) -> dict[tuple, dict]:
    """Per-(source, harness) counts over rollout records (turns not needed
    beyond the rollout-level fields)."""
    acc: dict[tuple, dict] = {}
    for r in records:
        s = acc.setdefault(key(r), defaultdict(int))
        s["rollouts"] += 1
        s[r["outcome"]] += 1
        s["turns"] += r["n_turns"]
        s["no_action"] += r["no_action_count"]
        s["cap_hits"] += r["cap_hit_count"]
        s["format_errors"] += r["format_error_count"]
        s["format_exits"] += r["format_exit"]
        s["onsets"] += r["n_loops"]
        s["onsets_norm"] += r["n_loops_norm"]
        s["loop_any"] += r["n_loops"] > 0
        if r["outcome"] == "failed":
            s["loop_failed"] += r["n_loops"] > 0
            s["ends_in_loop_failed"] += r["ends_in_loop"]
        s["completions"] += r["completion_turn_idx"] is not None
        s["premature"] += r["premature_finish"]
        s["reasoning_only_finish"] += (r["completion_turn_idx"] is not None
                                       and r["completion_visible_chars"] == 0)
    out: dict[tuple, dict] = {}
    for k, s in acc.items():
        n, failed = s["rollouts"], s["failed"]
        out[k] = {
            "rollouts": n, "solved": s["solved"], "failed": failed,
            "errored": s["errored"], "unscored": s["unscored"],
            "loop_rate_all": _pct(s["loop_any"], n),
            "loop_rate_failed": _pct(s["loop_failed"], failed),
            "onsets": s["onsets"], "onsets_norm": s["onsets_norm"],
            "ends_in_loop_failed": _pct(s["ends_in_loop_failed"], failed),
            "no_action_rate": _pct(s["no_action"], s["turns"]),
            "cap_hits": s["cap_hits"], "format_errors": s["format_errors"],
            "format_exits": s["format_exits"], "completions": s["completions"],
            "premature": s["premature"],
            "reasoning_only_finish": s["reasoning_only_finish"],
        }
    return out


def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "-"


def format_summary(table: dict[tuple, dict], key_names=("source", "harness")) -> str:
    cols = list(key_names) + list(SUMMARY_COLUMNS)
    rows = [[*map(str, k), *(str(v[c]) for c in SUMMARY_COLUMNS)]
            for k, v in sorted(table.items(), key=lambda kv: tuple(map(str, kv[0])))]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
              for i, c in enumerate(cols)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*cols), fmt.format(*("-" * w for w in widths))]
    lines += [fmt.format(*r) for r in rows]
    return "\n".join(lines)
