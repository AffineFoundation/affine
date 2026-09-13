"""Loop labels for the king seat's failed rollouts (data event 2026-09-11).

The reign-11 king fails by repeating itself: in its failed agent rollouts
49-100 % contain a loop (teacher 8-13 %), and 93 % of its in-loop thoughts
are byte-identical to an earlier turn. The `king_fail` group held almost
none of that material (1.1 % of its turns) because a repeated command is,
by construction, a "reference leaked into the prefix" for the slicer. The
`king_loop_onset` fold group keeps the FIRST turn of every loop -- the
state where the king starts to repeat -- so the duel asks the miner "what
do you do now?" exactly there. Measurement + proposal: king-loop-labels
report (internal/king-loops, 2026-09-11). No LLM judge: every label is a
string rule over the stored conversations.

Definitions (one reply = one turn; the observation of turn k is what came
back before turn k+1):
  repeat      the turn's action, normalized (lower-case, whitespace
              collapsed, digits -> 0, long hex -> H, directory prefixes
              collapsed), equals an earlier action of the rollout
  loop_onset  the first repeat whose observation equals (first 200
              normalized chars) the observation that earlier action got
  in_loop     the following turns until action AND observation both change
  escape      the first turn where both changed
  normal      everything else
Replies without exactly one action in the policy's dialect take no part
(the slicer never admits them); the loop state carries across them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from affine import dialects
from datagen.slicer import _normalize as normalize_fence

OBS_HEAD = 200
WS_RE = re.compile(r"\s+")
NUM_RE = re.compile(r"\d+")
HEX_RE = re.compile(r"\b[0-9a-f]{8,}\b")
PATH_RE = re.compile(r"(?:/[\w.\-@+]+){2,}/")

NORMAL = "normal"
ONSET = "loop_onset"
IN_LOOP = "in_loop"
ESCAPE = "escape"


ERROR_RE = re.compile(
    r"traceback|error|exception|no such file|not found|command not found|"
    r"permission denied|<returncode>[1-9]|exit code [1-9]|exit status [1-9]|"
    r"failed|fatal:|cannot |syntax|wasted call|is not a terminal|timed out|"
    r"killed|segmentation fault", re.I)
NUDGE_RE = re.compile(
    r"exactly one|format error|please always provide|did not include|"
    r"no (?:bash )?code block|your response must|must contain|"
    r"could not parse|invalid json|malformed|wasted call|"
    r"refer to that earlier|unknown tool|tool .* not found", re.I)
EMPTY_OBS_RE = re.compile(
    r"^(?:<returncode>\d+</returncode>\s*)?<output>\s*</output>$|"
    r"^\(?(?:no output|empty)\)?$|^current terminal screen:\s*$", re.I)
BAD_OBS = frozenset({"error", "empty", "nudge"})


def obs_kind(obs: str | None) -> str:
    """empty | nudge | error | ok | none -- what came back after a reply."""
    if obs is None:
        return "none"
    stripped = obs.strip()
    if not stripped or EMPTY_OBS_RE.match(norm_ws(stripped)):
        return "empty"
    head = stripped[:1500]
    if NUDGE_RE.search(head):
        return "nudge"
    if ERROR_RE.search(head):
        return "error"
    return "ok"


@dataclass(frozen=True)
class LoopLabel:
    label: str
    # For an onset / in-loop turn: the earlier turn whose action it repeats.
    repeats: int | None = None
    # The previous action-bearing turn got a bad observation (error / empty /
    # nudge) and this turn repeats its action anyway (normalized).
    persist: bool = False
    # Kind of the observation the PREVIOUS action-bearing turn received.
    prev_obs: str = "none"


def norm_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()


def norm_action(s: str) -> str:
    s = s.lower()
    s = PATH_RE.sub("/", s)
    s = HEX_RE.sub("H", s)
    s = NUM_RE.sub("0", s)
    return norm_ws(s)


def _terminus_commands(action_json: str) -> str:
    """A Terminus batch repeats when its keystrokes repeat; analysis/plan
    prose differs between otherwise identical batches."""
    try:
        obj = json.loads(action_json)
        cmds = obj.get("commands") or []
        keys = "\n".join(str(c.get("keystrokes", c)) if isinstance(c, dict)
                         else str(c) for c in cmds).strip()
        if keys:
            return keys
        return "<task_complete>" if obj.get("task_complete") else "<wait>"
    except (ValueError, AttributeError):
        return action_json


def reply_action(reply: str, action_kind: str) -> str:
    """The single action of a baked reply, or "" when it has none / several."""
    acts = dialects.get(action_kind).actions(normalize_fence(reply))
    if len(acts) != 1:
        return ""
    if action_kind == "terminus_json":
        return _terminus_commands(acts[0])
    return acts[0]


def observations(convs: list[list[dict]]) -> list[str | None]:
    """What came back after each reply: the non-assistant messages of the
    next conversation past its common prefix with this one (None for the
    last reply). Conversations are the baked root->reply paths, so a
    harness that rewrites or drops history is read the way the model saw
    it."""
    out: list[str | None] = []
    for k, conv in enumerate(convs):
        if k + 1 >= len(convs):
            out.append(None)
            continue
        nxt = convs[k + 1]
        n = 0
        while (n < len(conv) and n < len(nxt)
               and conv[n]["role"] == nxt[n]["role"]
               and conv[n]["content"] == nxt[n]["content"]):
            n += 1
        out.append("\n".join(m["content"] for m in nxt[n:-1]
                             if m["role"] != "assistant"))
    return out


def label_loops(convs: list[list[dict]], action_kind: str) -> list[LoopLabel]:
    """One label per conversation (reply), in reply order."""
    obs = observations(convs)
    labels: list[LoopLabel] = []
    obs_heads: list[str | None] = []
    norm_seen: dict[str, list[int]] = {}
    prev: tuple[int, str, str | None] | None = None  # (turn, a_norm, ohead)
    state = NORMAL
    for k, conv in enumerate(convs):
        reply = conv[-1]["content"] if conv and conv[-1]["role"] == "assistant" else ""
        act = reply_action(reply, action_kind) if reply else ""
        ohead = norm_ws(obs[k])[:OBS_HEAD] if obs[k] is not None else None
        obs_heads.append(ohead)
        if not act:
            labels.append(LoopLabel(NORMAL))
            continue
        a_norm = norm_action(act)
        earlier = norm_seen.get(a_norm) or []
        repeats = earlier[-1] if earlier else None
        same_obs = (repeats is not None and ohead is not None
                    and ohead == obs_heads[repeats])
        prev_kind = obs_kind(obs[prev[0]]) if prev is not None else "none"
        persist = prev is not None and prev_kind in BAD_OBS and a_norm == prev[1]
        if repeats is not None and same_obs:
            labels.append(LoopLabel(ONSET if state == NORMAL else IN_LOOP, repeats,
                                    persist, prev_kind))
            state = IN_LOOP
        elif state == IN_LOOP and prev is not None:
            changed_act = a_norm != prev[1]
            changed_obs = ohead is not None and ohead != prev[2]
            if changed_act and changed_obs:
                labels.append(LoopLabel(ESCAPE, None, persist, prev_kind))
                state = NORMAL
            elif changed_act and ohead is None:
                # Final turn: no observation, so an escape is undecidable.
                labels.append(LoopLabel(NORMAL, None, persist, prev_kind))
            else:
                labels.append(LoopLabel(IN_LOOP, repeats, persist, prev_kind))
        else:
            labels.append(LoopLabel(NORMAL, None, persist, prev_kind))
        norm_seen.setdefault(a_norm, []).append(k)
        prev = (k, a_norm, ohead)
    return labels
