"""A_match — discrete action agreement, verdict TELEMETRY only (2026-09-14).

Not a scoring term: nothing here enters the turn score, the margin or the
crown test. Published per side and per dialect so the operator can watch
how often a side's action is literally one of the teacher's reference
actions, and how often the teacher agrees with itself.

  A_match(side, turn) = share of the turn's valid reference actions that
                        equal the side's action after dialect-aware
                        normalisation (below)
  pair(turn)          = mean pairwise equality among the valid reference
                        actions — the teacher's self-agreement, i.e. the
                        "free credit" a deterministic-teacher turn hands out
  centred             = A_match − pair, per turn; the published mean is over
                        turns where both are defined

Normalisation is per action dialect (affine.dialects): a bash command is
compared whitespace-collapsed with quotes unified and a trailing ';'
dropped; a tool call as the sorted (name, arguments) list, XML or JSON; a
boxed answer with LaTeX decoration stripped; a Terminus batch as its
keystroke list plus the done flag. `text` (a free-form final reply) has no
normal form — A_match is undefined (None) on those turns. Ported from the
offline probe research/hints/amatch.py (PR #18) so the live numbers and the
probe agree.
"""

from __future__ import annotations

import json
import re

WS = re.compile(r"\s+")
FENCE = re.compile(r"```(?:bash|mswea_bash_command)[ \t]*\n(.*?)\n```", re.S)
XML_CALL = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
XML_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def norm_ws(s: str) -> str:
    return WS.sub(" ", s or "").strip()


def norm_bash(y: str) -> str | None:
    m = FENCE.search(y or "")
    cmd = m.group(1) if m else (y or "")
    cmd = norm_ws(cmd).replace('"', "'").rstrip(";").strip()
    return cmd or None


def norm_tool_call(y: str) -> str | None:
    calls = []
    for m in XML_CALL.finditer(y or ""):
        name = m.group(1).strip()
        args = {k.strip(): norm_ws(v) for k, v in XML_PARAM.findall(m.group(2))}
        calls.append((name, args))
    if not calls:
        body = re.sub(r"</?tool_call>", "", y or "").strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            return None
        for c in obj if isinstance(obj, list) else [obj]:
            if not isinstance(c, dict):
                continue
            name = c.get("name") or (c.get("function") or {}).get("name")
            args = c.get("arguments") or (c.get("function") or {}).get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": norm_ws(args)}
            calls.append((name, {str(k): norm_ws(str(v)) for k, v in (args or {}).items()}))
    if not calls:
        return None
    return json.dumps(calls, sort_keys=True, ensure_ascii=False)


def norm_boxed(y: str) -> str | None:
    m = BOXED.search(y or "")
    if not m:
        return None
    s = m.group(1)
    s = re.sub(r"\\(left|right|,|;|!|text|mathrm|displaystyle)", "", s)
    s = s.replace("$", "").replace(" ", "").rstrip(".").lower()
    return s or None


def norm_terminus(y: str) -> str | None:
    try:
        obj = json.loads(y)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", y or "", re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    cmds = obj.get("commands") if isinstance(obj, dict) else None
    if cmds is None:
        return None
    keys = []
    for c in cmds:
        if isinstance(c, dict):
            keys.append(norm_ws(str(c.get("keystrokes") or c.get("command") or "")))
        else:
            keys.append(norm_ws(str(c)))
    return json.dumps({"commands": keys, "done": bool(obj.get("task_complete"))}, ensure_ascii=False)


NORM = {"bash": norm_bash, "tool_call": norm_tool_call, "boxed": norm_boxed,
        "terminus_json": norm_terminus}


def norm_action(y: str | None, kind: str | None) -> str | None:
    """Canonical form of an action in its dialect; None when the dialect has
    no normal form (`text`) or the action does not parse."""
    fn = NORM.get(kind or "bash")
    return fn(y) if (fn and y is not None) else None


def pairwise(acts: list[str]) -> float | None:
    """Mean pairwise equality; None below two items."""
    if len(acts) < 2:
        return None
    n = len(acts)
    eq = sum(1 for i in range(n) for j in range(i + 1, n) if acts[i] == acts[j])
    return eq / (n * (n - 1) / 2)


def turn_agreement(y_side: str | None, ref_ys: list[str], kind: str | None
                   ) -> tuple[float | None, float | None]:
    """(A_match, pair) for one turn. A_match is None when the side's action
    or every reference has no normal form; pair is None below two valid refs."""
    refs = [n for n in (norm_action(y, kind) for y in ref_ys) if n is not None]
    pair = pairwise(refs)
    mine = norm_action(y_side, kind)
    if mine is None or not refs:
        return None, pair
    return sum(1 for r in refs if r == mine) / len(refs), pair


def summarize(rows: list[dict]) -> dict:
    """Side-level means from per-turn rows carrying `a_match` / `ref_pair`."""
    a = [r["a_match"] for r in rows if r.get("a_match") is not None]
    c = [r["a_match"] - r["ref_pair"] for r in rows
         if r.get("a_match") is not None and r.get("ref_pair") is not None]
    return {
        "a_match": sum(a) / len(a) if a else None,
        "a_match_n": len(a),
        "a_match_centered": sum(c) / len(c) if c else None,
    }
