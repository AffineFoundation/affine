"""Shared rendering + span helpers for the N1 stage-2 sets (Set A / Set B).

One candidate action y on a turn with prefix x is rendered exactly as the live
evalsrv forces an action (evalsrv.chat.force_text, thought_rendering
"as_generated", wvk 22):

    full = gen_prompt(x) + thought_body(z) + "\\n\\n" + y

with two renderings of the thought:
    no_thought    z = ""  -> body "\\n</think>"  (template's thinking-off form; the
                  N1 term lpC+(y|x) - lpC(y|x) has no thought conditioning, so
                  this is the primary rendering; identical to the sibling's
                  render_eval.py)
    with_thought  z = the candidate's own thought (king/challenger z_a, ref z);
                  the base-side action logprob then equals the live lpC(y|z)
                  echo (lpC_ya_za / lp_own) -> parity check for the endpoint.

Span levels, all as (start, end) char offsets, given relative to y (`rel`) and
absolute in the full text per rendering (`abs`):
    action   the whole y
    inner    y without its envelope: bash -> inside the ```bash fence;
             tool_call -> inside <tool_call>…</tool_call>; terminus_json -> the
             `commands` array; boxed -> inside \\boxed{}; text -> stripped reply
    body     the bytes the miner actually chooses (list of spans):
             bash/text -> = inner; tool_call XML -> every <parameter=…> value
             (JSON form -> the "arguments" value); terminus_json -> every
             "keystrokes" string value (falls back to the commands array when
             the batch is empty, e.g. task_complete); boxed -> = inner
Token rule downstream (score_sets.py): a token is scored iff its start offset
lies inside a span — the live evalsrv rule.

Pitfalls found on the stored duels (2026-09-21):
  * every live tool_call action is Qwen3 XML (<function=NAME><parameter=K>V
    </parameter>…), never JSON — 42% carry >1 parameter (Edit: file_path /
    old_string / new_string), so the body is a LIST of spans;
  * terminus `commands` may be [] (task_complete) — no keystrokes to score;
  * a side may answer a tool_call turn with prose (text fallback,
    text_fallback_at_tool_turns) — `y_kind` records the dialect the y actually
    parses in, and spans follow y_kind, not the turn's dialect.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
from affine import dialects  # noqa: E402  (live tree via common's sys.path)

SET_DIALECTS = ("bash", "tool_call", "terminus_json", "text")
SHELL_TOOL_NAMES = ("bash", "Bash", "terminal", "execute_bash", "shell", "run_command",
                    "run_terminal_cmd", "execute_command", "run_shell_command")
NO_THOUGHT_BODY = "\n" + C.THINK_CLOSE          # thought_body("") under as_generated

GROUP_OF_SOURCE = {
    **{s: "coding" for s in ("multiswe", "r2e_gym", "scaleswe", "swelego", "swerebench_v2", "swesmith")},
    **{s: "terminal" for s in ("affine_tmax", "terminal_bench_2", "terminal_lego")},
    **{s: "math" for s in ("affine_i3math", "affine_math")},
    **{s: "tool_use" for s in ("affine_agent", "affine_notool", "affine_tau2", "affine_tau2_synth",
                               "affine_when2call", "affine_wiki", "affine_sql")},
    **{s: "nl2repo" for s in ("affine_nl2lib", "nl2repobench")},
}


def group_of(source: str | None) -> str:
    return GROUP_OF_SOURCE.get(source or "", "general")


def depth_of(prefix: list[dict]) -> int:
    return sum(1 for m in prefix if m["role"] == "assistant")


# ------------------------------------------------------------------ classification
def classify_action(y: str, kind: str) -> str | None:
    """The dialect y parses in: `kind` when the whole y is one action of that
    dialect, "text" when it is a non-empty prose reply (fallback), else None."""
    if not y or not y.strip():
        return None
    spans = dialects.get(kind).spans(y)
    if len(spans) == 1 and y[spans[0][0]:spans[0][1]].strip() == y.strip():
        return kind
    return "text"          # any non-empty reply is a text action (fallback)


# ------------------------------------------------------------------ span finders
def _balanced_end(s: str, start: int) -> int:
    """End (exclusive) of the JSON value starting at s[start] ({, [ or ")."""
    ch = s[start]
    if ch == '"':
        i = start + 1
        while i < len(s):
            if s[i] == "\\":
                i += 2
                continue
            if s[i] == '"':
                return i + 1
            i += 1
        return -1
    if ch not in "{[":
        m = re.compile(r"[^,\]}\s]+").match(s, start)
        return m.end() if m else -1
    depth = 0
    in_str = False
    i = start
    while i < len(s):
        c = s[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _json_key_value_span(s: str, key: str, start: int = 0) -> tuple[int, int] | None:
    m = re.compile(r'"' + re.escape(key) + r'"\s*:\s*').search(s, start)
    if not m or m.end() >= len(s):
        return None
    e = _balanced_end(s, m.end())
    return (m.end(), e) if e > 0 else None


_PARAM_RE = re.compile(r"<parameter=[^>\n]+>(\n?)(.*?)(\n?)</parameter>", re.S)
_TC_INNER_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z_]*\n(.*?)\n?```\s*$", re.S)


def rel_spans(y: str, y_kind: str | None) -> dict:
    """{"action": [0, len], "inner": [s, e], "body": [[s, e], ...], "body_kind": str}
    relative to y. body_kind names the rule that produced `body`."""
    whole = (0, len(y))
    if y_kind == "bash":
        m = _FENCE_RE.match(y)
        inner = (m.start(1), m.end(1)) if m else whole
        return {"action": whole, "inner": inner, "body": [inner], "body_kind": "fence_inner"}
    if y_kind == "tool_call":
        m = _TC_INNER_RE.search(y)
        inner = (m.start(1), m.end(1)) if m else whole
        params = [(pm.start(2), pm.end(2)) for pm in _PARAM_RE.finditer(y, inner[0], inner[1])]
        if params:
            return {"action": whole, "inner": inner, "body": params, "body_kind": "xml_parameter_values"}
        args = _json_key_value_span(y, "arguments", inner[0])
        if args and args[1] <= inner[1]:
            return {"action": whole, "inner": inner, "body": [args], "body_kind": "json_arguments"}
        return {"action": whole, "inner": inner, "body": [inner], "body_kind": "envelope_inner"}
    if y_kind == "terminus_json":
        cmds = _json_key_value_span(y, "commands")
        inner = cmds or whole
        keys = []
        pos = inner[0]
        while True:
            ks = _json_key_value_span(y, "keystrokes", pos)
            if not ks or ks[0] >= inner[1]:
                break
            # inside the quotes only
            keys.append((ks[0] + 1, ks[1] - 1) if y[ks[0]] == '"' else ks)
            pos = ks[1]
        if keys:
            return {"action": whole, "inner": inner, "body": keys, "body_kind": "keystrokes_values"}
        return {"action": whole, "inner": inner, "body": [inner], "body_kind": "commands_array"}
    if y_kind == "boxed":
        s = y.find("\\boxed{")
        if s >= 0:
            depth = 0
            for i in range(s + len("\\boxed"), len(y)):        # braces only, as the live _boxed_finder
                if y[i] == "{":
                    depth += 1
                elif y[i] == "}":
                    depth -= 1
                    if depth == 0:
                        inner = (s + len("\\boxed{"), i)
                        return {"action": whole, "inner": inner, "body": [inner], "body_kind": "boxed_inner"}
        return {"action": whole, "inner": whole, "body": [whole], "body_kind": "whole"}
    # text (and anything else): the stripped reply
    lead = len(y) - len(y.lstrip())
    inner = (lead, lead + len(y.strip())) if y.strip() else whole
    return {"action": whole, "inner": inner, "body": [inner], "body_kind": "stripped_reply"}


# ------------------------------------------------------------------ rendering
def render_candidate(prompt: str, role: str, z: str | None, y: str, kind: str,
                     model: str | None = None, **extra) -> dict:
    """One candidate row: y, its own thought z (may be ""), y_kind, relative
    spans and, per rendering, the suffix appended to `prompt` plus absolute
    spans in prompt + suffix."""
    z = z or ""
    y_kind = classify_action(y, kind)
    rel = rel_spans(y, y_kind)
    body_wt, _ = C.thought_body(z)
    renders = {"no_thought": NO_THOUGHT_BODY + "\n\n" + y,
               "with_thought": body_wt + "\n\n" + y}
    abs_spans = {}
    for name, suffix in renders.items():
        a0 = len(prompt) + len(suffix) - len(y)
        abs_spans[name] = {"action_start": a0,
                           "action": [a0, a0 + len(y)],
                           "inner": [a0 + rel["inner"][0], a0 + rel["inner"][1]],
                           "body": [[a0 + s, a0 + e] for s, e in rel["body"]]}
    return {"role": role, "model": model, "y": y, "z": z, "y_kind": y_kind, "turn_kind": kind,
            "n_bytes_y": len(y.encode()), "n_bytes_body": sum(len(y[s:e].encode()) for s, e in rel["body"]),
            "rel": {"action": list(rel["action"]), "inner": list(rel["inner"]),
                    "body": [list(b) for b in rel["body"]], "body_kind": rel["body_kind"]},
            "render": {k: {"suffix": v} for k, v in renders.items()},
            "abs": abs_spans, **extra}


def full_text(prompt: str, cand: dict, rendering: str = "no_thought") -> str:
    return prompt + cand["render"][rendering]["suffix"]


# ------------------------------------------------------------------ attack rows
def tool_names(prefix: list[dict]) -> list[str]:
    sysm = next((m["content"] for m in prefix if m["role"] == "system"), "")
    i, j = sysm.find("<tools>"), sysm.find("</tools>")
    block = sysm[i:j] if i >= 0 else sysm
    return re.findall(r'"name":\s*"([^"]+)"', block)


def generic_action(kind: str, prefix: list[dict]) -> tuple[str, dict]:
    """Dialect-appropriate `ls -la` in the shape the LIVE data uses (tool_call
    = Qwen3 XML; terminus = a command batch)."""
    if kind == "bash":
        return "```bash\nls -la\n```", {}
    if kind == "terminus_json":
        y = json.dumps({"analysis": "Let me look at the current directory.",
                        "plan": "List the files to see what is here.",
                        "commands": [{"keystrokes": "ls -la\n", "duration": 1.0}]}, indent=2)
        return y, {}
    if kind == "tool_call":
        names = tool_names(prefix)
        name = next((n for n in SHELL_TOOL_NAMES if n in names), None)
        meta = {"attack_tool_in_schema": name is not None, "schema_tools": names[:12]}
        y = f"<tool_call>\n<function={name or 'bash'}>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"
        return y, meta
    return "ls -la", {}


def repeat_last_action(kind: str, prefix: list[dict]) -> str | None:
    """The previous assistant turn's action in the turn's dialect (text: its
    whole visible reply). None when the prefix has no such action."""
    for m in reversed(prefix):
        if m["role"] != "assistant":
            continue
        content = m["content"] or ""
        if content.endswith("\n" + C.THINK_CLOSE):
            content = content[: -len("\n" + C.THINK_CLOSE)]
        if kind == "text":
            return content.strip() or None
        y = dialects.last_action(content, kind)
        return y or None
    return None
