"""Action dialects: where a turn's action span begins and ends.

min(R,G) needs one thing from the turn format: a deterministic split of the
assistant turn into a thought `z` and an action `y`. R measures whether z
raises the teacher's likelihood of y, B measures whether the miner's own z
causes its own y, and G judges z on its own. None of that cares what the
action *looks like* — but all of it needs to know where the action is.

Historically that split was hardcoded to one closed ```bash block, which is
why corpus D could only ever hold shell-agent trajectories. A dialect makes
the delimiter pluggable while leaving the scoring math untouched:

    bash        ```bash ... ```            shell agents (swe, terminal, lean,
                                           prolog — anything with a container)
    tool_call   <tool_call> ... </tool_call>  tool-use / search envs
    boxed       \\boxed{...}               math / short-answer envs
    text        the whole visible reply    final reports / answers: the reply
                                           that ends a trajectory without a
                                           tool call (admitted wvk 13)
    terminus_json  one JSON object with    Terminus 2, the terminal-bench
                   "analysis"/"plan"/      reference agent (harbor): the reply
                   "commands"              IS the JSON; registered 2026-09-10,
                                           NOT admitted (staging only)

What deliberately does NOT vary per dialect: the thought channel rendering
(`</think>\\nTHOUGHT: {z}`, see evalsrv/chat.py). G compares the miner's
m = lpC(z_A|x) against the teacher's t_i = lpC(z_C^i|x) under that exact
rendering, so holding it fixed keeps the band comparable across dialects and
keeps every stored bash turn byte-identical on replay.

`system_marker` is load-bearing, not cosmetic: a miner can only be expected
to emit a dialect that the turn prefix asked for. The fold requires the
marker in the turn's system message, so an env whose prompt never states its
action contract is dropped rather than silently scored against a format the
model was never told about. `marker_roles` says which leading message may
carry it: the system message for every dialect that has one, plus the first
user message for `terminus_json` — Terminus 2 states its whole format
contract in the first user turn and sends a system message only when the
task supplies one.

`text` is the one dialect with no marker. Its contract — "when you are done,
say so in plain words" — is the default contract of every chat model and no
harness prompt spells it out, so the marker check is vacuous for it. The
fold compensates on the other side: a text turn is only ever the FINAL
sampled reply of a rollout that stopped by itself (`agent_completed`, not
finish=length, not max_turns), i.e. the reply the model chose to end on.
Why the dialect exists: under min(R,G) the visible message was never the
scored span, so a king trained against the score learned to put everything
in <think> and emit no visible text — Claude Code's compaction and
final-report prompts then get reasoning only (SWE-bench Pro post-mortem,
2026-09-08). With `text` admitted, the visible reply is an action: R asks
whether the miner's thought predicts the teacher's report, B whether its
thought causes its own, G judges the thought — the same three legs.

Admission is separate from parsing: `[dataset].allowed_action_kinds` in
affine.toml gates which dialects may enter live D. Parsing support here is a
prerequisite for that decision, not the decision itself.

Stdlib-only on purpose — the datagen box imports this without the validator's
dependency set.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

CONTRACT_TOML = Path(__file__).resolve().parents[1] / "affine.toml"

Span = tuple[int, int]
Finder = Callable[[str], list[Span]]

BASH_PATTERN = re.compile(r"```bash\n.*?\n```", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
BOXED_OPEN = "\\boxed{"


def _regex_finder(pattern: re.Pattern[str]) -> Finder:
    def find(text: str) -> list[Span]:
        return [m.span() for m in pattern.finditer(text)]
    return find


def _boxed_finder(text: str) -> list[Span]:
    """Spans of every `\\boxed{...}`, brace-balanced.

    Regex cannot match nested braces, and math answers routinely contain
    them (`\\boxed{\\frac{1}{2}}`). An unclosed `\\boxed{` yields no span at
    all, which is the same "no parsable action" outcome as an unterminated
    bash fence. An empty body (`\\boxed{}`) is not an answer either — models
    quote the instruction verbatim mid-thought ("put it in \\boxed{}"), and
    a truncated rollout must not get credit for that quote.
    """
    spans: list[Span] = []
    start = text.find(BOXED_OPEN)
    while start != -1:
        depth = 0
        for i in range(start + len(BOXED_OPEN) - 1, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    if text[start + len(BOXED_OPEN):i].strip():
                        spans.append((start, i + 1))
                    break
        start = text.find(BOXED_OPEN, start + len(BOXED_OPEN))
    return spans


def _text_finder(text: str) -> list[Span]:
    """One span: the visible reply without surrounding whitespace. An empty
    or whitespace-only reply has no action — the same forfeit as an
    unterminated fence. (On the duel side the text handed in here is what
    follows </think>, so an unclosed think block that never says anything
    visible is empty here too once require_think_close is on.)"""
    stripped = text.strip()
    if not stripped:
        return []
    start = len(text) - len(text.lstrip())
    return [(start, start + len(stripped))]


TERMINUS_REQUIRED_KEYS = ("analysis", "plan", "commands")


def _balanced_object_end(text: str, start: int) -> int:
    """Index one past the `}` closing the JSON object opened at `start`,
    string-aware, or -1 when it never closes."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _terminus_json_finder(text: str) -> list[Span]:
    """Spans of every top-level JSON object that is a Terminus 2 command
    batch: parses, is an object, and carries the three required keys with
    `commands` a list. Terminus 2 (harbor's terminal-bench agent) has no
    tool channel — the reply IS the action, a JSON object whose `analysis`
    and `plan` strings are the agent's visible reasoning and whose
    `commands` are tmux keystrokes. Prose around the object is tolerated by
    the harness ("warnings") and by this finder alike; a truncated object
    never closes and yields no span (forfeit, like an open fence)."""
    spans: list[Span] = []
    pos = text.find("{")
    while pos != -1:
        end = _balanced_object_end(text, pos)
        if end == -1:
            break
        try:
            obj = json.loads(text[pos:end])
        except ValueError:
            obj = None
        if (isinstance(obj, dict)
                and all(k in obj for k in TERMINUS_REQUIRED_KEYS)
                and isinstance(obj["commands"], list)):
            spans.append((pos, end))
            pos = text.find("{", end)
        else:
            pos = text.find("{", pos + 1)
    return spans


@dataclass(frozen=True)
class Dialect:
    id: str
    finder: Finder
    # "" = no marker required (text: the default contract of a chat model).
    system_marker: str
    label: str
    # A rollout in this dialect legitimately ends with a plain reply (agent
    # loops: the model stops calling tools and reports). The fold may then
    # record that final reply as a `text` turn. False for answer-format
    # dialects (boxed): a final reply without the format is a miss, not a
    # report.
    ends_in_text: bool = False
    # Which leading prefix message may state the contract (see module doc).
    marker_roles: tuple[str, ...] = ("system",)

    def spans(self, text: str) -> list[Span]:
        return self.finder(text)

    def actions(self, text: str) -> list[str]:
        return [text[s:e] for s, e in self.finder(text)]

    def system_ok(self, system_content: str) -> bool:
        if not self.system_marker:
            return True
        return self.system_marker in system_content.lower()

    def mandate_ok(self, prefix: list[dict]) -> bool:
        """The prefix states this dialect's contract: the marker appears in
        the first message of one of `marker_roles`. For ("system",) this is
        exactly the historical check — the first system message carries the
        marker, and a prefix with no system message is dropped (also for
        `text`, whose marker is vacuous)."""
        for role in self.marker_roles:
            first = next((m for m in prefix if m.get("role") == role), None)
            if first is not None and self.system_ok(str(first.get("content", ""))):
                return True
        return False


DIALECTS: dict[str, Dialect] = {
    d.id: d for d in (
        Dialect(
            id="bash",
            finder=_regex_finder(BASH_PATTERN),
            system_marker="bash",
            label="one closed ```bash block (shell agents)",
            ends_in_text=True,
        ),
        Dialect(
            id="tool_call",
            finder=_regex_finder(TOOL_CALL_PATTERN),
            system_marker="tool",
            label="one <tool_call>...</tool_call> block (tool use / search)",
            ends_in_text=True,
        ),
        Dialect(
            id="boxed",
            finder=_boxed_finder,
            system_marker="boxed",
            label="one \\boxed{...} answer (math / short answer)",
        ),
        Dialect(
            id="text",
            finder=_text_finder,
            system_marker="",
            label="the whole visible reply (final report / answer)",
        ),
        Dialect(
            id="terminus_json",
            finder=_terminus_json_finder,
            # Terminus 2's prompt: 'Format your response as JSON with the
            # following structure' + the "commands" key. Both words together
            # are the contract; "json" alone would match unrelated prompts.
            system_marker='"commands"',
            label='one Terminus 2 JSON command batch ("analysis"/"plan"/"commands")',
            marker_roles=("system", "user"),
        ),
    )
}

DEFAULT_KIND = "bash"
# The dialect a reply falls back to when it carries no action in its
# policy's dialect and ends the rollout (see the module docstring and
# datagen.slicer.slice_messages `text_final`).
TEXT_KIND = "text"


class UnknownDialect(ValueError):
    pass


def get(action_kind: str | None) -> Dialect:
    """Dialect for a turn's action_kind. None/empty means the bash default
    (pre-dialect corpus records carry no explicit kind)."""
    kind = action_kind or DEFAULT_KIND
    try:
        return DIALECTS[kind]
    except KeyError:
        raise UnknownDialect(
            f"unknown action_kind {kind!r}; registered: {sorted(DIALECTS)}"
        ) from None


def is_registered(action_kind: str | None) -> bool:
    return (action_kind or DEFAULT_KIND) in DIALECTS


def last_action(text: str, action_kind: str | None = DEFAULT_KIND) -> str:
    """The final complete action in `text`, or "" when none parses.

    Last (not first) match, matching the pre-dialect bash behavior: an agent
    may quote a command mid-thought before committing to the real one.
    """
    spans = get(action_kind).spans(text)
    if not spans:
        return ""
    s, e = spans[-1]
    return text[s:e]


def split_action(text: str, action_kind: str | None = DEFAULT_KIND
                 ) -> tuple[str, str]:
    """(text before the final action, the final action). ("", "") when none."""
    spans = get(action_kind).spans(text)
    if not spans:
        return "", ""
    s, e = spans[-1]
    return text[:s], text[s:e]


def count_actions(text: str, action_kind: str | None = DEFAULT_KIND) -> int:
    return len(get(action_kind).spans(text))


# -- corpus admission (fold prefilter) -----------------------------------------

def admitted_kinds(toml_path: Path | None = None) -> tuple[str, ...]:
    """``[dataset].allowed_action_kinds`` from affine.toml — the contract SSOT.

    Read at call time, not import time: the fold and the duel both consult
    the same file, and a missing key means the pre-dialect contract (bash).
    """
    raw = tomllib.loads((toml_path or CONTRACT_TOML).read_text())
    kinds = raw.get("dataset", {}).get("allowed_action_kinds", [DEFAULT_KIND])
    return tuple(str(k) for k in kinds)


def admission_reason(action_kind: str | None,
                     allowed: Iterable[str]) -> str | None:
    """Why a turn of this kind may not enter D, or None if it may."""
    kind = action_kind or DEFAULT_KIND
    if kind not in DIALECTS:
        return f"action_kind_unregistered:{kind}"
    if kind not in set(allowed):
        return f"action_kind_not_admitted:{kind}"
    return None


def reference_check(prefix: list[dict], reference_turn: str,
                    action_kind: str | None) -> tuple[str | None, str]:
    """Dialect-dependent half of the fold prefilter.

    Returns (drop_reason, action). drop_reason is None when the turn is a
    scorable reference: the turn's system message states the dialect's
    contract (``system_marker``) and the reference turn contains exactly one
    complete action. Callers run the leakage check on the returned action.
    Format-error replies and multi-action turns are real history (they stay
    in later prefixes) but not references.
    """
    d = get(action_kind)
    if not d.mandate_ok(prefix):
        return f"system_msg_no_{d.id}_mandate", ""
    acts = d.actions(reference_turn)
    if len(acts) != 1:
        return f"ref_{d.id}_actions={len(acts)}", ""
    return None, acts[0]
