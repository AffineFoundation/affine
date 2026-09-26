"""Bake a tool-using conversation into plain chat text the teacher's chat
template reproduces byte-for-byte.

Corpus D stores `[{role, content}]` prefixes and the duel renders them with
the teacher's chat template and NO `tools=` argument (evalsrv/chat.py). A
tool-use trajectory carries three things that rendering would otherwise
lose: the tool schemas the model was shown, the structured `tool_calls` on
assistant turns, and `role=tool` results. This module folds all three into
plain `content` so that

    template(baked_messages)  ==  template(original_messages, tools=tools)

holds exactly. That equality is the admission gate for every tool turn
(`parity_ok`): a prefix the model never actually saw is not a reference.

Nothing here hand-writes the teacher's format. Each piece is obtained by
rendering a minimal conversation through the template and cutting out the
span the template produced, so a template revision cannot silently
diverge from what the corpus stores — the parity check would fail loudly
instead.

Two template families are known (`TemplateStyle`, detected from a bare
render, never from the repo name):

* **qwen** (`Qwen/Qwen3.8-27B`, the genesis family): `<|im_start|>{role}\\n
  …<|im_end|>` blocks; the tools block is spliced into the system block;
  tool results render as a `user` block, so parity is exact.
* **glm** (`zai-org/GLM-5.3-Flash`, wvk 25): `<|system|>` / `<|user|>` /
  `<|assistant|>` markers with no end marker; a `<|system|>Reasoning
  Effort: …` preamble block of its own; the tools block is its OWN
  `<|system|>` block placed before the conversation's system message (the
  baker joins the two into ONE system message with the template's own
  marker between them -- the miner engines' genesis template refuses a
  second system message); tool calls render
  as `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
  inside the assistant block; tool results render under `<|observation|>`
  as `<tool_response>…</tool_response>` — a role plain messages cannot
  reach. Option (a) of the cutover plan (2026-09-26): the baker stores the
  `<tool_response>` text in a `user` turn and `parity_ok` compares
  everything except that one role marker (`role_marker_exempt`). Option
  (b) — a structured `duel_turns@v5` rendered with `tools=` at duel time —
  is the next swap's backlog.

Requires `transformers` (tokenizer only, no weights): the datagen box runs
this at slice time; the eval pod never imports it.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer

CONTRACT_TOML = Path(__file__).resolve().parents[1] / "affine.toml"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"          # Qwen's empty-reasoning stub (kept for callers)
_SENTINEL_USER = "\u2063affine-toolbake-user\u2063"


@dataclass(frozen=True)
class TemplateStyle:
    id: str
    # role -> the marker that opens that role's block
    role_markers: dict[str, str]
    # closes every block (qwen) or None = a block runs to the next role marker (glm)
    block_end: str | None
    # what the template inserts for an assistant turn without reasoning
    empty_think: str
    # the tools block is its own system block placed before the conversation's system message
    tools_own_system: bool
    # native role of tool results; None when the template renders them as `user`
    tool_result_role: str | None

    def start(self, role: str) -> str:
        return self.role_markers[role]


QWEN_STYLE = TemplateStyle(
    id="qwen",
    role_markers={r: f"{IM_START}{r}\n" for r in ("system", "user", "assistant", "tool")},
    block_end=IM_END,
    empty_think=EMPTY_THINK,
    tools_own_system=False,
    tool_result_role=None,
)
GLM_STYLE = TemplateStyle(
    id="glm",
    role_markers={"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>",
                  "observation": "<|observation|>"},
    block_end=None,
    empty_think="<think></think>",
    tools_own_system=True,
    tool_result_role="observation",
)
STYLES = (QWEN_STYLE, GLM_STYLE)


def teacher_repo(toml_path: Path | None = None) -> str:
    raw = tomllib.loads((toml_path or CONTRACT_TOML).read_text())
    return str(raw["teacher"]["repo"])


def openai_tool(tool: dict) -> dict:
    """verifiers trace `tools[i]` ({name, description, parameters}) → the
    OpenAI function-tool shape chat templates expect. Already-wrapped
    entries pass through."""
    if tool.get("type") == "function" and "function" in tool:
        return tool
    fn = {k: tool[k] for k in ("name", "description", "parameters") if k in tool}
    return {"type": "function", "function": fn}


def openai_tool_call(call: dict) -> dict:
    """verifiers trace tool_call ({id, name, arguments: json-str}) → OpenAI
    shape with `arguments` as a dict (the templates iterate the pairs)."""
    if "function" in call:
        fn = dict(call["function"])
    else:
        fn = {"name": call.get("name"), "arguments": call.get("arguments")}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {"_raw": args}
    fn["arguments"] = args if isinstance(args, dict) else {"_raw": args}
    return {"id": call.get("id", ""), "type": "function", "function": fn}


def detect_style(bare_render: str) -> TemplateStyle:
    """The template family, from the render of a one-message conversation."""
    if IM_START in bare_render:
        return QWEN_STYLE
    if GLM_STYLE.start("user") in bare_render:
        return GLM_STYLE
    raise ValueError("unknown chat template family; toolbake knows qwen (<|im_start|>) "
                     "and glm (<|user|>) blocks only")


def _block(rendered: str, role: str, occurrence: int = 0,
           style: TemplateStyle = QWEN_STYLE) -> str:
    """Body of the `occurrence`-th block of `role`."""
    marker = style.start(role)
    pos = -1
    for _ in range(occurrence + 1):
        pos = rendered.index(marker, pos + 1)
    start = pos + len(marker)
    if style.block_end is not None:
        return rendered[start:rendered.index(style.block_end, start)]
    ends = [rendered.find(m, start) for m in style.role_markers.values()]
    ends = [e for e in ends if e >= 0]
    return rendered[start:min(ends)] if ends else rendered[start:]


class ToolBaker:
    def __init__(self, tokenizer) -> None:
        self.tok = tokenizer
        bare = self.render([{"role": "user", "content": "u"}])
        self.style = detect_style(bare)
        # Whatever the template prepends on its own. Qwen3.8: a reasoning-
        # effort preamble INSIDE the system block (baked content must not
        # repeat it). GLM-5.3: a `<|system|>Reasoning Effort: …` block of its
        # own, before the conversation's system block (never part of baked
        # content; the conversation's system block is the NEXT one).
        self.preamble = _block(bare, "system", 0, self.style)

    @classmethod
    def from_pretrained(cls, repo: str | None = None, **kw) -> "ToolBaker":
        return cls(AutoTokenizer.from_pretrained(repo or teacher_repo(), **kw))

    def render(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        kw = {"tools": tools} if tools else {}
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **kw)

    def _body(self, rendered: str, role: str, occurrence: int = 0) -> str:
        return _block(rendered, role, occurrence, self.style)

    # -- the three baked pieces --------------------------------------------------

    def baked_system_messages(self, system_content: str | None, tools: list[dict]) -> list[dict]:
        """The system message(s) carrying the rendered tools block, so that
        re-rendering them without `tools=` reproduces the tools-aware render.
        qwen: one system message with the tools block spliced in. glm: the
        tools block as its own system message, followed by the original
        system message when there is one."""
        msgs = [{"role": "user", "content": "u"}]
        if system_content is not None:
            msgs.insert(0, {"role": "system", "content": system_content})
        rendered = self.render(msgs, tools)
        if not self.style.tools_own_system:
            body = self._body(rendered, "system")
            lead = self.preamble + "\n\n"
            if not body.startswith(lead):
                raise ValueError("template system block does not start with the "
                                 "bare preamble; baking rule no longer applies")
            return [{"role": "system", "content": body[len(lead):]}]
        # glm: block 0 = preamble, block 1 = tools, block 2 = the conversation's system
        if self._body(rendered, "system", 0) != self.preamble:
            raise ValueError("template no longer opens with the bare preamble block; "
                             "baking rule no longer applies")
        out = [{"role": "system", "content": self._body(rendered, "system", 1)}]
        if system_content is not None:
            if self._body(rendered, "system", 2) != system_content:
                raise ValueError("template altered the conversation's system content; "
                                 "baking rule no longer applies")
            out.append({"role": "system", "content": system_content})
        return out

    def baked_system(self, system_content: str | None, tools: list[dict]) -> str:
        """ONE system message carrying the tools block. qwen: the spliced
        block. glm: the tools body and the original system content joined by
        the template's own `<|system|>` marker (cut from the render, not
        hand-written), so the single message renders byte-identically to
        the two blocks the template emits. One message, not two, because
        the same prefix is rendered by the MINER engines under the genesis
        template, which refuses a second system message ("System message
        must be at the beginning" -- the Codex CLI finding of 2026-09-07).
        Under the genesis template the embedded marker is plain text."""
        msgs = self.baked_system_messages(system_content, tools)
        if len(msgs) == 1:
            return msgs[0]["content"]
        return self.style.start("system").join(m["content"] for m in msgs)

    def baked_assistant(self, content: str, tool_calls: list[dict]) -> str:
        """Assistant content with its tool calls rendered as the template's
        <tool_call> text (content first, then the calls)."""
        msgs = [{"role": "user", "content": "u"},
                {"role": "assistant", "content": content,
                 "tool_calls": [openai_tool_call(c) for c in tool_calls]}]
        body = self._body(self.render(msgs), "assistant")
        if body.startswith(self.style.empty_think):
            body = body[len(self.style.empty_think):]
        return body

    def baked_tool_results(self, results: list[dict],
                           tool_calls: list[dict] | None = None) -> str:
        """One user turn standing in for a run of consecutive `role=tool`
        messages (both templates merge them into a single block: qwen a
        `user` block, glm an `<|observation|>` block whose body is the
        `<tool_response>…</tool_response>` text). `tool_calls` = the calls
        of the assistant turn these results answer: GLM's template emits
        the results in the ORDER OF THE CALLS (matching `tool_call_id`),
        and Claude Code returns parallel results in completion order, so
        the synthetic render must carry the calls to reproduce that order
        (2026-09-26 parity harness: 405 of 3,540 claude_code paths)."""
        calls = [openai_tool_call(c) for c in (tool_calls or [])]
        msgs = [{"role": "user", "content": _SENTINEL_USER},
                {"role": "assistant", "content": "a", **({"tool_calls": calls} if calls else {})},
                *[{"role": "tool", "content": r.get("content") or "",
                   **({"tool_call_id": r["tool_call_id"]} if r.get("tool_call_id") else {}),
                   **({"name": r["name"]} if r.get("name") else {})}
                  for r in results]]
        rendered = self.render(msgs)
        if self.style.tool_result_role:
            return self._body(rendered, self.style.tool_result_role, 0)
        return self._body(rendered, "user", occurrence=1)

    # -- whole conversation -----------------------------------------------------

    def bake(self, messages: list[dict], tools: list[dict]) -> list[dict]:
        """Plain `[{role, content}]` (system/user/assistant only) for a
        conversation with tools. Input messages may carry `tool_calls`
        (assistant) and `role=tool` entries; other roles are dropped.
        Without tools and without tool traffic this is the identity."""
        tools = [openai_tool(t) for t in (tools or [])]
        out: list[dict] = []
        i = 0
        saw_system = False
        last_calls: list[dict] = []
        while i < len(messages):
            m = messages[i]
            role = m.get("role")
            if role == "system":
                saw_system = True
                content = m.get("content") or ""
                out.append({"role": "system",
                            "content": self.baked_system(content, tools) if tools else content})
                i += 1
                continue
            if role == "tool":
                j = i
                while j < len(messages) and messages[j].get("role") == "tool":
                    j += 1
                out.append({"role": "user",
                            "content": self.baked_tool_results(messages[i:j], last_calls)})
                i = j
                continue
            if role == "assistant":
                calls = m.get("tool_calls") or []
                last_calls = list(calls)
                content = m.get("content") or ""
                out.append({"role": "assistant",
                            "content": self.baked_assistant(content, calls)
                            if calls else content})
                i += 1
                continue
            if role == "user":
                out.append({"role": "user", "content": m.get("content") or ""})
            i += 1
        if tools and not saw_system:
            out.insert(0, {"role": "system", "content": self.baked_system(None, tools)})
        return out

    def _normalize_role_markers(self, rendered: str) -> str:
        """Fold the template's native tool-result role marker onto `user`
        (glm: `<|observation|>` → `<|user|>`). Applied to BOTH sides of the
        parity comparison, so a literal marker inside content compares
        unchanged and only the block role is exempt."""
        role = self.style.tool_result_role
        if not role:
            return rendered
        return rendered.replace(self.style.start(role), self.style.start("user"))

    def parity_ok(self, messages: list[dict], tools: list[dict],
                  baked: list[dict], role_marker_exempt: bool | None = None) -> bool:
        """The admission gate: the baked plain-text conversation renders to
        exactly the bytes the structured, tools-aware conversation does.

        `role_marker_exempt` (cutover option (a), 2026-09-26): for a
        template whose tool results live under a role plain messages cannot
        express (glm `<|observation|>`), compare with that one role marker
        folded onto `user` on both sides — everything else stays
        byte-exact. None = the style's default (True for glm, False for
        qwen, where plain equality already holds)."""
        structured = []
        for m in messages:
            mm = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("tool_calls"):
                mm["tool_calls"] = [openai_tool_call(c) for c in m["tool_calls"]]
            for k in ("tool_call_id", "name"):
                if m.get("role") == "tool" and m.get(k):
                    mm[k] = m[k]
            structured.append(mm)
        want = self.render(structured, [openai_tool(t) for t in tools] or None)
        got = self.render(baked)
        if got == want:
            return True
        if role_marker_exempt is None:
            role_marker_exempt = self.style.tool_result_role is not None
        if not role_marker_exempt:
            return False
        return self._normalize_role_markers(got) == self._normalize_role_markers(want)

    def parity_diff(self, messages: list[dict], tools: list[dict],
                    baked: list[dict]) -> tuple[int, str, str] | None:
        """First differing offset and the two 80-char windows after the role
        marker exemption, or None when parity holds (debugging / the
        parity harness)."""
        structured = []
        for m in messages:
            mm = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("tool_calls"):
                mm["tool_calls"] = [openai_tool_call(c) for c in m["tool_calls"]]
            for k in ("tool_call_id", "name"):
                if m.get("role") == "tool" and m.get(k):
                    mm[k] = m[k]
            structured.append(mm)
        want = self._normalize_role_markers(self.render(structured, [openai_tool(t) for t in tools] or None))
        got = self._normalize_role_markers(self.render(baked))
        if want == got:
            return None
        i = next((k for k, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
        return i, want[i:i + 80], got[i:i + 80]
