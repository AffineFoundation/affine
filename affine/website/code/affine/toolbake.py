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

Requires `transformers` (tokenizer only, no weights): the datagen box runs
this at slice time; the eval pod never imports it.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from transformers import AutoTokenizer

CONTRACT_TOML = Path(__file__).resolve().parents[1] / "affine.toml"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"
_SENTINEL_USER = "\u2063affine-toolbake-user\u2063"


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
    shape with `arguments` as a dict (Qwen's template iterates the pairs)."""
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


def _block(rendered: str, role: str, occurrence: int = 0) -> str:
    """Body of the `occurrence`-th `<|im_start|>{role}\\n…<|im_end|>` block."""
    marker = f"{IM_START}{role}\n"
    pos = -1
    for _ in range(occurrence + 1):
        pos = rendered.index(marker, pos + 1)
    start = pos + len(marker)
    return rendered[start:rendered.index(IM_END, start)]


class ToolBaker:
    def __init__(self, tokenizer) -> None:
        self.tok = tokenizer
        # Whatever the template prepends to every system block on its own
        # (Qwen3.8: the reasoning-effort preamble). It is already part of
        # every bash duel prompt, so baked content must not repeat it.
        self.preamble = self._system_body([{"role": "user", "content": "u"}])

    @classmethod
    def from_pretrained(cls, repo: str | None = None, **kw) -> "ToolBaker":
        return cls(AutoTokenizer.from_pretrained(repo or teacher_repo(), **kw))

    def render(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        kw = {"tools": tools} if tools else {}
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **kw)

    def _system_body(self, messages: list[dict], tools=None) -> str:
        return _block(self.render(messages, tools), "system")

    # -- the three baked pieces --------------------------------------------------

    def baked_system(self, system_content: str | None, tools: list[dict]) -> str:
        """System content carrying the rendered tools block. Re-rendering it
        without `tools=` must reproduce the tools-aware render."""
        msgs = [{"role": "user", "content": "u"}]
        if system_content is not None:
            msgs.insert(0, {"role": "system", "content": system_content})
        body = self._system_body(msgs, tools)
        lead = self.preamble + "\n\n"
        if not body.startswith(lead):
            raise ValueError("template system block does not start with the "
                             "bare preamble; baking rule no longer applies")
        return body[len(lead):]

    def baked_assistant(self, content: str, tool_calls: list[dict]) -> str:
        """Assistant content with its tool calls rendered as the template's
        <tool_call> text (content first, then the calls)."""
        msgs = [{"role": "user", "content": "u"},
                {"role": "assistant", "content": content,
                 "tool_calls": [openai_tool_call(c) for c in tool_calls]}]
        body = _block(self.render(msgs), "assistant")
        if body.startswith(EMPTY_THINK):
            body = body[len(EMPTY_THINK):]
        return body

    def baked_tool_results(self, results: list[dict]) -> str:
        """One user turn standing in for a run of consecutive `role=tool`
        messages (the template merges them into a single user block)."""
        msgs = [{"role": "user", "content": _SENTINEL_USER},
                {"role": "assistant", "content": "a"},
                *[{"role": "tool", "content": r.get("content") or "",
                   **({"tool_call_id": r["tool_call_id"]} if r.get("tool_call_id") else {}),
                   **({"name": r["name"]} if r.get("name") else {})}
                  for r in results]]
        return _block(self.render(msgs), "user", occurrence=1)

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
        while i < len(messages):
            m = messages[i]
            role = m.get("role")
            if role == "system":
                saw_system = True
                content = m.get("content") or ""
                out.append({"role": "system",
                            "content": self.baked_system(content, tools)
                            if tools else content})
                i += 1
                continue
            if role == "tool":
                j = i
                while j < len(messages) and messages[j].get("role") == "tool":
                    j += 1
                out.append({"role": "user",
                            "content": self.baked_tool_results(messages[i:j])})
                i = j
                continue
            if role == "assistant":
                calls = m.get("tool_calls") or []
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
            out.insert(0, {"role": "system",
                           "content": self.baked_system(None, tools)})
        return out

    def parity_ok(self, messages: list[dict], tools: list[dict],
                  baked: list[dict]) -> bool:
        """The admission gate: the baked plain-text conversation renders to
        exactly the bytes the structured, tools-aware conversation does."""
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
        return self.render(baked) == want
