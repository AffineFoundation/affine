"""Chat-protocol conformance probe — an ADMISSION check, not a score.

Question it answers: "does this checkpoint behave like a chat model when an
OpenAI-compatible client (Cursor, the chat pod, any SDK) talks to it?"
Concretely, with thinking on and a system prompt present, does every reply

  1. close its reasoning block — emit </think> — so a reasoning parser can
     separate hidden thought from the visible answer, and
  2. put a non-empty answer after it (text, or a well-formed tool call when
     the prompt calls for one)?

Why this exists (2026-09-07): the reign-8 king renders as an EMPTY message
in Cursor. With a system prompt it never emits </think>; the whole reply,
answer included, stays inside the think block, and vLLM's reasoning parser
files it all as `reasoning`. Nothing in the contract asked for the tag
(split_rollout treats it as optional; the scored body is built with our
own </think>), and D holds only agent-scaffold prompts, so miners drifted
off the chat protocol and we crowned them. Bench transcripts: genesis
closes </think> on 95–98% of replies, the teacher on 99%, kings of reigns
1–5 on 0–4%.

Precedent: the architecture pin — an admission rule, not a scoring change,
so no weight_version_key bump. Runs on the eval pod once the challenger is
served (it needs the model on a GPU), right after probe_injectable and
BEFORE the 1,300-turn scoring, so a rejected model costs minutes, not an
hour of teacher echoes. The eval slot itself is burned at enqueue by the
existing 1-hotkey-1-eval policy; that policy is untouched here.

Modes ([protocol_probe].mode in affine.toml):
  off      — nothing runs (default; staged)
  shadow   — runs, result published on the verdict, never rejects
  enforce  — runs; pass_rate < min_pass_rate rejects with
             rejection_reason = "protocol:<detail>"

The prompt set is fixed and public (below). Every prompt is Cursor-shaped
or plain-assistant-shaped — deliberately NOT the SWE scaffold D is made
of, because the failure is off-distribution collapse. Prompts are rendered
through the model's own chat template with thinking on and tool schemas
where the prompt has tools, exactly like a client would.

CLI (dry run against any OpenAI-compatible endpoint, e.g. the private
king pod; needs the server to expose raw text — pass --enable-thinking
when the server defaults thinking off):

    python -m evalsrv.protocol_probe --base-url https://host/v1 \
        --model affine-king --api-key $KEY --enable-thinking
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys

import httpx

from affine import dialects

from .chat import THINK_CLOSE

log = logging.getLogger("evalsrv.protocol_probe")

MODES = ("off", "shadow", "enforce")

# -- fixed prompt set ----------------------------------------------------------
# A compact IDE-agent system prompt in the shape clients send: identity,
# tool rules, communication rules. Not any vendor's verbatim text.
IDE_SYSTEM = (
    "You are an AI coding assistant, powered by a language model. You "
    "operate in an IDE and are pair programming with a USER to solve their "
    "coding task. You have tools available; when you need to inspect or "
    "change files, call exactly one tool per message and wait for its "
    "result before continuing. Never mention tool names to the USER. Keep "
    "answers concise. Use markdown; put file, directory and function names "
    "in backticks."
)
ASSISTANT_SYSTEM = "You are a helpful assistant."

IDE_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to the file."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List the entries of a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep_search",
        "description": "Search file contents with a regular expression.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Directory to search."}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "run_terminal_cmd",
        "description": "Run a shell command in the project directory.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}},
            "required": ["command"]}}},
]

# expects: "answer"  — non-empty visible text after </think>, no tool call
#          "tool"    — exactly one well-formed <tool_call> block after </think>
#          "any"     — either (the model may reasonably answer or call)
PROMPTS: list[dict] = [
    {"id": "ide_greeting", "expects": "answer",
     "messages": [{"role": "system", "content": IDE_SYSTEM},
                  {"role": "user", "content": "hello"}]},
    {"id": "ide_code_question", "expects": "answer",
     "messages": [{"role": "system", "content": IDE_SYSTEM},
                  {"role": "user", "content":
                   "Write a Python function that returns the n-th Fibonacci "
                   "number iteratively. Keep it short."}]},
    {"id": "ide_explain_snippet", "expects": "answer",
     "messages": [{"role": "system", "content": IDE_SYSTEM},
                  {"role": "user", "content":
                   "What does this do?\n\n```python\nfrom collections import "
                   "Counter\nc = Counter(words)\nprint(c.most_common(3))\n```"}]},
    {"id": "ide_followup", "expects": "answer",
     "messages": [{"role": "system", "content": IDE_SYSTEM},
                  {"role": "user", "content": "Which is faster in Python for "
                   "membership tests, a list or a set?"},
                  {"role": "assistant", "content": "A set. Membership on a set "
                   "is O(1) on average; on a list it is O(n)."},
                  {"role": "user", "content": "And a tuple?"}]},
    {"id": "ide_tool_read", "expects": "tool", "tools": IDE_TOOLS,
     "messages": [{"role": "system", "content": IDE_SYSTEM},
                  {"role": "user", "content":
                   "Open `src/app.py` and tell me what the main function does."}]},
    {"id": "ide_tool_search", "expects": "tool", "tools": IDE_TOOLS,
     "messages": [{"role": "system", "content": IDE_SYSTEM},
                  {"role": "user", "content":
                   "Find every place in the repo where `load_config` is called."}]},
    {"id": "ide_tool_answer_or_call", "expects": "any", "tools": IDE_TOOLS,
     "messages": [{"role": "system", "content": IDE_SYSTEM},
                  {"role": "user", "content":
                   "Run the tests and summarize the failures."}]},
    {"id": "assistant_greeting", "expects": "answer",
     "messages": [{"role": "system", "content": ASSISTANT_SYSTEM},
                  {"role": "user", "content": "hi there"}]},
    {"id": "assistant_bash", "expects": "answer",
     "messages": [{"role": "system", "content": ASSISTANT_SYSTEM},
                  {"role": "user", "content":
                   "Give me a bash one-liner that counts the lines in every "
                   "`.py` file under the current directory."}]},
    {"id": "no_system_greeting", "expects": "answer",
     "messages": [{"role": "user", "content": "hello"}]},
    # "code_block" cases (2026-09-27): reign 22's HumanEval fell 74.4 → 57.9
    # on a format habit that grew along the lineage — replies end with a
    # lone closing ``` and no opening ```python (close-only replies genesis
    # 0 → r20 4 → r21 35 → r22 61 of 164, correct code inside). The A leg is
    # blind to a few fence bytes; this asks for the fence directly. Pass =
    # exactly one balanced fenced block with a language tag (ide case); the
    # HumanEval-shaped case ("only the code") also accepts bare code — the
    # genesis answers it unfenced on 87 % of tasks — and fails only on a
    # malformed fence (close-only / unbalanced / untagged / several).
    {"id": "ide_code_block_only", "expects": "code_block",
     "messages": [{"role": "system", "content": IDE_SYSTEM},
                  {"role": "user", "content":
                   "Reply with only a Python code block — no text before or "
                   "after it. Implement `def is_palindrome(s: str) -> bool` "
                   "that ignores case and non-alphanumeric characters."}]},
    {"id": "assistant_code_block_only", "expects": "code_only",
     "messages": [{"role": "system", "content": ASSISTANT_SYSTEM},
                  {"role": "user", "content":
                   "Read the following function signature and docstring, and "
                   "fully implement the function described. Your response "
                   "should only contain the code for this function.\n"
                   "from typing import List\n\n\n"
                   "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
                   "    \"\"\" Check if in given list of numbers, are any two numbers closer "
                   "to each other than\n    given threshold.\n"
                   "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n"
                   "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n"
                   "    \"\"\"\n"}]},
    # Four more "only the code" tasks (public HumanEval prompts, verbatim from
    # the bench): the ones where the close-only habit showed most across the
    # reign-21/22 lineage (9/10, 8/10, 7/10, 7/10 of the last ten benched
    # models). One fixed task under-detects (reign 22 live: 1–2 of 12 replies
    # close-only on HumanEval/0 vs 37 % on the 164-task bench).
    {"id": "assistant_code_only_he70", "expects": "code_only",
     "messages": [{"role": "system", "content": ASSISTANT_SYSTEM},
                  {"role": "user", "content": "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n\ndef strange_sort_list(lst):\n    '''\n    Given list of integers, return list in strange order.\n    Strange sorting, is when you start with the minimum value,\n    then maximum of the remaining integers, then minimum and so on.\n\n    Examples:\n    strange_sort_list([1, 2, 3, 4]) == [1, 4, 2, 3]\n    strange_sort_list([5, 5, 5, 5]) == [5, 5, 5, 5]\n    strange_sort_list([]) == []\n    '''\n"}]},
    {"id": "assistant_code_only_he134", "expects": "code_only",
     "messages": [{"role": "system", "content": ASSISTANT_SYSTEM},
                  {"role": "user", "content": "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n\ndef check_if_last_char_is_a_letter(txt):\n    '''\n    Create a function that returns True if the last character\n    of a given string is an alphabetical character and is not\n    a part of a word, and False otherwise.\n    Note: \"word\" is a group of characters separated by space.\n\n    Examples:\n    check_if_last_char_is_a_letter(\"apple pie\") \u279e False\n    check_if_last_char_is_a_letter(\"apple pi e\") \u279e True\n    check_if_last_char_is_a_letter(\"apple pi e \") \u279e False\n    check_if_last_char_is_a_letter(\"\") \u279e False \n    '''\n"}]},
    {"id": "assistant_code_only_he69", "expects": "code_only",
     "messages": [{"role": "system", "content": ASSISTANT_SYSTEM},
                  {"role": "user", "content": "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n\ndef search(lst):\n    '''\n    You are given a non-empty list of positive integers. Return the greatest integer that is greater than \n    zero, and has a frequency greater than or equal to the value of the integer itself. \n    The frequency of an integer is the number of times it appears in the list.\n    If no such a value exist, return -1.\n    Examples:\n        search([4, 1, 2, 2, 3, 1]) == 2\n        search([1, 2, 2, 3, 3, 3, 4, 4, 4]) == 3\n        search([5, 5, 4, 4, 4]) == -1\n    '''\n"}]},
    {"id": "assistant_code_only_he103", "expects": "code_only",
     "messages": [{"role": "system", "content": ASSISTANT_SYSTEM},
                  {"role": "user", "content": "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n\ndef rounded_avg(n, m):\n    \"\"\"You are given two positive integers n and m, and your task is to compute the\n    average of the integers from n through m (including n and m). \n    Round the answer to the nearest integer and convert that to binary.\n    If n is greater than m, return -1.\n    Example:\n    rounded_avg(1, 5) => \"0b11\"\n    rounded_avg(7, 5) => -1\n    rounded_avg(10, 20) => \"0b1111\"\n    rounded_avg(20, 33) => \"0b11010\"\n    \"\"\"\n"}]},
]

# Prompt ids whose results are published but do not count toward pass_rate /
# passed ([protocol_probe].shadow_ids). A new case starts here, gets read on a
# few verdicts, and is promoted by removing it from the list (admission rule,
# no weight_version_key event).
DEFAULT_SHADOW_IDS = ("ide_code_block_only", "assistant_code_block_only",
                      "assistant_code_only_he70", "assistant_code_only_he134",
                      "assistant_code_only_he69", "assistant_code_only_he103")

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*([^`~\s]*)[^\n]*$")


def check_code_block(visible: str, *, fence_required: bool = True) -> dict:
    """Fence structure of a visible reply. Returns {ok, reasons, n_fences,
    n_blocks, language}. Pass = exactly one balanced fenced block whose opening
    fence carries a language tag and whose body is non-empty. With
    fence_required=False a reply with NO fence at all also passes (bare code
    is a fine answer to "only the code"; the genesis answers HumanEval that way
    on 87 % of tasks) — only malformed fences fail. Reasons:
      no_code_block        no fence line at all (bare code, or prose)
      close_only_fence     a single bare fence (the lineage habit: code, then ```)
      unbalanced_fence     an opening with no close / a tagged fence where a
                           close was expected / an odd leftover
      fence_no_language    a balanced block whose opening fence has no tag
      multiple_code_blocks more than one balanced block
      empty_code_block     the block has no content lines"""
    lines = visible.split("\n")
    fences: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _FENCE_RE.match(line)
        if m:
            fences.append((i, m.group(2)))
    reasons: list[str] = []
    blocks: list[tuple[int, int, str]] = []
    if not fences:
        if fence_required:
            reasons.append("no_code_block")
    elif len(fences) == 1:
        reasons.append("close_only_fence" if not fences[0][1] else "unbalanced_fence")
    else:
        open_: tuple[int, str] | None = None
        for idx, tag in fences:
            if open_ is None:
                open_ = (idx, tag)
            elif tag and open_[1]:
                # two tagged fences in a row: the first block never closed
                reasons.append("unbalanced_fence")
                open_ = (idx, tag)
            else:
                blocks.append((open_[0], idx, open_[1]))
                open_ = None
        if open_ is not None:
            reasons.append("unbalanced_fence")
        if len(blocks) > 1:
            reasons.append("multiple_code_blocks")
        for start, end, tag in blocks[:1]:
            if not tag:
                reasons.append("fence_no_language")
            if not any(l.strip() for l in lines[start + 1:end]):
                reasons.append("empty_code_block")
    reasons = list(dict.fromkeys(reasons))
    return {"ok": not reasons, "reasons": reasons, "n_fences": len(fences),
            "n_blocks": len(blocks), "language": blocks[0][2] if blocks else None}


# -- evaluation (pure) ----------------------------------------------------------
def evaluate(text: str, expects: str, *, think_stripped: bool = False,
             reasoning: str | None = None,
             structured_tool_calls: int = 0) -> dict:
    """Judge one raw completion. Returns {ok, reasons, think_closed,
    content_chars, tool_calls}.

    text: the completion as generated after the open <think> (the duel path)
    — or, for servers that run a reasoning parser (think_stripped=True), the
    visible `content` with `reasoning` supplied separately and any parsed
    `tool_calls` counted in structured_tool_calls.
    """
    reasons: list[str] = []
    if think_stripped:
        # A parser-side split cannot show the tag. The parser only emits
        # visible content / tool calls once it has seen </think>, so any
        # visible output means the block was closed; reasoning-only means
        # it was not (exactly the empty-Cursor-reply failure).
        del reasoning
        closed = bool(text.strip()) or structured_tool_calls > 0
        visible = text
    else:
        closed = THINK_CLOSE in text
        visible = text.split(THINK_CLOSE, 1)[1] if closed else ""
    if not closed:
        reasons.append("no_think_close")
    n_tool = structured_tool_calls + dialects.count_actions(visible, "tool_call")
    content = visible
    if n_tool:
        before, _ = dialects.split_action(visible, "tool_call")
        content = before
    content_chars = len(content.strip())
    if expects == "answer":
        if n_tool:
            reasons.append("unexpected_tool_call")
        if content_chars == 0 and closed:
            reasons.append("empty_content")
    elif expects == "tool":
        if n_tool == 0:
            reasons.append("no_tool_call")
        elif n_tool > 1:
            reasons.append("multiple_tool_calls")
    elif expects == "any":
        if n_tool == 0 and content_chars == 0 and closed:
            reasons.append("empty_content")
        if n_tool > 1:
            reasons.append("multiple_tool_calls")
    elif expects in ("code_block", "code_only"):
        # code_block: the prompt asked for a fenced block → exactly one, tagged.
        # code_only : the prompt asked for "only the code" → bare code or one
        #             tagged block; any malformed fence (close-only, unbalanced,
        #             untagged, several) fails.
        if n_tool:
            reasons.append("unexpected_tool_call")
        if closed:
            reasons.extend(check_code_block(
                content, fence_required=(expects == "code_block"))["reasons"])
    else:
        raise ValueError(f"unknown expects {expects!r}")
    return {"ok": not reasons, "reasons": reasons, "think_closed": closed,
            "content_chars": content_chars, "tool_calls": n_tool}


def _rates(results: list[dict]) -> dict:
    n = len(results)
    n_ok = sum(1 for r in results if r["ok"])
    by_reason: dict[str, int] = {}
    for r in results:
        for why in r["reasons"]:
            by_reason[why] = by_reason.get(why, 0) + 1
    return {"n": n, "n_ok": n_ok, "pass_rate": n_ok / n if n else 0.0,
            "think_close_rate": (sum(1 for r in results if r["think_closed"]) / n if n else 0.0),
            "by_reason": by_reason}


def summarize(results: list[dict], min_pass_rate: float,
              shadow_ids: tuple[str, ...] | list[str] = ()) -> dict:
    """pass_rate / passed over the ENFORCED prompts; results whose id is in
    shadow_ids are summarised separately under `shadow` (published, never
    decide). Results without an id (unit tests) count as enforced."""
    shadow = set(shadow_ids)
    enforced = [r for r in results if r.get("id") not in shadow]
    shadowed = [r for r in results if r.get("id") in shadow]
    out = _rates(enforced)
    out.update({
        "min_pass_rate": min_pass_rate,
        "passed": out["n"] > 0 and out["pass_rate"] >= min_pass_rate,
        "results": results,
    })
    if shadow:
        out["shadow"] = {"ids": sorted(shadow), **_rates(shadowed)}
    return out


def rejection_detail(summary: dict) -> str:
    top = sorted(summary["by_reason"].items(), key=lambda kv: -kv[1])[:2]
    why = ",".join(f"{k}={v}" for k, v in top) or "none"
    return (f"pass_rate={summary['pass_rate']:.2f}<{summary['min_pass_rate']:g}"
            f";{why}")


# -- eval-pod runner --------------------------------------------------------------
def probe_settings(raw: dict | None) -> dict:
    """[protocol_probe] with defaults. mode off = staged."""
    p = dict(raw or {})
    mode = str(p.get("mode", "off"))
    if mode not in MODES:
        raise ValueError(f"[protocol_probe].mode must be one of {MODES}, got {mode!r}")
    known = {pr["id"] for pr in PROMPTS}
    shadow_ids = tuple(str(x) for x in p.get("shadow_ids", list(DEFAULT_SHADOW_IDS)))
    unknown = [x for x in shadow_ids if x not in known]
    if unknown:
        raise ValueError(f"[protocol_probe].shadow_ids names unknown prompts: {unknown}")
    return {
        "mode": mode,
        "min_pass_rate": float(p.get("min_pass_rate", 0.9)),
        "n_samples": int(p.get("n_samples", 2)),
        "temperature": float(p.get("temperature", 0.7)),
        "max_tokens": int(p.get("max_tokens", 1024)),
        # prompt ids published but not counted toward pass_rate / passed
        "shadow_ids": shadow_ids,
        # completions per SHADOW prompt (0/absent = n_samples); a finer read per
        # verdict while a case is being judged for promotion (2026-09-27: 4).
        "shadow_n_samples": int(p.get("shadow_n_samples", 0) or 0),
    }


async def run_probe(model, settings: dict) -> dict:
    """Run the fixed prompt set against a served model (ModelPool / VllmModel
    with .complete). Returns summarize(...) plus the settings used."""
    sem = asyncio.Semaphore(8)

    async def one(prompt: dict, k: int) -> dict:
        async with sem:
            text = await model.complete(
                prompt["messages"], settings["temperature"],
                settings["max_tokens"], tools=prompt.get("tools"))
        r = evaluate(text, prompt["expects"])
        r.update({"id": prompt["id"], "sample": k,
                  "text_head": text[:200]})
        return r

    shadow = set(settings.get("shadow_ids", ()))
    n_shadow = int(settings.get("shadow_n_samples", 0) or 0) or settings["n_samples"]
    results = await asyncio.gather(*[
        one(p, k) for p in PROMPTS
        for k in range(n_shadow if p["id"] in shadow else settings["n_samples"])])
    out = summarize(list(results), settings["min_pass_rate"], settings.get("shadow_ids", ()))
    out["settings"] = {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in settings.items() if k != "mode"}
    out["mode"] = settings["mode"]
    return out


# -- CLI: any OpenAI-compatible endpoint --------------------------------------------
async def _cli(args: argparse.Namespace) -> int:
    settings = {"min_pass_rate": args.min_pass_rate, "n_samples": args.n_samples,
                "temperature": args.temperature, "max_tokens": args.max_tokens}
    headers = {"Authorization": f"Bearer {args.api_key}",
               "Content-Type": "application/json", "User-Agent": "affine-probe"}
    results: list[dict] = []
    prompts = [p for p in PROMPTS if not args.only or p["id"] in set(args.only)]
    shadow = set(args.shadow_ids or ())
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as http:
        for prompt in prompts:
            n_this = (args.shadow_n_samples or args.n_samples) if prompt["id"] in shadow else args.n_samples
            for k in range(n_this):
                body = {"model": args.model, "messages": prompt["messages"],
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "skip_special_tokens": False}
                if prompt.get("tools"):
                    body["tools"] = prompt["tools"]
                if args.enable_thinking:
                    body["chat_template_kwargs"] = {"enable_thinking": True}
                r = await http.post(f"{args.base_url.rstrip('/')}/chat/completions",
                                    json=body, headers=headers)
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning") or msg.get("reasoning_content")
                stripped = reasoning is not None and THINK_CLOSE not in content
                res = evaluate(content, prompt["expects"], think_stripped=stripped,
                               reasoning=reasoning,
                               structured_tool_calls=len(msg.get("tool_calls") or []))
                res.update({"id": prompt["id"], "sample": k,
                            "text_head": content[:200]})
                results.append(res)
                flag = "ok " if res["ok"] else "BAD"
                print(f"{flag} {prompt['id']:26s} #{k} {','.join(res['reasons']) or '-'}"
                      f"  content_chars={res['content_chars']} tool_calls={res['tool_calls']}")
    summary = summarize(results, args.min_pass_rate, tuple(args.shadow_ids or ()))
    summary["settings"] = settings
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=1))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=1)
    return 0 if summary["passed"] else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible base incl. /v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=os.environ.get("KING_API_KEY", "x"))
    ap.add_argument("--enable-thinking", action="store_true",
                    help="send chat_template_kwargs.enable_thinking=true")
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--min-pass-rate", type=float, default=0.9)
    ap.add_argument("--shadow-ids", nargs="*", default=list(DEFAULT_SHADOW_IDS),
                    help="prompt ids reported but not counted (default: the code_block cases)")
    ap.add_argument("--only", nargs="*", default=None, help="run only these prompt ids")
    ap.add_argument("--shadow-n-samples", type=int, default=0, help="samples per shadow prompt (0 = --n-samples)")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(_cli(args)))


if __name__ == "__main__":
    main()
