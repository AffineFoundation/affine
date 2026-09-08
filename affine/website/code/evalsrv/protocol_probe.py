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
]


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
    else:
        raise ValueError(f"unknown expects {expects!r}")
    return {"ok": not reasons, "reasons": reasons, "think_closed": closed,
            "content_chars": content_chars, "tool_calls": n_tool}


def summarize(results: list[dict], min_pass_rate: float) -> dict:
    n = len(results)
    n_ok = sum(1 for r in results if r["ok"])
    pass_rate = n_ok / n if n else 0.0
    by_reason: dict[str, int] = {}
    for r in results:
        for why in r["reasons"]:
            by_reason[why] = by_reason.get(why, 0) + 1
    return {
        "n": n, "n_ok": n_ok, "pass_rate": pass_rate,
        "min_pass_rate": min_pass_rate,
        "passed": n > 0 and pass_rate >= min_pass_rate,
        "think_close_rate": (sum(1 for r in results if r["think_closed"]) / n
                             if n else 0.0),
        "by_reason": by_reason,
        "results": results,
    }


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
    return {
        "mode": mode,
        "min_pass_rate": float(p.get("min_pass_rate", 0.9)),
        "n_samples": int(p.get("n_samples", 2)),
        "temperature": float(p.get("temperature", 0.7)),
        "max_tokens": int(p.get("max_tokens", 1024)),
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

    results = await asyncio.gather(*[
        one(p, k) for p in PROMPTS for k in range(settings["n_samples"])])
    out = summarize(list(results), settings["min_pass_rate"])
    out["settings"] = {k: v for k, v in settings.items() if k != "mode"}
    out["mode"] = settings["mode"]
    return out


# -- CLI: any OpenAI-compatible endpoint --------------------------------------------
async def _cli(args: argparse.Namespace) -> int:
    settings = {"min_pass_rate": args.min_pass_rate, "n_samples": args.n_samples,
                "temperature": args.temperature, "max_tokens": args.max_tokens}
    headers = {"Authorization": f"Bearer {args.api_key}",
               "Content-Type": "application/json", "User-Agent": "affine-probe"}
    results: list[dict] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as http:
        for prompt in PROMPTS:
            for k in range(args.n_samples):
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
    summary = summarize(results, args.min_pass_rate)
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
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(_cli(args)))


if __name__ == "__main__":
    main()
