"""Shared pieces of the coached-teacher recovery run.

Terms
  state        one prefix of a failed king rollout (everything the king saw
               before turn T) that a labeler flagged: loop onset, judged pivot,
               or a plain-teacher failure in the recoverable side-table.
  continuation the teacher run from the state to the end of the episode under
               the task's own harness (ops/recoverable plugin), or — ACP
               harnesses — a fresh run of the whole task (same-task proxy).
  coach        a frontier model that reads the king's WHOLE failed rollout (with
               outcome and, where judged, the reviewer's post-mortem) plus the
               teacher's continuation so far, and writes a short note the
               teacher reads at its next step. The note is appended to the last
               user / tool message of the request only; the harness never
               stores it, so the recorded trace stays hint-free.
  hint levels  fact (what visible facts matter, what to stop), plan (outcome-
               aware next aim, no literal command) and action (one concrete
               step) — the E1 levels; INJECT_DEFAULT says which are injected.
  gates        grounding: every entity the note names occurs in the teacher's
               visible context; leak: no 6-word window of the note comes from a
               king action the teacher has not seen (research/hints/hints.py).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import hints as H  # noqa: E402  (research/hints/hints.py: gates + OpenRouter call)

HINT_HEADER = "\n\n[Reviewer note]\n"
INJECT_DEFAULT = "fact+plan"
PROMPT_VERSION = "coach-v1"
THOUGHT_HEAD = 240
ACTION_HEAD = 500
OBS_HEAD = 400
KING_TRANSCRIPT_CAP = 45_000
CONT_TRANSCRIPT_CAP = 30_000
WS = re.compile(r"\s+")

COACH_SYSTEM = """You are a senior engineer coaching an AI agent in real time, with hindsight.

You see three things:
1. THE TASK.
2. A FAILED RUN: an earlier attempt at the same task by a weaker agent, turn by turn (thought head, action, observation head), its recorded outcome, and where available a reviewer's post-mortem. Turns marked VISIBLE are part of the current agent's own context; turns marked FUTURE are hindsight only — the current agent has never seen them.
3. THE CURRENT RUN: what the agent you are coaching has done since it took over. It is about to act again.

Write one coaching note the current agent will read, appended to its latest observation. Answer as a JSON object with keys "fact", "plan", "action":
- "fact": 1-3 sentences. Which facts ALREADY VISIBLE to the current agent matter now (a file, a test result, an error line, an observation), and what it should STOP doing (repeating a dead end of the failed run, repeating its own last command, exploring instead of editing, checking instead of finishing). No new information, no answer, no command.
- "plan": 2-3 sentences. What to aim for next at the level of a plan: which area to change or check, what to verify, and WHEN TO FINISH — if the work is done, say plainly that it should stop and submit / give its final answer now. No literal command, no patch text, no code.
- "action": 1 sentence. The concrete next step (may name a command, an edit, or "submit").
Rules:
- Every file, symbol, test name, number or observation you mention must already appear in the current agent's visible context (the failed run's VISIBLE turns and the current run). Do not quote or paraphrase text from FUTURE turns of the failed run; use them only to know which paths are dead ends and what the task actually needs.
- Never mention the failed run, the reviewer, hindsight, turn numbers, or the outcome. Write as a colleague looking over the shoulder.
- Plain language, no markdown, no bullet points. Output ONLY the JSON object."""

COACH_USER_TMPL = """=== THE TASK (head) ===
{task}

=== THE FAILED RUN ({visible_note}) ===
{king_transcript}
{post_mortem}
=== THE CURRENT RUN (the agent you coach; {cont_note}) ===
{cont_transcript}

Write the JSON object with "fact", "plan" and "action" for the current agent's next step."""

REWRITE_TMPL = """Your note names things the current agent cannot see in its context: {missing}.
Rewrite the JSON object so that every file, symbol, test name, number or quoted span it mentions appears in the agent's visible context, keeping the same intent. Output ONLY the JSON object."""


def norm(s: str) -> str:
    return WS.sub(" ", s or "").strip()


def sha(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", "replace")).hexdigest()[:16]


def message_text(content) -> str:
    """Flatten chat / Anthropic content (string or list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if not isinstance(part, dict):
                out.append(str(part))
                continue
            if part.get("type") in ("text", "input_text", "output_text"):
                out.append(str(part.get("text") or ""))
            elif part.get("type") == "tool_result":
                out.append(message_text(part.get("content")))
            elif part.get("type") == "tool_use":
                out.append(f"{part.get('name')}({json.dumps(part.get('input') or {})})")
            elif "text" in part:
                out.append(str(part["text"]))
        return "\n".join(out)
    return str(content)


def wire_reply_parts(m: dict) -> tuple[str, str]:
    """(thought, action) of one assistant wire message (OpenAI chat or
    Anthropic shapes)."""
    thought = str(m.get("reasoning_content") or m.get("reasoning") or "").strip()
    content = m.get("content")
    text_parts: list[str] = []
    calls: list[str] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "thinking":
                thought = (thought + "\n" + str(part.get("thinking") or "")).strip()
            elif part.get("type") == "tool_use":
                calls.append(f"{part.get('name')}({json.dumps(part.get('input') or {})})")
            elif part.get("type") in ("text", "output_text") or "text" in part:
                text_parts.append(str(part.get("text") or ""))
        text = "\n".join(text_parts)
    else:
        text = str(content or "")
    if "<think>" in text and "</think>" in text:
        t2, _, text = text.partition("</think>")
        thought = (thought + "\n" + t2.replace("<think>", "")).strip()
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        calls.append(f"{fn.get('name') or tc.get('name', '?')}({fn.get('arguments') or tc.get('arguments', '')})")
    action = text.strip()
    if calls:
        action = (action + "\n" if action else "") + "\n".join(calls)
    return thought, action


def compress_wire(messages: list[dict], start: int, cap: int = CONT_TRANSCRIPT_CAP) -> tuple[str, int]:
    """Compressed transcript of `messages[start:]`: one block per assistant
    reply (thought head, action, following observation head). Returns
    (text, n_replies)."""
    lines: list[str] = []
    n = 0
    i = start
    msgs = messages
    while i < len(msgs):
        m = msgs[i]
        if m.get("role") != "assistant":
            i += 1
            continue
        thought, action = wire_reply_parts(m)
        obs = []
        j = i + 1
        while j < len(msgs) and msgs[j].get("role") in ("user", "tool"):
            obs.append(message_text(msgs[j].get("content")))
            j += 1
        lines.append(f"--- step {n} ---")
        if thought:
            lines.append(f"thought: {norm(thought)[:THOUGHT_HEAD]}")
        lines.append(f"action: {action.strip()[:ACTION_HEAD]}")
        if obs:
            lines.append(f"observation: {norm(chr(10).join(obs))[:OBS_HEAD]}")
        n += 1
        i = j
    if not lines:
        return "(nothing yet — this is the agent's first step)", 0
    text = "\n".join(lines)
    if len(text) > cap:
        text = text[: cap // 3] + "\n[... middle of the run elided ...]\n" + text[-2 * cap // 3:]
    return text, n


def king_transcript_marked(transcript: str, turn_idx: int | None) -> str:
    """Mark the visible / future boundary in the hindsight transcript
    (research/hints/turnset.hindsight format: `--- turn i ---`)."""
    if turn_idx is None:
        return transcript
    marker = f"--- turn {turn_idx} ---"
    head = "[VISIBLE to the current agent: turns below this line up to the boundary]\n"
    if marker in transcript:
        return head + transcript.replace(
            marker, f"=========== BOUNDARY: the current agent took over here; turns below are "
                    f"FUTURE of the failed run (hindsight only) ===========\n{marker}", 1)
    return head + transcript + "\n=========== (the failed run ended before the boundary) ==========="


def build_coach_messages(ctx: dict, cont_transcript: str, n_cont: int) -> list[dict]:
    resume = ctx.get("resume_kind") != "same_task"
    if resume:
        visible_note = (f"turns before the boundary are VISIBLE to the current agent, "
                        f"which took over at turn {ctx['turn_idx']}; turns after it are FUTURE")
        king_text = king_transcript_marked(ctx["king_transcript"], int(ctx["turn_idx"]))
        cont_note = f"{n_cont} step(s) since it took over"
    else:
        visible_note = ("the current agent started the task from scratch and has seen NONE of "
                        "this run; every turn of it is FUTURE / hindsight only")
        king_text = ctx["king_transcript"]
        cont_note = f"{n_cont} step(s) since the start"
    pm = ""
    piv = ctx.get("pivot") or {}
    if piv.get("rationale") or piv.get("should_have"):
        pm = ("\nReviewer's post-mortem of the failed run (hindsight): "
              f"category {piv.get('failure_category')}; {piv.get('rationale') or ''} "
              f"It should have: {piv.get('should_have') or ''}\n")
    user = COACH_USER_TMPL.format(task=(ctx.get("task_prompt") or "")[:4000],
                                  visible_note=visible_note, king_transcript=king_text,
                                  post_mortem=pm, cont_note=cont_note,
                                  cont_transcript=cont_transcript)
    return [{"role": "system", "content": COACH_SYSTEM}, {"role": "user", "content": user}]


def compose_note(levels: dict, inject: str = INJECT_DEFAULT) -> str:
    parts = [levels.get(k, "") for k in inject.split("+") if levels.get(k)]
    return " ".join(p.strip() for p in parts if p).strip()


def gate(note: str, x_text: str, future_actions: list[str]) -> dict:
    g = H.grounding_check(note, x_text)
    leak = H.leak_check(note, future_actions, x_text)
    return {"grounding": g, "leak": leak, "n_sentences": H.sentence_count(note),
            "n_chars": len(note), "passed": bool(g["grounded"] and not leak["leaks_future"])}


def unit_stem(unit: str) -> str:
    return unit.replace(":", "_")
