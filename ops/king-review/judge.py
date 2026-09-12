"""Step 2 -- one structured judgment per sampled rollout from a long-context LLM.

Two-stage conversation per rollout so the judge commits to the pivot BEFORE
it knows the grade:
  stage 1 (blind): task + turn-indexed transcript -> pivotal turns (1-3, each
          with a one-line rationale and a confidence), trajectory summary,
          the repeated action if any
  stage 2 (revealed): outcome + stop condition + the teacher's rollout on the
          same task (compressed) -> failure_category (fixed list), free text,
          what_teacher_did_differently, recoverable_from_pivot

Every response is cached in `<out-dir>/cache/judgments.jsonl`, keyed by
rollout id + a hash of (model, prompt version, stage-1 prompt), so a re-run
costs nothing. Token usage and USD cost are logged per call (OpenRouter's
`usage.cost`) and summed in `<out-dir>/cost.json`; `--max-usd` stops
launching new rollouts once the sum passes the budget.

  OPENROUTER_API_KEY=... python judge.py --sample sample.jsonl --out-dir <dir> \
      --model deepseek/deepseek-v4-pro-0813 --concurrency 6 --max-usd 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx

from krlib import (TraceStore, clip, load_env_groups, norm_ws, parse_rollout,
                   read_jsonl, render_teacher_summary, render_transcript,
                   sha256_hex)

PROMPT_VERSION = "kr-v2"
DEFAULT_MODEL = "deepseek/deepseek-v4-pro-0813"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TRANSCRIPT_CHARS = 160_000

CATEGORIES = {
    "loop_after_ok": "repeats a command or check after an OK result instead of "
                     "moving on or finishing (done-blind)",
    "ignores_error": "sees an error / failing test and continues as if it passed, "
                     "or repeats the failing action unchanged",
    "wrong_plan": "a coherent but wrong approach (wrong file, wrong root cause, "
                  "wrong method) carried out without looping",
    "premature_finish": "declares done / submits / answers while the task is not "
                        "complete or the answer is wrong",
    "never_submits": "the work looks essentially done but the agent never issues "
                     "the completion action (submit, final answer, task_complete) "
                     "before the turn cap",
    "misreads_task": "misunderstands the task statement or requirements from the start",
    "tool_misuse": "wrong tool usage: bad arguments, wrong harness syntax, wrong path, "
                   "editing the wrong way",
    "format_error": "replies with no parseable action, several actions, or malformed "
                    "JSON, so the harness rejects or nudges",
    "context_loss": "forgets what it already did or learned (re-explores, contradicts "
                    "earlier findings) without exact repeats",
    "timeout_no_progress": "long reading / exploring stretches with no edits or "
                           "decisions, runs out of turns",
    "other": "none of the above (explain in category_note)",
}

SYSTEM_PROMPT = """You are a senior engineer reviewing the transcript of an autonomous AI agent (a language model) that worked on a task inside a harness (a coding agent, a terminal agent, a wiki-search tool loop, or a single-reply math solver).

The transcript is split into TURNS. Each turn is one reply of the agent and is labelled `### TURN <n>` (0-based). A turn shows the agent's private thinking (may be truncated), its visible text, the ACTION it issued (a shell command, a tool call, a JSON command batch, a final reply or a boxed answer) and the OBSERVATION the harness returned. Long observations are truncated with `[... N chars omitted ...]`. Stretches where the agent repeated the same action and got the same observation are collapsed into one block that names the turn range.

Your job is to find where the agent went wrong. Be concrete: cite turn numbers that exist in the transcript. Answer ONLY with a single JSON object, no prose around it."""

STAGE1_TEMPLATE = """## Task the agent was given (harness: {harness}, env: {source}, {n_turns} turns in total)
{task}

## Transcript
{transcript}

## Questions (answer as JSON)
Do NOT assume you know whether the task was solved. Judge from the transcript alone.

Return:
{{
  "trajectory_summary": "3-5 sentences: what the agent tried, in order, and how it ended",
  "agent_believed_done": true or false,
  "task_looks_solved_to_you": true, false or "unsure",
  "pivotal_turns": [
    {{"turn": <int>, "rationale": "one line: what went wrong AT this turn and why it matters", "should_have": "one line: what a strong engineer would have done at this turn instead", "confidence": <0.0-1.0>}}
  ],
  "repeated_action": "the exact action the agent kept re-issuing, if it repeated itself, else null",
  "first_useless_turn": <int or null: the first turn after which no further progress was made>
}}
Rules: 1 to 3 pivotal turns, most important first. A pivotal turn is the turn where the trajectory could still have been saved by a different decision -- the decision point, not the wreckage after it. Turn numbers must appear in the transcript (if the pivot is inside an omitted range, name the closest visible turn and lower the confidence). If you see nothing wrong, still name the turn you would scrutinise first, with a low confidence."""

STAGE2_TEMPLATE = """## The grade is now revealed
The environment graded this rollout as FAILED. Stop condition: `{stop_condition}`{cap_note}.
{teacher_block}
## Classify (answer as JSON)
Pick ONE primary failure_category from this fixed list:
{categories}

Return:
{{
  "failure_category": "<one key from the list>",
  "secondary_categories": ["<zero or more other keys that also apply>"],
  "category_note": "free text, at most 40 words: the specific way this rollout failed",
  "pivot_pattern": "at most 10 words, a GENERIC label for the decision error at the primary pivotal turn, reusable across rollouts (e.g. 'reruns git diff --stat after tests pass', 'edits without reading the failing test', 'answers with reasoning only, no boxed result')",
  "what_teacher_did_differently": "2-4 sentences if a teacher rollout was shown, else null",
  "recoverable_from_pivot": {{"estimate": true or false, "confidence": <0.0-1.0>, "why": "one line: could a strong model, continuing from the state just before the primary pivotal turn, still have solved the task?"}},
  "pivot_confirmed": true or false,
  "pivot_revision_note": "if knowing the grade changes your view of the pivot, say how in one line, else null",
  "revised_pivotal_turns": [
    {{"turn": <int>, "rationale": "one line", "should_have": "one line", "confidence": <0.0-1.0>}}
  ]
}}
`revised_pivotal_turns`: fill it ONLY if you gave no pivotal turns before or pivot_confirmed is false (for a single-reply rollout the turn is 0; the rationale must name the step inside the reply where the reasoning went wrong). Otherwise an empty list."""


def categories_block() -> str:
    return "\n".join(f"- `{k}`: {v}" for k, v in CATEGORIES.items())


def merge_usage(acc: dict, new: dict) -> dict:
    """Sum token counts and cost of a retried call so the ledger is honest."""
    if not acc:
        return dict(new)
    out = dict(new)
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        if isinstance(new.get(k), (int, float)) or isinstance(acc.get(k), (int, float)):
            out[k] = (new.get(k) or 0) + (acc.get(k) or 0)
    out["retried"] = True
    return out


def well_formed(rec: dict) -> bool:
    s1, s2 = rec.get("stage1") or {}, rec.get("stage2") or {}
    return ("pivotal_turns" in s1 and isinstance(s1.get("pivotal_turns"), list)
            and s2.get("failure_category") in CATEGORIES)


SCHEMA_NUDGE = ("Your previous reply did not follow the requested schema. Return ONLY the "
                "JSON object with exactly the keys described above (failure_category must be "
                "one of the listed keys).")


def extract_json(text: str) -> dict:
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        raise ValueError("no JSON object in response")
    return json.loads(m.group(0))


class Judge:
    def __init__(self, *, model: str, base_url: str, api_key: str, out_dir: Path,
                 concurrency: int, max_usd: float, price_in: float, price_out: float,
                 max_tokens: int = 6000, reasoning_effort: str | None = "low"):
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.out_dir = out_dir
        self.cache_path = out_dir / "cache" / "judgments.jsonl"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Off-schema judgments are not cached: the next run re-judges them.
        self.cache: dict[str, dict] = {r["cache_key"]: r for r in read_jsonl(self.cache_path)
                                       if well_formed(r)}
        self.sem = asyncio.Semaphore(concurrency)
        self.max_usd = max_usd
        self.price_in = price_in
        self.price_out = price_out
        self.spent_usd = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self.n_calls = 0
        self.lock = asyncio.Lock()
        self.stop = False

    async def chat(self, client: httpx.AsyncClient, messages: list[dict]) -> tuple[str, dict]:
        body = {"model": self.model, "messages": messages, "temperature": 0.0,
                "max_tokens": self.max_tokens, "response_format": {"type": "json_object"},
                "usage": {"include": True}}
        if self.reasoning_effort:
            # Reasoning models spend output tokens thinking; keep it short, the
            # judgment itself is the JSON object.
            body["reasoning"] = {"effort": self.reasoning_effort}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "HTTP-Referer": "https://affine.io", "X-Title": "affine king review"}
        delay = 3.0
        usage_acc: dict = {}
        for attempt in range(6):
            try:
                r = await client.post(f"{self.base_url}/chat/completions", json=body,
                                      headers=headers, timeout=900)
                if r.status_code in (429, 500, 502, 503, 504, 524):
                    raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:200]}",
                                                request=r.request, response=r)
                if r.status_code == 400 and ("response_format" in r.text
                                             or "reasoning" in r.text):
                    body.pop("response_format", None)
                    body.pop("reasoning", None)
                    continue
                r.raise_for_status()
                data = r.json()
                if "error" in data:
                    raise RuntimeError(str(data["error"])[:300])
                choice = data["choices"][0]
                msg = choice["message"]
                usage = merge_usage(usage_acc, data.get("usage") or {})
                content = msg.get("content") or ""
                if not content.strip() or choice.get("finish_reason") == "length":
                    # Thinking ate the output budget: pay once more with a bigger cap.
                    if body["max_tokens"] < 4 * self.max_tokens:
                        body["max_tokens"] = 4 * self.max_tokens
                        usage_acc = usage
                        continue
                    raise RuntimeError("empty or truncated content after retry")
                return content, usage
            except (httpx.HTTPError, RuntimeError, KeyError):
                if attempt == 5:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")

    def _account(self, usage: dict) -> float:
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        cost = usage.get("cost")
        if cost is None:
            cost = pt / 1e6 * self.price_in + ct / 1e6 * self.price_out
        self.tokens_in += pt
        self.tokens_out += ct
        self.spent_usd += float(cost)
        self.n_calls += 1
        return float(cost)

    async def judge_one(self, client: httpx.AsyncClient, item: dict, ts: TraceStore,
                        env_groups: dict[str, str], log) -> dict | None:
        env = ts.load_envelope(item["chunk"], item["line"])
        ro = parse_rollout(env, env_groups)
        transcript, tstats = render_transcript(ro, max_chars=MAX_TRANSCRIPT_CHARS)
        stage1 = STAGE1_TEMPLATE.format(
            harness=ro.harness, source=ro.source, n_turns=ro.n_turns,
            task=clip(norm_ws(ro.task_prompt), 4000, 500) or "(task text not stored)",
            transcript=transcript)
        cache_key = sha256_hex("|".join([self.model, PROMPT_VERSION, item["rollout_id"],
                                         SYSTEM_PROMPT, stage1]))
        if cache_key in self.cache:
            return self.cache[cache_key]
        if self.stop:
            return None
        teacher_block = ""
        teacher_meta = None
        if item.get("teacher"):
            tenv = ts.load_envelope(item["teacher"]["chunk"], item["teacher"]["line"])
            tro = parse_rollout(tenv, env_groups)
            same = "the SAME harness" if item["teacher"]["same_harness"] else \
                f"a DIFFERENT harness ({tro.harness})"
            teacher_block = (f"\n## The teacher model's rollout on the same task ({same})\n"
                             f"{render_teacher_summary(tro)}\n")
            teacher_meta = {"rollout_id": tro.rollout_id, "outcome": tro.outcome,
                            "harness": tro.harness, "n_turns": tro.n_turns,
                            "same_harness": item["teacher"]["same_harness"]}
        cap_note = ""
        if ro.stop_condition == "max_turns":
            cap_note = f" (the harness stopped it at the turn cap after {ro.n_turns} turns)"
        stage2 = STAGE2_TEMPLATE.format(stop_condition=ro.stop_condition or "unknown",
                                        cap_note=cap_note, teacher_block=teacher_block,
                                        categories=categories_block())
        async with self.sem:
            if self.stop:
                return None
            t0 = time.time()
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": stage1}]
            raw1, usage1 = await self.chat(client, messages)
            try:
                s1 = extract_json(raw1)
            except (ValueError, json.JSONDecodeError) as ex:
                s1 = {"parse_error": str(ex)}
            messages += [{"role": "assistant", "content": raw1},
                         {"role": "user", "content": stage2}]
            raw2, usage2 = await self.chat(client, messages)
            try:
                s2 = extract_json(raw2)
            except (ValueError, json.JSONDecodeError) as ex:
                s2 = {"parse_error": str(ex)}
            if s2.get("failure_category") not in CATEGORIES:
                # one corrective round inside the same conversation
                messages += [{"role": "assistant", "content": raw2},
                             {"role": "user", "content": SCHEMA_NUDGE}]
                raw2b, usage2b = await self.chat(client, messages)
                usage2 = merge_usage(usage2, usage2b)
                try:
                    s2 = extract_json(raw2b)
                    raw2 = raw2b
                except (ValueError, json.JSONDecodeError) as ex:
                    s2 = {**s2, "parse_error": str(ex)}
            async with self.lock:
                c1 = self._account(usage1)
                c2 = self._account(usage2)
                if self.spent_usd >= self.max_usd:
                    self.stop = True
            rec = {
                "cache_key": cache_key, "prompt_hash": cache_key[:16],
                "prompt_version": PROMPT_VERSION, "judge_model": self.model,
                "judged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "king": item["king"], "rollout_id": ro.rollout_id, "traj_id": ro.traj_id,
                "sid": ro.sid, "source": ro.source, "env_group": ro.env_group,
                "harness": ro.harness, "policy_id": ro.policy_id,
                "action_kind": ro.action_kind, "outcome": ro.outcome,
                "stop_condition": ro.stop_condition, "n_turns": ro.n_turns,
                "node_ids": [t.node_id for t in ro.turns],
                "det": {"loop_onsets": ro.loop_onsets(),
                        "in_loop": [t.idx for t in ro.turns if t.loop == "in_loop"],
                        "escape": [t.idx for t in ro.turns if t.loop == "escape"],
                        "no_action": [t.idx for t in ro.turns if t.no_action],
                        "token_cap": [t.idx for t in ro.turns if t.token_cap],
                        "completion": [t.idx for t in ro.turns if t.completion],
                        "last_turn": ro.n_turns - 1},
                "transcript_stats": tstats, "teacher": teacher_meta,
                "stage1": s1, "stage2": s2,
                "raw": {"stage1": raw1, "stage2": raw2},
                "usage": {"stage1": usage1, "stage2": usage2,
                          "cost_usd": round(c1 + c2, 5),
                          "seconds": round(time.time() - t0, 1)},
            }
            with open(self.cache_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.cache[cache_key] = rec
            log(f"  judged {ro.harness:<18} {ro.source:<16} {ro.n_turns:>3} turns "
                f"${c1 + c2:.3f}  total ${self.spent_usd:.2f} ({self.n_calls} calls)",
                flush=True)
            return rec

    async def run(self, sample: list[dict], ts: TraceStore, env_groups: dict[str, str],
                  log=print) -> list[dict]:
        async with httpx.AsyncClient() as client:
            tasks = [self.judge_one(client, item, ts, env_groups, log) for item in sample]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for item, r in zip(sample, results):
            if isinstance(r, Exception):
                log(f"  FAILED {item['rollout_id']}: {type(r).__name__}: {str(r)[:200]}")
            elif r is not None:
                out.append(r)
        return out


def load_api_key(env_names: list[str], key_file: str | None) -> str:
    if key_file:
        return Path(key_file).read_text().strip()
    for name in env_names:
        if os.environ.get(name):
            return os.environ[name]
    raise SystemExit(f"no API key: set one of {env_names} or pass --key-file")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY,OPENROUTER,ENGY_2",
                    help="comma list of env vars tried in order for the API key")
    ap.add_argument("--key-file", default=None, help="file holding the key (0600)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-usd", type=float, default=30.0)
    ap.add_argument("--price-in", type=float, default=0.579,
                    help="USD per 1M prompt tokens, used only if the API reports no cost")
    ap.add_argument("--price-out", type=float, default=1.738)
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--reasoning-effort", default="low",
                    help="OpenRouter reasoning effort for thinking models; 'none' to omit")
    args = ap.parse_args()

    sample = read_jsonl(args.sample)
    if args.limit:
        sample = sample[:args.limit]
    ts = TraceStore()
    env_groups = load_env_groups()
    judge = Judge(model=args.model, base_url=args.base_url,
                  api_key=load_api_key(args.key_env.split(","), args.key_file),
                  out_dir=args.out_dir, concurrency=args.concurrency, max_usd=args.max_usd,
                  price_in=args.price_in, price_out=args.price_out,
                  max_tokens=args.max_tokens,
                  reasoning_effort=None if args.reasoning_effort == "none" else args.reasoning_effort)
    n_cached = sum(1 for _ in judge.cache)
    print(f"judge: {len(sample)} rollouts, model {args.model}, {n_cached} cached judgments, "
          f"budget ${args.max_usd:.2f}", flush=True)
    results = asyncio.run(judge.run(sample, ts, env_groups))
    cost = {"judge_model": args.model, "n_rollouts_judged": len(results),
            "n_calls_this_run": judge.n_calls, "prompt_tokens_this_run": judge.tokens_in,
            "completion_tokens_this_run": judge.tokens_out,
            "usd_this_run": round(judge.spent_usd, 4),
            "usd_all_cached": round(sum(r["usage"]["cost_usd"] for r in results), 4),
            "stopped_on_budget": judge.stop}
    (args.out_dir / "cost.json").write_text(json.dumps(cost, indent=1))
    # Append-only ledger: one line per run, so spend on the shared key can be
    # attributed to this pipeline precisely.
    with open(args.out_dir / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "sample": str(args.sample), "n_sample": len(sample), **cost}) + "\n")
    print(json.dumps(cost, indent=1))


if __name__ == "__main__":
    main()
