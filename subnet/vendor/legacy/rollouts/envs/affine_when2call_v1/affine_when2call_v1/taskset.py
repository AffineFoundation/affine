"""affine-when2call-v1: When (not) to call tools — NVIDIA When2Call, TRAIN only.

Source: https://github.com/NVIDIA/When2Call (Ross, Mahabaleshwarka, Suhara,
NAACL 2025). Dataset nvidia/When2Call on the Hub, licence CC-BY-4.0 ("ready
for commercial use"); repository code Apache-2.0. The train sets were
generated from Salesforce/xlam-function-calling-60k (CC-BY-4.0) with
Mixtral-8x22B; the TEST sets were generated from BFCL v2 live and are NOT
used here — they are a held-out benchmark (the benchsuite worker's), and so
is BFCL itself.

Items: the `train_pref` split (9,000 rows, 8,295 unique questions). Each row
carries the user question, the tool schemas offered, and a `chosen_response`
that IS the label:
  tool_call         chosen = <TOOLCALL>[{name, arguments}]</TOOLCALL>  (3,000)
  request_for_info  chosen = a follow-up question                     (~2,850)
  cannot_answer     chosen = a refusal                                (~3,000)
The SFT split (15,000) holds only the two text classes and is not used in
v1; When2Call has no `direct` class in either train file. Rows whose chosen
text fits neither text class (~150) and the one question that also appears
in BFCL v3 (`EXCLUDE_QUESTIONS`) are dropped.

Grades (fold key `solved` first):
  solved  tool_call label: the FIRST assistant reply carries exactly one tool
          call with the gold name and every gold argument equal (string
          compare) — no extra tool call on the first turn;
          request_for_info: no tool call in the rollout AND the reply asks;
          cannot_answer: no tool call in the rollout (a refusal or a direct
          answer — these items were built by removing the tool from a
          tool_call item, so a self-contained "count the fruits" question is
          labelled cannot_answer; answering it is not a mistake for us)
  class_match  strict When2Call class match (tool name / ask / refuse)
  tool_called, name_match, pred_tool_call, pred_request_for_info,
  pred_cannot_answer, pred_direct   0/1 metrics (weight 0) for the per-class
          confusion on the board; the label itself rides on the task
          (`data.label`, and `repo = when2call/<label>` in the catalog)

Stops: the null harness ends on a prose reply (`agent_completed`); a
rollout that keeps calling tools stops at 4 turns under the name
`max_turns` (a clean failure for the fold). Task names `w2c-<sha1(question)
[:12]>` = catalog uid (rollouts/catalog.py `_when2call_meta`).
"""

from __future__ import annotations

import hashlib
import json
import re

import verifiers.v1 as vf
from datasets import load_dataset

from affine_when2call_v1.servers.tools import When2CallToolset, When2CallToolsetConfig

DATASET_NAME = "nvidia/When2Call"
DATASET_CONFIG = "train_pref"
SPLIT = "train"
DATASET_REVISION = "0582f7749df63a96fdc3070932e83e72396ace53"

TOOL_CALL = "tool_call"
ASK = "request_for_info"
REFUSE = "cannot_answer"
DIRECT = "direct"
LABELS = (TOOL_CALL, ASK, REFUSE)

# Exact question texts also present in BFCL v3 (held-out benchmark), found
# by normalized match over every BFCL_v3_*.json user prompt (2026-09-14).
EXCLUDE_QUESTIONS = frozenset({
    "Shuffle a deck of cards and draw 3 cards from the top.",
})

REFUSAL_MARKERS = (
    "unable to", "i can't", "i cannot", "apolog", "i'm sorry", "i am sorry",
    "not able to", "capability", "can't assist", "cannot assist", "can't help",
    "cannot help", "don't have access", "do not have access", "no access to",
    "not possible for me", "beyond my", "outside my",
)

SYSTEM = (
    "You are a helpful assistant with access to the tools listed here. "
    "Decide what the request needs: if one of the tools can fulfil it and "
    "the request gives every required argument, call that tool, once, with "
    "the arguments taken from the request; if a required argument is "
    "missing, do not guess it — ask the user for it in one short question; "
    "if no listed tool can do what is asked and it needs live data or an "
    "action you cannot perform, say so plainly in one or two sentences, "
    "without calling a tool. Think it through before you act."
)


def task_name(question: str) -> str:
    return "w2c-" + hashlib.sha1(question.strip().encode("utf-8")).hexdigest()[:12]


def parse_toolcall(text: str) -> dict | None:
    """The gold `<TOOLCALL>[{name, arguments}]</TOOLCALL>` -> the single call."""
    m = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", text, re.S)
    if not m:
        return None
    try:
        calls = json.loads(m.group(1))
    except ValueError:
        return None
    if isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict):
        return calls[0]
    return None


def text_class(text: str) -> str:
    """ask / refuse / direct for a prose reply, by the string rules that
    separate When2Call's own gold texts (99 % of 15k SFT answers)."""
    low = (text or "").lower().replace("\u2019", "'").replace("\u2018", "'").strip()
    asks = low.endswith("?") or bool(re.search(r"\?\s*$", low.split("\n")[-1] if low else ""))
    refuses = any(w in low for w in REFUSAL_MARKERS)
    if refuses and not asks:
        return REFUSE
    if asks:
        return ASK
    return DIRECT


def label_of(chosen: str) -> str | None:
    if parse_toolcall(chosen) is not None:
        return TOOL_CALL
    cls = text_class(chosen)
    return cls if cls in (ASK, REFUSE) else None


class When2CallData(vf.TaskData):
    label: str
    """When2Call's correct move: tool_call / request_for_info / cannot_answer."""
    gold_call: dict | None = None
    """{name, arguments} when the label is tool_call."""
    n_tools: int = 0


class When2CallTaskConfig(vf.TaskConfig):
    max_turns: int = 4
    tools: When2CallToolsetConfig = When2CallToolsetConfig()


def _tool_calls(msg) -> list[dict]:
    """[{name, arguments(dict)}] of one AssistantMessage (verifiers ToolCall:
    name + raw JSON arguments string)."""
    out = []
    for tc in getattr(msg, "tool_calls", None) or []:
        raw = getattr(tc, "arguments", None)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                raw = {"_raw": raw}
        out.append({"name": getattr(tc, "name", None),
                    "arguments": raw if isinstance(raw, dict) else {}})
    return out


def _same_value(a, b) -> bool:
    if a == b:
        return True
    return str(a).strip().lower() == str(b).strip().lower()


class When2CallTask(vf.Task[When2CallData, vf.State, When2CallTaskConfig]):
    @classmethod
    def toolsets(cls, config: When2CallTaskConfig) -> list[vf.Toolset]:
        return [When2CallToolset(config.tools)]

    @vf.stop
    async def max_turns(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= self.config.max_turns

    # -- what the model did -----------------------------------------------------
    def _first_calls(self, trace: vf.Trace) -> list[dict]:
        msgs = trace.assistant_messages
        return _tool_calls(msgs[0]) if msgs else []

    def _any_call(self, trace: vf.Trace) -> bool:
        return bool(trace.tool_messages) or any(_tool_calls(m) for m in trace.assistant_messages)

    def _pred(self, trace: vf.Trace) -> str:
        if self._any_call(trace):
            return TOOL_CALL
        reply = (trace.last_reply or "").strip()
        return text_class(reply) if reply else DIRECT

    # -- grades -----------------------------------------------------------------
    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace) -> float:
        """tool_call: the first reply's single call has the gold name and
        every gold argument. request_for_info: no tool call and the reply
        asks. cannot_answer: no tool call — a refusal OR a direct answer
        (When2Call built these by removing the tool from a tool_call item,
        so "count the fruits in this list" is labelled cannot_answer; a model
        that just answers it did the right thing for us, which is not to
        call). The strict class match is `class_match`."""
        label = self.data.label
        if label == TOOL_CALL:
            calls = self._first_calls(trace)
            gold = self.data.gold_call or {}
            if len(calls) != 1 or calls[0]["name"] != gold.get("name"):
                return 0.0
            got = calls[0]["arguments"]
            want = gold.get("arguments") or {}
            return float(all(k in got and _same_value(got[k], v) for k, v in want.items()))
        pred = self._pred(trace)
        if label == REFUSE:
            return float(pred in (REFUSE, DIRECT))
        return float(pred == label)

    @vf.reward(weight=0.0)
    async def class_match(self, trace: vf.Trace) -> float:
        """Strict When2Call class match: predicted move == label (a tool
        call counts when the name matches; arguments are `solved`'s job)."""
        label = self.data.label
        pred = self._pred(trace)
        if label == TOOL_CALL:
            gold = (self.data.gold_call or {}).get("name")
            return float(any(c["name"] == gold for c in self._first_calls(trace)))
        return float(pred == label)

    @vf.reward(weight=0.0)
    async def tool_called(self, trace: vf.Trace) -> float:
        return float(self._any_call(trace))

    @vf.reward(weight=0.0)
    async def name_match(self, trace: vf.Trace) -> float:
        gold = (self.data.gold_call or {}).get("name")
        return float(bool(gold) and any(c["name"] == gold for c in self._first_calls(trace)))

    @vf.reward(weight=0.0)
    async def pred_tool_call(self, trace: vf.Trace) -> float:
        return float(self._pred(trace) == TOOL_CALL)

    @vf.reward(weight=0.0)
    async def pred_request_for_info(self, trace: vf.Trace) -> float:
        return float(self._pred(trace) == ASK)

    @vf.reward(weight=0.0)
    async def pred_cannot_answer(self, trace: vf.Trace) -> float:
        return float(self._pred(trace) == REFUSE)

    @vf.reward(weight=0.0)
    async def pred_direct(self, trace: vf.Trace) -> float:
        return float(self._pred(trace) == DIRECT)


class When2CallConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names (`w2c-<sha12>`) to load; empty = the whole train_pref pool."""
    task: When2CallTaskConfig = When2CallTaskConfig()


def iter_rows():
    """(question, tools, label, gold_call) over the usable train_pref rows,
    first occurrence per question; the same walk the catalog builder does."""
    rows = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT, revision=DATASET_REVISION)
    seen: set[str] = set()
    for row in rows:
        msgs = row.get("messages") or []
        if not msgs or msgs[0].get("role") != "user":
            continue
        question = str(msgs[0].get("content") or "").strip()
        if not question or question in EXCLUDE_QUESTIONS or question in seen:
            continue
        chosen = str((row.get("chosen_response") or {}).get("content") or "")
        label = label_of(chosen)
        if label is None:
            continue
        tools = [t if isinstance(t, str) else json.dumps(t) for t in (row.get("tools") or [])]
        gold = parse_toolcall(chosen) if label == TOOL_CALL else None
        if label == TOOL_CALL and gold is None:
            continue
        seen.add(question)
        yield question, tools, label, gold


class When2CallTaskset(vf.Taskset[When2CallTask, When2CallConfig]):
    def load(self) -> list[When2CallTask]:
        want = set(self.config.tasks)
        tasks: list[When2CallTask] = []
        for i, (question, tools, label, gold) in enumerate(iter_rows()):
            name = task_name(question)
            if want and name not in want:
                continue
            cfg = self.config.task.model_copy(
                update={"tools": self.config.task.tools.model_copy(update={"tools": tools})})
            tasks.append(When2CallTask(
                When2CallData(idx=i, name=name, system_prompt=SYSTEM, prompt=question,
                              label=label, gold_call=gold, n_tools=len(tools)),
                cfg,
            ))
        return tasks
