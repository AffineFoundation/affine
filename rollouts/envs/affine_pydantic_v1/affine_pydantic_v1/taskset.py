"""affine-pydantic-v1: structured output - a JSON object validated against a per-row Pydantic model.

v1 port of the Hub's `primeintellect/pydantic-adherence` (v0): rows of
`justus27/pydantic-adherance-test` (train) carry a prompt and a
`verification_info` JSON with the Pydantic model source (`pydantic_config`)
and its class name (`model_name`). The reply's last JSON object (a ```json
fence or the last balanced `{...}`) is parsed and `model_validate`d; `correct`
= 1.0 iff it validates. Structured replies are a chat skill kings need and D
has nowhere else; the survey's 6-task teacher probe scored 3/6 (real headroom).

Shape: single turn under the `null` harness, `text` dialect (the whole visible
reply is the action). A real *system* message is added (the v0 env has none).
The model source executes in the eval process, as upstream does - it is the
dataset author's code, not the model's.

Task identity: `name = "pydantic-" + sha256(prompt)[:12]` (rollouts/catalog.py
`_pydantic_meta`).
"""

from __future__ import annotations

import hashlib
import json
import re
from types import ModuleType

import verifiers.v1 as vf
from datasets import load_dataset
from pydantic import BaseModel

DATASET = "justus27/pydantic-adherance-test"
SPLIT = "train"
SYSTEM = (
    "You answer with structured data. Read the request and the schema it "
    "describes, think it through, then reply with exactly one JSON object "
    "that satisfies the schema - inside a single ```json code block, with no "
    "other JSON in the reply. Your visible reply is graded by validating that "
    "object against the schema."
)
FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def task_name(prompt: str) -> str:
    return "pydantic-" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def last_json_block(text: str) -> str | None:
    matches = list(FENCE_RE.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    end = text.rfind("}")
    if end == -1:
        return None
    depth = 0
    for i in range(end, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                return text[i:end + 1].strip()
    return None


def extract_last_json(text: str) -> dict | None:
    block = last_json_block(text or "")
    if block is None:
        return None
    try:
        loaded = json.loads(block)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def load_model(code: str, model_name: str) -> type[BaseModel]:
    module = ModuleType("dyn_pydantic_cfg")
    exec(code, module.__dict__)  # noqa: S102 - dataset-authored schema code, as upstream
    cls = getattr(module, model_name, None)
    if cls is None or not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise RuntimeError(f"{model_name} not found or not a Pydantic BaseModel")
    cls.model_json_schema()
    return cls


class PydanticData(vf.TaskData):
    pydantic_config: str
    model_name: str


class PydanticTask(vf.Task[PydanticData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        parsed = extract_last_json(trace.last_reply or "")
        trace.info["json_found"] = parsed is not None
        if parsed is None:
            return 0.0
        model = load_model(self.data.pydantic_config, self.data.model_name)
        try:
            model.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001 - any validation failure is a miss
            trace.info["validation_error"] = str(exc)[:500]
            return 0.0
        return 1.0

    async def validate(self, runtime: vf.Runtime) -> bool:
        try:
            load_model(self.data.pydantic_config, self.data.model_name)
        except Exception:  # noqa: BLE001
            return False
        return True


class PydanticConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names to load (empty = the whole train split)."""
    task_system_prompt: str = SYSTEM


class PydanticTaskset(vf.Taskset[PydanticTask, PydanticConfig]):
    def load(self) -> list[PydanticTask]:
        cfg = self.config
        want = set(cfg.tasks)
        rows = load_dataset(DATASET, split=SPLIT)
        tasks: list[PydanticTask] = []
        for i, row in enumerate(rows):
            prompt = str(row["prompt"])
            name = task_name(prompt)
            if want and name not in want:
                continue
            info = json.loads(row["verification_info"])
            if not info.get("pydantic_config") or not info.get("model_name"):
                continue
            tasks.append(PydanticTask(
                PydanticData(idx=i, name=name, system_prompt=cfg.task_system_prompt, prompt=prompt,
                             pydantic_config=info["pydantic_config"], model_name=info["model_name"]),
                cfg.task,
            ))
        return tasks
