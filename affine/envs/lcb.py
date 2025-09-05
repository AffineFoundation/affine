from __future__ import annotations

import os
import time
import json
import affine as af
from typing import Any, Dict


_ENABLED = os.getenv("AFFINE_ENABLE_LCB", "false").lower() in {"1", "true", "yes", "on"}


if _ENABLED:
    # Dataset source: LiveCodeBench code_generation lite, release v6
    def _dataset_factory():
        return af.utils.R2BufferedDataset(
            dataset_name=os.getenv("AFFINE_LCB_DATASET", "livecodebench/code_generation_lite"),
            config=os.getenv("AFFINE_LCB_CONFIG", "release_v6"),
            split=os.getenv("AFFINE_LCB_SPLIT", "test"),
            buffer_size=int(os.getenv("AFFINE_LCB_BUFFER", "5")),
            max_batch=int(os.getenv("AFFINE_LCB_BATCH", "5")),
        )

    dataset = af.singleton("lcb-codegen", _dataset_factory)


    class LCB(af.BaseEnv):
        __version__: str = "0.0.1"

        def __init__(self):
            super().__init__()
            mem_mb = int(os.getenv("AFFINE_LCB_MEM_MB", "512"))
            timeout = int(os.getenv("AFFINE_LCB_TIMEOUT", "30"))
            cpu_time = int(os.getenv("AFFINE_LCB_CPU", "10"))
            self._executor = af.utils.ProgramExecutor(
                timeout=timeout, cpu_time=cpu_time, mem_bytes=mem_mb * 2**20
            )

        async def generate(self) -> af.Challenge:
            af.logger.trace("LCB.generate: fetching dataset row")
            row: Dict[str, Any] = await dataset().get()
            if row is None:
                raise RuntimeError("LCB: failed to fetch dataset row")

            # Expected fields from LCB-like rows (best-effort):
            qid = row.get("question_id") or row.get("id") or row.get("qid")
            content = (
                row.get("question_content")
                or row.get("prompt")
                or row.get("problem")
                or "Solve the following programming task."
            )
            starter = row.get("starter_code") or row.get("starter") or ""

            instructions = (
                "\n\n---\n"
                "You are an expert Python programmer.\n"
                "Return exactly one fenced code block:\n"
                "```python\n# your full solution here\n```\n\n"
                "Rules:\n"
                "- Read all input from STDIN (e.g., input() or sys.stdin).\n"
                "- Print only the required outputs to STDOUT.\n"
                "- No extra text, prompts, or explanations.\n"
            )

            prompt_parts = [str(content).rstrip()]
            if starter:
                prompt_parts.append("\n\nStarter code (optional):\n```python\n" + str(starter).rstrip() + "\n```")
            prompt_parts.append(instructions)
            prompt = "".join(prompt_parts)

            # Attach minimal metadata for evaluation
            row["_ts"] = time.time()
            row["_question_id"] = qid
            return af.Challenge(env=self, prompt=prompt, extra=row)

        async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
            af.logger.trace("LCB.evaluate: starting")
            from affine.utils.lcb_checker import evaluate_program  # local import to avoid cycles

            raw_reply = response.response or ""
            program = self._executor._strip_fences(raw_reply)

            try:
                score, details = await evaluate_program(self._executor, program, challenge.extra or {})
            except Exception as e:
                af.logger.warning(f"LCB.evaluate error: {e}")
                return af.Evaluation(env=self, score=0.0, extra={"error": str(e)})

            # Strict pass@1: 1.0 only if all public tests pass
            af.logger.trace(f"LCB.evaluate: score={score} details_passed={details.get('passed')} total={details.get('total')}")
            return af.Evaluation(env=self, score=1.0 if score >= 1.0 else 0.0, extra=details)

else:
    # Feature flag disabled: define no env class, but keep module importable
    pass

