"""gpqa-strict: upstream `gpqa` (research-environments) with the LLM-judge
fallback removed. The Affine benchmark suite uses no judge anywhere, so a
reply whose answer letter the deterministic MCQ regex cannot find scores 0.

Same data, same option shuffle (seed 0), same prompt as upstream, so the
number is comparable to a judge-off GPQA run elsewhere.
"""

from collections.abc import Iterator

import verifiers.v1 as vf
from gpqa.mcq import extract_mcq_answer
from gpqa.taskset import GPQAConfig, GPQAData, GPQATaskset


class GPQAStrictTask(vf.Task[GPQAData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return 1.0 if extract_mcq_answer(trace.last_reply) == self.data.answer else 0.0

    @vf.metric
    async def letter_found(self, trace: vf.Trace) -> float:
        return 1.0 if extract_mcq_answer(trace.last_reply) else 0.0


class GPQAStrictConfig(vf.TasksetConfig):
    diamond: bool = True


class GPQAStrictTaskset(vf.Taskset[GPQAStrictTask, GPQAStrictConfig]):
    def load(self) -> Iterator[GPQAStrictTask]:
        upstream = GPQATaskset(GPQAConfig(diamond=self.config.diamond))
        for task in upstream.load():
            yield GPQAStrictTask(task.data, self.config.task)
