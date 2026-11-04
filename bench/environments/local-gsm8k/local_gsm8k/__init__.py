"""local-gsm8k: the pinned Prime Hub env primeintellect/gsm8k (see bench/deploy/requirements-envs.lock), unchanged
(same dataset revision, prompt, task order and reward), except that each task's sandbox network policy is opened, and that a
reply too long to pass as one command-line argument is shortened to what verify.py would grade anyway.

Why the network: the Hub task declares network_allow=[] (no network inside its sandbox), which only the docker/prime runtimes
can enforce. Our harness is `null` (one chat turn, no tools, no code execution), so the sandbox does nothing but relay that one
chat call to the local vLLM endpoint; we run it as a plain subprocess on the bench node. Nothing about what the model sees or
how it is graded changes.

Why the long replies (2026-11-04, Phi-4-reasoning): the Hub reward passes the whole reply to verify.py as argv[2]; Linux caps
one argument at 128 KiB (MAX_ARG_STRLEN), so a ~32K-token reply raised "OSError: Argument list too long" and the rollout errored
instead of being graded. verify.py grades only the text after the LAST '####' (re.findall(r"####\\s*(.+)")[-1]), so an
oversized reply is passed as just that line - the same score. An oversized reply with no '####' line would be graded as one
>100 KB number, which can't match a GSM8K answer: it scores 0.0 without running verify.py."""
import re
from collections.abc import Iterator

import verifiers.v1 as vf
from gsm8k import GSM8KTaskset as _HubGSM8KTaskset
from gsm8k.taskset import GSM8KTask as _HubGSM8KTask
from gsm8k.taskset import VERIFY as _VERIFY  # the Hub's verify.py bytes, unchanged

MAX_ARG = 100_000  # bytes; below the kernel's 131072-byte MAX_ARG_STRLEN


def shorten(prediction):
    """The argv[2] verify.py gets: the reply itself, or for an oversized reply the last '#### N' line (None = no such line)."""
    if len((prediction or "").encode()) <= MAX_ARG:
        return prediction or ""
    m = re.findall(r"####\s*(.+)", prediction)
    return "#### " + m[-1].strip() if m else None


class LocalGSM8KTask(_HubGSM8KTask):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        pred = shorten(trace.last_reply)
        if pred is None:
            return 0.0
        result = await runtime.run_uv_script(_VERIFY, args=[self.data.answer, pred])
        if result.exit_code != 0:
            raise RuntimeError(f"verify.py failed: {result.stderr.strip()[-500:]}")
        lines = result.stdout.strip().splitlines()
        return float(lines[-1]) if lines else 0.0


class LocalGSM8KTaskset(_HubGSM8KTaskset):
    def load(self) -> Iterator:
        for task in super().load():
            data = task.data.model_copy(update={"network_allow": ["*"], "network_block": []})
            yield LocalGSM8KTask(data, task.config)


__all__ = ["LocalGSM8KTaskset", "LocalGSM8KTask", "shorten"]
