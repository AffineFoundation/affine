"""local-gsm8k: the pinned Prime Hub env primeintellect/gsm8k (see bench/deploy/requirements-envs.lock), unchanged
(same dataset revision, prompt, task order and reward), except that each task's sandbox network policy is opened.

Why: the Hub task declares network_allow=[] (no network inside its sandbox), which only the docker/prime runtimes can enforce.
Our harness is `null` (one chat turn, no tools, no code execution), so the sandbox does nothing but relay that one chat call
to the local vLLM endpoint; we run it as a plain subprocess on the bench node. Nothing about what the model sees or how it
is graded changes."""
from collections.abc import Iterator

from gsm8k import GSM8KTaskset as _HubGSM8KTaskset


class LocalGSM8KTaskset(_HubGSM8KTaskset):
    def load(self) -> Iterator:
        for task in super().load():
            data = task.data.model_copy(update={"network_allow": ["*"], "network_block": []})
            yield type(task)(data, task.config)


__all__ = ["LocalGSM8KTaskset"]
