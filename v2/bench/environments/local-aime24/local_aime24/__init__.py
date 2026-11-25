"""local-aime24: the pinned Prime Hub env primeintellect/aime24 (see bench/deploy/requirements-envs.lock), unchanged
(same dataset revision, prompt, task order and reward), except that each task's sandbox network policy is opened.

Why: the Hub task declares network_allow=[] (no network inside its sandbox), which only the docker/prime runtimes can enforce.
Our harness is `null` (one chat turn, no tools, no code execution), so the sandbox does nothing but relay that one chat call
to the local vLLM endpoint; we run it as a plain subprocess on the bench node. Nothing about what the model sees or how it
is graded changes."""
from collections.abc import Iterator

from aime24 import AIME24Taskset as _HubAIME24Taskset


class LocalAIME24Taskset(_HubAIME24Taskset):
    def load(self) -> Iterator:
        for task in super().load():
            data = task.data.model_copy(update={"network_allow": ["*"], "network_block": []})
            yield type(task)(data, task.config)


__all__ = ["LocalAIME24Taskset"]
