"""Abort a rollout that has collapsed into repeating one action.

A broken checkpoint can emit the same command forever: one merged model
answered every observation with a byte-identical reply calling an invented
shell command, 58 times in a row, until the step limit ended the task. At
temperature 0 there is no randomness to break out of that state, so every such
task burns its whole step budget to produce nothing. Detecting it early costs
a comparison per step and reclaims most of the rollout phase.

Healthy agents do not trip this: repeating one identical command eight times
in a row while the observation keeps changing is a stuck model, not progress.
"""

from __future__ import annotations

import os
import re

from minisweagent.exceptions import InterruptAgentFlow

DEFAULT_REPEAT_LIMIT = 8
EXIT_STATUS = "RepeatedAction"

# Permissive on purpose: catch the action whatever fence the model used, and
# fall back to the whole reply when nothing parses.
_ACTION = re.compile(r"```(?:bash|sh|mswea_bash_command)?\s*\n(.*?)\n```", re.S)


def _fingerprint(content: str) -> str:
    m = _ACTION.search(content)
    return " ".join((m.group(1) if m else content).split())


def repeat_limit() -> int:
    try:
        return int(os.environ.get("FASTBENCH_REPEAT_LIMIT", DEFAULT_REPEAT_LIMIT))
    except ValueError:
        return DEFAULT_REPEAT_LIMIT


def guard(base: type) -> type:
    """Wrap an agent class so identical consecutive actions end the task."""

    class LoopGuardAgent(base):  # type: ignore[valid-type,misc]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._limit = repeat_limit()
            self._last_fp = None
            self._repeats = 1

        def query(self) -> dict:
            message = super().query()
            if self._limit <= 1:
                return message
            fp = _fingerprint(message.get("content") or "")
            if fp and fp == self._last_fp:
                self._repeats += 1
            else:
                self._last_fp, self._repeats = fp, 1
            if self._repeats >= self._limit:
                # The reply is kept so the trajectory shows what it looped on.
                raise InterruptAgentFlow(
                    message,
                    {"role": "exit", "content": EXIT_STATUS,
                     "extra": {"exit_status": EXIT_STATUS, "submission": "",
                               "repeats": self._repeats}},
                )
            return message

    LoopGuardAgent.__name__ = f"LoopGuard{base.__name__}"
    return LoopGuardAgent
