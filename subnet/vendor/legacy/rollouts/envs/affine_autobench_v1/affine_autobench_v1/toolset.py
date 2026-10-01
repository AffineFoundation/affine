"""AutomationBench tool server with the register/_register API bridge and the
Artificial Analysis grade mirrored into the rollout state.

verifiers launches a task-scoped tool server as `python -m <module of the
toolset class>`, so the shim lives in its own module with a `run()` guard.
The pods' verifiers (a298bcf) calls `register(mcp)`; research-environments
b10db76 implements the newer `_register`, and without this bridge the server
died at start-up and the model saw zero tools (probe 2026-09-12: 12/12
rollouts hallucinated `<tool_call>` XML). Harmless on newer verifiers.

The live WorldState exists only in this process, so the guardrail / objective
split (state.py) is computed here after every `api_fetch`, next to upstream's
`partial_credit`, and travels to the host through the typed state.
"""

from __future__ import annotations

from typing import Callable

import verifiers.v1 as vf
from automationbench_v1.common import AutomationBenchToolsetConfig
from automationbench_v1.servers.toolset import AutomationBenchToolset

from affine_autobench_v1.state import AffineAutomationBenchState, aa_grade


class AffineAutomationBenchToolset(AutomationBenchToolset,
                                   vf.Toolset[AutomationBenchToolsetConfig, AffineAutomationBenchState]):
    def register(self, mcp) -> None:
        self._register(mcp)

    def _make_api_fetch(self) -> Callable:
        upstream = super()._make_api_fetch()

        def call(method: str, url: str, params: str | dict | None = None,
                 body: str | dict | None = None) -> str:
            result = upstream(method, url, params, body)
            broken, passed, total = aa_grade(
                self._task.assertions, self._task.initial_state, self._world)
            self.state.guardrail_broken = broken
            self.state.objectives_passed = passed
            self.state.objectives_total = total
            return result

        call.__name__ = upstream.__name__
        call.__doc__ = upstream.__doc__
        return call


if __name__ == "__main__":
    AffineAutomationBenchToolset.run()
