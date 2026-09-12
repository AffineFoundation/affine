"""AutomationBench tool server with the register/_register API bridge.

verifiers launches a task-scoped tool server as `python -m <module of the
toolset class>`, so the shim lives in its own module with a `run()` guard.
The pods' verifiers (a298bcf) calls `register(mcp)`; research-environments
b10db76 implements the newer `_register`, and without this bridge the server
died at start-up and the model saw zero tools (probe 2026-09-12: 12/12
rollouts hallucinated `<tool_call>` XML). Harmless on newer verifiers.
"""

from __future__ import annotations

from automationbench_v1.servers.toolset import AutomationBenchToolset


class AffineAutomationBenchToolset(AutomationBenchToolset):
    def register(self, mcp) -> None:
        self._register(mcp)


if __name__ == "__main__":
    AffineAutomationBenchToolset.run()
