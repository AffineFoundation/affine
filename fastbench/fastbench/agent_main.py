"""Run mini-swe-agent's SWE-bench batch command with the loop guard installed.

The batch runner instantiates ProgressTrackingAgent directly and ignores the
``agent_class`` config key that the single-instance runners honour, so the
guard is installed by swapping that attribute before the CLI builds any agent.
Every flag behaves exactly as ``mini-extra swebench``: this invokes the same
typer app.
"""

from __future__ import annotations

from minisweagent.run.benchmarks import swebench as _swebench

from .loopguard import guard


def main() -> None:
    _swebench.ProgressTrackingAgent = guard(_swebench.ProgressTrackingAgent)
    _swebench.app()


if __name__ == "__main__":
    main()
