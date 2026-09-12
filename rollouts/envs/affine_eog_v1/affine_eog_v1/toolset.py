"""EnterpriseOps-Gym tool server with the register/_register API bridge.

Own module with a `run()` guard because verifiers starts the task-scoped
server as `python -m <module of the toolset class>`. See
affine_autobench_v1/toolset.py for why the bridge is needed on the pods.
"""

from __future__ import annotations

from enterprise_ops_gym_v1.toolset import EnterpriseOpsToolset


class AffineEnterpriseOpsToolset(EnterpriseOpsToolset):
    def register(self, mcp) -> None:
        self._register(mcp)


if __name__ == "__main__":
    AffineEnterpriseOpsToolset.run()
