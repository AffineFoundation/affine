"""Per-task tool server for affine-agent-v1.

general_agent_v1's `GeneralAgentToolset` implements the MCP registration hook
as `_register(mcp)`; the verifiers checkout on the datagen pods
(prime-pilot/verifiers, 2026-08-08) calls `register(mcp)` after `setup_task`
and its default `register` only discovers `@vf.tool` methods — of which the
dynamic toolset has none. Probe 2026-09-07: the tool server started, advertised
zero tools, and the teacher answered without ever seeing them. This subclass
forwards the hook; everything else (task loading over `/task`, DB-hash state
push) is upstream's.
"""

from __future__ import annotations

from general_agent_v1.servers.toolset import GeneralAgentToolset


class AffineAgentToolset(GeneralAgentToolset):
    def register(self, mcp) -> None:
        self._register(mcp)


if __name__ == "__main__":
    AffineAgentToolset.run()
