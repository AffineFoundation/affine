# /// script
# requires-python = ">=3.12"
# dependencies = ["stirrup=={version}"]
# ///
"""Artificial Analysis' Stirrup agent loop driven against one OpenAI-compatible
endpoint, the way AA runs GDPval-AA / AA-Briefcase: one `code_exec` tool over a
local sandbox directory, a validating `finish(reason, paths)` tool and an
`abandon_task_finish(reason)` tool, up to --max-turns turns, deliverables copied
to --output-dir when the agent finishes.

Reference files: --input-dir (a directory in this runtime) is uploaded into
the agent's sandbox as `<sandbox>/<dirname>/...`; the literal `{{WORKDIR}}`
in the task prompt is replaced by the sandbox path so the prompt can list
absolute reference paths as AA's does.

Result marker: `<output-dir>/.stirrup_result.json` -- {finished, tool, reason,
paths, turns} -- for the task's grade.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.core.models import AssistantMessage, Tool, ToolResult, ToolUseCountMetadata
from stirrup.tools.code_backends import LocalCodeExecToolProvider
from stirrup.tools.finish import SIMPLE_FINISH_TOOL

RESULT_NAME = ".stirrup_result.json"
WORKDIR_TOKEN = "{{WORKDIR}}"


class AbandonParams(BaseModel):
    reason: Annotated[str, Field(description="Brief reason the task cannot be completed.")]


async def _abandon(params: AbandonParams) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content=params.reason, metadata=ToolUseCountMetadata(), success=True)


ABANDON_TOOL: Tool[AbandonParams, ToolUseCountMetadata] = Tool[AbandonParams, ToolUseCountMetadata](
    name="abandon_task_finish",
    description=("Give up on the task with a brief reason instead of submitting files. Only "
                 "when required inputs are missing, a hard dependency is unavailable, or the "
                 "request is incoherent - not to escape difficulty."),
    parameters=AbandonParams,
    executor=_abandon,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--system-prompt", default="")
    p.add_argument("--task", required=True)
    p.add_argument("--max-turns", type=int, default=250)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--context-window-tokens", type=int, default=131072)
    p.add_argument("--shell-timeout", type=int, default=600)
    p.add_argument("--workdir", required=True)
    p.add_argument("--input-dir", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--reasoning-effort", default="")
    return p.parse_args()


async def main() -> None:
    a = parse_args()
    Path(a.workdir).mkdir(parents=True, exist_ok=True)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = ChatCompletionsClient(
        model=a.model, max_tokens=a.max_tokens, context_window_tokens=a.context_window_tokens,
        base_url=a.base_url, api_key=a.api_key,
        reasoning_effort=a.reasoning_effort or None, max_retries=2)
    sandbox = LocalCodeExecToolProvider(temp_base_dir=a.workdir, shell_timeout=a.shell_timeout)
    agent = Agent(client=client, name="agent", max_turns=a.max_turns,
                  system_prompt=a.system_prompt or None, tools=[sandbox],
                  finish_tool=[SIMPLE_FINISH_TOOL, ABANDON_TOOL])
    inputs = [a.input_dir] if a.input_dir and os.path.isdir(a.input_dir) else None
    result = {"finished": False, "tool": None, "reason": None, "paths": [], "turns": 0}
    try:
        async with agent.session(output_dir=out, input_files=inputs,
                                 cache_on_interrupt=False, clear_cache_on_success=True) as session:
            task = a.task.replace(WORKDIR_TOKEN, str(sandbox.temp_dir))
            finish_params, history, metadata = await session.run(task)
            result["turns"] = sum(1 for turn in history for m in turn if isinstance(m, AssistantMessage))
            if finish_params is not None:
                result["finished"] = True
                result["tool"] = ("finish" if isinstance(finish_params, SIMPLE_FINISH_TOOL.parameters)
                                  else "abandon_task_finish")
                result["reason"] = getattr(finish_params, "reason", None)
                result["paths"] = list(getattr(finish_params, "paths", []) or [])
    finally:
        (out / RESULT_NAME).write_text(json.dumps(result))


if __name__ == "__main__":
    asyncio.run(main())
