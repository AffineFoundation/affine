# /// script
# requires-python = ">=3.12"
# dependencies = ["harbor=={version}"]
# ///
"""Resume a Terminus 2 (harbor) rollout from a stored prefix.

Mirrors verifiers' harnesses/terminus_2/program.py (same LocalEnvironment,
same agent construction) and adds two phases before the agent loop:

1. Replay: parse every assistant reply of the prefix with Terminus's own
   JSON parser and send its keystrokes into the fresh tmux session with the
   agent's own `_execute_commands` (durations capped at 60 s as in the
   original loop). The incremental pane buffer is then flushed so the
   teacher's first observation shows only its own commands.
2. Continue: seed the agent's Chat with the prefix minus its last user
   message, and run `_run_agent_loop` with that last user message as the
   prompt — exactly the call `run()` would have made next.

argv: --state-file --base-url --api-key --model --system-prompt --report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path, PurePosixPath

from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_2.terminus_2 import Command
from harbor.environments.base import ExecResult
from harbor.llms.chat import Chat
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths


class LocalEnvironment:
    default_user = None
    session_id = "verifiers"

    async def exec(self, command: str, cwd: str | None = None,
                   env: dict[str, str] | None = None,
                   timeout_sec: int | None = None,
                   user: str | int | None = None) -> ExecResult:
        _ = user
        result = await asyncio.to_thread(
            subprocess.run, command, shell=True, cwd=cwd,
            env={**os.environ, **(env or {})}, capture_output=True,
            text=True, timeout=timeout_sec, check=False)
        return ExecResult(stdout=result.stdout, stderr=result.stderr,
                          return_code=result.returncode)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-file", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--system-prompt", default="")
    ap.add_argument("--report", required=True)
    return ap.parse_args()


async def main() -> None:
    args = parse_args()
    state = json.loads(Path(args.state_file).read_text(encoding="utf-8"))
    messages = state["messages"]
    logs_dir = Path(os.environ["TMUX_TMPDIR"])
    logs_dir.mkdir(mode=0o700, exist_ok=True)
    EnvironmentPaths.agent_dir = PurePosixPath(logs_dir)

    agent = Terminus2(
        logs_dir=logs_dir, model_name=args.model, api_base=args.base_url,
        llm_kwargs={"custom_llm_provider": "openai", "api_key": args.api_key},
        record_terminal_session=False,
    )
    if args.system_prompt:
        call = agent._llm.call

        async def call_with_system_prompt(*a, message_history, **kw):
            return await call(*a, message_history=[
                {"role": "system", "content": args.system_prompt}, *message_history], **kw)

        agent._llm.call = call_with_system_prompt

    environment = LocalEnvironment()
    await agent.setup(environment)
    session = agent._session
    report: dict = {"replay": [], "replay_n": 0, "replay_seconds": 0.0}
    t0 = time.time()
    pending_completion = False
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        result = agent._parser.parse_response(m.get("content") or "")
        pending_completion = bool(result.is_task_complete) and not result.error
        if result.error:
            report["replay"].append({"msg_idx": i, "skipped": "parse_error"})
            continue
        commands = [Command(keystrokes=cmd.keystrokes,
                            duration_sec=min(cmd.duration, 60))
                    for cmd in result.commands]
        timed_out, _ = await agent._execute_commands(commands, session)
        report["replay"].append({"msg_idx": i, "n_commands": len(commands),
                                 "timed_out": timed_out})
        report["replay_n"] += 1
    # Drop everything the replay printed: the teacher sees the recorded
    # observation in the prefix, then only its own commands' output.
    await session.get_incremental_output()
    report["replay_seconds"] = round(time.time() - t0, 1)
    Path(args.report).write_text(json.dumps(report))

    agent._reset_per_run_state()
    agent._chat = Chat(agent._llm, interleaved_thinking=agent._interleaved_thinking)
    agent._chat._messages = [{"role": m["role"], "content": m.get("content") or ""}
                             for m in messages[:-1]]
    agent._context = AgentContext()
    agent._max_episodes = int(state.get("max_turns") or agent._max_episodes)
    agent._pending_completion = pending_completion
    if messages[-1].get("role") != "user":
        raise SystemExit("prefix must end with a user message")
    try:
        await agent._run_agent_loop(
            initial_prompt=messages[-1].get("content") or "",
            chat=agent._chat,
            original_instruction=state.get("task_prompt") or "",
        )
    finally:
        report["n_episodes"] = agent._n_episodes
        msgs = agent._chat.messages
        first = next((mm for mm in msgs[len(messages) - 1:] if mm.get("role") == "assistant"), None)
        report["first_reply"] = first.get("content") if first else None
        Path(args.report).write_text(json.dumps(report))


if __name__ == "__main__":
    asyncio.run(main())
