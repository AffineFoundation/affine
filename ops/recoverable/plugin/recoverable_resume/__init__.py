"""verifiers harness plugin: continue a stored rollout prefix with a new model.

Loaded by id (`--env.agent.harness.id recoverable_resume`) when this
package's parent directory is on PYTHONPATH of the verifiers eval process.
One eval = one task = one state (`--env.agent.harness.state_file`). The
task's own taskset, image, workdir and reward function are untouched, so the
continuation is graded exactly like a fresh rollout.

Per harness (`state["resume_kind"]`):

  textbased  mini-swe-agent textbased loop. Replays the prefix's executed
             ```mswea_bash_command blocks in the container, then seeds
             mini-swe's DefaultAgent with the prefix (programs/textbased_resume.py).
  bash       verifiers `bash` harness (bash + edit tools). Replays the
             prefix's tool calls with the program's semantics
             (programs/replay_tools.py), then launches the stock program on
             the prefix as a Messages prompt (`--initial-messages-file`).
  terminus   harbor Terminus 2. Replays keystrokes into a fresh tmux
             session, seeds the agent's Chat, resumes the agent loop
             (programs/terminus_resume.py).
  null       chat loop with MCP tools (wiki). Nothing to replay: the tools
             are pure functions of their arguments; the stock `null` program
             is launched on the prefix.

Every model call goes through the interception endpoint, so the trace
records only the continuation (the prefix is not re-sampled); `max_turns`
is the remaining budget the driver passes per state.
"""

from __future__ import annotations

import json
from pathlib import Path

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.bash import BashHarness, BashHarnessConfig
from verifiers.v1.harnesses.bash.harness import BASH_SYSTEM_PROMPT, EDIT_SYSTEM_PROMPT
from verifiers.v1.harnesses.null import NullHarness, NullHarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import (
    AssistantMessage,
    Messages,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from recoverable_resume import shield

__all__ = ["RecoverableResumeHarness"]

# Importing the plugin = running a continuation next to the datagen
# supervisor; keep its reaper off our containers (see shield.py).
shield.install()

PROGRAMS = Path(__file__).resolve().parent / "programs"
TEXTBASED_SOURCE = (PROGRAMS / "textbased_resume.py").read_text()
REPLAY_TOOLS_SOURCE = (PROGRAMS / "replay_tools.py").read_text()
TERMINUS_SOURCE = (PROGRAMS / "terminus_resume.py").read_text()


class RecoverableResumeConfig(HarnessConfig):
    state_file: str = ""
    """Host path of the state JSON written by ops/recoverable/states.py."""
    report_dir: str = ""
    """Host directory for the per-rollout replay/continuation report
    (`<trace id>.json`); empty = no report."""
    mini_swe_version: str = "2.4.6"
    harbor_version: str = "0.20.0"
    replay_timeout: int = 900
    """Seconds one replayed command may take (the original run allowed a day)."""


def to_messages(wire: list[dict]) -> Messages:
    out: Messages = []
    for m in wire:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "user":
            out.append(UserMessage(content=content))
        elif role == "assistant":
            calls = [ToolCall(id=tc["id"], name=tc["function"]["name"],
                              arguments=tc["function"]["arguments"])
                     for tc in (m.get("tool_calls") or [])]
            out.append(AssistantMessage(content=content or None,
                                        tool_calls=calls or None))
        elif role == "tool":
            out.append(ToolMessage(tool_call_id=m.get("tool_call_id") or "",
                                   content=content, name=m.get("name")))
    return out


class RecoverableResumeHarness(Harness[RecoverableResumeConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = False
    NEEDS_CONTAINER = False

    def __init__(self, config: RecoverableResumeConfig) -> None:
        super().__init__(config)
        self.state = json.loads(Path(config.state_file).read_text(encoding="utf-8"))
        self.kind = self.state["resume_kind"]
        inner_cfg = {"env": config.env, "forward_env": config.forward_env,
                     "tool_timeout": config.tool_timeout}
        self._bash = BashHarness(BashHarnessConfig(id="bash", edit=True, **inner_cfg))
        self._null = NullHarness(NullHarnessConfig(id="null", **inner_cfg))

    # -- sources -----------------------------------------------------------------

    def _textbased_source(self) -> str:
        return TEXTBASED_SOURCE.replace("{version}", self.config.mini_swe_version)

    def _terminus_source(self) -> str:
        return TERMINUS_SOURCE.replace("{version}", self.config.harbor_version)

    async def setup(self, runtime: Runtime) -> None:
        env = self.config.resolved_env
        if self.kind == "textbased":
            await runtime.prepare_uv_script(self._textbased_source(), env)
        elif self.kind == "bash":
            await runtime.prepare_uv_script(REPLAY_TOOLS_SOURCE, env)
            await self._bash.setup(runtime)
        elif self.kind == "terminus":
            await runtime.prepare_uv_script(self._terminus_source(), env)
        elif self.kind == "null":
            await self._null.setup(runtime)
        else:
            raise ValueError(f"unknown resume_kind {self.kind!r}")

    # -- helpers -------------------------------------------------------------------

    async def _write_state(self, runtime: Runtime, trace: Trace) -> str:
        path = f"/tmp/vf-resume-{trace.id}.json"
        await runtime.write(path, json.dumps(self.state).encode())
        return path

    async def _collect_report(self, runtime: Runtime, trace: Trace,
                              path: str, extra: dict | None = None) -> None:
        if not self.config.report_dir:
            return
        report: dict = {"state_id": self.state["state_id"], "trace_id": trace.id}
        try:
            report.update(json.loads((await runtime.read(path)).decode()))
        except Exception as e:  # noqa: BLE001 - the report is telemetry
            report["report_error"] = f"{type(e).__name__}: {e}"
        if extra:
            report.update(extra)
        out = Path(self.config.report_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{trace.id}.json").write_text(json.dumps(report))

    def _write_host_report(self, trace: Trace, report: dict) -> None:
        if not self.config.report_dir:
            return
        out = Path(self.config.report_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{trace.id}.json").write_text(json.dumps(
            {"state_id": self.state["state_id"], "trace_id": trace.id, **report}))

    def _split_system(self) -> tuple[str | None, list[dict]]:
        msgs = list(self.state["messages"])
        if msgs and msgs[0].get("role") == "system":
            return msgs[0].get("content") or "", msgs[1:]
        return None, msgs

    # -- launch --------------------------------------------------------------------

    async def launch(self, ctx: ModelContext, trace: Trace, runtime: Runtime,
                     endpoint: str, secret: str, mcp_urls: dict[str, str],
                     data: TaskData) -> ProgramResult:
        if self.kind == "textbased":
            return await self._launch_textbased(ctx, trace, runtime, endpoint, secret)
        if self.kind == "bash":
            return await self._launch_bash(ctx, trace, runtime, endpoint, secret,
                                           mcp_urls, data)
        if self.kind == "terminus":
            return await self._launch_terminus(ctx, trace, runtime, endpoint,
                                               secret, data)
        if self.kind == "null":
            return await self._launch_null(ctx, trace, runtime, endpoint, secret,
                                           mcp_urls, data)
        raise ValueError(f"unknown resume_kind {self.kind!r}")

    async def _launch_textbased(self, ctx, trace, runtime, endpoint, secret) -> ProgramResult:
        state_path = await self._write_state(runtime, trace)
        report_path = f"/tmp/vf-resume-{trace.id}.report.json"
        program = await runtime.prepare_uv_script(self._textbased_source(),
                                                  self.config.resolved_env)
        temperature = (self.state.get("sampling") or {}).get("temperature", 0.8)
        # `--key=value` form: the interception secret may start with "-".
        args = [f"--state-file={state_path}", f"--base-url={endpoint}",
                f"--api-key={secret}", f"--model={ctx.model}",
                f"--report={report_path}", f"--temperature={temperature}",
                f"--replay-timeout={self.config.replay_timeout}"]
        env = {**self.config.resolved_env, "MSWEA_CONFIGURED": "true",
               "MSWEA_SILENT_STARTUP": "true", "MSWEA_COST_TRACKING": "ignore_errors"}
        result = await runtime.run_program([*program, *args], env)
        await self._collect_report(runtime, trace, report_path,
                                   {"exit_code": result.exit_code,
                                    "stderr_tail": (result.stderr or "")[-1500:]})
        return result

    async def _launch_bash(self, ctx, trace, runtime, endpoint, secret, mcp_urls,
                           data) -> ProgramResult:
        state_path = await self._write_state(runtime, trace)
        report_path = f"/tmp/vf-resume-{trace.id}.report.json"
        replay = await runtime.prepare_uv_script(REPLAY_TOOLS_SOURCE,
                                                 self.config.resolved_env)
        rep = await runtime.run([*replay, "--state-file", state_path, "--report",
                                 report_path, "--timeout", str(self.config.replay_timeout)],
                                self.config.resolved_env)
        recorded_system, rest = self._split_system()
        expected = " ".join([BASH_SYSTEM_PROMPT, EDIT_SYSTEM_PROMPT])
        extra = {"replay_exit_code": rep.exit_code,
                 "replay_stderr_tail": (rep.stderr or "")[-1000:],
                 "system_prompt_match": recorded_system == expected}
        await self._collect_report(runtime, trace, report_path, extra)
        # The stock program re-creates the system prompt (bash + edit clause);
        # the recorded one is dropped so it is not duplicated.
        resumed = data.model_copy(update={"prompt": to_messages(rest),
                                          "system_prompt": None})
        return await self._bash.launch(ctx, trace, runtime, endpoint, secret,
                                       mcp_urls, resumed)

    async def _launch_null(self, ctx, trace, runtime, endpoint, secret, mcp_urls,
                           data) -> ProgramResult:
        recorded_system, rest = self._split_system()
        self._write_host_report(trace, {
            "system_prompt_match": recorded_system == data.system_prompt})
        resumed = data.model_copy(update={"prompt": to_messages(rest)})
        return await self._null.launch(ctx, trace, runtime, endpoint, secret,
                                       mcp_urls, resumed)

    async def _launch_terminus(self, ctx, trace, runtime, endpoint, secret,
                               data) -> ProgramResult:
        state_path = await self._write_state(runtime, trace)
        report_path = f"/tmp/vf-resume-{trace.id}.report.json"
        tmux_dir = f"/tmp/vf-terminus-2-{trace.id}"
        env = {**self.config.resolved_env, "TMUX_TMPDIR": tmux_dir}
        system_prompt, _ = self.resolve_text_prompt(data)
        args = [f"--state-file={state_path}", f"--base-url={endpoint}",
                f"--api-key={secret}", f"--model={ctx.model}",
                f"--system-prompt={system_prompt or ''}", f"--report={report_path}"]
        try:
            program = await runtime.prepare_uv_script(self._terminus_source(),
                                                      self.config.resolved_env)
            result = await runtime.run_program([*program, *args], env)
            await self._collect_report(runtime, trace, report_path,
                                       {"exit_code": result.exit_code,
                                        "stderr_tail": (result.stderr or "")[-1500:]})
            return result
        finally:
            try:
                await runtime.run(
                    ["sh", "-c",
                     'tmux kill-server >/dev/null 2>&1 || true; rm -rf "$TMUX_TMPDIR"'],
                    {"TMUX_TMPDIR": tmux_dir})
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
