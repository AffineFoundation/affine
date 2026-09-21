"""Stirrup — Artificial Analysis' agent harness — as a verifiers harness.

Why: AA-Briefcase + GDPval-AA are 25 % of the AA Intelligence Index and D
held no state of their shape (one `code_exec` tool, hundreds of turns, source
files in a sandbox, deliverable FILES, an explicit `finish(paths)` call). The
king never-submits / loops on long runs, and the only lever the meter has is
teacher-shaped prefixes at those states. This harness makes the teacher (and
the king seat) play the tasks under Stirrup itself, through the interception
server like every other program harness, so the recorded trajectories are the
prompts Stirrup sends — tool schemas, turn-budget warnings and all.

Stirrup speaks OpenAI Chat Completions with a `base_url`, so the program is a
uv script (`program.py`, `stirrup=={version}`) launched in the task runtime
with the interception endpoint + secret, the same way the Terminus 2 harness
runs harbor. No MCP: the sandbox tool is Stirrup's own local executor over a
temp directory under `workdir`; the task stages reference files under
`input_dir` (uploaded into the sandbox as `<sandbox>/<dirname>/...`, the
prompt's `{{WORKDIR}}` token is replaced by the sandbox path) and reads the
deliverables the agent submitted from `output_dir`.

Lives in the affine repo under rollouts/harnesses/stirrup_agent and is
installed editable on the datagen pods (ops/king-datagen/deploy_pods.sh);
harness id `stirrup_agent` in policies.toml.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

__all__ = ["StirrupAgentHarness", "StirrupAgentHarnessConfig"]

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()

WORKDIR = "/workspace/stirrup"
INPUT_DIR = "/workspace/reference_files"
OUTPUT_DIR = "/workspace/output"
RESULT_PATH = f"{OUTPUT_DIR}/.stirrup_result.json"


class StirrupAgentHarnessConfig(HarnessConfig):
    version: str = Field(default="0.2.0", pattern=r"^[A-Za-z0-9._+-]+$")
    """Stirrup release to install, pinned for reproducibility."""
    max_turns: int = 250
    """AA's GDPval-AA budget (Briefcase runs 500). One turn = one assistant
    message and its tool calls; Stirrup warns the model over the last turns."""
    max_tokens: int = 16384
    """Per-reply output cap the client sends."""
    context_window_tokens: int = 131072
    """Stirrup summarises history past ~70 % of this."""
    shell_timeout: int = 600
    """Seconds per `code_exec` command (AA: 10 minutes)."""
    workdir: str = WORKDIR
    input_dir: str = INPUT_DIR
    output_dir: str = OUTPUT_DIR
    reasoning_effort: str = ""
    """Passed as `reasoning_effort` when set (the teacher ignores it)."""


class StirrupAgentHarness(Harness[StirrupAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = False

    async def setup(self, runtime: Runtime) -> None:
        source = PROGRAM_SOURCE.replace("{version}", self.config.version)
        await runtime.prepare_uv_script(source, self.config.resolved_env)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        if self.config.disabled_tools:
            raise ValueError("stirrup_agent does not support disabling tools")
        system_prompt, prompt = self.resolve_text_prompt(data)
        if prompt is None:
            raise ValueError("stirrup_agent requires a task prompt")
        cfg = self.config
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--system-prompt={system_prompt or ''}",
            f"--task={prompt}",
            f"--max-turns={cfg.max_turns}",
            f"--max-tokens={cfg.max_tokens}",
            f"--context-window-tokens={cfg.context_window_tokens}",
            f"--shell-timeout={cfg.shell_timeout}",
            f"--workdir={cfg.workdir}",
            f"--input-dir={cfg.input_dir}",
            f"--output-dir={cfg.output_dir}",
            f"--reasoning-effort={cfg.reasoning_effort}",
        ]
        source = PROGRAM_SOURCE.replace("{version}", cfg.version)
        program = await runtime.prepare_uv_script(source, cfg.resolved_env)
        return await runtime.run_program([*program, *args], dict(cfg.resolved_env))
