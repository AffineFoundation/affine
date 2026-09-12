"""mini-swe-agent driven in text-based mode: THOUGHT + one fenced bash block
per turn (mini_textbased config, litellm_textbased model class) instead of the
built-in harness's native tool calls. Trace messages then match the affine
corpus action contract after the standard mswea fence normalization.

Lives in the affine repo under rollouts/harnesses/mini_swe_textbased and is
installed editable on the datagen pods at
/root/prime-pilot/mini-swe-textbased (ops/king-datagen/deploy_pods.sh ships
this file there).

Fence contract (2026-09-11): the packaged `mini_textbased` config asks for
```mswea_bash_command fences, but the corpus stores every such action with
the fence rewritten to ```bash (datagen.slicer FOREIGN_FENCE_RE — what the
`bash` dialect parser expects), so a miner distilled from D writes ```bash
and the unmodified harness rejects it (RepeatedFormatError on 8 % of the
reign-11 king's replies, 68 of 298 finished rollouts). The action regex
below accepts both fences; the prompt templates are unchanged, so the
teacher keeps writing the mswea fence and D is stored exactly as before."""

from verifiers.v1.clients import ModelContext
from verifiers.v1.harnesses.mini_swe_agent.harness import (
    PROGRAM_SOURCE,
    MiniSWEAgentHarness,
)
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

__all__ = ["MiniSWETextbasedHarness", "ACTION_REGEX"]

# mini-swe-agent's litellm_textbased default is ```mswea_bash_command only;
# ```bash is what D stores and what distilled miners write. Same span rules
# as the default (optional trailing whitespace after the fence word, body up
# to the next closing fence); exactly one match per reply is still required.
ACTION_REGEX = r"```(?:bash|mswea_bash_command)\s*\n(.*?)\n```"


class MiniSWETextbasedHarness(MiniSWEAgentHarness):
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
        _, prompt = self.resolve_text_prompt(data)
        source = PROGRAM_SOURCE.replace("{version}", self.config.version)
        args = [
            "--model",
            ctx.model,
            "--model-class",
            "litellm_textbased",
            "--task",
            prompt,
            "--exit-immediately",
            "--yolo",
            "-c",
            "mini_textbased",
            "-c",
            "agent.cost_limit=0",
            "-c",
            "environment.timeout=86400",
            "-c",
            "model.cost_tracking=ignore_errors",
            "-c",
            f"model.action_regex={ACTION_REGEX}",
            "-c",
            "model.model_kwargs.custom_llm_provider=openai",
            "-c",
            "model.model_kwargs.temperature=0.7",
            "-c",
            f"model.model_kwargs.api_base={endpoint}",
            "-c",
            f"model.model_kwargs.api_key={secret}",
        ]
        env = {
            **self.config.resolved_env,
            "MSWEA_CONFIGURED": "true",
            "MSWEA_SILENT_STARTUP": "true",
        }
        program = await runtime.prepare_uv_script(source, self.config.resolved_env)
        return await runtime.run_program([*program, *args], env)
