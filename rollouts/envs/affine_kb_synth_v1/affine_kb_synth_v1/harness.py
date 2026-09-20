"""τ² harness for the KB-backed synth domains: affine_tau2_synth_v1's harness
(DeepSeek customer, fold-clean stops, generic example values) whose program
entry installs the knowledge-base environment (kb.py) for the task's domain
before τ²'s `run_task`, so the agent sees τ³'s header + `KB_search` instead of
the policy text. `--env.agent.harness.kb false` runs the plain synth
environment through this harness (A/B control)."""

from __future__ import annotations

import json
import os
import sys

from affine_tau2_synth_v1.taskset import SYNTH_DATA_DIR  # noqa: F401  - sets TAU2_DATA_DIR first

import verifiers.v1 as vf  # noqa: E402
from affine_tau2_synth_v1 import harness as synth  # noqa: E402
from tau2.data_model.simulation import SimulationRun  # noqa: E402
from tau2.data_model.tasks import Task as TauTask  # noqa: E402
from tau2.utils.utils import DATA_DIR  # noqa: E402
from tau2_bench_v1 import harness as base  # noqa: E402

from affine_tau2_v1 import harness as telecom  # noqa: E402


class AffineKBSynthHarnessConfig(synth.AffineTau2SynthHarnessConfig):
    kb: bool = True
    """Move the domain policy into the knowledge base behind KB_search (false = plain synth prompt)."""


class AffineKBSynthHarness(synth.AffineTau2SynthHarness, vf.Harness[AffineKBSynthHarnessConfig]):
    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):  # type: ignore[override]
        del mcp_urls
        cfg = self.config
        user_key = os.environ.get(cfg.user_key_env, "")
        if not user_key:
            raise RuntimeError(f"user simulator key env {cfg.user_key_env!r} is not set")
        model = ctx.model.rsplit("/", 1)[-1]
        agent_llm = f"openai/{ctx.model.removeprefix('openai/')}"
        agent_args: dict[str, object] = {"api_base": endpoint, "api_key": secret, "timeout": None}
        if model.startswith("gpt-"):
            agent_llm = f"openai/responses/{model}"
            agent_args["include"] = ["reasoning.encrypted_content"]
        run_config = {
            "domain": data.domain,
            "task": {
                **data.model_dump(mode="json", include=set(TauTask.model_fields) - {"description"}),
                "description": data.tau_description.model_dump(mode="json") if data.tau_description else None,
            },
            "agent_llm": agent_llm,
            "agent_llm_args": agent_args,
            "user_llm": f"openai/{cfg.user_model}",
            "user_llm_args": {"temperature": cfg.user_temperature, "api_base": cfg.user_base_url,
                              "api_key": user_key, "timeout": 86400},
            "example_values": cfg.example_values,
            "kb": cfg.kb,
        }
        result = await runtime.run_program(
            [sys.executable, "-m", __name__],
            {**cfg.env, base._RUN_CONFIG: json.dumps(run_config), "TAU2_DATA_DIR": str(DATA_DIR)},
        )
        if result.exit_code != 0:
            return result
        lines = result.stdout.splitlines()
        simulation = SimulationRun.model_validate_json(next(
            line.removeprefix(base._RESULT_PREFIX) for line in reversed(lines) if line.startswith(base._RESULT_PREFIX)))
        simulation.id = trace.id
        trace.info["tau2"] = {"simulation": simulation.model_dump(mode="json")}
        kb_line = next((line for line in lines if line.startswith(_KB_PREFIX)), None)
        if kb_line:
            trace.info["kb"] = json.loads(kb_line.removeprefix(_KB_PREFIX))
        trace.stop(telecom.stop_condition(simulation.termination_reason))
        return result


_KB_PREFIX = "AFFINE_KB_INFO "

__all__ = ["AffineKBSynthHarness", "AffineKBSynthHarnessConfig"]


if __name__ == "__main__":
    from tau2.data_model.message import AssistantMessage, Message  # noqa: PLC0415
    from tau2.run import run_task  # noqa: PLC0415
    from tau2.user.base import UserState  # noqa: PLC0415
    from tau2.utils import llm_utils  # noqa: PLC0415

    from affine_kb_synth_v1 import kb  # noqa: PLC0415

    _tau_to_litellm_messages = llm_utils.to_litellm_messages
    _tau_flip_roles = UserState.flip_roles

    def _to_litellm_messages(messages: list[Message]) -> list[dict]:
        converted = _tau_to_litellm_messages(messages)
        for index, message in enumerate(messages):
            if isinstance(message, AssistantMessage) and message.raw_data:
                converted[index] = message.raw_data["message"]
        return converted

    def _flip_roles(self: UserState):
        flipped = _tau_flip_roles(self)
        for message, flipped_message in zip(self.messages, flipped, strict=True):
            if isinstance(flipped_message, AssistantMessage):
                flipped_message.raw_data = message.raw_data
        return flipped

    setattr(llm_utils, "to_litellm_messages", _to_litellm_messages)
    setattr(UserState, "flip_roles", _flip_roles)

    config = json.loads(os.environ[base._RUN_CONFIG])
    if config.get("example_values", True):
        synth.apply_example_values(config["domain"])
    if config.get("kb", True):
        info = kb.install(config["domain"])
    else:
        info = {"domain": config["domain"], "kb": False}
    simulation = run_task(
        domain=config["domain"],
        task=TauTask.model_validate(config["task"]),
        agent="llm_agent",
        user="user_simulator",
        llm_agent=config["agent_llm"],
        llm_args_agent=config["agent_llm_args"],
        llm_user=config["user_llm"],
        llm_args_user=config["user_llm_args"],
        max_steps=500,
    )
    print(f"{_KB_PREFIX}{json.dumps(info)}")
    print(f"{base._RESULT_PREFIX}{simulation.model_dump_json()}")
