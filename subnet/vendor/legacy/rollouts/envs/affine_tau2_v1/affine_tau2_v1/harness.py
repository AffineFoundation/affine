"""τ² harness with a configurable, cheap user simulator.

prime-envs' `Tau2Harness` hard-wires the customer to `gpt-4.1` through Prime
inference (or `OPENAI_API_KEY`). For datagen the customer is a cheap
OpenAI-compatible model - DeepSeek V4.1 Flash on Engy by default
(~$0.001 per task at τ²'s 10-30 short user turns) - selected by
`--env.agent.harness.user-model` / `user-base-url` / `user-key-env`, or the
environment variables `TAU2_USER_MODEL` / `TAU2_USER_BASE_URL` /
`TAU2_USER_KEY_ENV`. Everything else (τ²'s orchestrator as a subprocess, the
agent routed through the verifiers interception endpoint, the simulation
stored in `trace.info["tau2"]`, the `user_completed` / `tau2_<reason>` stop)
is the upstream harness; only the user LLM block is replaced.

Example values in the tool schemas (2026-09-18, docs/tau2-airline-failures.md):
the airline trap is `get_user_details(user_id="sara_doe_496")` - the king
fills the slot with the EXAMPLE value the schema carries ("such as
'sara_doe_496'"). τ²'s telecom tool docstrings carry no examples, so by
default this harness adds airline-style examples to the identifier
parameters (`phone_number: ... such as '555-123-4567'`, `customer_id: ...
such as 'C1234'`, ...) before the tools are built - the same realistic bait,
in the domain where the pool is disjoint from the benchmark. The values are
not in the database; a call with one of them fails like any invented id.
`--env.agent.harness.example-values false` restores the upstream schemas.

The program entry (`python -m affine_tau2_v1.harness`) reproduces the
upstream one (its litellm message / role patches) plus the example-value
patch, then runs τ²'s `run_task`.
"""

from __future__ import annotations

import json
import os
import sys

import verifiers.v1 as vf
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task as TauTask
from tau2_bench_v1 import harness as base
from tau2_bench_v1.taskset import Tau2Data

DEFAULT_USER_MODEL = "deepseek-v4.1-flash"
DEFAULT_USER_BASE_URL = "https://api.engy.ai/v1"
DEFAULT_USER_KEY_ENV = "ENGY"


class AffineTau2HarnessConfig(base.Tau2HarnessConfig):
    user_model: str = os.environ.get("TAU2_USER_MODEL", DEFAULT_USER_MODEL)
    """OpenAI-compatible model id that plays the customer."""
    user_base_url: str = os.environ.get("TAU2_USER_BASE_URL", DEFAULT_USER_BASE_URL)
    user_key_env: str = os.environ.get("TAU2_USER_KEY_ENV", DEFAULT_USER_KEY_ENV)
    """Environment variable holding the user model's API key (never the key itself)."""
    user_temperature: float = 0.0
    example_values: bool = True
    """Add airline-style example values to the identifier parameters of the telecom tools."""


class AffineTau2Harness(base.Tau2Harness, vf.Harness[AffineTau2HarnessConfig]):
    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace[Tau2Data],
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: Tau2Data,
    ) -> vf.ProgramResult:
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
            "user_llm_args": {
                "temperature": cfg.user_temperature,
                "api_base": cfg.user_base_url,
                "api_key": user_key,
                "timeout": 86400,
            },
            "example_values": cfg.example_values,
        }
        result = await runtime.run_program(
            [sys.executable, "-m", __name__],
            {**cfg.env, base._RUN_CONFIG: json.dumps(run_config)},
        )
        if result.exit_code != 0:
            return result
        simulation = SimulationRun.model_validate_json(next(
            line.removeprefix(base._RESULT_PREFIX)
            for line in reversed(result.stdout.splitlines())
            if line.startswith(base._RESULT_PREFIX)))
        simulation.id = trace.id
        trace.info["tau2"] = {"simulation": simulation.model_dump(mode="json")}
        trace.stop(stop_condition(simulation.termination_reason))
        return result


# τ² termination -> the stop names the corpus fold understands
# (affine.corpus.view CLEAN_STOP_CONDITIONS). The conversation ending on
# purpose - the customer says goodbye (USER_STOP) or the agent closes /
# transfers (AGENT_STOP) - is `agent_completed`, so the agent's final message
# to the customer is the rollout's deliberate final reply. Running out of
# steps or making too many tool errors is the agent not finishing
# (`max_turns`), a clean failure the king seat exists to capture - NOT an
# infrastructure error (the upstream `tau2_too_many_errors` stop made every
# such king rollout `errored` and the fold dropped it, 11 / 16 in the
# 2026-09-17 probe). Real errors (invalid messages, LLM / tool exceptions)
# keep the upstream `tau2_<reason>` name and stay errored.
CLEAN_STOPS = {
    TerminationReason.USER_STOP: "agent_completed",
    TerminationReason.AGENT_STOP: "agent_completed",
    TerminationReason.MAX_STEPS: "max_turns",
    TerminationReason.TOO_MANY_ERRORS: "max_turns",
}


def stop_condition(reason: TerminationReason) -> str:
    return CLEAN_STOPS.get(reason, f"tau2_{reason.value}")


__all__ = ["AffineTau2Harness", "AffineTau2HarnessConfig"]


# Airline-style example values for the telecom identifier parameters
# (parameter name -> example), spliced into the `Args:` lines of the
# TelecomTools docstrings that τ² parses into the tool schemas.
TELECOM_EXAMPLES = {
    "phone_number": "555-123-4567",
    "customer_id": "C1234",
    "line_id": "L5678",
    "plan_id": "P1002",
    "id": "C1234",
    "full_name": "Jane Doe",
}


def with_example_values(doc: str, examples: dict[str, str] = TELECOM_EXAMPLES) -> str:
    out = []
    for line in doc.splitlines():
        stripped = line.strip()
        name, sep, rest = stripped.partition(":")
        if sep and name in examples and rest.strip() and "such as" not in rest and "e.g." not in rest and line.startswith(" "):
            line = line.rstrip().rstrip(".") + f", such as '{examples[name]}'."
        out.append(line)
    return "\n".join(out)


def apply_example_values() -> int:
    from tau2.domains.telecom.tools import TelecomTools  # noqa: PLC0415 - program entry only

    n = 0
    for attr in dir(TelecomTools):
        fn = getattr(TelecomTools, attr, None)
        doc = getattr(fn, "__doc__", None)
        if not callable(fn) or not doc or attr.startswith("_"):
            continue
        new = with_example_values(doc)
        if new != doc:
            fn.__doc__ = new
            n += 1
    return n


if __name__ == "__main__":
    from tau2.data_model.message import AssistantMessage, Message  # noqa: PLC0415
    from tau2.run import run_task  # noqa: PLC0415
    from tau2.user.base import UserState  # noqa: PLC0415
    from tau2.utils import llm_utils  # noqa: PLC0415

    # Upstream patches: keep LiteLLM's raw assistant messages when τ² rebuilds
    # the history (reasoning survives across turns) - identical to
    # tau2_bench_v1.harness.__main__.
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
        apply_example_values()
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
    print(f"{base._RESULT_PREFIX}{simulation.model_dump_json()}")
