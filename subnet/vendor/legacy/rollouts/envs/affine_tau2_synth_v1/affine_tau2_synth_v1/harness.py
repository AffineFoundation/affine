"""τ² harness for the tau2-synth domains: affine_tau2_v1's harness (Engy DeepSeek
customer, fold-clean stops) with the synth data dir and a GENERIC example-value
patch.

Example values (docs/tau2-airline-failures.md): the king fills identifier
slots with the example the schema shows. The synth tool docstrings carry no
examples, so before τ² builds the tools this program adds one to every
identifier parameter of the domain's toolkit classes - `patron_id: ... such
as 'PAT042'`, `email: ... such as 'jane.doe@email.com'`, `phone: ... such as
'555-123-4567'`. Id examples copy the domain database's own prefix (PAT, MEM,
CUST, ORD, ...) with a number that does not exist in the database, so a call
with one fails like any invented id. `--env.agent.harness.example-values
false` restores the upstream schemas.
"""

from __future__ import annotations

import json
import os
import re
import sys

from affine_tau2_synth_v1.taskset import SYNTH_DATA_DIR  # noqa: F401  - sets TAU2_DATA_DIR first

import verifiers.v1 as vf  # noqa: E402
from tau2.data_model.simulation import SimulationRun  # noqa: E402
from tau2.data_model.tasks import Task as TauTask  # noqa: E402
from tau2.utils.utils import DATA_DIR  # noqa: E402
from tau2_bench_v1 import harness as base  # noqa: E402

from affine_tau2_v1 import harness as telecom  # noqa: E402

ID_KEYS = re.compile(r"(^|_)id$")
FIXED_EXAMPLES = {
    "email": "jane.doe@email.com",
    "phone": "555-123-4567",
    "phone_number": "555-123-4567",
    "full_name": "Jane Doe",
    "name": "Jane Doe",
    "vin": "1HGCM82633A004352",
}


class AffineTau2SynthHarnessConfig(telecom.AffineTau2HarnessConfig):
    pass


class AffineTau2SynthHarness(telecom.AffineTau2Harness, vf.Harness[AffineTau2SynthHarnessConfig]):
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
        }
        result = await runtime.run_program(
            [sys.executable, "-m", __name__],
            {**cfg.env, base._RUN_CONFIG: json.dumps(run_config), "TAU2_DATA_DIR": str(DATA_DIR)},
        )
        if result.exit_code != 0:
            return result
        simulation = SimulationRun.model_validate_json(next(
            line.removeprefix(base._RESULT_PREFIX)
            for line in reversed(result.stdout.splitlines())
            if line.startswith(base._RESULT_PREFIX)))
        simulation.id = trace.id
        trace.info["tau2"] = {"simulation": simulation.model_dump(mode="json")}
        trace.stop(telecom.stop_condition(simulation.termination_reason))
        return result


__all__ = ["AffineTau2SynthHarness", "AffineTau2SynthHarnessConfig"]


def _real_values(db: object, out: dict[str, set[str]]) -> None:
    if isinstance(db, dict):
        for k, v in db.items():
            if isinstance(v, str) and (ID_KEYS.search(k) or k in FIXED_EXAMPLES):
                out.setdefault(k, set()).add(v)
            _real_values(v, out)
    elif isinstance(db, list):
        for x in db:
            _real_values(x, out)


def domain_examples(domain: str) -> dict[str, str]:
    """param name -> example value, built from the domain's db.json id prefixes."""
    path = DATA_DIR / "tau2" / "domains" / domain / "db.json"
    real: dict[str, set[str]] = {}
    if path.exists():
        _real_values(json.loads(path.read_text()), real)
    examples = dict(FIXED_EXAMPLES)
    for key, values in real.items():
        if key in FIXED_EXAMPLES:
            continue
        sample = sorted(values)[0]
        m = re.match(r"^([A-Za-z][A-Za-z\-]*?)(\d+)$", sample)
        if not m:
            continue
        prefix, digits = m.group(1), m.group(2)
        for n in (42, 77, 123, 4242, 9001):
            cand = f"{prefix}{str(n).zfill(len(digits))}"
            if cand not in values:
                examples[key] = cand
                break
    if "id" not in examples:
        first = next((v for k, v in examples.items() if ID_KEYS.search(k)), None)
        if first:
            examples["id"] = first
    return examples


def apply_example_values(domain: str) -> int:
    import importlib  # noqa: PLC0415 - program entry only

    from tau2.environment.toolkit import ToolKitBase  # noqa: PLC0415

    examples = domain_examples(domain)
    n = 0
    for mod_name in ("tools", "user_tools"):
        try:
            mod = importlib.import_module(f"tau2.domains.{domain}.{mod_name}")
        except ModuleNotFoundError:
            continue
        for obj in vars(mod).values():
            if not (isinstance(obj, type) and issubclass(obj, ToolKitBase) and obj is not ToolKitBase):
                continue
            for attr in dir(obj):
                fn = getattr(obj, attr, None)
                doc = getattr(fn, "__doc__", None)
                if attr.startswith("_") or not callable(fn) or not doc:
                    continue
                new = telecom.with_example_values(doc, examples)
                if new != doc:
                    fn.__doc__ = new
                    n += 1
    return n


if __name__ == "__main__":
    from tau2.data_model.message import AssistantMessage, Message  # noqa: PLC0415
    from tau2.run import run_task  # noqa: PLC0415
    from tau2.user.base import UserState  # noqa: PLC0415
    from tau2.utils import llm_utils  # noqa: PLC0415

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
        apply_example_values(config["domain"])
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
