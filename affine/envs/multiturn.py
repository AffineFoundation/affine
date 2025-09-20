from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import affine as af
from .remote_qs import RemoteEnvQS

ConversationMessage = Dict[str, Any]

# Import initial prompts (conversation_start) from agentenv_affine
try:
	from agentenv_affine.ded_env_client import AffineDedEnvClient as _DED_CLI  # type: ignore
except Exception:
	_DED_CLI = None  # type: ignore
try:
	from agentenv_affine.hvm_env_client import AffineHvmEnvClient as _HVM_CLI  # type: ignore
except Exception:
	_HVM_CLI = None  # type: ignore
try:
	from agentenv_affine.abd_env_client import AffineAbdEnvClient as _ABD_CLI  # type: ignore
except Exception:
	_ABD_CLI = None  # type: ignore
try:
	from agentenv_affine.sat_env_client import AffineSatEnvClient as _SAT_CLI  # type: ignore
except Exception:
	_SAT_CLI = None  # type: ignore

_CONV_START: Dict[str, Tuple[ConversationMessage, ConversationMessage]] = {}
if _DED_CLI and hasattr(_DED_CLI, "conversation_start"):
	_CONV_START["DED"] = tuple(_DED_CLI.conversation_start)  # type: ignore[arg-type]
if _HVM_CLI and hasattr(_HVM_CLI, "conversation_start"):
	_CONV_START["HVM"] = tuple(_HVM_CLI.conversation_start)  # type: ignore[arg-type]
if _ABD_CLI and hasattr(_ABD_CLI, "conversation_start"):
	_CONV_START["ABD"] = tuple(_ABD_CLI.conversation_start)  # type: ignore[arg-type]
if _SAT_CLI and hasattr(_SAT_CLI, "conversation_start"):
	_CONV_START["SAT"] = tuple(_SAT_CLI.conversation_start)  # type: ignore[arg-type]


def get_conversation_start(env_key: str) -> List[ConversationMessage]:
	"""Returns the list of the 2 initial messages (human, gpt) for the given environment.
	If missing, provides a minimal fallback.
	"""
	pair = _CONV_START.get(env_key)
	if pair:
		return [pair[0], pair[1]]
	# Generic fallback if not defined
	return [
		{"from": "human", "loss": None, "value": "You are an agent. Follow the instructions and act accordingly."},
		{"from": "gpt", "loss": False, "value": "Ok."},
	]


async def run_multiturn(
	*,
	env_key: str,
	env: RemoteEnvQS,
	llm_call: Callable[[List[ConversationMessage]], "Awaitable[str]"],
	max_rounds: int = 30,
	env_id: int | None = None,
) -> af.Evaluation:
	"""Generic multi-turn loop:
	- initializes the environment (create + first observation) or reset existing env_id
	- maintains a conversation history similar to AgentGym
	- queries the LLM and calls /step until done or max_rounds
	Returns an Evaluation (score + extras)
	"""
	# init
	conv: List[ConversationMessage] = list(get_conversation_start(env_key))

	# Determine initial observation via reset or create
	if env_id is not None:
		try:
			res0 = await env.reset(env_id)
			obs = res0.get("observation") if isinstance(res0, dict) else None
			if not obs:
				obs = await env.observation(env_id)
			chal = af.Challenge(env=env, prompt=obs, extra={"id": env_id})
		except Exception:
			# Fallback to creating a new episode if reset requires extra params or failed
			chal = await env.generate()
			env_id = int(chal.extra.get("id"))
			obs = chal.prompt
	else:
		chal = await env.generate()  # performs create + initial observation
		env_id = int(chal.extra.get("id"))
		obs = chal.prompt

	conv.append({"from": "human", "loss": None, "value": obs})

	reward = 0.0
	done = False
	rounds = 0
	extra_last: Dict[str, Any] = {}

	while not done and rounds < max_rounds:
		# 1) model proposes an action
		action = await llm_call(conv)
		conv.append({"from": "gpt", "loss": True, "value": action})
		# 2) step in the environment
		res = await env.step(env_id, action)
		obs = res.get("observation", "")
		reward = float(res.get("reward", 0.0))
		done = bool(res.get("done", False))
		extra_last = {k: v for k, v in res.items() if k not in ("observation", "reward", "done")}
		# 3) add the observation to the transcript
		conv.append({"from": "human", "loss": None, "value": obs})
		rounds += 1

	return af.Evaluation(env=env, score=reward, extra={"rounds": rounds, **extra_last})


async def run_multiturn_with_chutes(
	*,
	env_key: str,
	env: RemoteEnvQS,
	model: str,
	slug: str,
	max_rounds: int = 30,
	env_id: int | None = None,
) -> af.Evaluation:
	"""Ready-to-use multi-turn with the chutes backend (affine.query).
	Serializes the conversation as a transcript and calls `affine.query` at each turn.
	"""
	from .. import query  # chutes chat API wrapper

	async def _llm_call(conv: List[ConversationMessage]) -> str:
		transcript = []
		for m in conv:
			role = m.get("from") or ("assistant" if m.get("loss") else "user")
			val = m.get("value") or ""
			transcript.append(f"{role.upper()}: {val}")
		prompt = "\n".join(transcript)
		r = await query(prompt, model, slug, timeout=180)
		return r.response or ""

	return await run_multiturn(env_key=env_key, env=env, llm_call=_llm_call, max_rounds=max_rounds, env_id=env_id)
