from __future__ import annotations

import anyio
from typing import Any, Dict, Optional

import affine as af
import json

try:
	import quixand as qs  # type: ignore
except Exception:  # pragma: no cover
	qs = None  # type: ignore

from pydantic import PrivateAttr


class RemoteEnvQS(af.BaseEnv):
	"""Environment client that calls HTTP endpoints from inside
	the container via Quixand Proxy (no exposed port).
	
	Expected: a Quixand Sandbox launched with uvicorn serving the FastAPI app
	from agentenv (e.g., agentenv_affine.ded_server:app) listening on 8000.
	"""
	_sbx: Any = PrivateAttr(default=None)
	_timeout: int = PrivateAttr(default=300)

	def __init__(self, sandbox: "qs.Sandbox", timeout: int = 300):  # type: ignore[name-defined]
		super().__init__()
		self._sbx = sandbox
		self._timeout = timeout

	async def _run(self, *, path: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None) -> Any:
		def _call():
			# Convert path like "/observation?id=123" into endpoint and params
			endpoint_and_query = path.lstrip("/")
			if "?" in endpoint_and_query:
				endpoint, query = endpoint_and_query.split("?", 1)
				from urllib.parse import parse_qsl
				params = dict(parse_qsl(query))
			else:
				endpoint, params = endpoint_and_query, {}
			if payload:
				params.update(payload)
			func = getattr(self._sbx.proxy, endpoint)
			return func(_timeout=self._timeout, _ensure_ready=True, **params)
		return await anyio.to_thread.run_sync(_call)

	# AgentGym-like endpoints
	async def create(self) -> int:
		res = await self._run(path="/create", method="POST")
		return int(res["id"]) if isinstance(res, dict) and "id" in res else int(res)

	async def observation(self, env_id: int) -> str:
		return await self._run(path=f"/observation?id={env_id}", method="GET")

	async def available_actions(self, env_id: int) -> list[str]:
		res = await self._run(path=f"/available_actions?id={env_id}", method="GET")
		return res if isinstance(res, list) else []

	async def step(self, env_id: int, action: str) -> Dict[str, Any]:
		# Ensure action is JSON-safe even if proxy builds JSON naively.
		# Using json.dumps wraps/escapes quotes and backslashes so the outer JSON stays valid.
		safe_action = json.dumps(action)
		return await self._run(path="/step", method="POST", payload={"id": env_id, "action": safe_action})

	async def reset(self, env_id: int, **kwargs: Any) -> Dict[str, Any]:
		payload = {"id": env_id, **kwargs}
		return await self._run(path="/reset", method="POST", payload=payload)

	async def detail(self, env_id: int) -> Dict[str, Any]:
		res = await self._run(path=f"/detail?id={env_id}", method="GET")
		return res if isinstance(res, dict) else {}

	# Affine contract
	async def generate(self) -> af.Challenge:
		env_id = await self.create()
		obs = await self.observation(env_id)
		return af.Challenge(env=self, prompt=obs, extra={"id": env_id})

	async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
		env_id = int(challenge.extra.get("id"))
		res = await self.step(env_id, response.response or "")
		score = float(res.get("reward", 0.0))
		return af.Evaluation(env=self, score=score, extra={k: v for k, v in res.items() if k != "reward"})
