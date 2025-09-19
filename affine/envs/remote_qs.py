from __future__ import annotations

import anyio
from typing import Any, Dict, Optional

import affine as af

try:
	import quixand as qs  # type: ignore
except Exception:  # pragma: no cover
	qs = None  # type: ignore


class RemoteEnvQS(af.BaseEnv):
	"""Environment client that calls HTTP endpoints from inside
	the container via Quixand Proxy (no exposed port).
	
	Expected: a Quixand Sandbox launched with uvicorn serving the FastAPI app
	from agentenv (e.g., agentenv_affine.ded_server:app) listening on 8000.
	"""
	def __init__(self, sandbox: "qs.Sandbox", timeout: int = 300):  # type: ignore[name-defined]
		super().__init__()
		self.sbx = sandbox
		self.timeout = timeout

	async def _run(self, *, path: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None) -> Any:
		def _call():
			return self.sbx.proxy.run(path=path, method=method, payload=payload, timeout=self.timeout, ensure_ready=True)
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
		return await self._run(path="/step", method="POST", payload={"id": env_id, "action": action})

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
