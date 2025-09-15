from __future__ import annotations
import os
import aiohttp
from typing import Any, Dict
import affine as af

# Mapping of envs -> default URL (overridable via environment variables)
ENV_DEFAULTS = {
    "DED": os.getenv("DED_SERVER_URL", "http://127.0.0.1:8010"),
    "HVM": os.getenv("HVM_SERVER_URL", "http://127.0.0.1:8011"),
    "ABD": os.getenv("ABD_SERVER_URL", "http://127.0.0.1:8012"),
    "SAT": os.getenv("SAT_SERVER_URL", "http://127.0.0.1:8013"),
}

class RemoteEnv(af.BaseEnv):
    env_key: str
    base: str
    timeout: int
    def __init__(self, env_key: str):
        super().__init__(
            env_key=env_key,
            base=ENV_DEFAULTS[env_key],
            timeout=int(os.getenv(f"{env_key}_TIMEOUT", "300")),
        )

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as sess:
            async with sess.post(f"{self.base}/{path}", json=payload) as r:
                r.raise_for_status(); return await r.json()

    async def _get_text(self, path: str, params: Dict[str, Any]) -> str:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as sess:
            async with sess.get(f"{self.base}/{path}", params=params) as r:
                r.raise_for_status(); return await r.text()

    async def _create(self) -> int:
        res = await self._post("create", {})
        if isinstance(res, dict) and "id" in res:
            return int(res["id"])  # AgentGym style
        if isinstance(res, int):
            return int(res)
        raise RuntimeError(f"Unexpected create() response: {res}")

    async def generate(self) -> af.Challenge:
        env_id = await self._create()
        prompt = await self._get_text("observation", {"id": env_id})
        return af.Challenge(env=self, prompt=prompt, extra={"id": env_id, "server": self.base})

    async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
        env_id = challenge.extra.get("id")
        res = await self._post("step", {"id": env_id, "action": response.response or ""})
        score = float(res.get("reward", 0.0))
        return af.Evaluation(env=self, score=score, extra={k: v for k, v in res.items() if k not in ("reward",)})

# Thin subclasses (public names preserved)
class DED(RemoteEnv):
    def __init__(self): super().__init__("DED")
class HVM(RemoteEnv):
    def __init__(self): super().__init__("HVM")
class ABD(RemoteEnv):
    def __init__(self): super().__init__("ABD")
class SAT(RemoteEnv):
    def __init__(self): super().__init__("SAT")

#DED = _DED
#HVM = _HVM
#ABD = _ABD
#SAT = _SAT 