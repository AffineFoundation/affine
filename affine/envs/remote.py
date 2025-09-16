from __future__ import annotations
import os
import aiohttp
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable
import affine as af

@dataclass
class EnvSpec:
    name: str
    base: str
    # Paths (relative to base). If prefix is set, paths are prefix/<path>
    create_path: str = 'create'
    obs_path: str = 'observation'
    step_path: str = 'step'
    reset_path: Optional[str] = None
    # HTTP method for observation (most are GET)
    obs_method: str = 'GET'
    # Identifier key used in requests (id vs env_idx, etc.)
    id_key: str = 'id'
    # Step action key
    action_key: str = 'action'
    # Create payload style
    create_send_id: bool = False           # include {id: value} in create body
    create_id_factory: Optional[Callable[[], int]] = None  # if send_id, how to choose
    create_ret: str = 'json_id'            # 'json_id' or 'int'
    create_ret_key: str = 'id'            # if json_id, key name
    # Reset hook (optional)
    reset_required: bool = False
    reset_payload_factory: Optional[Callable[[int], Dict[str, Any]]] = None
    # Multi-turn rollout
    max_steps: int = 1

class UniversalRemoteEnv(af.BaseEnv):
    step_mode: bool = True

    def __init__(self, spec: EnvSpec):
        super().__init__()
        self.spec = spec
        self._timeout = int(os.getenv(f"{spec.name}_TIMEOUT", '300'))

    async def _request(self, method: str, path: str, *, params: Dict[str, Any] | None = None, json: Dict[str, Any] | None = None) -> Dict[str, Any] | str:
        url = f"{self.spec.base.rstrip('/')}/{path.lstrip('/')}"
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            if method.upper() == 'GET':
                async with sess.get(url, params=params) as r:
                    r.raise_for_status()
                    ctype = r.headers.get('Content-Type', '')
                    if 'application/json' in ctype:
                        return await r.json()
                    return await r.text()
            else:
                async with sess.post(url, json=json) as r:
                    r.raise_for_status()
                    ctype = r.headers.get('Content-Type', '')
                    if 'application/json' in ctype:
                        return await r.json()
                    return await r.text()

    def _ppath(self, local: str) -> str:
        # pass through; callers provide full paths or prefixed already
        return local

    async def _create(self) -> int:
        payload = {}
        if self.spec.create_send_id:
            val = self.spec.create_id_factory() if self.spec.create_id_factory else int(os.getenv(f"{self.spec.name}_CREATE_ID", '0'))
            payload[self.spec.id_key] = val
        data = await self._request('POST', self._ppath(self.spec.create_path), json=payload)
        if self.spec.create_ret == 'int':
            env_id = int(data)
        else:
            assert isinstance(data, dict), f"Expected JSON dict from create, got: {type(data)}"
            env_id = int(data[self.spec.create_ret_key])
        # optional reset
        if self.spec.reset_required and self.spec.reset_path:
            body = self.spec.reset_payload_factory(env_id) if self.spec.reset_payload_factory else {self.spec.id_key: env_id}
            await self._request('POST', self._ppath(self.spec.reset_path), json=body)
        return env_id

    async def _observation(self, env_id: int) -> str:
        params = {self.spec.id_key: env_id}
        if self.spec.obs_method.upper() == 'GET':
            res = await self._request('GET', self._ppath(self.spec.obs_path), params=params)
        else:
            res = await self._request('POST', self._ppath(self.spec.obs_path), json=params)
        return res if isinstance(res, str) else str(res)

    async def _step(self, env_id: int, action: str) -> Dict[str, Any]:
        body = {self.spec.id_key: env_id, self.spec.action_key: action}
        res = await self._request('POST', self._ppath(self.spec.step_path), json=body)
        assert isinstance(res, dict), f"Expected JSON dict from step, got: {type(res)}"
        return res

    async def generate(self) -> af.Challenge:
        env_id = await self._create()
        prompt = await self._observation(env_id)
        return af.Challenge(env=self, prompt=prompt, extra={'id': env_id, 'server': self.spec.base})

    async def rollout(self, miner: af.Miner, challenge: af.Challenge) -> af.Result:
        env_id = int(challenge.extra.get('id'))
        obs = challenge.prompt
        total_reward = 0.0
        transcript = []
        latency = 0.0
        for t in range(1, max(1, self.spec.max_steps) + 1):
            resp = await af.query(obs, model=miner.model or '', slug=miner.slug or 'llm')
            latency += float(resp.latency_seconds)
            action = resp.response or ''
            step = await self._step(env_id, action)
            obs2 = step.get('observation', '')
            reward = float(step.get('reward', 0.0))
            done = bool(step.get('done', False))
            total_reward = reward  # per-step reward semantics: last step carries final reward
            transcript.append({'t': t, 'observation': obs, 'action': action, 'reward': reward, 'done': done})
            obs = obs2
            if done:
                break
        response = af.Response(response=transcript[-1]['action'] if transcript else '', latency_seconds=latency, attempts=len(transcript), model=miner.model or '', error=None, success=bool(transcript))
        evaluation = af.Evaluation(env=self, score=total_reward, extra={'transcript': transcript})
        return af.Result(miner=miner, challenge=challenge, response=response, evaluation=evaluation)

# --------- Built-in specs (existing defaults) ---------------------------------
ENV_DEFAULTS = {
    'DED': os.getenv('DED_SERVER_URL', 'http://127.0.0.1:8010'),
    'HVM': os.getenv('HVM_SERVER_URL', 'http://127.0.0.1:8011'),
    'ABD': os.getenv('ABD_SERVER_URL', 'http://127.0.0.1:8012'),
    'SAT': os.getenv('SAT_SERVER_URL', 'http://127.0.0.1:8013'),
}

_DEF_SPEC = lambda name: EnvSpec(name=name, base=ENV_DEFAULTS[name], max_steps=1)

# SearchQA spec (AgentGym)
SEARCHQA_SPEC = EnvSpec(
    name='SEARCHQA',
    base=os.getenv('SEARCHQA_SERVER_URL') or os.getenv('AFFINE_SEARCHQA_URL', 'http://127.0.0.1:36001'),
    create_path='create',
    obs_path='observation',
    step_path='step',
    id_key='env_idx',
    create_send_id=True,
    create_id_factory=lambda: int(os.getenv('SEARCHQA_ITEM_ID', '0')) or __import__('random').randint(0, 221327),
    create_ret='int',
    max_steps=int(os.getenv('SEARCHQA_MAX_STEPS', '6')),
)

# LMRL-Gym: Maze and Wordle (share same base URL; specify via LMRLGYM_SERVER_URL)
LMRLGYM_BASE = os.getenv('LMRLGYM_SERVER_URL', 'http://127.0.0.1:8014')
LMRL_MAZE_SPEC = EnvSpec(
    name='LMRL_MAZE',
    base=LMRLGYM_BASE,
    create_path='maze/create',
    obs_path='maze/observation',
    step_path='maze/step',
    reset_path='maze/reset',
    id_key='id',
    create_send_id=False,
    create_ret='json_id',
    create_ret_key='id',
    reset_required=True,
    reset_payload_factory=lambda eid: {'id': eid, 'game': int(os.getenv('LMRL_MAZE_GAME', '0'))},
    max_steps=int(os.getenv('LMRL_MAZE_MAX_STEPS', '20')),
)
LMRL_WORDLE_SPEC = EnvSpec(
    name='LMRL_WORDLE',
    base=LMRLGYM_BASE,
    create_path='wordle/create',
    obs_path='wordle/observation',
    step_path='wordle/step',
    reset_path='wordle/reset',
    id_key='id',
    create_send_id=False,
    create_ret='json_id',
    create_ret_key='id',
    reset_required=True,
    reset_payload_factory=lambda eid: {'id': eid, 'seed': int(os.getenv('LMRL_WORDLE_SEED', '0'))},
    max_steps=int(os.getenv('LMRL_WORDLE_MAX_STEPS', '6')),
)

# Tool Academia (open-ended tool-use task)
ACADEMIA_SPEC = EnvSpec(
    name='ACADEMIA',
    base=os.getenv('ACADEMIA_SERVER_URL', 'http://127.0.0.1:8030'),
    create_path='create',
    obs_path='observation',
    step_path='step',
    id_key='env_idx',
    create_send_id=True,
    create_id_factory=lambda: int(os.getenv('ACADEMIA_ITEM_ID', '0')),
    create_ret='int',
    max_steps=int(os.getenv('ACADEMIA_MAX_STEPS', '5')),
)

# Thin subclasses matching public names
class DED(UniversalRemoteEnv):
    def __init__(self): super().__init__(_DEF_SPEC('DED'))
class HVM(UniversalRemoteEnv):
    def __init__(self): super().__init__(_DEF_SPEC('HVM'))
class ABD(UniversalRemoteEnv):
    def __init__(self): super().__init__(_DEF_SPEC('ABD'))
class SAT(UniversalRemoteEnv):
    def __init__(self): super().__init__(_DEF_SPEC('SAT'))
class SEARCHQA(UniversalRemoteEnv):
    def __init__(self): super().__init__(SEARCHQA_SPEC)
class LMRL_MAZE(UniversalRemoteEnv):
    def __init__(self): super().__init__(LMRL_MAZE_SPEC)
class LMRL_WORDLE(UniversalRemoteEnv):
    def __init__(self): super().__init__(LMRL_WORDLE_SPEC)
class ACADEMIA(UniversalRemoteEnv):
    def __init__(self): super().__init__(ACADEMIA_SPEC)
