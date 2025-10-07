from typing import Any, Mapping, Dict, List, Optional

import requests
from requests.exceptions import RequestException
from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput

class SearchQAEnvClient(BaseEnvClient):
    conversation_start = (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value":"""You must always reason inside <think>...</think> first; if you lack knowledge, issue a <search>...</search> and then stop; do not generate <information> or <answer> yet; wait for external input between <information>...</information> before continuing; resume only when new <information> is given; do not skip steps or anticipate answers early.""",
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
    )

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        self.info = {}
        self.env_ids = {}

    def create(self) -> str:
        data = dict()
        data['id'] = 0
        ok = requests.post(
            f"{self.env_server_base}/create",
            json=data,
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        env_id = ok.json()
        self.info[env_id] = {}
        self.env_ids[env_id] = True
        
        return env_id

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Dict[str, Any], env_idx: str = None) -> Dict[str, Any]:
        if env_idx is not None:
            data["env_idx"] = env_idx
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str, env_idx: str = None) -> Dict[str, Any]:
        params = {}
        if env_idx is not None:
            params["env_idx"] = env_idx
        res = requests.get(
            f"{self.env_server_base}/{path}",
            params=params,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self, env_idx: str) -> Dict[str, Any]:
        question = self._get("observation", env_idx=env_idx)
        if env_idx not in self.info:
            self.info[env_idx] = {}
        self.info[env_idx]["observation"] = question
        
        return question

    def step(self, env_idx: str, action: str) -> StepOutput:
        # action is the original output of llm
        # print(f"Action: {action}")
        response = self._post("step", {"action": action}, env_idx=env_idx)
        # print(response)

        if env_idx not in self.info:
            self.info[env_idx] = {}
            
        self.info[env_idx]["observation"] = response.get("observation")
        self.info[env_idx]["reward"] = response.get("reward")
        self.info[env_idx]["done"] = response.get("done")
        
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, env_idx: str, id: int = 0) -> Dict[str, Any]:
        response = self._post("reset", {"id": id}, env_idx=env_idx)
        if env_idx not in self.info:
            self.info[env_idx] = {}
            
        self.info[env_idx].update(response)
        
        return response
    
    def close(self, env_idx: str):
        try:
            response = self._post("close", {}, env_idx=env_idx)
        except:
            response = None
        if env_idx in self.info:
            del self.info[env_idx]
        if env_idx in self.env_ids:
            del self.env_ids[env_idx]
            
        return response

class SearchQATask(BaseTask):
    env_client_cls = SearchQAEnvClient
    env_name = "SearchQA"

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
