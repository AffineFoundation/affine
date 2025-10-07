from typing import Any, Mapping, Dict

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


class SqlGymEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "value": "Given you a description of a SQlite database system, I will ask you a question, then you should help me operate the SQLite database with SQL to answer the question.\n\nYou have to explain the problem and your solution to me and write down your thoughts.\nAfter thinking and explaining thoroughly, you should give a SQL statement to solve the question.\n\nyour response should be like this:\nThought: Your thought here.\n\nAction: ```sql\nSELECT * FROM table WHERE condition;\n```\n\nYou MUST put SQL in markdown format without any other comments. Your SQL should be in one line. Every time you can only execute one SQL statement.",
                "loss": None,
            }
        ),
        ConversationMessage({"from": "gpt", "value": "Ok.", "loss": False}),
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
        ok = requests.post(
            f"{self.env_server_base}/create",
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
        max_retries = 5
        for _ in range(max_retries):
            res = requests.post(
                f"{self.env_server_base}/{path}",
                json=data,
                timeout=self.timeout,
            )
            if res.status_code == 503:
                import time

                time.sleep(0.1)
            elif res.status_code == 200:
                break
            else:
                print("---------------------")
                print(res.status_code)
                print(data)
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

    def step(self, env_idx: str, action: str) -> StepOutput:
        action = action.split("```sql")[-1].split("```")[0].strip()
        response = self._post("step", {"action": action}, env_idx=env_idx)

        if env_idx not in self.info:
            self.info[env_idx] = {}
            
        self.info[env_idx]["state"] = response.get("state")
        self.info[env_idx]["reward"] = response.get("reward")
        self.info[env_idx]["done"] = response.get("done")
        
        return StepOutput(
            state=response["state"],
            reward=response["reward"],
            done=response["done"],
        )

    def observe(self, env_idx: str) -> Dict[str, Any]:
        response = self._get("observation", env_idx=env_idx)

        if env_idx not in self.info:
            self.info[env_idx] = {}
        self.info[env_idx]["observation"] = response
        
        return response

    def reset(self, env_idx: str, idx: int = 0) -> Dict[str, Any]:
        response = self._post("reset", {"item_id": idx}, env_idx=env_idx)

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


class SqlGymTask(BaseTask):
    env_client_cls = SqlGymEnvClient
    env_name = "SQLGym"

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
