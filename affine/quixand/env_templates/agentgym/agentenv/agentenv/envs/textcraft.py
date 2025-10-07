from typing import Any, Mapping, Dict

import re

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


class TextCraftEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": 'You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format "craft [target object] using [input ingredients]".\nEvery round I will give you an observation, you have to respond an action based on the state and instruction. You can "get" an object (ingredients) from the inventory or the environment, look-up the game inventory by "inventory", or "craft" (target) using any of the crafting commands.\nYour output must strictly follow this format:"Thought:\nyour thoughts.\n\nAction:\nyour next action"\n\nReminder: \n1. Always specify the quantity when using "get" and "craft" commands. - Example of get: get 1 lapis lazuli - Example1 of craft: craft 1 blue dye using 1 lapis lazuli - Example2 of craft: craft 1 golden carrot using 8 gold nugget, 1 carrot\n2. When using "get" command, do not specify whether the item comes from the inventory or the environment.\n3. You can use ONLY crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use special types of the same ingredient e.g. "dark oak planks" in the command instead.\n\n',
            }
        ),
        ConversationMessage(
            {
                "from": "gpt",
                "loss": False,
                "value": "OK. I'll follow your instructions and try my best to solve the task.",
            }
        ),
    )

    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        minecraft_dir: str = "agentenv_textcraft/",
        commands: str = None,
        goal: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        self.minecraft_dir = minecraft_dir
        self.commands = commands
        self.goal = goal

        self.info = {}
        self.env_ids = {}  # Store all created environment IDs
    
    def create(self) -> str:
        dir_info = {"minecraft_dir": self.minecraft_dir, "commands": self.commands, "goal": self.goal}
        ok = requests.post(
            f"{self.env_server_base}/create", timeout=self.timeout, json=dir_info
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        env_id = ok["id"]
        self.info[env_id] = {
            "observation": ok["observation"],
            "reward": 0,
            "done": False,
        }
        self.env_ids[env_id] = True

        return env_id

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Dict[str, Any], env_idx: str = None) -> Dict[str, Any]:
        # textcraft uses 'id' as the parameter name
        if env_idx is not None:
            data["id"] = env_idx
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
            params["id"] = env_idx
        res = requests.get(
            f"{self.env_server_base}/{path}",
            params=params,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self, env_idx: str) -> str:
        return self.info.get(env_idx, {}).get("observation", "")

    def step(self, env_idx: str, action: str) -> StepOutput:
        action_matches = re.findall(r"Action:\s*(.*?)(?=\n|$)", action, re.DOTALL)
        if len(action_matches) > 1:
            return StepOutput(
                state="Error: Only one 'Action' is allowed per response. Please adjust your response.",
                reward=0,
                done=False,
            )
        action = action_matches[-1] if action_matches else ""
        action = re.sub(r"[^A-Za-z0-9, ]+", "", action)
        action = " ".join(action.split()).strip()
        response = self._post("step", {"action": action}, env_idx=env_idx)

        if env_idx not in self.info:
            self.info[env_idx] = {}

        self.info[env_idx] = {
            "observation": response["observation"],
            "reward": response["reward"],
            "done": response["done"],
        }
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, env_idx: str, idx: int = 0) -> Dict[str, Any]:
        response = self._post("reset", {"data_idx": idx}, env_idx=env_idx)

        if env_idx not in self.info:
            self.info[env_idx] = {}

        self.info[env_idx].update(
            {
                "observation": response["observation"],
                "reward": 0,
                "done": False,
            }
        )
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

class TextCraftTask(BaseTask):
    env_client_cls = TextCraftEnvClient
    env_name = "TextCraft"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
