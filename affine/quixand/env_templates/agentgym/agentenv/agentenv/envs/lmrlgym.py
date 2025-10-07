import json
from typing import Any, Mapping
from typing import Dict

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput



# ----------------------------------------
# maze
# ----------------------------------------


class MazeEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {"from": "human", "loss": None, "value": "You are an expert maze solver."}
        ),
        ConversationMessage(
            {
                "from": "gpt",
                "loss": False,
                "value": "OK. I'll follow your instructions and try my best to solve the task.",
            }
        ),
    )
    _fully_first_observation = """\
Your objective is to reach the goal in as few steps as possible. At each step you will be given information about where the goal is, your current position,
and the walls that surround you. 

When you move right you increase your y position by 1, when you move down you increase your x position by 1. 

Here is an example.

```
environment: The goal is at position 8, 6. Your current position is at position 5, 6. There is a wall above you.
action: move left
environment: The goal is at position 8, 6. Your current position is at position 5, 5. There are walls above you, below you.
action: move left
environment: The goal is at position 8, 6. Your current position is at position 5, 4. There are walls above you, below you.
action: move up
environment: The goal is at position 8, 6. Your current position is at position 5, 4. There are walls above you, below you.
action: move left
environment: The goal is at position 8, 6. Your current position is at position 5, 3. There are walls to your left, below you.
action: move down
environment: The goal is at position 8, 6. Your current position is at position 5, 3. There are walls to your left, below you.
action: move left
environment: The goal is at position 8, 6. Your current position is at position 5, 3. There are walls to your left, below you.
action: move down
environment: The goal is at position 8, 6. Your current position is at position 5, 3. There are walls to your left, below you.
action: move left
environment: The goal is at position 8, 6. Your current position is at position 5, 3. There are walls to your left, below you.
action: move right
environment: The goal is at position 8, 6. Your current position is at position 5, 4. There are walls above you, below you.
action: move down
environment: The goal is at position 8, 6. Your current position is at position 5, 4. There are walls above you, below you.
action: move right
environment: The goal is at position 8, 6. Your current position is at position 5, 5. There are walls above you, below you.
action: move right
environment: The goal is at position 8, 6. Your current position is at position 5, 6. There is a wall above you.
action: move down
environment: The goal is at position 8, 6. Your current position is at position 6, 6. There are walls to your right, to your left.
action: move down
environment: The goal is at position 8, 6. Your current position is at position 7, 6. There are walls to your right, to your left.
action: move right
environment: The goal is at position 8, 6. Your current position is at position 7, 6. There are walls to your right, to your left.
action: move down
environment: Success
```

Your possible actions are "move up", "move down", "move left", "move right". Formally, your return should be in this format:
Thought:\n<Your Thought>\n\nAction:\n<Your Action>

Now let's start a new game. Return your action and your thought in the format above strictly. Now, make the optimal action given the current environment state:
""".strip()
    _partially_first_observation = """\
Your objective is to reach the goal in as few steps as possible. At each step you will see your move history, and the walls that surround you.

Here is an example. 
```
environment: There are walls above you, below you.
action: move up
environment: There are walls above you, below you.
action: move left
environment: There is a wall above you.
action: move left
environment: There are walls above you, below you.
action: move up
environment: There are walls above you, below you.
action: move up
environment: There are walls above you, below you.
action: move up
environment: There are walls above you, below you.
action: move right
environment: There is a wall above you.
action: move down
environment: There are walls to your right, to your left.
action: move down
environment: There are walls to your right, to your left.
action: move right
environment: There are walls to your right, to your left.
action: move down
environment: Success
```

Your possible actions are "move up", "move down", "move left", "move right". Formally, your return should be in this format:
Thought:\n<Your Thought>\n\nAction:\n<Your Action>

Now let's start a new game. Return your action and your thought in the format above strictly. Now, make the optimal action given the current environment state:
""".strip()

    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        self.info = {}
        self.env_ids = {}

    def create(self) -> str:
        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        print(ok)
        env_id = ok["id"]
        self.info[env_id] = {
            "reward": 0,
            "done": False,
        }
        self.env_ids[env_id] = True

        return env_id
        
    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Dict[str, Any], env_idx: str = None) -> Dict[str, Any]:
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
        print(action)
        if action.endswith("</s>"):
            action = action[:-5]
        _action = action.split("Action:")
        if len(_action) > 1:
            action = _action[1].strip()
        else:
            action = _action[0].strip()
        print(f"Action: {action}")
        response = self._post("step", {"action": action}, env_idx=env_idx)
        print(response)

        if env_idx not in self.info:
            self.info[env_idx] = {"reward": 0, "done": False}
            
        self.info[env_idx].update(
            {
                "observation": response["observation"],
                "reward": self.info[env_idx]["reward"] + response["reward"],
                "done": response["done"],
            }
        )
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, env_idx: str, idx: int = 0) -> Dict[str, Any]:
        response = self._post("reset", {"game": idx}, env_idx=env_idx)
        print(response)
        self.first_observation = self._fully_first_observation
        response["observation"] = (
            self.first_observation + "\n" + response["observation"]
        )
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


class MazeTask(BaseTask):
    env_client_cls = MazeEnvClient
    env_name = "LMRL-Gym.maze"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)


# ----------------------------------------
# wordle
# ----------------------------------------


class WordleEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {"from": "human", "loss": None, "value": "You are an expert wordle player."}
        ),
        ConversationMessage(
            {
                "from": "gpt",
                "loss": False,
                "value": "OK. I'll follow your instructions and try my best to solve the task.",
            }
        ),
    )
    first_observation = """\
Welcome to the game of Wordle. Your objective is to guess a hidden 5 letter word. You have 6 attempts to guess it correctly and you should try to guess it in as few attempts as possible. When guessing the word, you should format your word as a space separated sequence of letters, like "s h i r e" for example. After guessing the word, you will receive feedback from the game environment in the form of a sequence of 5 space separated letters like "b y g g b", where each letter indicates some information about the hidden word. The environment will return one of three letters – "b", "g", or "y" – for each letter in the word you guessed. We describe the meaning of each letter below:

"b": If the environment returns a "b", it means that the letter at that position in your guessed word is not in the hidden word.
"y": If the environment returns a "y", it means that the letter at that position in your guessed word is in the hidden word but is not in the correct position.
"g": If the environment returns a "g", it means that the letter at that position in your guessed word is in the hidden word and is in the correct position.

As a note, if you guess an invalid word (e.g. not a 5 letter word or a word not in the vocabulary), the environment will respond with an "invalid word" message. In general though, you should use this information returned by the environment to update your belief about what the hidden word might be and adjust your next guess accordingly.

Here is the complete list of valid vocabulary words that are accepted by the game:
```
{{vocab}}
```

Here is an example. If the current status of the game is given as:
```
guess 1: p a n i c
feedback 1: b b y b b
guess 2: f e l o n
feedback 2: g b b y g
```
Based on the feedback from the environment, you know that the first letter is "f", the last letter is "n", and there is an "o" somewhere in the word, but it is not in the second to last position. You also know that there is not a "p", "a", "i", "c", "e", or "l" in the word. Knowing this, you might guess the next word to be:
Thought:\nI know that the first letter is "f", the last letter is "n", and there is an "o" somewhere in the word, but it is not in the second to last position. I also know that there is not a "p", "a", "i", "c", "e", or "l" in the word. A good word from the vocabulary to try might therefore be \"f r o w n\", since it is in the vocabulary, meets all known letter constraints, and we get to gain more information about the position of "o". Therefore this is a good guess to try next.\n\nAction:\nf r o w n

Formally, your return should be in this format:
Thought:\n<Your Thought>\n\nAction:\n<The Word You Guess>

The guessed word is in the vocabulary, meets all known letter constraints, and we get to gain more information about the position of "o", so it is a good guess to try next.

Now let's start a new game. Remember, the word you guess should be strictly in the vocabulary. You should return your thought and your word strictly in the formation mentioned above.
""".strip()

    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        
        # Initialize info dict to store multiple environment instances
        self.info = {}
        self.env_ids = {}  # Store all created environment IDs
        self.vocab = None  # Store vocab once fetched
    
    def create(self) -> str:
        """Create a new environment instance and return env_id"""
        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        print(ok)
        env_id = ok["id"]
        
        # Fetch vocab if not already fetched
        if self.vocab is None:
            self.vocab = self._get("filtered_vocab", env_idx=env_id)
        
        # Initialize info for this environment instance
        self.info[env_id] = {
            "observation": self.first_observation.replace(
                "{{vocab}}", "\n".join(self.vocab)
            ),
            "vocab": self.vocab,
            "reward": 0,
            "done": False,
        }
        self.env_ids[env_id] = True
        print(self.info[env_id]["observation"])
        
        return env_id

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Dict[str, Any], env_idx: str = None) -> Dict[str, Any]:
        # lmrlgym uses 'id' as the parameter name
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
        print(action)
        if action.endswith("</s>"):
            action = action[:-5]
        _action = action.split("Action:")
        if len(_action) > 1:
            action = _action[1].strip()
        else:
            action = _action[0].strip()
        print(f"Action: {action}")
        response = self._post("step", {"action": action}, env_idx=env_idx)
        print(response)
        
        # Update info for the corresponding env_idx
        if env_idx not in self.info:
            self.info[env_idx] = {"reward": 0, "done": False, "vocab": self.vocab}
            
        self.info[env_idx].update(
            {
                "observation": response["observation"],
                "reward": self.info[env_idx]["reward"] + response["reward"],
                "done": response["done"],
            }
        )
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, env_idx: str, idx: int = 0) -> Dict[str, Any]:
        # env_idx is the environment instance ID
        response = self._post("reset", {"seed": idx}, env_idx=env_idx)
        
        # Update info for the corresponding env_idx
        if env_idx not in self.info:
            self.info[env_idx] = {"vocab": self.vocab}
            
        self.info[env_idx].update(
            {
                "observation": self.first_observation.replace(
                    "{{vocab}}", "\n".join(self.vocab if self.vocab else [])
                ),
                "reward": 0,
                "done": False,
            }
        )
        return response
    
    def close(self, env_idx: str):
        """Close the specified environment instance and clean up resources"""
        try:
            response = self._post("close", {}, env_idx=env_idx)
        except:
            response = None
            
        # Clean up data in info
        if env_idx in self.info:
            del self.info[env_idx]
        if env_idx in self.env_ids:
            del self.env_ids[env_idx]
            
        return response


class WordleTask(BaseTask):
    env_client_cls = WordleEnvClient
    env_name = "LMRL-Gym.wordle"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
