from typing import Any, Callable, Mapping, Optional, Sequence, List, Union

from . import APIAgent, BaseEnvClient
from .types import ConversationMessage, APIConversationMessage, ExperienceOutput, APIExperienceOutput


class BaseTask:
    env_client_cls: Callable
    env_name: str

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int = 1,
    ) -> None:
        """
        Initializes the Task object.

        Args:
            client_args (Mapping[str, Any]): A mapping of client arguments.
            n_clients (int, optional): The number of clients. Defaults to 1. Larger than 1 for batch generation. Batch generation is not implemented yet.
        """
        if self.env_client_cls is None or self.env_name is None:
            raise NotImplementedError
        self.clients = [self.env_client_cls(**client_args) for _ in range(n_clients)]
        self.len = len(self.clients[0])

    def _generate_experience_one(
        self,
        agent: APIAgent,
        client: BaseEnvClient,
        idx: int,
        generation_config = None,
        max_rounds: Optional[int] = None,
    ) -> ExperienceOutput:
        env_id = client.create()
        client.reset(env_id, idx)
        reward = 0.0
        done = False
        state = client.observe(env_id)
        if isinstance(agent, APIAgent):
            conversation = [APIConversationMessage({"role": "user", "content": client.conversation_start[0]["value"], "reasoning_content": None}),
                            APIConversationMessage({"role": "assistant", "content": client.conversation_start[1]["value"], "reasoning_content": None}),
                            APIConversationMessage({"role": "user", "content": state, "reasoning_content": None})]
        else:
            raise NotImplementedError
        rounds = 0

        while not done:
            if isinstance(agent, APIAgent):
                generated_text, generated_reasoning_text = agent.generate(conversation)
                conversation.append(
                    APIConversationMessage(
                        {"role": "assistant", "content": generated_text, "reasoning_content": generated_reasoning_text}
                    )
                )
            else:
                raise NotImplementedError

            step_output = client.step(env_id, generated_text)
            state, reward, done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )

            if isinstance(agent, APIAgent):
                conversation.append(
                    APIConversationMessage(
                        {"role": "user", "content": state, "reasoning_content": None}
                    )
                )
            else:
                raise NotImplementedError

            rounds += 1
            if max_rounds is not None and rounds >= max_rounds:
                break
        if hasattr(client, "close"):
            client.close(env_id)

        if isinstance(agent, APIAgent):
            return APIExperienceOutput(
                conversation=conversation,
                reward=reward,
            )
        else:
            raise NotImplementedError

    def _generate_experience_batch(
        self,
        agent: APIAgent,
        idxs: Sequence[int],
        generation_config = None,
        max_rounds: Optional[int] = None,
    ) -> List[ExperienceOutput]:
        client = self.clients[0]
        result = [
            self._generate_experience_one(
                agent=agent,
                client=client,
                idx=idx,
                generation_config=generation_config,
                max_rounds=max_rounds,
            )
            for idx in idxs
        ]
        return result

    def generate_experience(
        self,
        agent: APIAgent,
        idxs: Union[Sequence[int], int],
        generation_config = None,
        max_rounds: Optional[int] = None,
    ) -> List[ExperienceOutput]:
        if isinstance(idxs, int):
            idxs = [idxs]

        return self._generate_experience_batch(
            agent=agent,
            idxs=idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
        )
