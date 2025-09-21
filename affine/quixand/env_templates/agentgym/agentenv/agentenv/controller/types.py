from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, TypedDict, List

ConversationMessage = TypedDict(
    "ConversationMessage", {"from": str, "loss": Optional[bool], "value": str}
)

APIConversationMessage = TypedDict(
    "APIConversationMessage", {"role": str, "content": str, "reasoning_content": Optional[str]}
)

TokenizedConversationOutput = TypedDict(
    "TokenizedConversationOutput",
    {
        "text": str,
        "input_ids": Sequence[int],
        "action_mask": Sequence[int],
    },
)


class ActionFormat(Enum):
    REACT = "react"
    FUNCTION_CALLING = "function_calling"
    CODE_AS_ACTION = "code_as_action"


class InferenceEngine(Enum):
    DEFAULT = "default"
    VLLM = "vllm"


@dataclass
class StepOutput:
    state: str
    reward: float
    done: bool


@dataclass
class ExperienceOutput:
    conversation: List[ConversationMessage]
    reward: float
    text: str
    seq_ids: List[int]
    attention_mask: List[int]
    action_mask: List[int]


@dataclass
class APIExperienceOutput:
    conversation: List[ConversationMessage]
    reward: float


@dataclass
class ActionWithTought:
    thought: str
    action: str


@dataclass
class EvaluationOutput:
    experiences: List[ExperienceOutput]
    score: float
    success: float


@dataclass
class Function():
    name: str
    arguments: str


@dataclass
class ChatCompletionMessageToolCall():
    # tool_call id
    id: str

    # extracted tool calls
    function: Function
