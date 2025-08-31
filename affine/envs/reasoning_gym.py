import time
import random
import affine as af
import reasoning_gym as rg
from typing import Optional, List, Union


class ReasoningGym(af.BaseEnv):
    __version__: str = "0.0.1"
    dataset_names: List[str]
    difficulty: Optional[str]
    seed: Optional[int]

    def __init__(
        self,
        dataset_names: Optional[Union[str, List[str]]] = None,
        difficulty: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        # Handle both single dataset string and list of datasets
        if dataset_names is None:
            dataset_names = [
                "aiw",
                "arc_agi",
                "advanced_geometry",
                "basic_arithmetic",
                "boxnet",
                "countdown",
            ]
        elif isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        super().__init__(
            dataset_names=dataset_names, difficulty=difficulty, seed=seed, **kwargs
        )

    async def generate(self):
        selected_dataset = random.choice(self.dataset_names)

        config = {}
        if self.difficulty:
            config["difficulty"] = self.difficulty
        if self.seed is not None:
            config["seed"] = self.seed + hash(selected_dataset)

        dataset = rg.create_dataset(selected_dataset, **config)

        # Get first sample from the generator
        try:
            sample = next(iter(dataset))
        except StopIteration:
            raise ValueError(f"No samples generated for dataset {selected_dataset}")

        prompt = sample.get("prompt", sample.get("question", str(sample)))

        return af.Challenge(
            env=self,
            prompt=prompt,
            extra={
                "sample": sample,
                "dataset_name": selected_dataset,
                "timestamp": time.time(),
            },
        )

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        sample = challenge.extra["sample"]
        dataset_name = challenge.extra["dataset_name"]
        score_fn = rg.get_score_answer_fn(dataset_name)
        user_answer = response.response or ""

        try:
            score = score_fn(sample, user_answer)
            score_value = float(score) if isinstance(score, (int, float, bool)) else 0.0
        except Exception as e:
            score_value = 0.0

        return af.Evaluation(
            env=self,
            score=score_value,
            extra={
                "sample": sample,
                "user_answer": user_answer,
                "dataset_name": challenge.extra["dataset_name"],
                "expected": sample.get("answer", sample.get("solution", None)),
            },
        )
