"""Shared config for the scoring harness. Runs ON the pod (localhost vLLM endpoints)."""

from dataclasses import dataclass, field


@dataclass
class ModelCfg:
    name: str            # short id used in results
    repo: str            # HF repo (also the vLLM served-model name)
    port: int
    family: str          # "glm" | "qwen"
    gpus: str            # CUDA_VISIBLE_DEVICES for serving
    tp: int = 1


TEACHER = ModelCfg(
    name="glm45air",
    repo="zai-org/GLM-4.5-Air-FP8",
    port=8000,
    family="glm",
    gpus="0,1",
    tp=2,
)

# Kings: one B300 each (35B-A3B BF16 ~70GB). Serve up to 6 at once on GPUs 2-7.
KING_BENCH = {  # swe-rebench scores from albedo.tech/data/model-scores.json
    "genesis": 58.2, "XCIX": 39.8, "I": 38.4, "II": 37.2, "VIII": 36.2,
    "XCIV": 36.2, "XLI": 34.2, "XI": 33.6, "XLII": 33.4, "VII": 33.2,
    "III": 32.8, "XL": 32.2, "C": 32.0, "V": 32.0, "XCVI": 32.0,
    "IV": 29.2, "XLIII": 28.8, "XLVII": 28.2, "IX": 27.8, "XLIV": 27.6,
    "VI": 27.4, "XLVIII": 27.0, "L": 26.8, "X": 26.4, "XLV": 26.0,
    "XLIX": 25.6, "XC": 16.0, "XLVI": 13.2, "CI": 12.4, "LI": 11.6,
}


def king_cfg(suffix: str, port: int, gpu: str) -> ModelCfg:
    return ModelCfg(
        name=f"king-{suffix}",
        repo=f"dendriteholdings/albedo-qwen3.6-35b-king-{suffix}",
        port=port,
        family="qwen",
        gpus=gpu,
        tp=1,
    )


@dataclass
class RunCfg:
    n_teacher_samples: int = 4
    n_miner_samples: int = 4
    temperature: float = 0.8
    max_thought_tokens: int = 1024
    max_action_tokens: int = 768
    max_concurrency: int = 48
    term_weights: dict = field(default_factory=lambda: {
        "L1": 1.0, "L2": 1.0, "L3": 1.0, "L4": 0.5, "L5": 0.5,
    })
