"""BRUMO 2025: 30 public problems, attributed to MathArena and BRUMO.

Dataset: https://huggingface.co/datasets/MathArena/brumo_2025
Pack licence: CC BY-NC-SA 4.0. Problems are ordered by ``problem_idx``. The
pinned public revision, AIME instruction and symbolic rule-based reward are
explicit; one user turn and no system prompt.
"""

from affine_math_grader import MathContestTaskset


class BRUMO2025Taskset(MathContestTaskset):
    DATASET_NAME = "MathArena/brumo_2025"
    DATASET_SPLIT = "train"
    DATASET_REVISION = "6de9bc0987f5710c6376ee980e1483dbeb55cfda"
    DATASET_COUNT = 30
