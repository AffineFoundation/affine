"""CMIMC 2025: 40 public problems, attributed to MathArena and CMIMC.

Dataset: https://huggingface.co/datasets/MathArena/cmimc_2025
Pack licence: CC BY-NC-SA 4.0. Problems are ordered by ``problem_idx``. The
pinned public revision, AIME instruction and symbolic rule-based reward are
explicit; one user turn and no system prompt.
"""

from affine_math_grader import MathContestTaskset


class CMIMC2025Taskset(MathContestTaskset):
    DATASET_NAME = "MathArena/cmimc_2025"
    DATASET_SPLIT = "train"
    DATASET_REVISION = "7b3ca7c4b417d31b1f97f781f39fc4f97f1b9185"
    DATASET_COUNT = 40
