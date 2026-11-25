"""HMMT November 2025: 30 public problems, attributed to MathArena and HMMT.

Dataset: https://huggingface.co/datasets/MathArena/hmmt_nov_2025
Pack licence: CC BY-NC-SA 4.0. Problems are ordered by ``problem_idx``. The
pinned public revision, AIME instruction and symbolic rule-based reward are
explicit; one user turn and no system prompt.
"""

from affine_math_grader import MathContestTaskset


class HMMTNov2025Taskset(MathContestTaskset):
    DATASET_NAME = "MathArena/hmmt_nov_2025"
    DATASET_SPLIT = "train"
    DATASET_REVISION = "118dbfb45c4c9467c672268ed55166642897aa46"
    DATASET_COUNT = 30
