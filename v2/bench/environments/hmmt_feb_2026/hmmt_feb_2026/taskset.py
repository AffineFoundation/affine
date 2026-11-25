"""HMMT February 2026: 33 public problems, attributed to MathArena and HMMT.

Dataset: https://huggingface.co/datasets/MathArena/hmmt_feb_2026
Pack licence: CC BY-NC-SA 4.0. Problems are ordered by ``problem_idx``. The
pinned public revision, AIME instruction and symbolic rule-based reward are
explicit; one user turn and no system prompt.
"""

from affine_math_grader import MathContestTaskset


class HMMTFeb2026Taskset(MathContestTaskset):
    DATASET_NAME = "MathArena/hmmt_feb_2026"
    DATASET_SPLIT = "train"
    DATASET_REVISION = "589a53fe152febc0f4f99f3e561007bf8da8e664"
    DATASET_COUNT = 33
