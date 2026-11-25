"""BeyondAIME: 100 public problems from ByteDance Seed, licensed CC0 1.0.

Dataset: https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME
The test split's pinned parquet row order is deterministic and is preserved;
there is no ``problem_idx`` sort. Integer gold answers are converted to strings.
One user turn, no system prompt, AIME instruction and rule-based math reward.
"""

from affine_math_grader import MathContestTaskset


class BeyondAIMETaskset(MathContestTaskset):
    DATASET_NAME = "ByteDance-Seed/BeyondAIME"
    DATASET_SPLIT = "test"
    DATASET_REVISION = "c705198ae1043810b1e1693bd879250b51a7a523"
    DATASET_COUNT = 100
    SORT_BY_INDEX = False
