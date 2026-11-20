"""GSM8K-Platinum: the cleaned GSM8K test set, with the existing local GSM8K grader.

The dataset and prompt stay fixed: madrylab/gsm8k-platinum, config main, split test,
at DATASET_REVISION, in file order. LocalGSM8KTask supplies the Hub wheel's unchanged
verify.py, its long-reply handling and the opened network policy used by local-gsm8k.
"""
from collections.abc import Iterator

from gsm8k.taskset import GSM8KData, GSM8KTaskset as _HubGSM8KTaskset, SYSTEM
from local_gsm8k import LocalGSM8KTask

DATASET_NAME = "madrylab/gsm8k-platinum"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"
DATASET_REVISION = "e762492455a1cf7967de89f05b6bef72fc713b66"


def build_tasks(rows, config) -> Iterator[LocalGSM8KTask]:
    """Build tasks in dataset file order; this also makes dataset loading replaceable in offline tests."""
    for i, row in enumerate(rows):
        data = GSM8KData(
            idx=i,
            prompt=f"{SYSTEM}\n\n{row['question']}",
            answer=row["answer"].split("####")[-1].strip(),
        )
        data = data.model_copy(update={"network_allow": ["*"], "network_block": []})
        yield LocalGSM8KTask(data, config)


class LocalGSM8KPlatinumTaskset(_HubGSM8KTaskset):
    def load(self) -> Iterator[LocalGSM8KTask]:
        from datasets import load_dataset

        rows = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, revision=DATASET_REVISION)
        return build_tasks(rows, self.config.task)


__all__ = ["LocalGSM8KPlatinumTaskset", "build_tasks"]
