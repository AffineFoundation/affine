"""Where the E codes of a round are. Read-only: it reads the sealed results of the box from the store.

    python e_codes.py r000001 r000002

It needs the store variables of ``run`` (SN120_R2_...). For each LoRA it prints the
count of each code, the first and the last task with an E, how many E each
server had (a task goes to server ``index % GPUS``), and what null M did on
the same tasks.
"""

import collections
import json
import os
import sys

from sn120_v2.r2 import R2Store


def show(round_id: str, key: str, result: dict, gpus: int) -> None:
    null = result["null_outcomes"][0]
    print(f"{round_id} {key}")
    print(f"  null M: {dict(collections.Counter(null.rstrip('-')))}")
    for entry in result["entries"]:
        for text in entry["outcomes"]:
            n = len(text.rstrip("-"))
            at = [i for i in range(n) if text[i] == "E"]
            print(f"  queue_pos {entry['queue_pos']}: tasks {n}, codes {dict(collections.Counter(text[:n]))}")
            if not at:
                continue
            tail = sum(i >= n - len(at) for i in at)
            print(f"    E: {len(at)}; first at task {at[0]}, last at task {at[-1]}; {tail} of them are among the last {len(at)} tasks")
            print(f"    E per server: {dict(sorted(collections.Counter(i % gpus for i in at).items()))}")
            print(f"    null M on the E tasks: {dict(collections.Counter(null[i] for i in at))}")


def main() -> None:
    store, gpus = R2Store.from_env(), int(os.environ.get("GPUS", "8"))
    for round_id in sys.argv[1:]:
        keys = [key for key in store.list(f"private/rounds/{round_id}/") if key.endswith("/result.json")]
        if not keys:
            print(f"{round_id}: no sealed result in the store")
        for key in keys:
            show(round_id, key, json.loads(store.get(key)), gpus)


if __name__ == "__main__":
    main()
