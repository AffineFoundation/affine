"""affine-mrcr-v1: MRCR v2 long-context coreference, answered from a file in the sandbox.

Wrapper over prime-envs' `mrcr_v2_v1` (OpenAI MRCR v2, the public GCS CSVs):
a long synthetic chat transcript is uploaded to `/workspace/context.txt`,
the agent scans it from the shell / a REPL, finds the k-th occurrence the
question names ("the 3rd poem about a product launch") and writes it,
prefixed by the 12-char hash, to `/workspace/answer.txt` (or says it last);
scored by the official SequenceMatcher ratio gated on the hash prefix
(`correct`; the fold's rule is score >= 1.0 -- see `exact_match`).

Changes for the duel corpus (env wave 5, 2026-09-20, docs/env-targets-for-
lagging-axes.md §7.2):

  * several (needle_count, context_range) BUCKETS in one taskset, each a
    separate public CSV of ~100 rows; `buckets: list[str]` of `<n>n-<range>`;
  * the kingboard's MRCR card is the 8-needle 64k-128k CSV
    (`bench:mrcr-v2`, 103 rows). That bucket is REFUSED here by construction
    (`CARD_BUCKET`), so D and the card never share a row. Every other bucket
    is a disjoint sample of the same generator -- the telecom `full \\ base`
    kind of decision, recorded in the source stanza;
  * `tasks: list[str]` selector by `name = "mrcr-<n>n-<range>-<row index:04d>"`
    (rollouts/catalog.py `build_mrcr_catalog` enumerates the same names
    from the CSVs' row counts without loading the contexts into the catalog);
  * a real *system* message (the base ships none; the shell harness supplies
    the dialect word) that carries the "do not print the whole file" rule
    the newer upstream INSTRUCTIONS have and the pods' pinned copy lacks.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import verifiers.v1 as vf
from mrcr_v2_v1.taskset import (
    BASE_URL,
    INSTRUCTIONS,
    RANGE_PATTERNS,
    WORKDIR,
    MRCRConfig,
    MRCRData,
    MRCRTask,
    download_cached,
)

# The benchmarked bucket (ops/benchsuite suite.toml `mrcr-v2`: 8 needles,
# 64k-128k). Never a D source.
CARD_BUCKET = "8n-64k-128k"
# Default D buckets: same needle densities and context sizes around the card,
# never the card's own CSV. ~100 rows each.
DEFAULT_BUCKETS = ["4n-32k-64k", "8n-32k-64k", "2n-64k-128k", "4n-64k-128k", "8n-16k-32k"]

SYSTEM = (
    "You answer questions about a long conversation transcript that is stored in the sandbox "
    "at /workspace/context.txt. Do not print the whole file into the conversation: search it "
    "with shell commands or Python, keep command output to small matching excerpts, enumerate "
    "every candidate before choosing the one the question asks for, then write ONLY the final "
    "answer (the 12-character hash prefix followed immediately by the requested content) to "
    "/workspace/answer.txt and print it as your last message."
)


def parse_bucket(bucket: str) -> tuple[int, str]:
    """`"4n-32k-64k"` -> (4, "32k-64k"); refuses the card bucket and unknown ranges."""
    if bucket == CARD_BUCKET:
        raise ValueError(f"{bucket} is the kingboard's MRCR card bucket; it stays out of D")
    head, _, rng = bucket.partition("n-")
    if not head.isdigit() or int(head) not in (2, 4, 8) or rng not in RANGE_PATTERNS:
        raise ValueError(f"bad MRCR bucket {bucket!r}; want <2|4|8>n-<range>, range in {sorted(RANGE_PATTERNS)}")
    return int(head), rng


def bucket_url(needles: int, rng: str) -> str:
    return f"{BASE_URL}/mrcr_v2p1_{needles}needle_{RANGE_PATTERNS[rng]}_dynamic_fewshot_text_style_fast.csv"


def load_rows(bucket: str) -> list[dict]:
    """Rows of one bucket's public CSV (cached like the base taskset)."""
    needles, rng = parse_bucket(bucket)
    got = download_cached(bucket_url(needles, rng))
    text = Path(got).read_text(encoding="utf-8") if isinstance(got, Path) else str(got)
    csv.field_size_limit(sys.maxsize)
    return list(csv.DictReader(io.StringIO(text)))


def task_name(bucket: str, index: int) -> str:
    return f"mrcr-{bucket}-{index:04d}"


def question_of(row: dict) -> str:
    question = row["view_ops"].strip()
    if question.startswith("User: "):
        question = question[len("User: "):]
    if question.endswith("Assistant:"):
        question = question[: -len("Assistant:")].strip()
    return question


class AffineMRCRConfig(MRCRConfig):
    tasks: list[str] = []
    """Task names to load (empty = every row of every bucket)."""
    buckets: list[str] = list(DEFAULT_BUCKETS)
    """`<needles>n-<context range>` CSVs to load; the card bucket is refused."""
    task_system_prompt: str = SYSTEM


class MRCRTaskset(vf.Taskset[MRCRTask, AffineMRCRConfig]):
    def load(self) -> list[MRCRTask]:
        cfg = self.config
        want = set(cfg.tasks)
        tasks: list[MRCRTask] = []
        for bucket in cfg.buckets:
            parse_bucket(bucket)  # refuse the card bucket before any download
            for i, row in enumerate(load_rows(bucket)):
                name = task_name(bucket, i)
                if want and name not in want:
                    continue
                tasks.append(MRCRTask(
                    MRCRData(
                        idx=i, name=name, system_prompt=cfg.task_system_prompt,
                        prompt=f"{question_of(row)}\n\n{INSTRUCTIONS}",
                        answer=row["answer"], context=row["queries"], workdir=WORKDIR,
                    ),
                    cfg.task,
                ))
        return tasks
