"""Estimate old GSM8K results on GSM8K-Platinum without a GPU or store access.

The Platinum questions are byte-identical to original test questions. Reuse the
old per-task scores for the 1,199 whose answer is unchanged; the ten revised
answers give a lower bound (all wrong) and an upper bound (all right). This is
an estimate, not a substitute for a new rollout on the cleaned dataset.

  python -m affine_bench.backfill_gsm8k_platinum results/ [--out estimates.json]

results/ contains <row>/gsm8k.per_task.jsonl files. By default the two datasets
are read from Hugging Face. --original-jsonl and --platinum-jsonl (together)
make the command completely offline; each row needs question and answer.
"""
import argparse
import json
import math
from pathlib import Path

ORIGINAL_DATASET = "openai/gsm8k"
PLATINUM_DATASET = "madrylab/gsm8k-platinum"
PLATINUM_REVISION = "e762492455a1cf7967de89f05b6bef72fc713b66"


def _answer(row):
    return row["answer"].split("####")[-1].strip()


def _questions(rows, name):
    questions = [row["question"] for row in rows]
    if len(set(questions)) != len(questions):
        raise ValueError(f"{name} has duplicate questions; cannot map them unambiguously")
    return {question: i for i, question in enumerate(questions)}


def map_questions(original_rows, platinum_rows, require_complete=True):
    """Map original indices to Platinum indices by exact question text."""
    original_rows, platinum_rows = list(original_rows), list(platinum_rows)
    original = _questions(original_rows, "original GSM8K")
    platinum = _questions(platinum_rows, "GSM8K-Platinum")
    if not set(platinum) <= set(original):
        raise ValueError("a Platinum question is missing from the original GSM8K test split")
    mapping = {original[row["question"]]: i for i, row in enumerate(platinum_rows)}
    revised = {i for i, j in mapping.items() if _answer(original_rows[i]) != _answer(platinum_rows[j])}
    if require_complete and (len(original_rows), len(platinum_rows), len(revised)) != (1319, 1209, 10):
        raise ValueError(
            "expected 1,319 original and 1,209 Platinum questions with 10 revised answers; "
            f"got {len(original_rows)}, {len(platinum_rows)}, {len(revised)}"
        )
    return mapping, revised


def _scores(path):
    scores = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        idx, n, correct = record["idx"], record["n"], record["correct"]
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or idx in scores:
            raise ValueError(f"{path}: original task indices must be unique nonnegative integers")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (n, correct)):
            raise ValueError(f"{path}: task {idx} has invalid n or correct")
        if n < 0 or not 0 <= correct <= n:
            raise ValueError(f"{path}: task {idx} must have n >= 0 and 0 <= correct <= n")
        # node_run.summarize treats an errored rollout as a failed sample: n=0 has correct=0.
        scores[idx] = correct / n if n else 0.0
    return scores


def _backfill_with_mapping(path, original_rows, platinum_rows, mapping, revised, unchanged):
    scores = _scores(path)
    if set(scores) != set(range(len(original_rows))):
        raise ValueError(f"{path}: expected every original task index 0..{len(original_rows) - 1}")
    unchanged_score = sum(scores[i] for i in unchanged) / len(unchanged)
    revised_scores = sum(scores[i] for i in revised)
    return {
        "row": path.parent.name,
        "original": {
            "tasks": len(scores),
            "score": sum(scores.values()) / len(scores),
        },
        "platinum_unchanged": {
            "tasks": len(unchanged),
            "score": unchanged_score,
        },
        "platinum_bounds": {
            "tasks": len(platinum_rows),
            "lower": (unchanged_score * len(unchanged)) / len(platinum_rows),
            "upper": (unchanged_score * len(unchanged) + len(revised)) / len(platinum_rows),
            "revised_tasks": len(revised),
            "old_revised_score": revised_scores / len(revised) if revised else None,
        },
    }


def backfill_row(path, original_rows, platinum_rows, require_complete=True):
    """Return one row's original score, unchanged-question score and all-set bounds."""
    mapping, revised = map_questions(original_rows, platinum_rows, require_complete)
    unchanged = sorted(set(mapping) - revised)
    return _backfill_with_mapping(path, original_rows, platinum_rows, mapping, revised, unchanged)


def backfill(directory, original_rows, platinum_rows, require_complete=True):
    # Map once for the whole directory, then apply it to each row's old scores.
    mapping, revised = map_questions(original_rows, platinum_rows, require_complete)
    original_rows, platinum_rows = list(original_rows), list(platinum_rows)
    unchanged = sorted(set(mapping) - revised)
    files = sorted(directory.glob("*/gsm8k.per_task.jsonl"))
    if not files:
        raise ValueError(f"no <row>/gsm8k.per_task.jsonl files in {directory}")
    return [_backfill_with_mapping(path, original_rows, platinum_rows, mapping, revised, unchanged) for path in files]


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv=None, load_datasets=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="directory containing <row>/gsm8k.per_task.jsonl")
    parser.add_argument("--out", type=Path, help="also write this JSON report (never writes to the benchmark store)")
    parser.add_argument("--original-jsonl", type=Path, help="offline original rows with question and answer")
    parser.add_argument("--platinum-jsonl", type=Path, help="offline Platinum rows with question and answer")
    args = parser.parse_args(argv)
    if bool(args.original_jsonl) != bool(args.platinum_jsonl):
        parser.error("--original-jsonl and --platinum-jsonl must be given together")
    if load_datasets is not None:
        original, platinum = load_datasets()
    elif args.original_jsonl:
        original, platinum = _jsonl(args.original_jsonl), _jsonl(args.platinum_jsonl)
    else:
        from datasets import load_dataset

        original = load_dataset(ORIGINAL_DATASET, "main", split="test")
        platinum = load_dataset(PLATINUM_DATASET, "main", split="test", revision=PLATINUM_REVISION)
    report = {
        "method": "exact question-text mapping; unchanged answers reused, revised answers bounded 0/1",
        "original_dataset": f"{ORIGINAL_DATASET} main/test",
        "platinum_dataset": f"{PLATINUM_DATASET}@{PLATINUM_REVISION} main/test",
        "rows": backfill(args.directory, original, platinum, require_complete=load_datasets is None),
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
