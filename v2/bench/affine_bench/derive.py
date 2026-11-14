"""A result at a smaller count of samples, computed from a finished run at a larger count. Host side; pure arithmetic.

Why: a benchmark's count of samples for each problem can be cut (ArXivMath, 2026-11-14: 16 -> 2). A row whose run at the
old count was in work is not run again. Its value at the new count is computed from that run, so every row has a value
of the same kind. The value of the full run stays with the row (`score_full`, `samples_full`).

What the host has of a run: one line per problem (`per_task.jsonl`) with the count of right samples, not each sample.
The server gets no seed, so the samples of a problem are independent draws and no sample has a number. The pick below
uses the counts only, so it does not depend on the order in which the samples ended.

The rule `pick-of-full/1`, fixed on 2026-11-14 before the full run of any row had ended:
  - It applies only when the full run is whole: every problem has all its samples and none ended with an error.
  - Put the samples of a problem in the order: right ones first, then wrong ones. Number them 0 .. full-1.
  - h = sha256("<CONSTANT>|<benchmark id>|<idx>") as bytes, where idx is the problem's row number in the pinned dataset.
    The first picked number is  int(h[0:8], big-endian) mod full.  Each next one comes from the next 8 bytes, mod the
    count of numbers that are not picked yet, counted over those numbers in rising order.
  - A picked sample is right when its number is below the problem's count of right samples.
  - The problem's value is (right picked samples) / pick. The score is the mean over the problems.
The numbers depend on the problem only, never on the result, so the pick is a fair draw of `pick` of the `full` samples."""
import hashlib, json, math

RULE = "pick-of-full/1"
CONSTANT = "sn120-v2-bench pick-of-full/1 2026-11-14"


def picked(bid, idx, full, pick):
    """The numbers of the samples that the rule takes for one problem."""
    h = hashlib.sha256(f"{CONSTANT}|{bid}|{idx}".encode()).digest()
    free = list(range(full)); out = []
    for i in range(pick):
        out.append(free.pop(int.from_bytes(h[8 * i:8 * i + 8], "big") % len(free)))
    return out


def from_full(r, per_task, pick, version, problems):
    """(result at `pick` samples a problem, None), or (None, why) when the full run is not whole. `r` is the full run's
    summary, `per_task` the text of its per-problem lines, `version` the bench_version of the result to make, `problems`
    the count of problems of the benchmark (its `problems` setting). cap_rate and the token counts stay those of the full run."""
    full = int(r.get("rollouts") or 0)
    if not 0 < pick < full or pick > 4:
        return None, f"the run has {full} samples a problem: no pick of {pick}"
    if not problems:
        return None, "the settings do not say how many problems the benchmark has"
    try:
        rows = [json.loads(line) for line in per_task.splitlines() if line.strip()]
    except ValueError:
        return None, "a per-problem line is not readable"
    if len(rows) != problems or len({p.get("idx") for p in rows}) != problems or r.get("samples") != problems * full:
        return None, f"{r.get('samples')} samples of {len(rows)} problems are there, not {problems * full} of {problems}: the full run is not whole"
    means = []
    for p in rows:
        n, c = p.get("n"), p.get("correct")
        if n != full or p.get("errors"):
            return None, f"problem {p.get('idx')} has {n} of {full} samples ({p.get('errors')} with an error): the full run is not whole"
        if not isinstance(c, (int, float)) or abs(c - round(c)) > 1e-9 or not 0 <= round(c) <= full:
            return None, f"problem {p.get('idx')} has no whole count of right samples"
        means.append(sum(k < round(c) for k in picked(r["benchmark"], p["idx"], full, pick)) / pick)
    m = len(means); score = sum(means) / m
    se = math.sqrt(sum((x - score) ** 2 for x in means) / (m - 1) / m) if m > 1 else None
    out = dict(r, rollouts=pick, score=score, se=se, samples=m * pick, samples_expected=m * pick, errors=0, errored_tasks=0, complete=True,
               pass_at_k=sum(x > 0 for x in means) / m, bench_version=version,
               derived_from=r.get("bench_version"), derived_by=RULE, score_full=r.get("score"), se_full=r.get("se"), samples_full=r.get("samples"))
    return out, None
