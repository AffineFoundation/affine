"""Per-task cross-check: verifiers MATH-500 run vs the calibrated SN120 runner (20261031-math500-comparable).

usage: python xcheck_math500.py <our per_task.jsonl> <calibrated grades.jsonl> [--out file.json]
Our idx = row of HuggingFaceH4/MATH-500 test split at the pinned revision; calibrated ids are its unique_id.
Writes only aggregates + the ids of the most divergent tasks (MATH-500 is public; no full-MATH-test data here).
"""
import argparse, json, math, urllib.request

REV = "6e4ed1a2"  # registry pin (bench/config/benchmarks.toml, math500)


def rows(rev):
    meta = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/HuggingFaceH4/MATH-500/revision/{rev}"))
    sha = meta["sha"]
    url = f"https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/{sha}/test.jsonl"
    return sha, [json.loads(l) for l in urllib.request.urlopen(url).read().decode().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ours"); ap.add_argument("calib"); ap.add_argument("--out")
    a = ap.parse_args()
    sha, ds = rows(REV)
    ours = {ds[r["idx"]]["unique_id"]: r for r in map(json.loads, open(a.ours)) if r["n"]}
    cal = {r["id"]: r for r in map(json.loads, open(a.calib)) if r.get("set") == "math500" and r["n"]}
    ids = sorted(set(ours) & set(cal))
    d = [(ours[i]["correct"] / ours[i]["n"]) - (cal[i]["c"] / cal[i]["n"]) for i in ids]
    po = sum(ours[i]["correct"] / ours[i]["n"] for i in ids) / len(ids)
    pc = sum(cal[i]["c"] / cal[i]["n"] for i in ids) / len(ids)
    md = sum(d) / len(d); se = math.sqrt(sum((x - md) ** 2 for x in d) / (len(d) - 1) / len(d))
    tok_o = sum(ours[i]["tokens"] / ours[i]["n"] for i in ids) / len(ids); tok_c = sum(cal[i]["tok"] / cal[i]["n"] for i in ids) / len(ids)
    worst = sorted(ids, key=lambda i: (ours[i]["correct"] / ours[i]["n"]) - (cal[i]["c"] / cal[i]["n"]))
    out = dict(dataset_sha=sha, tasks=len(ids), ours=po, calibrated=pc, diff_pp=100 * md, diff_se_pp=100 * se,
               ours_rollouts=sum(ours[i]["n"] for i in ids), calibrated_rollouts=sum(cal[i]["n"] for i in ids),
               tasks_ours_better=sum(x > 1e-9 for x in d), tasks_cal_better=sum(x < -1e-9 for x in d),
               mean_tokens_ours=tok_o, mean_tokens_calibrated=tok_c,
               most_negative=[(i, round(ours[i]["correct"] / ours[i]["n"], 3), round(cal[i]["c"] / cal[i]["n"], 3)) for i in worst[:8]],
               most_positive=[(i, round(ours[i]["correct"] / ours[i]["n"], 3), round(cal[i]["c"] / cal[i]["n"], 3)) for i in worst[-8:]])
    s = json.dumps(out, indent=1); print(s)
    if a.out:
        open(a.out, "w").write(s)


if __name__ == "__main__":
    main()
