"""Stage 1: build the Bradley-Terry pair dataset for a black-box discriminator.

The proposal being tested: score the miner's ACTION y_A, which the contract
currently does not score at all. Today y_A only appears in the B gate
(did z cause y?) and the leakage check (was y smuggled into z?). Neither asks
whether the action is any good. swe_rebench measures exactly that, and
`rt7_full_panel.py` shows Reason carries no detectable signal about it.

A discriminator trained to separate teacher actions from miner actions on the
same prefix gives an action-quality proxy that needs no ground-truth pass/fail,
and needs no teacher logprobs — so it also survives a move to a closed teacher
where Reason becomes uncomputable.

This script only builds the dataset. It runs on CPU against artifacts already
on disk; nothing is queried and no model is loaded.

Sources, all local:
  ../affine/state/evals/chal-*.json.gz   pair records: z_a, y_a, teacher refs
  research/data/*/chunks/*.jsonl.gz      corpus trajectories, for the prefix

Layout note: the artifacts store one miner rollout against k teacher refs per
turn, so z_a is constant within a turn and only the teacher reference varies.
Rows are therefore emitted per (turn, model) with the k teacher actions inline,
which is k times smaller than one row per pair and loses nothing. Prefixes are
written once to a separate deduped file because turns recur across duels.

Splits are BY REPO AND BY TIME: repos are ordered by first appearance and cut
chronologically, so no repo straddles the split and the test set also measures
whether the discriminator survives population drift. Training on a repo that
appears in the evaluation panel would measure memorization.

The cut is applied to the LABELLED repos only. Bench runs lag submission — the
newest 70 repos carry 1 swe label between them — so cutting the full repo list
chronologically dumps the unlabelled tail into test and leaves nothing to
score. Labelled repos are the scarce resource and the evaluation panel, so
they are split evenly by default; every unlabelled repo goes to train, where
it is still useful as discriminator training data.

Usage:
  cd research && python scripts/build_disc_pairs.py
  cd research && python scripts/build_disc_pairs.py --turns-per-duel 400
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "affine"))

from affine.corpus.materialize import materialize_turn  # noqa: E402

EVALS_GLOB = "../affine/state/evals/chal-*.json.gz"
EVALS_INDEX = "../affine/state/evals/index.jsonl"
BENCH_HISTORY = "../affine/state/bench_history.jsonl"
CORPUS_GLOB = "data/*/chunks/*.jsonl.gz"
OUT_DIR = Path("data/disc_pairs")
SHARD_ROWS = 20_000
SUITE = "swe_rebench_lite"


def load_corpus_index():
    """traj_id -> trajectory record, from every locally cached corpus chunk."""
    traj = {}
    for path in sorted(glob.glob(CORPUS_GLOB)):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("traj_id"):
                    traj[rec["traj_id"]] = rec
    return traj


def load_benched_repos():
    """Repos with at least one ok bench run — the Stage 3 evaluation panel."""
    out = set()
    try:
        fh = open(BENCH_HISTORY)
    except FileNotFoundError:
        return out
    for line in fh:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("suite") == SUITE and (row.get("result") or {}).get("ok"):
            out.add((row.get("repo") or "").lower())
    return out


def resolve_prefix(traj_index, turn_id):
    """Rebuild the turn prefix with the production materializer, or None."""
    if ":" not in turn_id:
        return None
    traj_id, idx = turn_id.rsplit(":", 1)
    traj = traj_index.get(traj_id)
    if not traj:
        return None
    try:
        want = int(idx)
    except ValueError:
        return None
    meta = next((t for t in traj.get("turns", [])
                 if int(t.get("turn_idx", -1)) == want), None)
    if not meta:
        return None
    try:
        return materialize_turn(traj, meta)
    except Exception:
        return None


def duel_rows(artifact, turn_ids_wanted):
    """Emit (turn, model) rows carrying z_a, y_a and the k teacher actions."""
    refs = artifact.get("teacher_refs") or {}
    req = artifact.get("request") or {}
    out = []
    for role, key, repo_key, rev_key in (
            ("challenger", "challenger_rows", "challenger_repo", "challenger_revision"),
            ("king", "king_rows", "king_repo", "king_revision")):
        repo = (req.get(repo_key) or "").lower()
        if not repo:
            continue
        for row in artifact.get(key) or []:
            tid = row.get("turn_id")
            if tid not in turn_ids_wanted or not row.get("valid", True):
                continue
            pairs = row.get("pairs") or []
            tref = refs.get(tid) or []
            if not pairs or not tref:
                continue
            # z_a/y_a are constant across pairs within a turn (1 miner rollout).
            z_a = pairs[0].get("z_a") or ""
            y_a = pairs[0].get("y_a") or ""
            if not y_a:
                continue
            teacher_y = [r.get("y") or "" for r in tref if r.get("y")]
            if not teacher_y:
                continue
            out.append({
                "turn_id": tid,
                "role": role,
                "repo": repo,
                "revision": req.get(rev_key) or "",
                "z_a": z_a,
                "y_a": y_a,
                "teacher_y": teacher_y,
                "teacher_z": [r.get("z") or "" for r in tref],
                "n_exact_match": sum(1 for y in teacher_y if y.strip() == y_a.strip()),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns-per-duel", type=int, default=200,
                    help="deterministic subsample per duel; 0 = all turns")
    ap.add_argument("--seed", type=int, default=120)
    ap.add_argument("--train-frac", type=float, default=0.50,
                    help="chronological fraction of LABELLED repos kept for "
                         "train; unlabelled repos always go to train")
    ap.add_argument("--require-prefix", action="store_true",
                    help="drop turns whose prefix cannot be resolved locally")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N duels")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("indexing local corpus ...")
    traj_index = load_corpus_index()
    benched = load_benched_repos()
    print(f"  {len(traj_index)} trajectories cached, {len(benched)} benched repos")

    at_by_chal = {}
    try:
        for line in open(EVALS_INDEX):
            r = json.loads(line)
            if r.get("challenge_id"):
                at_by_chal[r["challenge_id"]] = r.get("at", "")
    except FileNotFoundError:
        pass

    paths = sorted(glob.glob(EVALS_GLOB))
    if args.limit:
        paths = paths[:args.limit]

    rows = []
    first_seen = {}
    epoch_counts = Counter()
    prefix_needed = set()
    stats = Counter()

    for path in paths:
        try:
            art = json.loads(gzip.open(path).read())
        except Exception:
            stats["duels_unreadable"] += 1
            continue
        if not art.get("challenger_rows") or not art.get("king_rows"):
            stats["duels_incomplete"] += 1
            continue

        cid = Path(path).name.split(".")[0]
        at = at_by_chal.get(cid, "")
        epoch = (art.get("slice") or {}).get("corpus_epoch")

        tids = list(art.get("turn_ids") or [])
        if args.turns_per_duel and len(tids) > args.turns_per_duel:
            rng = random.Random(f"{args.seed}:{cid}")
            tids = rng.sample(tids, args.turns_per_duel)
        wanted = set(tids)

        got = duel_rows(art, wanted)
        if not got:
            stats["duels_no_rows"] += 1
            continue
        stats["duels_used"] += 1
        epoch_counts[epoch] += 1

        for r in got:
            r["challenge_id"] = cid
            r["at"] = at
            r["corpus_epoch"] = epoch
            prefix_needed.add(r["turn_id"])
            repo = r["repo"]
            if repo not in first_seen or (at and at < first_seen[repo]):
                first_seen[repo] = at
        rows.extend(got)

    if not rows:
        raise SystemExit("no rows built — check EVALS_GLOB path (run from research/)")

    # ---- prefixes, written once and deduped --------------------------------
    print(f"resolving {len(prefix_needed)} distinct turn prefixes ...")
    prefixes, missing = {}, set()
    for tid in prefix_needed:
        t = resolve_prefix(traj_index, tid)
        if t is None:
            missing.add(tid)
        else:
            prefixes[tid] = {"turn_id": tid, "prefix": t["prefix"],
                             "reference_turn": t["reference_turn"],
                             "instance_id": t.get("instance_id", ""),
                             "action_kind": t.get("action_kind", "bash")}
    print(f"  resolved {len(prefixes)}, missing {len(missing)}")

    if args.require_prefix and missing:
        before = len(rows)
        rows = [r for r in rows if r["turn_id"] not in missing]
        stats["rows_dropped_no_prefix"] = before - len(rows)
        first_seen = {r: a for r, a in first_seen.items()
                      if any(x["repo"] == r for x in rows)}

    # ---- chronological split, applied to the labelled repos ----------------
    # Unlabelled repos cannot be scored in Stage 3, so they are never worth
    # spending on test; they go to train as extra discriminator data.
    labelled = sorted((r for r in first_seen if r in benched),
                      key=lambda r: (first_seen[r] or "", r))
    cut = int(len(labelled) * args.train_frac)
    split_of = {r: ("train" if i < cut else "test") for i, r in enumerate(labelled)}
    for r in first_seen:
        split_of.setdefault(r, "train")
    for r in rows:
        r["split"] = split_of[r["repo"]]

    # ---- write --------------------------------------------------------------
    for old in out_dir.glob("pairs_*.jsonl.gz"):
        old.unlink()
    rows.sort(key=lambda r: (r["split"], r["at"], r["challenge_id"], r["turn_id"]))
    shards = []
    for i in range(0, len(rows), SHARD_ROWS):
        name = f"pairs_{i // SHARD_ROWS:04d}.jsonl.gz"
        with gzip.open(out_dir / name, "wt") as fh:
            for r in rows[i:i + SHARD_ROWS]:
                fh.write(json.dumps(r) + "\n")
        shards.append(name)
    with gzip.open(out_dir / "turns.jsonl.gz", "wt") as fh:
        for t in prefixes.values():
            fh.write(json.dumps(t) + "\n")

    # ---- diagnostics --------------------------------------------------------
    # If miner actions already equal teacher actions verbatim, a discriminator
    # has nothing to separate and the whole approach is dead. Measure it here,
    # before anyone rents a GPU.
    n_ref = sum(len(r["teacher_y"]) for r in rows)
    n_exact = sum(r["n_exact_match"] for r in rows)
    fully_matched = sum(1 for r in rows if r["n_exact_match"] == len(r["teacher_y"]))
    any_matched = sum(1 for r in rows if r["n_exact_match"] > 0)

    by_split = Counter(r["split"] for r in rows)
    repos_by_split = defaultdict(set)
    for r in rows:
        repos_by_split[r["split"]].add(r["repo"])
    test_benched = repos_by_split["test"] & benched
    train_benched = repos_by_split["train"] & benched

    manifest = {
        "generated_by": "scripts/build_disc_pairs.py",
        "args": vars(args),
        "shards": shards,
        "turns_file": "turns.jsonl.gz",
        "n_rows": len(rows),
        "n_turns_with_prefix": len(prefixes),
        "n_turns_missing_prefix": len(missing),
        "prefix_coverage": len(prefixes) / max(1, len(prefix_needed)),
        "duels": dict(stats),
        "epochs": {str(k): v for k, v in epoch_counts.most_common()},
        "split": {
            "rows": dict(by_split),
            "repos": {k: len(v) for k, v in repos_by_split.items()},
            "benched_repos_train": len(train_benched),
            "benched_repos_test": len(test_benched),
            "test_repos": sorted(repos_by_split["test"]),
        },
        "action_overlap": {
            "teacher_refs_total": n_ref,
            "exact_match_refs": n_exact,
            "exact_match_rate": n_exact / max(1, n_ref),
            "rows_all_refs_matched": fully_matched,
            "rows_any_ref_matched": any_matched,
            "rows_any_ref_matched_rate": any_matched / max(1, len(rows)),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    ov = manifest["action_overlap"]
    print(f"""
built {len(rows)} (turn, model) rows -> {out_dir}/
  shards            {len(shards)}  x up to {SHARD_ROWS} rows
  prefixes          {len(prefixes)} resolved ({manifest['prefix_coverage']:.1%} coverage)
  duels used        {stats['duels_used']}

split (chronological, by repo — no repo straddles)
  train             {by_split['train']:>7} rows  {len(repos_by_split['train']):>4} repos  \
{len(train_benched):>4} with swe labels
  test              {by_split['test']:>7} rows  {len(repos_by_split['test']):>4} repos  \
{len(test_benched):>4} with swe labels

ACTION OVERLAP — the go / no-go diagnostic
  teacher refs matched verbatim by the miner   {ov['exact_match_rate']:.1%}
  rows where >=1 teacher ref matched verbatim  {ov['rows_any_ref_matched_rate']:.1%}
  rows where ALL teacher refs matched          {ov['rows_all_refs_matched']}

  A discriminator separates teacher actions from miner actions. Where the two
  strings are identical there is nothing to separate, so this rate is an upper
  bound on how much of the panel can carry signal. High rate => stop here.""")


if __name__ == "__main__":
    main()
