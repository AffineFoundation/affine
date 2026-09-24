"""Autofill: turn the coverage report into launched work.

The nightly check only *listed* missing cells (2026-09-15/16), and nothing
launched them — Jacob, 2026-09-17 00:21 UTC: "tonnes of env and benchmarks
missing ... should not be the case". This script closes the loop: for every
row in scope (teacher, genesis, reference models, kings with reign >=
COVERAGE_MIN_KING_REIGN) it queues one benchmark pass per missing cell group
and starts one env backfill per row with env gaps.

    python autofill.py            # queue / start everything missing (idempotent)
    python autofill.py --dry-run  # print the plan only

Benchmark cells -> ops/coverage/bench_queue.py entries (one Lium pod each,
the queue's budget cap applies):
  chat sets missing      -> `lium --no-sandbox` pass on those envs, merged into the row's card
  SWE-bench / miniF2F    -> `lium` pass (chat = humaneval) with the sandbox sets forced
  other sandbox missing  -> `agentic`-mode pass on those envs (docker cells), merged
  agentic sets missing   -> `agentic` pass on those envs, merged
Env cells -> ops/coverage/start_env_backfill.sh (rent a serving box, run
rollouts.backfill on the backfill pod for the missing + low-n sources), at most
ENV_MAX_LIVE boxes at once (ledger state/backfill_pods.json).

Idempotent: a cell that is running, or whose row already has a pending /
running queue entry covering it, or an env row with a live driver, is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from coverage import classify, load_matrix, MIN_KING_REIGN  # noqa: E402

STATE_DIR = HERE / "state"
QUEUE_PATH = STATE_DIR / "bench_queue.json"
LEDGER_PATH = STATE_DIR / "backfill_pods.json"
CARDS_DIR = REPO / "affine" / "state" / "benchsuite"
SUITE_TOML = REPO / "ops" / "benchsuite" / "suite.toml"
VALIDATOR_STATE = REPO / "affine" / "state" / "state.json"
PY = str(REPO / ".venv" / "bin" / "python")
TEACHER_REF = os.environ.get("COVERAGE_TEACHER_REF", "hf://Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
GENESIS_REF = "hf://Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0"
FULL_SANDBOX_TRIGGERS = {"minif2f"}
# benchmark columns autofill never queues: Gaia2 ambiguity runs from the box through the
# benchsuite worker's ask-queue (ARE), not as a pod pass (2026-09-21)
BENCH_EXCLUDE = set((os.environ.get("COVERAGE_BENCH_EXCLUDE") or "gaia2-ambiguity").split(","))
FAST_MIN_MISSING = int(os.environ.get("COVERAGE_FAST_MIN_MISSING", "8"))   # >= this many benchmark cells missing -> one fast pass   # only the Prime-sandbox set needs the full sandbox phase; docker sets run as cells
ENV_MAX_LIVE = int(os.environ.get("COVERAGE_ENV_MAX_LIVE", "3"))   # serving boxes (not the driver pod); pro6000 first
# The env backfill needs the driver pod (rollouts.backfill runs there). It is rented by
# the datagen worker; when it is gone (72 h TTL took affine-backfill-2 on 2026-09-19)
# renting serving boxes only burns money: five idle boxes, $35, that afternoon.
BACKFILL_DRIVER_PTR = Path.home() / "subnet120" / "affine" / "state" / "pods" / "backfill_driver.json"   # kept current by the datagen worker


def _driver_default() -> str:
    """affine-backfill-3 (no Lium TTL, 2026-09-19 22:19 UTC); the pointer file wins when the pod is replaced."""
    try:
        return str(json.loads(BACKFILL_DRIVER_PTR.read_text())["ssh"])
    except (OSError, ValueError, KeyError):
        return "root@94.70.140.197:20099"


BACKFILL_POD = os.environ.get("COVERAGE_BACKFILL_POD") or _driver_default()
ENV_CONTAINERS = int(os.environ.get("COVERAGE_ENV_CONTAINERS", "12"))


def suite_lists() -> tuple[set, set, set]:
    s = tomllib.load(SUITE_TOML.open("rb"))["modes"]
    return set(s.get("chat_envs") or []), set(s.get("sandbox_envs") or []), set(s.get("agentic_std_envs") or [])


def king_refs() -> dict[str, str]:
    """digest12 -> full sha256 for every reign in state.json."""
    try:
        st = json.loads(VALIDATOR_STATE.read_text())
    except (OSError, ValueError):
        return {}
    k = st.get("king") or {}
    return {str(x.get("revision") or "")[:12]: str(x.get("revision") or "")
            for x in [k, *(k.get("previous") or [])] if x.get("revision")}


def row_ref(row: dict, kings: dict[str, str]) -> str | None:
    if row["kind"] == "teacher":
        return TEACHER_REF
    if row["kind"] == "genesis":
        return GENESIS_REF
    if row["kind"] == "king":
        return kings.get(row.get("digest12") or "")
    # reference rows (occamy-1.0, ...): the card knows the hf ref
    for rid in row.get("cards") or []:
        try:
            k = json.loads((CARDS_DIR / f"{rid}.json").read_text()).get("king") or {}
        except (OSError, ValueError):
            continue
        if str(k.get("repo") or "").startswith("hf://"):
            return k["repo"]
        if k.get("hf_repo") and k.get("hf_revision"):
            return f"hf://{k['hf_repo']}@{k['hf_revision']}"
    return None


def row_label(row: dict) -> str:
    if row["kind"] == "teacher":
        return "teacher"
    if row["kind"] == "genesis":
        return "genesis"
    if row["kind"] == "king":
        return str(row.get("reign"))
    return str(row.get("label") or row.get("key")).lower().replace(" ", "-")


def queue_entries() -> list[dict]:
    try:
        return json.loads(QUEUE_PATH.read_text())
    except (OSError, ValueError):
        return []


def covered_by_queue(q: list[dict], ref: str, envs: set[str]) -> bool:
    """A pending / running entry for this ref already carries every env."""
    for e in q:
        if e.get("status") not in ("pending", "running") or e.get("ref") != ref:
            continue
        have = set((e.get("chat_envs") or "").split(",")) - {""}
        if e.get("sandbox", True) and e.get("mode") == "lium":
            have |= suite_lists()[1]
        if envs <= have:
            return True
    return False


def add(ref: str, label: str, mode: str, envs: list[str], sandbox: bool, merge_into: str | None,
        merge_as: str, priority: int, note: str, dry: bool, bucket: str = "fill") -> None:
    cmd = [PY, str(HERE / "bench_queue.py"), "add", "--ref", ref, "--label", label, "--mode", mode,
           "--priority", str(priority), "--note", note, "--bucket", bucket]
    if envs:
        cmd += ["--chat-envs", ",".join(sorted(envs))]
    if not sandbox:
        cmd.append("--no-sandbox")
    if merge_into:
        cmd += ["--env", f"BENCHSUITE_MERGE_INTO={merge_into}", "--env", f"BENCHSUITE_MERGE_AS={merge_as}"]
    print(("DRY " if dry else "") + " ".join(cmd[2:]))
    if not dry:
        subprocess.run(cmd, check=False, cwd=str(HERE))


def driver_pod_reachable() -> bool:
    host, _, port = BACKFILL_POD.rpartition(":")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
                        "-p", port, host, "tmux -V >/dev/null 2>&1; test -x /root/rollouts/run_backfill.sh"],
                       capture_output=True, text=True)
    return r.returncode == 0


def live_env_drivers() -> set[str]:
    """digest12s with an unreleased serving box in the ledger."""
    try:
        return {r["model"] for r in json.loads(LEDGER_PATH.read_text())
                if not r.get("released_at") and r.get("purpose") == "model"}
    except (OSError, ValueError, KeyError):
        return set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-env", action="store_true", help="benchmark cells only")
    args = ap.parse_args()
    rep = classify(load_matrix())
    matrix_rows = {r["key"]: r for r in load_matrix()["rows"]}
    chat, sandbox, agentic = suite_lists()
    kings = king_refs()
    q = queue_entries()
    live = live_env_drivers()
    driver_ok = args.no_env or driver_pod_reachable()
    if not driver_ok:
        print(f"env backfill: driver pod {BACKFILL_POD} unreachable — no serving box is rented "
              "(ask the datagen worker for a new backfill pod; set COVERAGE_BACKFILL_POD)", file=sys.stderr)
    n_bench = n_env = 0
    for r in rep["rows"]:
        row = matrix_rows[r["key"]]
        ref = row_ref(row, kings)
        label = row_label(row)
        if not ref:
            print(f"{r['label']}: no model reference known; cannot fill", file=sys.stderr)
            continue
        merge_into = (row.get("cards") or [None])[0]
        merge_as = "teacher" if row["kind"] == "teacher" else "king"
        missing = set(r["missing_bench"]) - BENCH_EXCLUDE   # running cells are not in missing_bench
        if row.get("current"):
            # the sitting king's card is the watcher's crown pass (fast mode, 5 pods): never
            # queue a second pass for it; its gaps close when the watcher's pass ends
            if missing:
                print(f"{r['label']}: sitting king — {len(missing)} cells left to the watcher's pass")
            missing = set()
        if r.get("running_bench") and missing:
            # a pass is still landing cells on this row: wait for it before judging gaps
            print(f"{r['label']}: {len(r['running_bench'])} cells running; gap check deferred")
            missing = set()
        if row.get("inflight"):
            # a benchsuite pass (watcher / by hand / fast) is running for this row: its
            # cells land on their own; queueing them again doubled reign 18 on 09-19
            print(f"{r['label']}: benchsuite pass {row['inflight'].get('run_id')} in flight; nothing queued")
            missing = set()
        groups = []
        # a row with no card yet, or missing most benchmark columns, gets the whole card in
        # ONE `fast` pass (5 pods in parallel, Daytona for SWE-bench; speed-up budget 2026-09-19)
        if missing and (not row.get("cards") or len(missing) >= FAST_MIN_MISSING):
            if covered_by_queue(q, ref, missing) or any(e.get("status") in ("pending", "running") and e.get("ref") == ref
                                                        and e.get("mode") == "fast" for e in q):
                print(f"{r['label']}: fast pass already queued / running")
            else:
                add(ref, label, "fast", [], False, None, merge_as, 20, f"autofill {r['label']}: whole card (fast, {len(missing)} cells missing)", args.dry_run, bucket="speedup")
                n_bench += 1
            missing = set()
        if missing & chat:
            groups.append(("lium", sorted(missing & chat), False, "chat sets"))
        sb = missing & sandbox
        if sb & FULL_SANDBOX_TRIGGERS:
            groups.append(("lium", ["humaneval"], True, "sandbox sets (miniF2F needs the full sandbox phase)"))
        elif sb:
            groups.append(("agentic", sorted(sb), False, "docker sandbox cells (SWE-bench / long-context) as cells"))
        ag = missing & agentic
        if ag:
            groups.append(("agentic", sorted(ag), False, "agentic sets"))
        rest = missing - chat - sandbox - agentic
        if rest:
            groups.append(("agentic", sorted(rest), False, "other cells"))
        for i, (mode, envs, sbx, why) in enumerate(groups):
            need = set(sb) if sbx else set(envs)
            if covered_by_queue(q, ref, need):
                print(f"{r['label']}: {why} already queued / running")
                continue
            add(ref, label, mode, envs, sbx, merge_into if not sbx else None, merge_as,
                20 + i, f"autofill {r['label']}: {why}", args.dry_run)
            n_bench += 1
        env_gaps = list(r["missing_env"]) + [x["env"] for x in r["low_env"]]
        d12 = row.get("digest12") or ""
        if env_gaps and not args.no_env and row["kind"] != "teacher":
            if d12 in live:
                print(f"{r['label']}: env driver live, {len(env_gaps)} gaps pending")
            elif not driver_ok:
                print(f"{r['label']}: env gaps {len(env_gaps)} — deferred, no driver pod")
            elif len(live) >= ENV_MAX_LIVE:
                print(f"{r['label']}: env backfill deferred ({len(live)} boxes live >= {ENV_MAX_LIVE})")
            else:
                cmd = [str(HERE / "rent_and_backfill.sh"), ref, d12, label, ",".join(env_gaps), str(ENV_CONTAINERS)]
                print(("DRY " if args.dry_run else "") + " ".join(cmd))
                if not args.dry_run:
                    subprocess.Popen(["setsid", "bash", *cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
                    live.add(d12)
                n_env += 1
        elif env_gaps and row["kind"] == "teacher":
            print(f"Teacher: env gaps {env_gaps} -> teacher backfill (Engy endpoint) is the datagen worker's run: "
                  f"run_backfill.sh --teacher --sources {','.join(env_gaps)}")
    print(f"autofill: {n_bench} benchmark passes queued, {n_env} env backfills started")
    return 0


if __name__ == "__main__":
    sys.exit(main())
