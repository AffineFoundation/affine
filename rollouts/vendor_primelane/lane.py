"""Continuous Prime-taskset datagen lane (second lane on the datagen pod).

Cycle: pick a taskset by weighted deficit -> select the next unprocessed
tasks (deterministic seeded shuffle of the catalog) -> run one batch through
verifiers v1 (local docker, mini_swe_textbased harness) -> convert finished
traces to corpus turns (resolved score is telemetry, not a keep-gate) -> validate against the refresh prefilter contract ->
queue + upload staging shards -> mark state -> prune the batch's images.

Restart-safe: outcomes live in state.jsonl (written only after a batch's
traces are parsed), sliced-but-not-uploaded turns survive in
pending_turns.jsonl / outbox/, and a crash mid-batch just means the
un-marked tasks are re-selected. Supervised by bootstrap.sh.

  python -m primelane.lane           # the service
  python -m primelane.lane --once    # one batch, then exit
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from datagen.providers import looks_like_provider_failure
from datagen.uploader import TurnUploader, shard_sha256

from primelane.catalog import load_catalog
from primelane.convert import convert_run, validate_records
from primelane.lanespec import LANE_IMAGE_PREFIXES, SPECS, panel_keys

log = logging.getLogger("primelane")

DEFAULT_PROVIDERS = [
    {"name": "engy", "model": "glm-5.2",
     "base_url": "https://api.engy.ai/v1", "key_env": "ENGY",
     "label": "engy/glm-5.2"},
    {"name": "openrouter", "model": "z-ai/glm-4.5-air",
     "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER",
     "label": "openrouter/z-ai/glm-4.5-air"},
]

MAX_CONSECUTIVE_FAILS = 3
FAIL_SLEEP_S = 600
POOL_EXHAUSTED_SLEEP_S = 6 * 3600
# After this many consecutive primary-provider failures, route batches to the
# fallback for PRIMARY_COOLDOWN_BATCHES before re-trying the primary.
PRIMARY_STRIKES = 2
PRIMARY_COOLDOWN_BATCHES = 6


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class LaneConfig:
    tasksets: tuple[tuple[str, int], ...]
    langs: frozenset[str] | None      # None = all languages
    providers: tuple[dict, ...]
    batch_size: int
    concurrency: int
    max_turns: int
    rollout_timeout_s: int
    batch_timeout_s: int
    data_dir: Path
    verifiers_dir: Path
    hf_repo: str
    upload_min_turns: int
    upload_interval_s: int
    prune_images: bool
    seed: int


def load_lane_config() -> LaneConfig:
    raw_ts = os.environ.get(
        "LANE_TASKSETS", "scaleswe:4,swerebench_v2:2,r2e_gym:1,multiswe:1")
    tasksets = []
    for entry in raw_ts.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, _, w = entry.partition(":")
        if name not in SPECS:
            sys.exit(f"unknown taskset {name!r}; known: {sorted(SPECS)}")
        tasksets.append((name, int(w or 1)))
    raw_langs = os.environ.get("LANE_LANGS", "python").strip().lower()
    langs = None if raw_langs in ("all", "*") else frozenset(
        s.strip() for s in raw_langs.split(",") if s.strip())
    providers = json.loads(os.environ["LANE_PROVIDERS"]) \
        if os.environ.get("LANE_PROVIDERS") else DEFAULT_PROVIDERS
    providers = [p for p in providers if os.environ.get(p["key_env"])]
    if not providers:
        sys.exit("no lane provider has its key env set (fail-closed)")
    return LaneConfig(
        tasksets=tuple(tasksets),
        langs=langs,
        providers=tuple(providers),
        batch_size=_int("LANE_BATCH_SIZE", 10),
        # Capacity coordination: the main datagen loop shares this pod (its
        # rollout containers + eval workers); keep the lane's slice modest.
        concurrency=_int("LANE_CONCURRENCY", 10),
        max_turns=_int("LANE_MAX_TURNS", 80),
        rollout_timeout_s=_int("LANE_ROLLOUT_TIMEOUT_S", 3600),
        batch_timeout_s=_int("LANE_BATCH_TIMEOUT_S", 7200),
        data_dir=Path(os.environ.get("LANE_DATA_DIR", "/root/prime-lane/data")),
        verifiers_dir=Path(os.environ.get(
            "LANE_VERIFIERS_DIR", "/root/prime-pilot/verifiers")),
        hf_repo=os.environ.get("LANE_HF_REPO", "unconst/affine-datagen-turns"),
        upload_min_turns=_int("LANE_UPLOAD_MIN_TURNS", 400),
        upload_interval_s=_int("LANE_UPLOAD_INTERVAL_S", 3600),
        prune_images=os.environ.get("LANE_PRUNE_IMAGES", "1") not in
        ("0", "false", "no", ""),
        seed=_int("LANE_SEED", 0),
    )


# -- state ---------------------------------------------------------------------

class LaneState:
    def __init__(self, path: Path):
        self.path = path
        self.done: dict[str, set[str]] = {}
        if path.exists():
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.done.setdefault(rec["taskset"], set()).add(rec["uid"])

    def mark(self, taskset: str, uid: str, outcome: str, **extra) -> None:
        rec = {"taskset": taskset, "uid": uid, "outcome": outcome,
               "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               **extra}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.done.setdefault(taskset, set()).add(uid)


# -- selection -------------------------------------------------------------------

def ordered_uids(cfg: LaneConfig, name: str, catalog: list[dict]) -> list[dict]:
    rows = [r for r in catalog
            if cfg.langs is None or (r.get("language") or "") in cfg.langs]
    rng = random.Random(f"{cfg.seed}:{name}")
    rng.shuffle(rows)
    return rows


def pick_taskset(cfg: LaneConfig, counts: dict[str, int],
                 remaining: dict[str, int]) -> str | None:
    """Largest deficit vs target share, among tasksets with work left."""
    cands = [(n, w) for n, w in cfg.tasksets if remaining.get(n, 0) > 0]
    if not cands:
        return None
    total_w = sum(w for _, w in cands)
    total_c = sum(counts.get(n, 0) for n, _ in cands) + 1
    return max(cands, key=lambda nw: (
        nw[1] / total_w * total_c - counts.get(nw[0], 0), nw[1], nw[0]))[0]


# -- batch execution --------------------------------------------------------------

def eval_cmd(cfg: LaneConfig, name: str, provider: dict,
             uids: list[str], run_dir: Path) -> list[str]:
    spec = SPECS[name]
    cmd = [
        "nice", "-n", "10", "uv", "run", "eval", spec.taskset_id,
        "-n", str(len(uids)),
        "-m", provider["model"],
        "--client.base-url", provider["base_url"],
        "--client.api-key-var", provider["key_env"],
        "--env.agent.harness.id", "mini_swe_textbased",
        "--env.agent.runtime.type", "docker",
        "--env.agent.max-turns", str(cfg.max_turns),
        "--env.agent.timeout.setup", "1800",
        "--env.agent.timeout.rollout", str(cfg.rollout_timeout_s),
        "--env.agent.timeout.scoring", "1800",
        "--push", "False", "--rich", "False",
        "-c", str(cfg.concurrency),
        "-o", str(run_dir),
    ]
    if spec.select == "tasks":
        # Harbor filters on the task directory basename; convert still keys
        # on TaskData.name (e.g. "terminal-bench/<dir>").
        task_ids = uids
        if name == "terminal_bench_2":
            task_ids = [u.rsplit("/", 1)[-1] for u in uids]
        cmd.extend(["--env.taskset.tasks", *task_ids])
    else:
        # swesmith task names are "{lang}:{instance_id}"; the HF filter
        # matches bare instance_id across language shards.
        filter_ids = uids
        if name == "swesmith":
            filter_ids = [u.split(":", 1)[1] if ":" in u else u for u in uids]
        id_set = "{" + ",".join(repr(u) for u in filter_ids) + "}"
        filter_expr = f"lambda row: row[{spec.uid_field!r}] in {id_set}"
        if spec.pass_dataset:
            cmd.extend([
                "--env.taskset.dataset-name", spec.dataset,
                "--env.taskset.split", spec.split,
            ])
        elif spec.split:
            cmd.extend(["--env.taskset.split", spec.split])
        cmd.extend(["--env.taskset.filter-fn", filter_expr])
    cmd.extend(spec.extra_flags)
    return cmd


def build_terminal_lego_images(batch: list[dict]) -> tuple[list[dict], list[str]]:
    """Local-build each Terminal-Lego task image; return (ok_batch, failed_uids)."""
    ok: list[dict] = []
    failed: list[str] = []
    for row in batch:
        uid = row["uid"]
        image = row.get("image") or ""
        task_dir = Path(row.get("task_dir") or "")
        dockerfile = task_dir / "environment" / "Dockerfile"
        context = task_dir / "environment"
        if not image or not dockerfile.is_file():
            log.warning("terminal_lego %s: missing image/Dockerfile", uid)
            failed.append(uid)
            continue
        # Skip rebuild if the tag already exists locally.
        probe = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, timeout=60)
        if probe.returncode == 0:
            ok.append(row)
            continue
        log.info("docker build %s <- %s", image, dockerfile)
        proc = subprocess.run(
            ["docker", "build", "-t", image, "-f", str(dockerfile),
             str(context)],
            capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            tail = (proc.stdout or "")[-500:] + (proc.stderr or "")[-500:]
            log.error("docker build failed for %s: %s", uid, tail)
            failed.append(uid)
            continue
        ok.append(row)
    return ok, failed


def run_eval(cfg: LaneConfig, cmd: list[str]) -> int:
    env = dict(os.environ)
    env["PATH"] = f"{Path.home()}/.local/bin:" + env.get("PATH", "")
    log.info("eval: %s", " ".join(cmd[:8]) + " ...")
    proc = subprocess.Popen(
        cmd, cwd=str(cfg.verifiers_dir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)
    t0 = time.time()
    tail: list[str] = []
    while True:
        line = proc.stdout.readline()  # type: ignore[union-attr]
        if line:
            tail.append(line)
            tail = tail[-60:]
        elif proc.poll() is not None:
            break
        if time.time() - t0 > cfg.batch_timeout_s:
            log.warning("batch timeout after %ds; killing eval",
                        cfg.batch_timeout_s)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(10)
            return -2
    code = proc.returncode
    if code != 0:
        log.error("eval exited %s; tail:\n%s", code, "".join(tail[-25:]))
    return code


def reap_lane_containers() -> None:
    """Remove leftover containers from this lane's image namespaces only —
    the main loop's swerebench/sweb.eval containers never match."""
    try:
        out = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.ID}} {{.Image}}"],
            capture_output=True, text=True, timeout=60).stdout
        stale = [line.split()[0] for line in out.splitlines()
                 if len(line.split()) == 2
                 and line.split()[1].startswith(LANE_IMAGE_PREFIXES)]
        if stale:
            subprocess.run(["docker", "rm", "-f", *stale],
                           capture_output=True, timeout=120)
            log.info("reaped %d leftover lane container(s)", len(stale))
    except Exception:
        log.warning("lane container reap failed", exc_info=True)


def prune_batch_images(images: list[str]) -> None:
    for image in set(images):
        if not image:
            continue
        try:
            subprocess.run(["docker", "rmi", "-f", image],
                           capture_output=True, timeout=120)
        except Exception:
            log.warning("prune of %s failed", image, exc_info=True)


def batch_provider_suspect(per_task: list[dict], run_dir: Path) -> bool:
    """A batch whose traces are missing, or where nothing resolved and most
    rollouts errored with provider-failure signatures, is retried on the
    fallback endpoint."""
    if not (run_dir / "traces.jsonl").exists() or not per_task:
        return True
    errored = [r for r in per_task if r.get("error")]
    if any(r["resolved"] == 1.0 for r in per_task):
        return False
    if len(errored) < max(1, len(per_task) // 2):
        return False
    blob = " ".join(str(r.get("error") or "") + str(r.get("stop") or "")
                    for r in errored)
    return looks_like_provider_failure(blob) or len(errored) == len(per_task)


# -- upload ----------------------------------------------------------------------

def flush_uploads(cfg: LaneConfig, pending: Path,
                  uploader: TurnUploader) -> None:
    outbox = cfg.data_dir / "outbox"
    cut_state = cfg.data_dir / "upload_state.json"
    last_cut = 0.0
    if cut_state.exists():
        last_cut = float(json.loads(cut_state.read_text()).get("last_cut", 0))
    n_pending = 0
    if pending.exists():
        with open(pending, "rb") as f:
            n_pending = sum(1 for line in f if line.strip())
    if n_pending and (n_pending >= cfg.upload_min_turns
                      or time.time() - last_cut >= cfg.upload_interval_s):
        outbox.mkdir(parents=True, exist_ok=True)
        sha = shard_sha256(pending)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        dest = outbox / f"prime-turns-{stamp}-{sha[:12]}.jsonl"
        pending.rename(dest)
        cut_state.write_text(json.dumps({"last_cut": time.time()}))
        log.info("cut shard %s (%d turns)", dest.name, n_pending)
    shards = sorted(outbox.glob("*.jsonl")) if outbox.is_dir() else []
    for shard in shards:
        try:
            uploader.upload_shard(shard)
        except Exception:
            log.warning("upload of %s failed; will retry next cycle",
                        shard.name, exc_info=True)
            break
        shard.unlink()


# -- main ------------------------------------------------------------------------

def archive_traces(cfg: LaneConfig, run_dir: Path, tag: str) -> None:
    src = run_dir / "traces.jsonl"
    if not src.exists():
        return
    dest_dir = cfg.data_dir / "trace_archive"
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, gzip.open(dest_dir / f"{tag}.jsonl.gz",
                                           "wb") as fout:
        shutil.copyfileobj(fin, fout)


def process_batch(cfg: LaneConfig, name: str, batch: list[dict],
                  state: LaneState, pending: Path, panel: tuple,
                  primary_down: list[int]) -> bool:
    """Run one batch end to end. Returns True if traces were produced."""
    spec = SPECS[name]
    panel_ids, panel_repos, panel_bare = panel
    tag = f"{name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    run_base = cfg.data_dir / "runs" / tag
    uids = [r["uid"] for r in batch]
    meta_by_uid = {r["uid"]: r for r in batch}
    reap_lane_containers()

    if spec.local_docker_build:
        batch, build_failed = build_terminal_lego_images(batch)
        for uid in build_failed:
            state.mark(name, uid, "error", detail="docker_build_failed")
        uids = [r["uid"] for r in batch]
        meta_by_uid = {r["uid"]: r for r in batch}
        if not batch:
            log.error("all terminal_lego docker builds failed; skipping batch")
            return False

    providers = list(cfg.providers)
    if primary_down[0] > 0 and len(providers) > 1:
        primary_down[0] -= 1
        providers = providers[1:] + providers[:1]
        log.info("primary on cooldown (%d batches left); trying %s first",
                 primary_down[0], providers[0]["name"])

    per_task: list[dict] = []
    used = None
    for attempt, provider in enumerate(providers):
        run_dir = run_base / provider["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        code = run_eval(cfg, eval_cmd(cfg, name, provider, uids, run_dir))
        records, per_task = convert_run(
            run_dir, spec.name, provider["label"], meta_by_uid,
            panel_ids, panel_repos, panel_bare)
        suspect = code != 0 or batch_provider_suspect(per_task, run_dir)
        log.info("batch %s via %s: exit=%s tasks=%d resolved=%d raw_turns=%d "
                 "in %.0fs%s", tag, provider["name"], code, len(per_task),
                 sum(1 for r in per_task if r["resolved"] == 1.0),
                 len(records), time.time() - t0,
                 " [provider-suspect]" if suspect else "")
        if not suspect or attempt == len(providers) - 1:
            used = (provider, run_dir, records)
            if suspect and provider is cfg.providers[0]:
                primary_down[0] = PRIMARY_COOLDOWN_BATCHES
            break
        if provider is cfg.providers[0]:
            primary_down[0] = PRIMARY_COOLDOWN_BATCHES
        log.warning("retrying batch on fallback provider")

    provider, run_dir, records = used
    if not per_task and not (run_dir / "traces.jsonl").exists():
        shutil.rmtree(run_base, ignore_errors=True)
        return False

    kept, drops = validate_records(records, panel_ids, panel_repos,
                                   panel_bare)
    if drops:
        log.info("validation drops: %s", drops)
    if kept:
        pending.parent.mkdir(parents=True, exist_ok=True)
        with open(pending, "a", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    kept_turns_by_uid: dict[str, int] = {}
    for rec in kept:
        # sid is embedded in traj_id; recover per-task counts via instance_id
        kept_turns_by_uid[rec["instance_id"]] = \
            kept_turns_by_uid.get(rec["instance_id"], 0) + 1
    seen_uids = set()
    for row in per_task:
        uid = row["uid"]
        seen_uids.add(uid)
        sid = meta_by_uid.get(uid, {}).get("sid", "")
        outcome = ("resolved" if row["resolved"] == 1.0
                   else "unresolved" if row["resolved"] == 0.0
                   else "error")
        state.mark(name, uid, outcome,
                   detail=row.get("error") or row.get("stop") or "",
                   n_turns=kept_turns_by_uid.get(sid, row["n_records"]),
                   provider=provider["label"],
                   cost_usd=row.get("cost_usd", 0.0),
                   prompt_tokens=row.get("prompt_tokens", 0),
                   completion_tokens=row.get("completion_tokens", 0),
                   cached_tokens=row.get("cached_tokens", 0),
                   n_calls=row.get("n_calls", 0),
                   agent_wall_s=row.get("agent_wall_s"))
    # Tasks the eval never produced a trace for (killed batch, load error):
    # left unmarked so they are re-selected later.
    missing = [u for u in uids if u not in seen_uids]
    if missing:
        log.info("%d task(s) produced no trace; will be re-selected",
                 len(missing))

    archive_traces(cfg, run_dir, tag)
    shutil.rmtree(run_base, ignore_errors=True)
    if cfg.prune_images:
        prune_batch_images([r["image"] for r in batch])
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true",
                    help="process a single batch, then exit")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_lane_config()
    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN missing (fail-closed: turns could never upload)")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(cfg.data_dir / "runs", ignore_errors=True)
    log.info("prime lane starting: tasksets=%s langs=%s providers=%s "
             "batch=%d conc=%d hf_repo=%s",
             cfg.tasksets, sorted(cfg.langs) if cfg.langs else "all",
             [p["name"] for p in cfg.providers], cfg.batch_size,
             cfg.concurrency, cfg.hf_repo)

    state = LaneState(cfg.data_dir / "state.jsonl")
    pending = cfg.data_dir / "pending_turns.jsonl"
    uploader = TurnUploader(cfg.hf_repo, private=True)
    panel = panel_keys()

    pools = {name: ordered_uids(cfg, name, load_catalog(name))
             for name, _ in cfg.tasksets}
    for name, rows in pools.items():
        n_done = len(state.done.get(name, set()))
        log.info("taskset %s: %d selectable tasks (%d already processed)",
                 name, len(rows), n_done)

    counts: dict[str, int] = {}
    fails = 0
    primary_down = [0]  # batches left on fallback after primary strikes
    while True:
        remaining = {
            name: sum(1 for r in rows
                      if r["uid"] not in state.done.get(name, set()))
            for name, rows in pools.items()
        }
        name = pick_taskset(cfg, counts, remaining)
        if name is None:
            log.info("all pools exhausted; sleeping %ds",
                     POOL_EXHAUSTED_SLEEP_S)
            if args.once:
                break
            flush_uploads(cfg, pending, uploader)
            time.sleep(POOL_EXHAUSTED_SLEEP_S)
            pools = {n: ordered_uids(cfg, n, load_catalog(n))
                     for n, _ in cfg.tasksets}
            continue
        done = state.done.get(name, set())
        batch = [r for r in pools[name] if r["uid"] not in done]
        batch = batch[: cfg.batch_size]
        counts[name] = counts.get(name, 0) + 1
        log.info("cycle: taskset=%s batch=%d remaining=%s",
                 name, len(batch), remaining)

        ok = process_batch(cfg, name, batch, state, pending, panel,
                           primary_down)
        if ok:
            fails = 0
        else:
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                log.error("%d consecutive batch failures; sleeping %ds",
                          fails, FAIL_SLEEP_S)
                time.sleep(FAIL_SLEEP_S)
                fails = 0
        flush_uploads(cfg, pending, uploader)
        if args.once:
            break


if __name__ == "__main__":
    main()
