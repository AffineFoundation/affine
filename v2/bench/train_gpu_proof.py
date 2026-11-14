"""GPU proof of the training job at two commits: one long-row batch and one short batch on one GPU.

This file is a tool for the operator. It is not part of the GPU code and it
changes no digest. It trains nothing that a round uses: every file it writes
is under ``OUT``.

Run it from inside the checkout, with the Python of the virtual environment:

    python train_gpu_proof.py

Settings are environment variables. Each has a default; the script prints
the value that it uses and stops when a path does not exist.

    REPO         the git checkout (default: the checkout of the current directory)
    BASE_DIR     the bf16 head on the local disk (default: see ``head_dir``)
    GPU          the index of the one GPU to use (default 0). It must be free.
    OUT          a new scratch directory (default /tmp/sn120-train-proof-<time>)
    LONG_TOKENS  length of the long row in tokens (default 36864, the training cap)
    RUNS         which runs to do (default "3 1 2 4 5 6": the long row at the new commit first)
    TIMEOUT_S    time limit of one run in seconds (default 900)
    NEW, OLD     the two commits
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file

from sn120_v2.gpu import plan, train_job
from sn120_v2.params import Recipe

NEW = os.environ.get("NEW", "efeaa435957d582a34ee54fbe229ec54b3e67ca2")  # with gradient checkpointing (PR 192)
OLD = os.environ.get("OLD", "5bc36ed86aa180faf07155bf987c4a1f8b3a42f5")  # without it
DIGESTS = {
    "efeaa435957d582a34ee54fbe229ec54b3e67ca2": "e4ae285f4b60121b0737d04d193fec92291df3189d739e07138df6b4ceaf697d",
    "5bc36ed86aa180faf07155bf987c4a1f8b3a42f5": "843a15349669fdc4a586c8351d6c16205fd1ccb74e779c457eca96161d5a413e",
}
GENESIS_BASE_HASH = "87de8304e9d33dde39b81a41fdbbf306426ea6fd993b7d85332e6a48e8a4bbae"
SHORT_TOKENS = 1000
# number -> (name, commit, batch, recipe changes). "repeat" must give the bytes of run 1; "recipe" checks the norm rule on a GPU.
RUNS = {
    1: ("short-new", NEW, "short", {}),
    2: ("short-old", OLD, "short", {}),
    3: ("long-new", NEW, "long", {}),
    4: ("long-old", OLD, "long", {}),
    5: ("short-new-repeat", NEW, "short", {}),
    6: ("short-new-lr1e-4-steps2", NEW, "short", {"lora_lr": 1e-4, "steps": 2}),
}


def say(text: str) -> None:
    print(text, flush=True)


def stop(text: str) -> None:
    say(f"STOP: {text}")
    say("RESULT: FAIL. The proof did not run; see the STOP line.")
    sys.exit(2)


def head_dir() -> Path:
    """The bf16 head. ``BASE_DIR`` wins; else the two places that SMOKE.md names."""
    places = [os.environ.get("BASE_DIR")]
    if os.environ.get("SN120_WORK_DIR"):
        places.append(str(Path(os.environ["SN120_WORK_DIR"]) / "cache" / GENESIS_BASE_HASH))
    if os.environ.get("SMOKE"):
        places.append(str(Path(os.environ["SMOKE"]) / "genesis" / "base"))
    for place in places:
        if place and (Path(place) / "config.json").is_file():
            return Path(place)
    stop(f"no head found in {[p for p in places if p]}. Set BASE_DIR to the directory of the bf16 head (it holds config.json and the safetensors files).")


def text_of(tokenizer, n_tokens: int, salt: int) -> str:
    """Lines of sums, about ``n_tokens`` tokens long. The same call gives the same text.

    The job trains only on tokens that the head could sample (its top 20 and
    its top 95%). A sum is text that a math head gives a high probability,
    so these rows train; rows of random words would give no gradient.
    """
    lines = "".join(f"{(i * 7919 + salt * 131) % 997} + {i} = {(i * 7919 + salt * 131) % 997 + i}\n" for i in range(n_tokens // 4 + 8))
    return tokenizer.decode(tokenizer.encode(lines, add_special_tokens=False)[:n_tokens])


def batch_rows(tokenizer, kind: str, long_tokens: int) -> list[dict]:
    """The rows of a batch, in the form the trainer gives to the job. No miner data."""
    rows = []
    if kind == "long":  # one row at the given length: a long prompt and an output that fills the rest
        out_tokens = min(32000, long_tokens * 7 // 8)
        rows.append({"prompt": text_of(tokenizer, max(8, long_tokens - out_tokens - 16), 1), "output": text_of(tokenizer, out_tokens, 2), "reward": 1.0, "advantage": 1.0})
    for i in range(3 if kind == "long" else 8):
        sign = 1.0 if i % 2 else -1.0
        rows.append({"prompt": f"Add the numbers of list {i // 4}.", "output": text_of(tokenizer, SHORT_TOKENS - 50 * i, 10 + i), "reward": sign, "advantage": sign})
    return rows


def prepare(job_dir: Path, kind: str, changes: dict) -> None:
    """Write ``job.json`` and ``facts.json``. Runs with the code of one commit on PYTHONPATH."""
    base_dir = Path(os.environ["PROOF_BASE_DIR"])
    recipe = replace(Recipe(), **changes, **json.loads(os.environ.get("PROOF_RECIPE", "{}")))
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    rows = batch_rows(tokenizer, kind, int(os.environ["PROOF_LONG_TOKENS"]))
    encoded, dropped = train_job.encode(tokenizer, rows, recipe)
    job = {"base_dir": str(base_dir), "base_hash": "proof", "dtype": os.environ.get("PROOF_DTYPE", "bfloat16"), "recipe": recipe.to_dict(), "seed": 120, "rows": rows}
    (job_dir / "lora").mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
    facts = {"code_digest": plan.code_digest(), "code": str(Path(train_job.__file__).parent), "rows": len(encoded), "dropped": dropped}
    facts["longest_row_tokens"] = max(len(row["prompt_ids"]) + len(row["tokens"]) for row in encoded)
    (job_dir / "facts.json").write_text(json.dumps(facts), encoding="utf-8")


def cpu_job(job_dir: Path) -> None:
    """Test of this script on a machine with no GPU only (PROOF_CPU=1). The real job refuses the CPU."""
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    (job_dir / "report.json").write_text(json.dumps(train_job.run(job, job_dir / "lora", devices=["cpu"])), encoding="utf-8")


def gpu_used_mib(gpu: int) -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)], capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def last_line(path: Path) -> str:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except OSError:
        return "(no log)"
    return lines[-1][:300] if lines else "(empty log)"


def source_of(repo: Path, out: Path, commit: str) -> Path:
    """A second working copy of one commit under ``out``. The checkout of the box does not move."""
    tree = out / f"src-{commit[:8]}"
    if not tree.exists():
        if subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True).returncode != 0:
            subprocess.run(["git", "-C", str(repo), "fetch", "origin", commit], check=False)
        done = subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(tree), commit], capture_output=True, text=True)
        if done.returncode != 0:
            stop(f"git cannot make a working copy of {commit}: {done.stderr.strip()[-300:]}")
    return tree / "v2"


def one_run(name: str, commit: str, kind: str, changes: dict, repo: Path, out: Path, base_dir: Path, gpu: int, cpu: bool) -> dict:
    job_dir = out / name
    job_dir.mkdir(parents=True)
    # The environment of a job of the trainer: one GPU, and a kernel cache of its own (here under OUT).
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TRITON_CACHE_DIR=str(out / "triton"))
    (out / "triton").mkdir(exist_ok=True)
    env.update(PYTHONPATH=str(source_of(repo, out, commit)), PROOF_BASE_DIR=str(base_dir), PROOF_LONG_TOKENS=os.environ.get("LONG_TOKENS", "36864"))
    me = str(Path(__file__).resolve())
    ready = subprocess.run([sys.executable, me, "prepare", str(job_dir), kind, json.dumps(changes)], env=env, capture_output=True, text=True)
    if ready.returncode != 0:
        stop(f"{name}: the job file was not made: {ready.stderr.strip()[-600:]}")
    facts = json.loads((job_dir / "facts.json").read_text(encoding="utf-8"))
    if commit in DIGESTS and facts["code_digest"] != DIGESTS[commit]:
        stop(f"{name}: the code under {facts['code']} has digest {facts['code_digest']}, not the digest of {commit[:8]}")
    say(f"run {name}: commit {commit[:8]}, code digest {facts['code_digest'][:12]}, {facts['rows']} rows, longest row {facts['longest_row_tokens']} tokens. Started.")
    argv = [sys.executable, me, "cpu-job", str(job_dir)] if cpu else [sys.executable, "-m", "sn120_v2.gpu.train_job", str(job_dir)]
    peak, done = [gpu_used_mib(gpu) or 0], threading.Event()

    def sample() -> None:
        while not done.wait(0.5):
            peak[0] = max(peak[0], gpu_used_mib(gpu) or 0)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started, timed_out = time.monotonic(), False
    with (job_dir / "job.log").open("wb") as log:
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(float(os.environ.get("TIMEOUT_S", "900")))
        except subprocess.TimeoutExpired:
            proc.kill()
            code, timed_out = proc.wait(), True
    done.set()
    sampler.join()
    result = {"run": name, "commit": commit[:8], "longest_row_tokens": facts["longest_row_tokens"], "wall_s": round(time.monotonic() - started, 1), "peak_gpu_mib": peak[0] or None}
    try:
        report = json.loads((job_dir / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {}
    if code == 0 and report.get("status") == "ok":
        result.update(finished=True, **{key: report.get(key) for key in ("load_s", "step_s", "step_norm", "step_norm_before_cap", "forward_tokens", "rows", "truncated_rows", "loss")})
        result["lora_sha256"] = hashlib.sha256((job_dir / "lora" / "adapter_model.safetensors").read_bytes()).hexdigest()
    else:
        reason = "ran past TIMEOUT_S" if timed_out else report.get("reason") or last_line(job_dir / "job.log")
        result.update(finished=False, exit_code=code, last_line=reason)
    say("RUN " + json.dumps(result, sort_keys=True))
    return result


def differing_share(out: Path, one: str, two: str) -> float:
    """Share of the lora_B numbers that differ between two runs. A first step moves every number by +lr or -lr."""
    a, b = (load_file(str(out / name / "lora" / "adapter_model.safetensors")) for name in (one, two))
    keys = [key for key in a if ".lora_B." in key]
    return sum(int((a[key] != b[key]).sum()) for key in keys) / sum(a[key].numel() for key in keys)


def verdict(results: dict[str, dict], out: Path) -> None:
    good = lambda name: name in results and results[name]["finished"]  # noqa: E731
    faults, notes = [], []
    for name in ("short-new", "long-new", "short-new-repeat", "short-new-lr1e-4-steps2"):
        if name in results and not good(name):
            faults.append(f"{name} did not finish at the new commit: {results[name]['last_line']}")
        elif name in results and not results[name]["step_norm"] > 0:
            faults.append(f"{name} gave a norm of zero: no token trained")
    if good("short-new") and good("short-old"):
        new, old = results["short-new"], results["short-old"]
        if not (new["step_norm"] > 0 and old["step_norm"] > 0):
            faults.append("the short batch gave a norm of zero: no token trained, so the run proves nothing")
        ratio = new["step_norm"] / (old["step_norm"] or 1.0)
        notes.append(f"short batch, new against old: step_norm {new['step_norm']:.6f} against {old['step_norm']:.6f} (ratio {ratio:.6f}); step_s {new['step_s']:.1f} against {old['step_s']:.1f} (ratio {new['step_s'] / old['step_s']:.2f}); peak {new['peak_gpu_mib']} against {old['peak_gpu_mib']} MiB; share of lora_B numbers that differ {differing_share(out, 'short-new', 'short-old'):.6f}")
        if abs(ratio - 1) > 1e-3:
            faults.append(f"the norm of the short batch differs between the commits by more than 0.1%: ratio {ratio:.6f}")
    elif "short-old" in results and not good("short-old"):
        faults.append(f"short-old did not finish, so there is no comparison: {results['short-old']['last_line']}")
    if good("short-new") and good("short-new-repeat"):
        same = results["short-new"]["lora_sha256"] == results["short-new-repeat"]["lora_sha256"]
        notes.append("a second run of the short batch at the new commit gives " + ("the same LoRA bytes" if same else f"other LoRA bytes (share of lora_B numbers that differ {differing_share(out, 'short-new', 'short-new-repeat'):.6f}); the old commit was not run two times, so this does not say which commit causes it"))
    if good("short-new") and good("short-new-lr1e-4-steps2"):
        notes.append(f"norm with lora_lr 1e-4 and steps 2 against the default recipe: {results['short-new-lr1e-4-steps2']['step_norm'] / (results['short-new']['step_norm'] or 1.0):.3f} times (the CPU test head gave 19.93 times)")
    if good("long-new") and "long-old" in results:
        new, old = results["long-new"], results["long-old"]
        if old["finished"]:
            notes.append(f"the long row ({new['longest_row_tokens']} tokens) ALSO finished at the old commit: this GPU does not need the change for this length. Peak {new['peak_gpu_mib']} against {old['peak_gpu_mib']} MiB; step_s {new['step_s']:.1f} against {old['step_s']:.1f}; step_norm {new['step_norm']:.6f} against {old['step_norm']:.6f}")
        else:
            notes.append(f"the long row ({new['longest_row_tokens']} tokens) finished at the new commit (peak {new['peak_gpu_mib']} MiB, step_s {new['step_s']:.1f}) and FAILED at the old commit: {old['last_line']}")
    for note in notes:
        say("NOTE: " + note)
    for fault in faults:
        say("FAULT: " + fault)
    needed = [name for name in ("short-new", "short-old", "long-new") if name not in results]
    if faults:
        say("RESULT: FAIL. Do not take efeaa435 on the strength of this run.")
    elif needed:
        say(f"RESULT: NOT COMPLETE. No fault in the runs that were done, but these runs are missing: {', '.join(needed)}.")
    else:
        say("RESULT: PASS. The training job of efeaa435 finished the long-row batch and the short batch on a GPU, and the norm of the short batch is the norm of 5bc36ed8.")


def main() -> None:
    repo = Path(os.environ.get("REPO") or subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip() or ".")
    if not (repo / "v2" / "sn120_v2").is_dir():
        stop(f"REPO={repo} is not the checkout (no v2/sn120_v2 in it). Set REPO to the directory of the git checkout.")
    base_dir = head_dir()
    gpu, cpu = int(os.environ.get("GPU", "0")), os.environ.get("PROOF_CPU") == "1"
    out = Path(os.environ.get("OUT") or f"/tmp/sn120-train-proof-{int(time.time())}")
    out.mkdir(parents=True, exist_ok=False)
    say(f"REPO={repo}  BASE_DIR={base_dir}  GPU={gpu}  OUT={out}  python={sys.executable}")
    say(f"torch {torch.__version__}, transformers {transformers.__version__}")
    if not cpu:
        used = gpu_used_mib(gpu)
        if used is None:
            stop(f"nvidia-smi gives no memory figure for GPU {gpu}")
        names = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader", "-i", str(gpu)], capture_output=True, text=True).stdout.strip()
        say(f"GPU: {names}")
        if used > 1000 and os.environ.get("FORCE") != "1":
            stop(f"GPU {gpu} has {used} MiB in use. Stop what runs on it, or choose a free GPU with GPU=<index>. FORCE=1 runs all the same.")
    results = {}
    for number in (int(n) for n in os.environ.get("RUNS", "3 1 2 4 5 6").split()):
        name, commit, kind, changes = RUNS[number]
        results[name] = one_run(name, commit, kind, changes, repo, out, base_dir, gpu, cpu)
    (out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True), encoding="utf-8")
    for commit in {RUNS[n][1] for n in RUNS}:
        subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(out / f"src-{commit[:8]}")], capture_output=True)
    say(f"All RUN lines are also in {out}/results.json; each run keeps its job.log under {out}/<run>/.")
    verdict(results, out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        prepare(Path(sys.argv[2]), sys.argv[3], json.loads(sys.argv[4]))
    elif len(sys.argv) > 1 and sys.argv[1] == "cpu-job":
        cpu_job(Path(sys.argv[2]))
    else:
        main()
