#!/usr/bin/env python3
"""Reference local Generator pool search using vLLM and the published king.

This is a deliberately simple offline reference, not the validator's scorer.
It samples a reproducible subset of the pinned prior pool, requests eight
independent king rollouts per task, grades with the production math grader,
and ranks IDs by closeness of the measured pass rate to 0.5. The report and
ids-file output are separate. No submission is signed or uploaded.

Prerequisites: the v2 ``gpu``, ``envs`` and ``generator`` extras (including
vLLM 0.32.0, math-verify 0.9.0 and pyarrow). Start vLLM yourself, or let this
script start one local BF16 server. The server is always bound to 127.0.0.1.
The king base is a folded BF16 head, with no LoRA and no FP8 weights/KV.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import random
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from sn120_v2.canonical import canonical_json
from sn120_v2.errors import ConfigError, InfraError, V2Error
from sn120_v2.generator.constants import KING, POOL_CACHE_ENV, SAMPLING
from sn120_v2.generator.pool import PriorPool
from sn120_v2.types import PASS

PUBLIC_BASE = "https://affine.io/public/read_links.json"
PUBLIC_HEAD = "https://affine.io/public/head.json"
PUBLIC_MINING = "https://affine.io/mining.json"
END_OF_TURN = "<|im_end|>"
ROLLOUT_FORMAT = "affine-generator-rollouts-v1"
ROLLOUTS_PER_TASK = 8


def rollout_metadata(
    ids: Sequence[str], king: str, model: str, seed: int, max_tokens: int = SAMPLING.max_tokens,
) -> dict[str, Any]:
    """Everything a resumed journal must match; it never includes a local URL."""
    return {
        "king": king,
        "model": model,
        "ids": list(ids),
        "pool_sample_seed": seed,
        "rollout_seed_base": seed,
        "rollouts_per_task": ROLLOUTS_PER_TASK,
        "max_tokens": max_tokens,
        "temperature": SAMPLING.temperature,
        "top_p": SAMPLING.top_p,
        "top_k": SAMPLING.top_k,
        "thinking": SAMPLING.thinking,
        "skip_special_tokens": False,
    }


def journal_header(path: Path) -> Mapping[str, Any]:
    """The run identity of a journal, read before deciding whether a server is needed."""
    if not path.exists():
        return {}
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024 * 1024:
        raise ConfigError("rollout journal must be a regular file no larger than 64 GiB")
    try:
        with path.open(encoding="utf-8") as stream:
            row = json.loads(stream.readline(1024 * 1024 + 1))
        metadata = row["metadata"]
        if row.get("format") != ROLLOUT_FORMAT or not isinstance(metadata, dict) or type(metadata.get("king")) is not str:
            raise ValueError("invalid journal header")
        return metadata
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise ConfigError(f"cannot resume rollout journal {path}: {type(exc).__name__}: {exc}") from exc


def resume_rollouts(path: Path, metadata: Mapping[str, Any]) -> dict[tuple[str, int], tuple[str, bool]]:
    """Read a complete, compatible JSONL journal; refuse truncation or mixed runs."""
    outputs: dict[tuple[str, int], tuple[str, bool]] = {}
    if not path.exists():
        return outputs
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or json.loads(lines[0]) != {"format": ROLLOUT_FORMAT, "metadata": dict(metadata)}:
            raise ConfigError("rollout journal header does not match this king, model, pool sample, seed and sampling settings")
        for line in lines[1:]:
            row = json.loads(line)
            if set(row) != {"id", "sample", "seed", "output", "capped", "finish_reason"}:
                raise ValueError("unexpected rollout fields")
            task_id, sample = row["id"], row["sample"]
            index = metadata["ids"].index(task_id) if isinstance(task_id, str) and task_id in metadata["ids"] else -1
            if index < 0 or type(sample) is not int or not 0 <= sample < ROLLOUTS_PER_TASK:
                raise ValueError("rollout is outside the requested pool sample")
            key = (task_id, sample)
            expected_seed = int(metadata["rollout_seed_base"]) + index * ROLLOUTS_PER_TASK + sample
            if key in outputs or row["seed"] != expected_seed or type(row["seed"]) is not int:
                raise ValueError("duplicate rollout or wrong deterministic seed")
            if type(row["output"]) is not str or type(row["capped"]) is not bool or row["finish_reason"] not in ("stop", "length"):
                raise ValueError("rollout output or finish metadata has the wrong type")
            if row["capped"] != (row["finish_reason"] == "length"):
                raise ValueError("rollout cap and finish metadata disagree")
            outputs[key] = (row["output"], row["capped"])
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise ConfigError(f"cannot resume rollout journal {path}: {type(exc).__name__}: {exc}") from exc
    print(f"resumed {len(outputs)} saved rollouts from {path}", flush=True)
    return outputs


class RolloutJournal:
    """Append and fsync every raw completed rollout before grading can fail."""

    def __init__(self, path: Path, metadata: Mapping[str, Any], outputs: Mapping[tuple[str, int], tuple[str, bool]], resume: bool = False) -> None:
        self.path, self._outputs = path, outputs
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        if existed and (not path.is_file() or path.stat().st_size > 64 * 1024 * 1024 * 1024):
            raise ConfigError("rollout journal must be a regular file no larger than 64 GiB")
        self._stream = path.open("a" if resume else "x", encoding="utf-8")
        if not resume or not existed:
            self.write({"format": ROLLOUT_FORMAT, "metadata": dict(metadata)})

    def write(self, row: Mapping[str, Any]) -> None:
        self._stream.write(canonical_json(row).decode("utf-8") + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def save(self, key: tuple[str, int], value: tuple[str, bool], index: int, seed: int) -> None:
        text, capped = value
        self.write({
            "id": key[0], "sample": key[1], "seed": seed + index * ROLLOUTS_PER_TASK + key[1],
            "output": text, "capped": capped, "finish_reason": "length" if capped else "stop",
        })
        self._outputs[key] = value

    def close(self) -> None:
        self._stream.close()


def read_json(source: str) -> Any:
    """Read a small public JSON file (or explicit local test fixture)."""
    if "://" not in source:
        return json.loads(Path(source).read_text())
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ConfigError("public king metadata must use HTTPS without credentials")
    request = urllib.request.Request(source, headers={"User-Agent": "affine-generator-search/1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as reply:  # noqa: S310 - public king metadata
            raw = reply.read(16 * 1024 * 1024 + 1)
    except (OSError, ValueError) as exc:
        raise InfraError(f"cannot read public king metadata {source}: {type(exc).__name__}") from exc
    if len(raw) > 16 * 1024 * 1024:
        raise InfraError("public king metadata is unexpectedly large")
    return json.loads(raw)


def sample_ids(registry: PriorPool, sample_size: int, seed: int) -> tuple[str, ...]:
    """Uniformly sample the registry without minting or altering a prior ID."""
    if not registry._locations:
        raise ConfigError("build the pool ID index first: generator_miner.py pool --index --cache DIR")
    if type(sample_size) is not int or not 1 <= sample_size <= len(registry._locations):
        raise ConfigError("sample size must be between 1 and the number of published IDs")
    return tuple(sorted(random.Random(seed).sample(sorted(registry._locations), sample_size)))


def request_body(model: str, messages: list[dict[str, str]], seed: int, max_tokens: int = SAMPLING.max_tokens) -> dict[str, Any]:
    """The same OpenAI-compatible body as the v2 production evaluator."""
    return {
        "model": model,
        "messages": messages,
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": SAMPLING.temperature,
        "top_p": SAMPLING.top_p,
        "top_k": SAMPLING.top_k,
        "skip_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": SAMPLING.thinking},
    }


def post_chat(url: str, body: Mapping[str, Any], timeout: float) -> tuple[str, bool]:
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=canonical_json(body),
        headers={"Content-Type": "application/json", "User-Agent": "affine-generator-search/1"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:  # noqa: S310 - explicit local vLLM endpoint
            response = json.load(reply)
        choice = response["choices"][0]
        text = choice["message"]["content"]
        finish = choice["finish_reason"]
        if not isinstance(text, str) or finish not in ("stop", "length"):
            raise ValueError("unexpected vLLM reply")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InfraError(f"vLLM generation failed: {type(exc).__name__}") from exc
    if finish == "stop" and text.endswith(END_OF_TURN):
        text = text[: -len(END_OF_TURN)]
    return text, finish == "length"


def wait_http(url: str, process: subprocess.Popen[bytes] | None, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise InfraError(f"vLLM exited with status {process.returncode}; inspect its local server log")
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as reply:  # noqa: S310
                if reply.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise InfraError("vLLM did not become ready before the startup timeout")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def local_server(
    model_dir: Path | None,
    url: str,
    gpu: int,
    max_model_len: int,
    startup_timeout: float,
    extra_args: Sequence[str],
) -> Iterator[str]:
    """Optionally run a BF16-only vLLM server; never expose it off localhost."""
    if model_dir is None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ConfigError("the reference runner permits only a localhost vLLM endpoint")
        yield url
        return
    if url != "auto":
        raise ConfigError("when --model-dir is supplied, use --url auto or leave it unset")
    port = free_port()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_dir),
        "--served-model-name",
        "king",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        KING.dtype.lower(),
        "--kv-cache-dtype",
        KING.kv_cache_dtype.lower(),
        "--max-model-len",
        str(max_model_len),
        "--enable-prefix-caching",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        *extra_args,
    ]
    process = subprocess.Popen(command, env=env)  # noqa: S603 - fixed vLLM invocation, user model directory
    address = f"http://127.0.0.1:{port}/v1"
    try:
        wait_http(address, process, startup_timeout)
        yield address
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def fetch_king(work: Path, mining_source: str, links_source: str) -> tuple[str, Path]:
    """Reuse v2's hash-verified fetch of the published folded base, never a substitute model."""
    from sn120_v2.replicate import Contract, fetch

    contract = Contract.load(mining_source)
    links = read_json(links_source)
    if not isinstance(links, dict):
        raise ConfigError("read_links.json must be an object")
    if links.get("head_version_id") != contract.head.version_id:
        raise ConfigError("read_links.json and head.json name different kings")
    if not contract.manifest.lora_files:
        # An empty lora_files map is expected for a folded base. A nonempty map
        # is still checked by HeadManifest, and fetch downloads only the folded base.
        pass
    if any("fp8" in str(name).lower() for name in contract.manifest.base_files):
        raise ConfigError("the published king base appears to contain FP8 weights; use the BF16 folded head")
    if KING.fp8:
        raise ConfigError("Generator king must never use FP8 weights or KV cache")
    result = fetch(contract, work, links_source=links_source)
    base = Path(result["dir"])
    if not base.is_dir():
        raise InfraError("the verified king base was not materialized")
    print(f"king: {contract.head.version_id}", flush=True)
    print(f"verified folded BF16 base: {base}", flush=True)
    return contract.head.version_id, base


def grade_rollouts(
    registry: PriorPool,
    ids: Sequence[str],
    outputs: Mapping[tuple[str, int], tuple[str, bool]],
    workers: int,
    timeout_s: float,
    grader_factory: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Use the official grader process pool; never discard a failed rollout."""
    if grader_factory is None:
        from sn120_v2.envs.math_grader import GradePool

        grader_factory = GradePool
    grader = grader_factory(workers=workers, timeout_s=timeout_s)
    try:
        tasks = [registry.tasks((task_id,))[0] for task_id in ids]
        jobs = [
            (task, outputs[(task.task_id, sample)][0], outputs[(task.task_id, sample)][1])
            for task in tasks
            for sample in range(8)
        ]
        grades = grader.grade_many(jobs)
    finally:
        grader.close()
    rows = []
    for index, task_id in enumerate(ids):
        mine = grades[index * 8 : (index + 1) * 8]
        passes = sum(grade.code == PASS for grade in mine)
        rate = passes / len(mine)
        rows.append({
            "id": task_id,
            "n": len(mine),
            "s": passes,
            "local_pass_rate": rate,
            "distance_to_half": abs(rate - 0.5),
            "outcomes": [grade.code for grade in mine],
            "cap_hits": sum(grade.cap_hit for grade in mine),
            "grader_errors": sum(grade.code == "E" for grade in mine),
        })
    return sorted(rows, key=lambda row: (row["distance_to_half"], -row["n"], row["id"]))


def score(
    registry: PriorPool,
    ids: Sequence[str],
    url: str,
    model: str,
    seed: int,
    workers: int,
    http_workers: int,
    timeout_s: float,
    grade_timeout_s: float,
    generate: Callable[[str, Mapping[str, Any], float], tuple[str, bool]] = post_chat,
    grader_factory: Callable[..., Any] | None = None,
    save_rollouts: Path | None = None,
    resume: bool = False,
    king: str = "",
) -> list[dict[str, Any]]:
    """Generate eight deterministic-seed draws per task, then rank locally.

    A rollout journal keeps every raw completion on disk. ``resume`` loads only
    a journal with the exact same generation metadata and skips completed draws.
    """
    metadata = rollout_metadata(ids, king, model, seed)
    if resume and save_rollouts is None:
        raise ConfigError("--resume needs --save-rollouts")
    outputs = resume_rollouts(save_rollouts, metadata) if resume and save_rollouts is not None else {}
    journal = RolloutJournal(save_rollouts, metadata, outputs, resume=resume) if save_rollouts is not None else None
    try:
        def one(task_id: str, index: int, sample: int) -> tuple[tuple[str, int], tuple[str, bool]]:
            messages = registry.messages(task_id)
            body = request_body(model, messages, seed=seed + index * ROLLOUTS_PER_TASK + sample)
            return (task_id, sample), generate(url, body, timeout_s)

        pending = [(index, task_id, sample) for index, task_id in enumerate(ids) for sample in range(ROLLOUTS_PER_TASK) if (task_id, sample) not in outputs]
        with concurrent.futures.ThreadPoolExecutor(max_workers=http_workers) as executor:
            futures = {executor.submit(one, task_id, index, sample): (index, task_id) for index, task_id, sample in pending}
            try:
                for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    key, value = future.result()
                    index, _ = futures[future]
                    if journal is not None:
                        journal.save(key, value, index, seed)
                    else:
                        outputs[key] = value
                    if number % ROLLOUTS_PER_TASK == 0:
                        print(f"generated {number}/{len(futures)} rollouts", flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
    finally:
        if journal is not None:
            journal.close()

    # Keep all grades, including grader failures and cap hits, in the denominator.
    return grade_rollouts(registry, ids, outputs, workers, grade_timeout_s, grader_factory=grader_factory)


def write_outputs(report: Path, ids_file: Path, rows: Sequence[dict[str, Any]], count: int, king: str, settings: Mapping[str, Any]) -> None:
    """Write a JSONL report and a separate ids-only file."""
    if not 1 <= count <= 32:
        raise ConfigError("the ids file must contain between 1 and 32 IDs")
    if len(rows) < count:
        raise ConfigError("not enough sampled tasks for the requested ids file")
    selected = rows[:count]
    report.parent.mkdir(parents=True, exist_ok=True)
    ids_file.parent.mkdir(parents=True, exist_ok=True)
    report_tmp = report.with_name(report.name + ".tmp")
    ids_tmp = ids_file.with_name(ids_file.name + ".tmp")
    with report_tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            enriched = {"king": king, "settings": dict(settings)} | dict(row)
            stream.write(canonical_json(enriched).decode() + "\n")
    ids_tmp.write_text("".join(row["id"] + "\n" for row in selected), encoding="utf-8")
    report_tmp.replace(report)
    ids_tmp.replace(ids_file)
    print(f"wrote {ids_file}: {count} IDs ranked by local pass rate closest to 0.5")
    print(f"wrote {report}: ranked JSONL, one task per line")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path(os.environ.get(POOL_CACHE_ENV, "~/.cache/affine/generator")).expanduser())
    p.add_argument("--work", type=Path, default=Path("./generator-search-work"), help="verified king cache and work directory")
    p.add_argument("--model-dir", type=Path, help="already fetched folded king base; skips public download")
    p.add_argument("--king", default="", help="record a head version for --model-dir; no model substitution")
    p.add_argument("--mining-json", default=PUBLIC_MINING)
    p.add_argument("--read-links", default=PUBLIC_BASE)
    p.add_argument("--url", default="auto", help="existing localhost /v1 endpoint, or auto with --model-dir")
    p.add_argument("--model", default="king", help="served model name at the vLLM endpoint")
    p.add_argument("--sample-size", type=int, default=64)
    p.add_argument("--count", type=int, default=32, help="number of IDs in the output file (1..32)")
    p.add_argument("--seed", type=int, default=1201, help="pool sample seed; rollout seeds are seed + sample index")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--http-workers", type=int, default=16)
    p.add_argument("--grade-workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--request-timeout", type=float, default=3600.0)
    p.add_argument("--grade-timeout", type=float, default=20.0)
    p.add_argument("--startup-timeout", type=float, default=1800.0)
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--vllm-arg", action="append", default=[], help="extra argument passed to the local vLLM server; repeat as needed")
    p.add_argument("--report", type=Path, default=Path("generator-search.jsonl"))
    p.add_argument("--ids-file", type=Path, default=Path("generator-task-ids.txt"))
    p.add_argument("--save-rollouts", type=Path, help="durable raw-rollout JSONL, fsynced after every completion")
    p.add_argument("--resume", action="store_true", help="reuse completed draws from --save-rollouts and grade without regenerating")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if not 1 <= args.count <= 32:
            raise ConfigError("--count must be 1..32")
        if args.sample_size < args.count:
            raise ConfigError("--sample-size must be at least --count")
        if not 1 <= args.http_workers <= 256 or not 1 <= args.grade_workers <= 256:
            raise ConfigError("worker counts must be 1..256")
        if args.resume and args.save_rollouts is None:
            raise ConfigError("--resume needs --save-rollouts")
        registry = PriorPool(args.cache, build_index=False)
        ids = sample_ids(registry, args.sample_size, args.seed)
        saved = journal_header(args.save_rollouts) if args.resume and args.save_rollouts is not None else {}
        king = saved.get("king", "") if not args.king else args.king
        if saved and saved.get("king", "") != king:
            raise ConfigError("--king does not match the rollout journal")
        model_dir = args.model_dir
        metadata = rollout_metadata(ids, king, args.model, args.seed)
        outputs = resume_rollouts(args.save_rollouts, metadata) if args.resume else {}
        fully_saved = len(outputs) == len(ids) * ROLLOUTS_PER_TASK
        if fully_saved:
            print("all rollouts are saved; grading without fetching or starting vLLM", flush=True)
            rows = grade_rollouts(registry, ids, outputs, args.grade_workers, args.grade_timeout)
        else:
            if model_dir is None:
                king, model_dir = fetch_king(args.work, args.mining_json, args.read_links)
                if saved and king != saved["king"]:
                    raise ConfigError("the published king changed; use the original --model-dir or a new rollout journal")
            elif not king:
                raise ConfigError("--model-dir requires --king HEAD_VERSION so the report names the measured king")
        settings = {
            "temperature": SAMPLING.temperature,
            "top_p": SAMPLING.top_p,
            "top_k": SAMPLING.top_k,
            "max_tokens": SAMPLING.max_tokens,
            "thinking": SAMPLING.thinking,
            "rollouts_per_task": 8,
            "pool_sample_seed": args.seed,
            "dtype": KING.dtype,
            "kv_cache_dtype": KING.kv_cache_dtype,
            "note": "Local estimates only. The validator samples the king frozen at the epoch deadline with commit-reveal seeds; local acceptance is not guaranteed.",
        }
        if not fully_saved:
            with local_server(model_dir, args.url, args.gpu, args.max_model_len, args.startup_timeout, args.vllm_arg) as url:
                rows = score(
                    registry,
                    ids,
                    url,
                    args.model,
                    args.seed,
                    args.grade_workers,
                    args.http_workers,
                    args.request_timeout,
                    args.grade_timeout,
                    save_rollouts=args.save_rollouts,
                    resume=args.resume,
                    king=king,
                )
        write_outputs(args.report, args.ids_file, rows, args.count, king, settings)
        return 0
    except (V2Error, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"search failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
