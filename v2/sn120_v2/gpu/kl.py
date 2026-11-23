"""Full-vocabulary KL(new || head) on an entry's own scored sample-0 rollouts.

The box runs this after vLLM has released its GPUs and before it seals the
result. Prompts are rendered with the head chat template; completions are
re-tokenized with the head tokenizer because the OpenAI serving reply does
not retain token ids. Both models run the trainer's teacher-forced forward,
with the output layer and fp32 log-softmax in bounded position chunks.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from ..errors import InfraError
from ..rollouts import capture_file
from ..types import ERROR, NOT_RUN, NULL_MODEL, SCORED
from .batch import CHAT_TEMPLATE_KWARGS, END_OF_TURN
from .plan import BoxConfig, log_phase
from .procs import Child
from .resident import resident_head
from .trainer import job_env

LOGIT_CHUNK = 2048
ROWS_PER_LOAD = 16
TOKENIZATION = "head chat-template(prompt) + head-tokenizer(completion_text); <|im_end|> restored for uncapped replies"


def selected_rollouts(capture: Path, outcomes: Sequence[str], round_seed: str, lora_hash: str, count: int) -> list[dict]:
    """A deterministic, ordered subsample of scored, non-error sample-0 rows."""
    if not outcomes:
        raise InfraError("KL needs the entry's sample-0 eval outcomes")
    scored = len(outcomes[0].rstrip(NOT_RUN))
    rows: dict[int, dict] = {}
    try:
        with capture.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                pos = row.get("task_pos")
                if type(pos) is not int or not 0 <= pos < scored or row.get("sample_index") != 0:
                    continue
                if row.get("outcome") in (ERROR, NOT_RUN) or row["outcome"] != outcomes[0][pos]:
                    continue
                if isinstance(row.get("prompt"), list) and isinstance(row.get("completion_text"), str):
                    rows[pos] = row
    except OSError as exc:
        raise InfraError(f"KL cannot read rollout capture: {exc}") from exc
    eligible = [rows[pos] for pos in sorted(rows)]
    if not eligible:
        raise InfraError("KL found no captured, scored sample-0 rollouts")
    seed = hashlib.sha256((round_seed + lora_hash + "kl").encode()).digest()
    chosen = random.Random(int.from_bytes(seed, "big")).sample(eligible, min(count, len(eligible)))
    return sorted(chosen, key=lambda row: row["task_pos"])


def tokenize_rows(tokenizer: Any, rows: Sequence[dict], max_tokens: int = 0) -> list[dict]:
    eos = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    if type(eos) is not int or eos < 0:
        raise InfraError(f"the head tokenizer has no {END_OF_TURN} token")
    encoded = []
    for row in rows:
        text = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False, **CHAT_TEMPLATE_KWARGS)
        prompt = tokenizer.encode(text, add_special_tokens=False)
        completion = tokenizer.encode(row["completion_text"], add_special_tokens=False)
        if row.get("outcome") != "C":
            completion = [*completion, eos]
        if max_tokens > 0:
            # A token cap is a strict prefix: never manufacture an end token at the cut.
            completion = completion[:max_tokens]
        if not completion:
            raise InfraError(f"KL rollout at task position {row['task_pos']} has no completion tokens")
        encoded.append({"prompt_ids": prompt, "completion_ids": completion})
    return encoded


def kl_from_log_probs(new_log_probs, head_log_probs):
    """KL(new || head) at each position, with fp32 probabilities and logs."""
    return (new_log_probs.exp() * (new_log_probs - head_log_probs)).sum(dim=-1)


def _completion_hidden(model: Any, row: dict, device: str):
    """Run one model trunk exactly once for a teacher-forced sequence."""
    import torch

    ids = row["prompt_ids"] + row["completion_ids"]
    start = len(row["prompt_ids"]) - 1
    return model.base_model(input_ids=torch.tensor([ids], device=device), use_cache=False).last_hidden_state[0, start:-1]


def _completion_log_probs(model: Any, hidden: Any, chunk: int):
    """Chunk only the output projection and fp32 log-softmax, never the trunk."""
    import torch

    output = model.get_output_embeddings()
    for offset in range(0, hidden.shape[0], chunk):
        logits = output(hidden[offset:offset + chunk])
        yield torch.log_softmax(logits.float(), dim=-1)


def sequence_kl(new: Any, head: Any, row: dict, device: str = "cpu", chunk: int = LOGIT_CHUNK) -> tuple[float, int]:
    """Summed completion-position KL in nats for one sequence; no [T,V] tensor."""
    import torch

    if chunk < 1:
        raise InfraError("KL output chunk must be positive")
    total = 0.0  # Python float64: a real one-step update can be only ~1e-3 nats/token.
    with torch.no_grad():
        new_hidden = _completion_hidden(new, row, device)
        head_hidden = _completion_hidden(head, row, device)
        for new_logs, head_logs in zip(_completion_log_probs(new, new_hidden, chunk),
                                       _completion_log_probs(head, head_hidden, chunk), strict=True):
            total += float(kl_from_log_probs(new_logs, head_logs).sum(dtype=torch.float64).item())
    if not math.isfinite(total) or total < 0:
        raise InfraError(f"KL produced an invalid sequence sum: {total}")
    return total, len(row["completion_ids"])


def summarize(sequences: Sequence[float], tokens: int) -> dict[str, Any]:
    if not sequences or tokens < 1 or not all(math.isfinite(value) and value >= 0 for value in sequences):
        raise InfraError("KL summary needs finite nonnegative sequences and completion tokens")
    return {
        "kl_seq": math.fsum(sequences) / len(sequences),
        "kl_seq_se": statistics.stdev(sequences) / math.sqrt(len(sequences)) if len(sequences) > 1 else 0.0,
        "kl_tok": math.fsum(sequences) / tokens,
        "kl_k": len(sequences),
        "kl_tokens": tokens,
    }


def measure_rows(new: Any, head: Any, rows: Sequence[dict], device: str = "cpu", chunk: int = LOGIT_CHUNK) -> dict[str, Any]:
    sequences, tokens = [], 0
    for row in rows:
        value, count = sequence_kl(new, head, row, device, chunk)
        sequences.append(value)
        tokens += count
    return summarize(sequences, tokens)


def _load_model(path: Path, dtype: str):
    from .train_job import leave_room_for_the_driver, load_head

    leave_room_for_the_driver("cuda:0")
    return load_head(str(path), dtype).to("cuda:0").eval().requires_grad_(False)


def run_worker(job_dir: Path) -> None:
    """One visible GPU, a resident head, and its assigned candidate directories."""
    import gc

    import torch
    import transformers

    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    head_path = Path(job["head_dir"])
    head = None
    for item in job["items"]:
        out = job_dir / f"result-{item['shard']}.json"
        candidate = encoded = None
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(head_path, local_files_only=True)
            encoded = tokenize_rows(tokenizer, item["rows"], item.get("max_tokens", 0))
            if head is None:
                head = _load_model(head_path, job["dtype"])
            candidate_path = Path(item["candidate_dir"])
            candidate = head if candidate_path == head_path else _load_model(candidate_path, job["dtype"])
            started = time.monotonic()
            # Even null calibration performs two separate forward invocations.
            sequences, tokens = [], 0
            for row in encoded:
                value, count = sequence_kl(candidate, head, row, "cuda:0")
                sequences.append(value)
                tokens += count
            result = {"metrics": {"sequences": sequences, "kl_tokens": tokens},
                      "wall_s": time.monotonic() - started, "tokenization": TOKENIZATION}
        except Exception as exc:  # one entry's OOM/exception never kills its peers
            result = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            del candidate, encoded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        tmp.replace(out)


class KLRunner:
    """Parent-side process lifecycle. Every expected measurement fails closed independently."""

    def __init__(self, config: BoxConfig, resident: Any, spawn=Child):
        self.config, self.resident, self.spawn = config, resident, spawn

    def measure(self, ticket: Any, entries: Sequence[Any], null_outcomes: Sequence[str], tested: Any) -> dict[str, Any]:
        params = ticket.params
        started = time.monotonic()
        wall_started = time.time()
        if not params.eval_merged:
            raise InfraError("KL measurement requires the merged candidate that was actually served")
            raise InfraError("KL measurement requires the merged candidate that was actually served")
        head_dir = resident_head(self.resident, ticket.head)
        root = self.config.work_dir / "kl" / ticket.round_id
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        jobs: list[dict] = []
        try:
            jobs.append(dict(key="null", candidate_dir=str(head_dir), max_tokens=params.kl_max_tokens,
                             rows=selected_rollouts(capture_file(self.config.work_dir / "rollouts", ticket.round_id, NULL_MODEL),
                                                    null_outcomes, ticket.draw.round_seed, NULL_MODEL, params.kl_rollouts)))
        except Exception as exc:
            log_phase(self.config.work_dir, "kl_null", round_id=ticket.round_id, error=f"{type(exc).__name__}: {exc}")
        answers: dict[str, dict] = {}
        expected: dict[str, int] = {}
        for entry in entries:
            if entry.status != SCORED:
                continue
            try:
                proof = tested.get(entry.queue_pos)
                if proof is None or not proof.path.is_dir():
                    raise InfraError(f"KL has no tested merged base for entry {entry.queue_pos}")
                rows = selected_rollouts(capture_file(self.config.work_dir / "rollouts", ticket.round_id, entry.lora.lora_hash),
                                         entry.outcomes, ticket.draw.round_seed, entry.lora.lora_hash, params.kl_rollouts)
                jobs.append(dict(key=str(entry.queue_pos), candidate_dir=str(proof.path), max_tokens=params.kl_max_tokens, rows=rows))
                expected[str(entry.queue_pos)] = len(rows)
            except Exception as exc:
                answers[str(entry.queue_pos)] = {"error": f"{type(exc).__name__}: {exc}"}
        # Interleave complete rollouts over GPUs, then keep up to 16 together
        # per model load. Every entry with >=8 rollouts can use every GPU;
        # larger entries also run concurrently on each card.
        shards = []
        shard_ids = set()
        for job in jobs:
            for gpu in range(params.n_gpus):
                rows = job["rows"][gpu::params.n_gpus]
                for offset in range(0, len(rows), ROWS_PER_LOAD):
                    part = rows[offset:offset + ROWS_PER_LOAD]
                    shards.append(job | {"rows": part, "shard": f"{job['key']}:{part[0]['task_pos']}"})
                    shard_ids.add(shards[-1]["shard"])
        if len(shards) != len(shard_ids):
            raise InfraError("KL shard identities are not unique")
        assignments = {str(gpu): [item for item in shards[gpu::params.n_gpus] if item["key"] != "null"]
                       for gpu in range(params.n_gpus)}
        assignments = {gpu: items for gpu, items in assignments.items() if items}
        # Put the two-forward null calibration on its own card, with a resident
        # head and no candidate loading. Candidate shards use the other cards.
        for item in shards:
            if item["key"] == "null":
                assignments.setdefault("null", []).append(item)
        # The independent phase cap may never consume the seal/promotion margin.
        deadline = min(wall_started + params.kl_time_limit_s,
                       min(ticket.expires_at, ticket.issued_at + params.round_time_limit_s) - 60)

        def run(gpu: str, items: list[dict]) -> None:
            gpu_started = time.monotonic()
            tokens = 0
            error = None
            completed: dict[str, dict] = {}
            directory = root / f"gpu{gpu}"
            directory.mkdir()
            (directory / "job.json").write_text(json.dumps({"head_dir": str(head_dir), "dtype": params.dtype, "items": items}), encoding="utf-8")
            proc = None
            try:
                logs = self.config.work_dir / "logs"
                logs.mkdir(parents=True, exist_ok=True)
                env = job_env([int(gpu) if gpu != "null" else 0], self.config.work_dir) | {"HF_HUB_OFFLINE": "1"}
                proc = self.spawn([sys.executable, "-m", "sn120_v2.gpu.kl", str(directory)], env, logs / f"kl-{ticket.round_id}-gpu{gpu}.log")
                remaining = deadline - time.time()
                code = proc.wait(max(0.0, remaining))
                if code is None:
                    raise TimeoutError(f"KL exceeded its {params.kl_time_limit_s}s phase cap or the promotion deadline")
                for item in items:
                    try:
                        raw = (directory / f"result-{item['shard']}.json").read_text(encoding="utf-8")
                        result = json.loads(raw)
                        if not isinstance(result, dict) or ("metrics" in result) == ("error" in result):
                            raise InfraError("KL child must return exactly metrics or error")
                        if "metrics" in result:
                            metrics = result["metrics"]
                            if not isinstance(metrics, dict):
                                raise InfraError("KL child returned invalid metrics")
                            sequences, tokens = metrics.get("sequences"), metrics.get("kl_tokens")
                            if (not isinstance(sequences, list) or len(sequences) != len(item["rows"])
                                    or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                                           for value in sequences)
                                    or type(tokens) is not int or tokens < len(item["rows"])):
                                raise InfraError("KL child returned an invalid shard metric set")
                        completed[item["shard"]] = result
                    except (OSError, ValueError) as exc:
                        completed[item["shard"]] = {"error": f"KL child exited with code {code} without a readable result: {exc}"}
                    except Exception as exc:
                        completed[item["shard"]] = {"error": f"{type(exc).__name__}: {exc}"}
                answers.update(completed)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                for item in items:
                    answers[item["shard"]] = {"error": error}
            finally:
                if proc is not None:
                    try:
                        proc.stop()
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        for item in items:
                            answers[item["shard"]] = {"error": error}
                wall_s = time.monotonic() - gpu_started
                tokens = sum(result.get("metrics", {}).get("kl_tokens", 0) for result in completed.values())
                log_phase(self.config.work_dir, "kl_gpu", round_id=ticket.round_id,
                          gpu=int(gpu) if gpu != "null" else 0, worker=gpu,
                          wall_s=wall_s, tokens=tokens, tokens_s=tokens / wall_s if wall_s > 0 else 0.0,
                          model_tokens=2 * tokens,  # two teacher-forced forwards per completion token
                          shards=len(items), error=error)

        try:
            if assignments:
                with ThreadPoolExecutor(len(assignments)) as pool:
                    list(pool.map(lambda pair: run(*pair), assignments.items()))
        finally:
            shutil.rmtree(root, ignore_errors=True)
        output = {"entries": {}, "null": None, "wall_s": time.monotonic() - started}
        for job in jobs:
            key = job["key"]
            pieces = [answers.get(item["shard"]) for item in shards if item["key"] == key]
            try:
                if any(piece is None or "metrics" not in piece for piece in pieces):
                    error = next((piece.get("error") for piece in pieces if piece is not None and "metrics" not in piece), "KL returned no measurement")
                    raise InfraError(error)
                sequences = [value for piece in pieces for value in piece["metrics"]["sequences"]]
                tokens = sum(piece["metrics"]["kl_tokens"] for piece in pieces)
                if key != "null" and len(sequences) != expected[key]:
                    raise InfraError("KL child measured a different number of rollouts")
                metrics = summarize(sequences, tokens)
                wall_s = math.fsum(piece.get("wall_s", 0.0) for piece in pieces)
                if key == "null":
                    output["null"] = metrics
                else:
                    output["entries"][int(key)] = metrics
                log_phase(self.config.work_dir, "kl_entry" if key != "null" else "kl_null", round_id=ticket.round_id,
                          queue_pos=None if key == "null" else int(key), gpu_wall_s=wall_s,
                          max_tokens=params.kl_max_tokens, tokenization=TOKENIZATION, **metrics)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if key == "null":
                    log_phase(self.config.work_dir, "kl_null", round_id=ticket.round_id, error=error)
                else:
                    output["entries"][int(key)] = {"error": error}
                    log_phase(self.config.work_dir, "kl_entry", round_id=ticket.round_id, queue_pos=int(key), error=error)
        for key, answer in answers.items():
            if ":" not in key:
                output["entries"][int(key)] = answer
                log_phase(self.config.work_dir, "kl_entry", round_id=ticket.round_id, queue_pos=int(key), error=answer["error"])
        log_phase(self.config.work_dir, "kl", round_id=ticket.round_id, wall_s=output["wall_s"],
                  tokens=sum(metrics.get("kl_tokens", 0) for metrics in output["entries"].values()),
                  max_tokens=params.kl_max_tokens, time_limit_s=params.kl_time_limit_s)
        return output


if __name__ == "__main__":
    run_worker(Path(sys.argv[1]))
