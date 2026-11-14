#!/usr/bin/env python3
"""Decode-speed microbench for the eval servers. Run it on a spare GPU box, never on the box of a live round.

    python v2/bench/decode_microbench.py

It starts one vLLM server per config, each on a GPU of its own, with the server command and the request
body of the box's own code (``sn120_v2/gpu/evaluator.py``). Each config changes one thing of the base.
It measures the tokens per second of one stream, alone and among 20 streams, at a context above 32,000
tokens, counted after the prompt pass. It also tests whether a seeded request gives the same tokens
again, among other streams, and in each config.

It writes only below one temp directory. It reads no state, journal, store or chain. It opens no
connection except to its own servers on this machine. It stops its servers at the end and on Ctrl-C.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import http.client
import importlib.metadata
import json
import os
import random
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the checkout's v2/: the box code must come from this checkout
try:  # the first check of the script reports a failure here in one line
    import numpy

    from sn120_v2.gpu.evaluator import ServeSettings, chat_body, server_argv, server_env
    from sn120_v2.gpu.procs import Child
    from sn120_v2.params import Params
    from sn120_v2.types import NULL_MODEL

    IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - only on a broken checkout or install
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

VLLM_VERSION = "0.32.0"
N_ADAPTERS = 5  # r000002 had 5 LoRAs
STREAMS = 20
CONTEXT_TOKENS = 34000  # the prompt of each stream; the task asks for a context above 32,000
DECODE_TOKENS = 512
SEED = 120
WORDS = (
    "time year people way day man thing woman life child world school state family student group country problem hand part place "
    "case week company system program question work government number night point home water room mother area money story fact month "
    "lot right study book eye job word business issue side kind head house service friend father power hour game line end member law "
    "car city community name president team minute idea kid body information back parent face others level office door health person "
    "art war history party result change morning reason research girl guy moment air teacher force education foot boy age policy music "
    "market sense nation plan college interest death experience effect class control field development role effort rate heart drug show "
    "leader light voice wife police mind price report decision son view relationship town road arm difference value building action season"
).split()

# name -> (what it changes, what it proves). The base is the box's own command; each other config changes one thing.
CONFIGS: dict[str, tuple[str, str]] = {
    "base": ("the box's own command", "b, c: today's speed with 1 and with 5 adapters loaded"),
    "a": ("no --enable-lora", "a: the cost of the LoRA support itself, even for M alone"),
    "d": ("--max-loras 8", "d: whether the size of the LoRA table (32 against 8) costs time"),
    "e": ("--compilation-config cudagraph_mode FULL", "e: whether full CUDA graphs run with LoRA and what they give"),
    "f": ("--speculative-config ngram, 2 tokens", "f: speculative decoding without a draft model"),
    "g": ("--max-lora-rank = recipe rank", "g: whether the rank room (64 against the recipe's rank) costs time"),
    "h": ("--enforce-eager", "h: the speed with no CUDA graph at all, as a reference for the graph gain of today"),
    "f2": ("--speculative-config ngram_gpu, 2 tokens", "f2: the GPU form of the n-gram method"),
}


def fail(line: str) -> None:
    print(f"STOP: {line}", flush=True)
    raise SystemExit(2)


def set_flag(argv: list[str], flag: str, value: str | None) -> list[str]:
    """``argv`` with the value of ``flag`` replaced. ``value=None`` removes the flag and its value."""
    at = argv.index(flag)
    return argv[:at] + ([] if value is None else [flag, value]) + argv[at + 2 :]


def config_argv(name: str, head: str, port: int, params: Params) -> list[str]:
    base = server_argv(head, port, params)
    for flag in ("--enable-lora", "--max-loras", "--max-lora-rank", "--max-num-seqs", "--enable-prefix-caching", "--dtype", "--max-model-len"):
        if flag not in base:
            fail(f"the box's server command has no {flag}: the configs of this script no longer fit it")
    if name == "base":
        return base
    if name == "a":
        argv = set_flag(set_flag(base, "--max-loras", None), "--max-lora-rank", None)
        argv.remove("--enable-lora")
        return argv
    if name == "d":
        return set_flag(base, "--max-loras", "8")
    if name == "e":
        return base + ["--compilation-config", json.dumps({"cudagraph_mode": "FULL"})]
    if name == "f":
        return base + ["--speculative-config", json.dumps({"method": "ngram", "num_speculative_tokens": 2, "prompt_lookup_max": 4, "prompt_lookup_min": 2})]
    if name == "f2":
        return base + ["--speculative-config", json.dumps({"method": "ngram_gpu", "num_speculative_tokens": 2, "prompt_lookup_max": 4, "prompt_lookup_min": 2})]
    if name == "g":
        return set_flag(base, "--max-lora-rank", str(params.recipe.rank))
    if name == "h":
        return base + ["--enforce-eager"]
    raise ValueError(name)


# ---------------------------------------------------------------- adapters


def safetensors_header(path: Path) -> dict[str, Any]:
    with open(path, "rb") as file:
        (size,) = struct.unpack("<Q", file.read(8))
        return json.loads(file.read(size))


def target_shapes(head: Path, targets: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    """``{module path: (out, in)}`` of every 2-D weight of the head whose module name is a target (the rule of the trainer)."""
    shapes = {}
    for file in sorted(head.glob("*.safetensors")):
        for name, entry in safetensors_header(file).items():
            parts = name.split(".")
            if name != "__metadata__" and parts[-1] == "weight" and len(parts) > 1 and parts[-2] in targets and len(entry["shape"]) == 2:
                shapes[name[: -len(".weight")]] = (entry["shape"][0], entry["shape"][1])
    return shapes


def write_adapter(out_dir: Path, shapes: dict[str, tuple[int, int]], rank: int, alpha: float, targets: tuple[str, ...], seed: int) -> int:
    """A made-up LoRA in the layout of the trainer (PEFT config, float32 safetensors). Returns the bytes written."""
    rng = numpy.random.default_rng(seed)
    header, blobs, offset = {}, [], 0
    for path in sorted(shapes):
        out_features, in_features = shapes[path]
        a = (rng.random((rank, in_features), dtype=numpy.float32) * 2 - 1) / numpy.float32(in_features**0.5)
        b = rng.standard_normal((out_features, rank), dtype=numpy.float32) * numpy.float32(1e-3)
        for name, value in (("lora_A", a), ("lora_B", b)):
            blob = value.astype("<f4").tobytes()
            header[f"base_model.model.{path}.{name}.weight"] = {"dtype": "F32", "shape": list(value.shape), "data_offsets": [offset, offset + len(blob)]}
            blobs.append(blob)
            offset += len(blob)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "adapter_model.safetensors", "wb") as file:
        file.write(struct.pack("<Q", len(text)) + text)
        for blob in blobs:
            file.write(blob)
    config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": "microbench", "r": rank, "lora_alpha": alpha}
    config.update(lora_dropout=0.0, bias="none", target_modules=list(targets), inference_mode=True)
    (out_dir / "adapter_config.json").write_text(json.dumps(config, sort_keys=True))
    return 8 + len(text) + offset


# ---------------------------------------------------------------- requests and timing


def post_json(port: int, path: str, body: dict | None, timeout_s: float) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_s)
    try:
        conn.request("POST" if body is not None else "GET", path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
        reply = conn.getresponse()
        data = reply.read()
        if reply.status >= 400:
            raise OSError(f"HTTP {reply.status}: {data[:200].decode('utf-8', 'replace')}")
        return json.loads(data) if data.strip().startswith(b"{") else {}
    finally:
        conn.close()


def prompt_words(stream: int, count: int) -> str:
    rng = random.Random(f"{SEED}/{stream}")
    return " ".join(rng.choice(WORDS) for _ in range(count))


def messages_for(stream: int, words: int) -> list[dict[str, str]]:
    # The words come first, so two streams share no prefix beyond the tokens of the chat template.
    return [{"role": "user", "content": prompt_words(stream, words) + "\n\nWrite a long story that uses these words in this order."}]


def request_body(model: str, stream: int, words: int, params: Params, max_tokens: int) -> dict:
    body = chat_body(model, messages_for(stream, words), SEED + stream, params)  # the sampling of the eval, from the box code
    body.update(max_tokens=max_tokens, ignore_eos=True, return_token_ids=True)  # a fixed length, so the speeds can be compared
    return body


def stream_chat(port: int, body: dict, timeout_s: float = 900.0) -> dict[str, Any]:
    """One streamed chat request. Returns the arrival time and the token count of each chunk, the token ids and the text."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_s)
    sent = time.monotonic()
    events: list[tuple[float, int]] = []
    ids: list[int] = []
    text: list[str] = []
    usage: dict[str, Any] = {}
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        reply = conn.getresponse()
        if reply.status >= 400:
            raise OSError(f"HTTP {reply.status}: {reply.read()[:200].decode('utf-8', 'replace')}")
        for raw in reply:
            line = raw.strip()
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            chunk = json.loads(line[5:])
            now = time.monotonic()
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = "".join(str(delta.get(key) or "") for key in ("reasoning_content", "reasoning", "content"))
                got = choice.get("token_ids") or delta.get("token_ids") or []
                if piece or got:
                    events.append((now, len(got) or 1))
                    ids.extend(got)
                    text.append(piece)
    finally:
        conn.close()
    return {"sent": sent, "events": events, "ids": ids, "text": "".join(text), "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}


def decode_rate(result: dict[str, Any], start: float | None = None, end: float | None = None) -> float | None:
    """Tokens per second after the prompt pass: the tokens after the first chunk, over the time from the first chunk on."""
    events = result["events"]
    if len(events) < 2:
        return None
    total = result["completion_tokens"] or sum(count for _, count in events)
    scale = total / max(1, sum(count for _, count in events))  # a chunk whose token count is not known counts as one token
    first = events[0][0] if start is None else max(start, events[0][0])
    last = events[-1][0] if end is None else min(end, events[-1][0])
    tokens = sum(count for at, count in events if first < at <= last) * scale
    return tokens / (last - first) if last - first > 0.5 else None


def same(left: dict[str, Any], right: dict[str, Any], prefix: bool = False) -> dict[str, Any]:
    """Whether two replies are equal, and the first position that differs (token position when the server gave token ids).

    With ``prefix``, ``right`` may be longer: only its start, as long as ``left``, is compared.
    """
    unit = "token" if left["ids"] and right["ids"] else "character"
    a, b = (left["ids"], right["ids"]) if unit == "token" else (left["text"], right["text"])
    if prefix:
        b = b[: len(a)]
    if a == b:
        return {"equal": True, "unit": unit, "length": len(a)}
    first = next((at for at, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return {"equal": False, "unit": unit, "first_difference": first, "length": [len(a), len(b)]}


def median(values: list[float]) -> float | None:
    values = sorted(values)
    return values[len(values) // 2] if len(values) % 2 else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2 if values else None


def steps_counter(port: int) -> float | None:
    """The number of decode steps the server ran so far (count of the histogram ``vllm:iteration_tokens_total``)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    try:
        conn.request("GET", "/metrics")
        found = re.findall(r"^vllm:iteration_tokens_total_count(?:\{[^}]*\})? ([0-9.e+]+)$", conn.getresponse().read().decode("utf-8", "replace"), re.M)
        return sum(float(value) for value in found) if found else None
    except (OSError, ValueError):
        return None
    finally:
        conn.close()


def measure_target(port: int, models: list[str], words: int, params: Params, streams: int = STREAMS, decode_tokens: int = DECODE_TOKENS) -> dict[str, Any]:
    """One stream alone (two times), then ``streams`` streams side by side. Stream ``i`` uses ``models[i % len(models)]``."""
    model_of = [models[index % len(models)] for index in range(streams)]
    with ThreadPoolExecutor(streams) as pool:  # the prompt pass, kept by the prefix cache, so all streams decode at the same time later
        warm = list(pool.map(lambda index: post_json(port, "/v1/chat/completions", request_body(model_of[index], index, words, params, 1), 900.0), range(streams)))
    prompt_tokens = [int((reply.get("usage") or {}).get("prompt_tokens") or 0) for reply in warm]
    body = request_body(model_of[0], 0, words, params, decode_tokens)
    alone = [stream_chat(port, body), stream_chat(port, body)]
    for many_tokens in (decode_tokens, 8 * decode_tokens):
        steps_before, wall = steps_counter(port), time.monotonic()
        with ThreadPoolExecutor(streams) as pool:
            many = list(pool.map(lambda index: stream_chat(port, request_body(model_of[index], index, words, params, many_tokens)), range(streams)))
        wall, steps_after = time.monotonic() - wall, steps_counter(port)
        start = max((result["events"][0][0] for result in many if result["events"]), default=0.0)
        end = min((result["events"][-1][0] for result in many if result["events"]), default=0.0)
        window = end - start
        if window > 2.0:  # else the prompt passes of some streams ran beside the decode of others: one more try, with longer replies
            break
    rates = [rate for rate in (decode_rate(result, start, end) if window > 2.0 else decode_rate(result) for result in many) if rate]
    return {
        "prompt_tokens_min": min(prompt_tokens),
        "context_above_32000": min(prompt_tokens) > 32000,
        "alone_tok_s": [round(rate, 1) if rate else None for rate in map(decode_rate, alone)],
        "alone_prompt_pass_s": [round(result["events"][0][0] - result["sent"], 2) if result["events"] else None for result in alone],
        "many_streams": streams,
        "many_tok_s_per_stream_median": round(median(rates), 1) if rates else None,
        "many_tok_s_per_stream_min": round(min(rates), 1) if rates else None,
        "many_tok_s_sum": round(sum(rates), 1) if rates else None,
        "many_window_s": round(window, 2),  # the time in which all streams decode; below 2 s the whole decode time of each stream is used
        "many_steps": steps_after - steps_before if steps_before is not None and steps_after is not None else None,
        "many_wall_s": round(wall, 2),
        "many_decode_tokens": many_tokens,
        "completion_tokens": alone[0]["completion_tokens"],
        "same_alone_twice": same(alone[0], alone[1]),
        "same_alone_and_among_many": same(alone[0], many[0], prefix=True),
        "_reply": {"ids": alone[0]["ids"], "text": alone[0]["text"]},
    }


# ---------------------------------------------------------------- one server


def start_facts(log: str) -> dict[str, Any]:
    """The facts of the start that the server log gives."""
    captured: dict[str, int] = {}
    for mode, total in re.findall(r"\((PIECEWISE|FULL)\)[^\r\n]*?\d+/(\d+)", log):
        captured[mode] = max(captured.get(mode, 0), int(total))
    modes = re.findall(r"cudagraph_mode['\"]?\s*[=:]\s*<?CUDAGraphMode\.(\w+)", log)
    finished = re.search(r"Graph capturing finished in (\d+) secs, took ([-\d.]+) GiB", log)
    speculative = re.search(r"speculative_config=(None|\w+\([^)]{0,120})", log)
    return {
        "graph_mode": modes[-1] if modes else None,
        "graph_sizes_captured": captured or None,
        "graph_capture_s": int(finished.group(1)) if finished else None,
        "enforce_eager": (re.findall(r"enforce_eager=(\w+)", log) or [None])[-1],
        "attention_backends": sorted(set(re.findall(r"\b(FLASHINFER|FLASH_ATTN|TRITON_ATTN|FLEX_ATTENTION|XFORMERS)\b", log))) or None,
        "speculative_config": speculative.group(1) if speculative else None,
        "runner_v1_fallback": bool(re.search(r"[Ff]alling back to (the )?V1|V1 model runner", log)),
        "warnings_cudagraph": sorted({line.strip()[-160:] for line in re.findall(r"WARNING[^\r\n]*[Cc]udagraph[^\r\n]*", log)})[:3],
    }


def error_line(log: str, head: str) -> str:
    lines = [line.strip() for line in log.splitlines() if re.search(r"Error|error:|Exception|CUDA out of memory|unrecognized arguments", line)]
    line = lines[-1] if lines else (log.strip().splitlines() or ["no output"])[-1]
    return re.sub(r"\d+\.\d+\.\d+\.\d+", "<address>", line.replace(head, "<head>"))[-240:]


def worker(args: argparse.Namespace) -> int:
    """Start one server, measure it, write ``<temp>/<config>.json``, stop the server."""
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    name, temp, params = args.worker, Path(args.temp), Params()
    work, log_path = temp / name, temp / "logs" / f"{name}.log"
    work.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"config": name, "change": CONFIGS[name][0], "ok": False, "phases": {}}
    server = None
    read_log = lambda: log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""  # noqa: E731
    try:
        argv = [sys.executable, *config_argv(name, args.head, args.port, params)[1:]]
        out["flags"] = [flag for flag in argv[argv.index("--port") + 2 :]]
        started = time.monotonic()
        server = Child(argv, server_env(args.gpu, work), log_path)
        (temp / f"{name}.pid").write_text(str(server.proc.pid))
        while True:
            try:
                post_json(args.port, "/health", None, 10.0)
                break
            except (OSError, ValueError):
                if not server.alive():
                    raise RuntimeError("the server stopped at the start: " + error_line(read_log(), args.head)) from None
                if time.monotonic() - started > args.start_timeout_s:
                    raise RuntimeError(f"the server was not ready after {args.start_timeout_s} s") from None
                time.sleep(3.0)
        out["start_s"] = round(time.monotonic() - started, 1)
        out["start"] = start_facts(read_log())
        try:  # the tokens of one word, so the prompt has the wanted size
            sample = prompt_words(0, 2000)
            per_word = int(post_json(args.port, "/tokenize", {"model": NULL_MODEL, "prompt": sample}, 60.0)["count"]) / 2000
        except (OSError, KeyError, ValueError, TypeError):
            per_word = 1.0
        words = min(int(args.context_tokens / per_word), int((params.max_context_tokens - args.decode_tokens - 512) / max(per_word, 1.6)) if per_word == 1.0 else 10**9)
        out["tokens_per_word"] = round(per_word, 3)
        lora = "--enable-lora" in argv
        adapters = [f"lora{index}" for index in range(1, N_ADAPTERS + 1)]
        phases: list[tuple[str, int, dict[str, list[str]]]] = [("no_lora", 0, {"M": [NULL_MODEL]})]
        if lora:
            mix = {"M": [NULL_MODEL], "one_adapter": [adapters[0]], "mix_M_and_5": [NULL_MODEL, *adapters]}
            phases = [(f"{N_ADAPTERS}_adapters", N_ADAPTERS, mix)]
            if name == "base":
                phases.insert(0, ("1_adapter", 1, {"M": [NULL_MODEL], "one_adapter": [adapters[0]]}))
        loaded = 0
        for phase, wanted, targets in phases:
            for adapter in adapters[loaded:wanted]:
                post_json(args.port, "/v1/load_lora_adapter", {"lora_name": adapter, "lora_path": str(temp / "adapters" / adapter)}, 600.0)
            loaded = max(loaded, wanted)
            out["phases"][phase] = {}
            for target, models in targets.items():
                try:
                    out["phases"][phase][target] = measure_target(args.port, models, words, params, args.streams, args.decode_tokens)
                except Exception as exc:  # noqa: BLE001 - one failed target must not stop the others
                    out["phases"][phase][target] = {"error": error_line(f"{type(exc).__name__}: {exc}", args.head)}
                if not server.alive():
                    raise RuntimeError("the server stopped during the measurement: " + error_line(read_log(), args.head))
        out["ok"] = True
    except KeyboardInterrupt:
        out["error"] = "interrupted"
    except Exception as exc:  # noqa: BLE001 - the cause goes to the table in one line
        out["error"] = error_line(f"{type(exc).__name__}: {exc}", args.head)
    finally:
        if server is not None:
            server.stop()
        shutil.rmtree(work, ignore_errors=True)
        (temp / f"{name}.json").write_text(json.dumps(out))
    return 0 if out["ok"] else 1


# ---------------------------------------------------------------- the whole run


def free_gpus(limit_mib: int) -> tuple[list[int], list[str]]:
    report = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60, check=True).stdout
    rows = [[cell.strip() for cell in line.split(",")] for line in report.splitlines() if line.strip()]
    return [int(row[0]) for row in rows if int(row[2]) < limit_mib], [f"{row[1]} {int(row[3]) // 1024} GiB" for row in rows]


def find_head(given: str | None) -> Path:
    roots = [given] if given else [os.environ.get("SN120_WORK_DIR", ""), "/root/sn120-run/work", *glob.glob("/root/*/work"), *glob.glob("/workspace/*/work"), *glob.glob("/tmp/*/work")]
    found: dict[Path, int] = {}
    for root in filter(None, roots):
        for config in [Path(root) / "config.json", *Path(root).glob("cache/*/config.json")]:
            size = sum(file.stat().st_size for file in config.parent.glob("*.safetensors")) if config.is_file() else 0
            if size and (config.parent / "tokenizer_config.json").is_file():
                found[config.parent.resolve()] = size
    if not found:
        fail("no head found (a directory with config.json, tokenizer_config.json and *.safetensors); give it with --head DIR or set SN120_WORK_DIR")
    return min(found, key=lambda path: found[path])  # the bf16 head is the smaller one; the fp32 master is two times its size


def cell(value: Any) -> str:
    return "-" if value is None else str(value)


def table(results: dict[str, dict[str, Any]], compare: dict[str, Any]) -> str:
    rows = [("config", "phase", "streams on", "1 alone", "1 of 20", "sum of 20", "window s", "ctx", "twice", "in 20", "vs base")]
    failed = [f"{name}: FAILED: {result.get('error')}" for name, result in results.items() if not result.get("ok")]
    for name, result in results.items():
        for phase, targets in result["phases"].items():
            for target, got in targets.items():
                if "error" in got:
                    failed.append(f"{name} {phase} {target}: FAILED: {got['error']}")
                    continue
                equal = lambda item: "-" if item is None else "same" if item["equal"] else f"differ@{item['first_difference']}"  # noqa: E731
                rows.append((name, phase, target, cell(got["alone_tok_s"][0]), cell(got["many_tok_s_per_stream_median"]), cell(got["many_tok_s_sum"]), cell(got["many_window_s"]), cell(got["prompt_tokens_min"]),
                             equal(got["same_alone_twice"]), equal(got["same_alone_and_among_many"]), equal(compare.get(f"{name}/{phase}/{target}"))))
    widths = [max(len(row[at]) for row in rows) for at in range(len(rows[0]))]
    return "\n".join(["  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() for row in rows] + failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", help="directory of the head (default: found below SN120_WORK_DIR or /root/sn120-run/work)")
    parser.add_argument("--configs", default=",".join(CONFIGS), help="configs to run, in order of importance (default: all)")
    parser.add_argument("--gpus", help="GPU numbers to use, for example 0,1,2 (default: every GPU that is free)")
    parser.add_argument("--streams", type=int, default=STREAMS)
    parser.add_argument("--context-tokens", type=int, default=CONTEXT_TOKENS)
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    parser.add_argument("--start-timeout-s", type=int, default=900)
    parser.add_argument("--first-port", type=int, default=8300)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--temp", help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if IMPORT_ERROR:
        fail(f"the checkout or the install is not complete ({IMPORT_ERROR}); run with the Python of the box install, from the checkout")
    if args.worker:
        return worker(args)

    # The checks of the first seconds. Each one stops the script with one line.
    params, names = Params(), [name for name in args.configs.split(",") if name]
    if unknown := [name for name in names if name not in CONFIGS]:
        fail(f"unknown config {unknown}; known: {', '.join(CONFIGS)}")
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        fail("vLLM is not installed in this Python; run with the Python of the box install")
    try:
        free, kinds = free_gpus(ServeSettings().free_limit_mib)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        fail(f"nvidia-smi gave no GPU list ({type(exc).__name__})")
    if args.gpus:
        free = [gpu for gpu in free if gpu in {int(item) for item in args.gpus.split(",")}]
    if not free:
        fail("no free GPU (each GPU has a process on it); stop the other work or give --gpus")
    head = find_head(args.head)
    for name in names:
        config_argv(name, str(head), 0, params)  # stops when the box command lost a flag
    shapes = target_shapes(head, params.recipe.target_modules)
    if not shapes:
        fail("the head has no weight with the name of a target module of the recipe")
    body = chat_body(NULL_MODEL, [], 0, params)
    waves = [names[at : at + len(free)] for at in range(0, len(names), len(free))]
    print(f"decode microbench: vLLM {version}{'' if version == VLLM_VERSION else f' (WARNING: the box has {VLLM_VERSION})'}; {len(kinds)} GPUs ({kinds[0]}), {len(free)} free; "
          f"head with {len(shapes)} target layers; {len(names)} configs in {len(waves)} wave(s); expected time {6 + 7 * len(waves)} to {10 + 12 * len(waves)} minutes", flush=True)

    temp = Path(tempfile.mkdtemp(prefix="decode-microbench-"))
    (temp / "logs").mkdir()
    children: list[subprocess.Popen] = []

    def stop(*_: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    results: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    try:
        size = sum(write_adapter(temp / "adapters" / f"lora{index}", shapes, params.recipe.rank, params.recipe.alpha, params.recipe.target_modules, SEED + index) for index in range(1, N_ADAPTERS + 1))
        print(f"{N_ADAPTERS} made-up adapters written (rank {params.recipe.rank}, {size / 2**20:.0f} MiB in all); temp directory {temp}", flush=True)
        for number, wave in enumerate(waves, 1):
            print(f"wave {number}: starting {', '.join(wave)} (one GPU each); the servers need some minutes to start", flush=True)
            children = []
            for at, name in enumerate(wave):
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", name, "--temp", str(temp), "--head", str(head), "--gpu", str(free[at]), "--port", str(args.first_port + at)]
                command += ["--streams", str(args.streams), "--context-tokens", str(args.context_tokens), "--decode-tokens", str(args.decode_tokens), "--start-timeout-s", str(args.start_timeout_s)]
                children.append(subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=open(temp / "logs" / f"{name}.worker.log", "ab")))
            pending = dict(zip(wave, children))
            while pending:
                time.sleep(5.0)
                for name in [name for name, child in pending.items() if child.poll() is not None]:
                    del pending[name]
                    path = temp / f"{name}.json"
                    results[name] = json.loads(path.read_text()) if path.exists() else {"config": name, "ok": False, "phases": {}, "error": "the worker stopped with no result"}
                    print(f"  {name}: {'done' if results[name]['ok'] else 'FAILED: ' + str(results[name].get('error'))} ({time.monotonic() - started:.0f} s)", flush=True)
    except KeyboardInterrupt:
        print("interrupted: stopping the servers", flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
        for child in children:
            try:
                child.wait(90)
            except subprocess.TimeoutExpired:
                child.kill()
        for pid_file in temp.glob("*.pid"):  # a server that outlived its worker
            try:
                os.killpg(int(pid_file.read_text()), signal.SIGKILL)
            except (OSError, ValueError):
                pass
            pid_file.unlink()
        shutil.rmtree(temp / "adapters", ignore_errors=True)
    for name in names:  # a config that was interrupted still shows what it measured
        if name not in results and (temp / f"{name}.json").exists():
            results[name] = json.loads((temp / f"{name}.json").read_text())

    # Each config against the base: the same seeded request, alone, on M and on one adapter.
    replies = {f"{name}/{phase}/{target}": got.pop("_reply") for name, result in results.items() for phase, targets in result["phases"].items() for target, got in targets.items() if "_reply" in got}
    last = f"{N_ADAPTERS}_adapters"
    compare = {}
    for key, reply in replies.items():
        name, phase, target = key.split("/")
        reference = replies.get(f"base/{last}/{target}" if target != "mix_M_and_5" else f"base/{last}/M")
        if reference is not None and (name, phase) != ("base", last):
            compare[key] = same(reference, reply)
    for key, reply in replies.items():
        name, phase, target = key.split("/")
        results[name]["phases"][phase][target]["reply_sha256"] = hashlib.sha256(json.dumps(reply["ids"] or reply["text"]).encode()).hexdigest()[:16]
    report = {
        "script": "decode_microbench", "vllm": version, "gpus": kinds, "gpus_free": len(free), "wall_s": round(time.monotonic() - started),
        "sampling": {key: body[key] for key in ("temperature", "top_p", "top_k", "seed") if key in body} | {"seed": f"{SEED} + stream number", "ignore_eos": True},
        "decode_tokens": args.decode_tokens, "context_tokens_wanted": args.context_tokens, "adapters": {"count": N_ADAPTERS, "rank": params.recipe.rank, "layers": len(shapes), "made_up": True},
        "proves": {name: CONFIGS[name][1] for name in names}, "same_as_base": compare, "configs": results,
    }
    (temp / "result.json").write_text(json.dumps(report, indent=1))
    print("\ntokens per second after the prompt pass; 'window s' = seconds in which all 20 streams decode (below 2: the 20-stream speeds are not clean);\n'twice' = the same request alone two times; 'in 20' = alone against among 20; 'vs base' = against the base with 5 adapters")
    print(table(results, compare))
    print(f"\nJSON: {temp / 'result.json'}   server logs: {temp / 'logs'}   (paste the table and the JSON file)", flush=True)
    return 0 if results and all(result["ok"] for result in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
