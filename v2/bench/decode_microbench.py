#!/usr/bin/env python3
"""Decode-speed microbench for the eval servers. Run it on a spare GPU box, never on the box of a live round.

    python v2/bench/decode_microbench.py

It starts one vLLM server per config, each on a GPU of its own, with the server command and the request
body of the box's own code (``sn120_v2/gpu/evaluator.py``) and the LoRA table of a round (``--table``,
6). Each config changes one thing of the base. It measures the tokens per second of one stream, alone
and among N streams (``--streams``), at a given context (``--context-tokens``), counted after the prompt
pass. The cases are a first look (M and all 6 LoRAs in flight) and later looks (M and 2, M and 1). It
also tests whether a seeded request gives the same tokens again, among other streams, and in each
config, and it prints the start time and the memory of each server.

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
N_ADAPTERS = 6  # batch_cap: the most LoRAs of one ticket
ADAPTER_STRENGTH = 1e-2  # the factor of the matrix B of a made-up adapter
FLAG = "--specialize-active-lora"
STREAMS = 20
CONTEXT_TOKENS = 34000  # the prompt of each stream; the task asks for a context above 32,000
DECODE_TOKENS = 512
REQUEST_TIMEOUT_S = 3600  # one request; the prompt passes of many streams with a long context run one after another
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
    "base": ("the box's own command, table of the round", "today's speed at the table of the round"),
    "flag": (FLAG, "LoRA kernels sized for the adapters in the step, not for the table + 1"),
    "rank": ("--max-lora-rank = recipe rank, no " + FLAG, "whether the rank room (64 against the rank of the LoRAs) costs time"),
    "both": (FLAG + " and --max-lora-rank = recipe rank", "the two together"),
    "noflag": ("no " + FLAG, "for a checkout whose own command has the flag: the command without it"),
    "a": ("no --enable-lora", "a: the cost of the LoRA support itself, even for M alone"),
    "d": ("--max-loras 32", "d: the table of 32, as before 5bc36ed8"),
    "i": ("--max-loras 32 and " + FLAG, "i: the flag with the table of 32"),
    "e": ("--compilation-config cudagraph_mode FULL", "e: whether full CUDA graphs run with LoRA and what they give"),
    "f": ("--speculative-config ngram, 2 tokens", "f: speculative decoding without a draft model"),
    "g": ("--max-lora-rank = recipe rank", "g: whether the rank room (64 against the recipe's rank) costs time"),
    "h": ("--enforce-eager", "h: the speed with no CUDA graph at all, as a reference for the graph gain of today"),
    "f2": ("--speculative-config ngram_gpu, 2 tokens", "f2: the GPU form of the n-gram method"),
}
DEFAULT_CONFIGS = ["base", "flag", "rank", "both"]


def fail(line: str) -> None:
    print(f"STOP: {line}", flush=True)
    raise SystemExit(2)


def set_flag(argv: list[str], flag: str, value: str | None) -> list[str]:
    """``argv`` with the value of ``flag`` replaced. ``value=None`` removes the flag and its value."""
    at = argv.index(flag)
    return argv[:at] + ([] if value is None else [flag, value]) + argv[at + 2 :]


def with_flag(argv: list[str], on: bool) -> list[str]:
    """``argv`` with the bare flag FLAG present or absent."""
    argv = [item for item in argv if item != FLAG]
    return argv + [FLAG] if on else argv


def config_argv(name: str, head: str, port: int, params: Params, table: int = N_ADAPTERS) -> list[str]:
    base = server_argv(head, port, params)
    for flag in ("--enable-lora", "--max-loras", "--max-lora-rank", "--max-num-seqs", "--enable-prefix-caching", "--dtype", "--max-model-len"):
        if flag not in base:
            fail(f"the box's server command has no {flag}: the configs of this script no longer fit it")
    base = set_flag(base, "--max-loras", str(table))  # the table of a round: the entries of the ticket
    if name == "base":
        return base
    if name == "flag":
        return with_flag(base, True)
    if name == "noflag":
        return with_flag(base, False)
    if name == "rank":
        return with_flag(set_flag(base, "--max-lora-rank", str(params.recipe.rank)), False)
    if name == "both":
        return with_flag(set_flag(base, "--max-lora-rank", str(params.recipe.rank)), True)
    if name == "a":
        argv = set_flag(set_flag(base, "--max-loras", None), "--max-lora-rank", None)
        argv.remove("--enable-lora")
        return argv
    if name == "d":
        return set_flag(base, "--max-loras", "32")
    if name == "i":
        return with_flag(set_flag(base, "--max-loras", "32"), True)
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


def write_adapter(out_dir: Path, shapes: dict[str, tuple[int, int]], rank: int, alpha: float, targets: tuple[str, ...], seed: int, strength: float = ADAPTER_STRENGTH) -> int:
    """A made-up LoRA in the layout of the trainer (PEFT config, float32 safetensors). Returns the bytes written."""
    rng = numpy.random.default_rng(seed)
    header, blobs, offset = {}, [], 0
    for path in sorted(shapes):
        out_features, in_features = shapes[path]
        a = (rng.random((rank, in_features), dtype=numpy.float32) * 2 - 1) / numpy.float32(in_features**0.5)
        b = rng.standard_normal((out_features, rank), dtype=numpy.float32) * numpy.float32(strength)  # strong enough to change the tokens of M soon
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


def stream_chat(port: int, body: dict, timeout_s: float = 900.0, started: threading.Event | None = None) -> dict[str, Any]:
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
                    if started is not None and len(events) >= 8:
                        started.set()  # this stream is in the decode batch now
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


def measure_target(port: int, models: list[str], words: int, params: Params, streams: int = STREAMS, decode_tokens: int = DECODE_TOKENS, timeout_s: float = REQUEST_TIMEOUT_S) -> dict[str, Any]:
    """One stream alone (two times), then ``streams`` streams side by side. Stream ``i`` uses ``models[i % len(models)]``."""
    model_of = [models[index % len(models)] for index in range(streams)]
    with ThreadPoolExecutor(streams) as pool:  # the prompt pass, kept by the prefix cache, so all streams decode at the same time later
        warm = list(pool.map(lambda index: post_json(port, "/v1/chat/completions", request_body(model_of[index], index, words, params, 1), timeout_s), range(streams)))
    prompt_tokens = [int((reply.get("usage") or {}).get("prompt_tokens") or 0) for reply in warm]
    body = request_body(model_of[0], 0, words, params, decode_tokens)
    alone = [stream_chat(port, body, timeout_s), stream_chat(port, body, timeout_s)]
    alone_lora = stream_chat(port, request_body(model_of[1], 1, words, params, decode_tokens), timeout_s) if len(models) > 1 and streams > 1 else None
    for many_tokens in (decode_tokens, 8 * decode_tokens):
        steps_before, wall = steps_counter(port), time.monotonic()
        with ThreadPoolExecutor(streams) as pool:
            many = list(pool.map(lambda index: stream_chat(port, request_body(model_of[index], index, words, params, many_tokens), timeout_s), range(streams)))
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
        "alone_lora_tok_s": round(decode_rate(alone_lora) or 0.0, 1) if alone_lora else None,
        "alone_prompt_pass_s": [round(result["events"][0][0] - result["sent"], 2) if result["events"] else None for result in alone],
        "many_streams": streams,
        "many_tok_s_per_stream_median": round(median(rates), 1) if rates else None,
        "many_tok_s_per_stream_min": round(min(rates), 1) if rates else None,
        "many_tok_s_sum": round(sum(rates), 1) if rates else None,
        "many_window_s": round(window, 2),  # the time in which all streams decode; below 2 s the whole decode time of each stream is used
        "many_steps": steps_after - steps_before if steps_before is not None and steps_after is not None else None,
        "many_ms_per_step": round(1000 * wall / (steps_after - steps_before), 2) if steps_before is not None and steps_after and steps_after > steps_before else None,
        "many_wall_s": round(wall, 2),
        "many_decode_tokens": many_tokens,
        "completion_tokens": alone[0]["completion_tokens"],
        "same_alone_twice": same(alone[0], alone[1]),
        "same_alone_and_among_many": same(alone[0], many[0], prefix=True),
        "_reply": {"ids": alone[0]["ids"], "text": alone[0]["text"]},
        "_reply_lora": {"ids": alone_lora["ids"], "text": alone_lora["text"]} if alone_lora else None,
    }


# ---------------------------------------------------------------- the correctness test of the flag

CORRECT_TOKENS = 256  # the reply of each request of a case: all have this length, so all are in the decode batch together
CORRECT_M_ROWS = 4  # requests of M in flight in the run "with M"
CORRECT_LEAVE_BY = 30  # a LoRA shows the fault only if its text without M leaves M's text soon: below this token
CORRECT_PROMPT_WORDS = 48  # far below the 512 tokens of the largest recorded graph: each decode step uses a recorded graph


def fault_counts(table: int) -> list[int]:
    """The k for which vLLM 0.32.0 records a graph of its own below the table + 1: powers of two up to the table."""
    return [k for k in range(1, table + 1) if k & (k - 1) == 0]


def correct_body(model: str, stream: int, params: Params, tokens: int) -> dict:
    body = request_body(model, stream, CORRECT_PROMPT_WORDS, params, tokens)
    body.update(temperature=0.0)  # greedy and a fixed seed: a change of the text is a change of the model's output
    return body


def run_case(port: int, probes: list[tuple[str, int]], m_rows: int, params: Params, m_first: bool = True, tokens: int = CORRECT_TOKENS) -> dict[str, dict[str, Any]]:
    """One case on one server: the ``probes`` (LoRA, stream) all in flight at once, with ``m_rows`` requests of M.

    Every request has a fixed length (``ignore_eos``). No other request runs on the server during the case.
    ``m_first``: the M requests start first and end last, so M is in flight for the whole reply of each probe. But the
    probes can come into the batch one step apart, so the first and last steps can hold fewer than k LoRAs.
    Not ``m_first``: the probes start first; M comes in when all probes write and ends before a probe ends. So each step
    with M holds exactly the k LoRAs. Returns ``{model: reply}`` of the probes.
    """
    m_tokens = tokens + 96 if m_first else tokens // 2
    m_events, probe_events = [threading.Event() for _ in range(m_rows)], [threading.Event() for _ in probes]
    with ThreadPoolExecutor(m_rows + len(probes)) as pool:
        send_m = lambda: [pool.submit(stream_chat, port, correct_body(NULL_MODEL, 200 + at, params, m_tokens), 900.0, m_events[at]) for at in range(m_rows)]  # noqa: E731
        send_probes = lambda: [pool.submit(stream_chat, port, correct_body(model, stream, params, tokens), 900.0, probe_events[at]) for at, (model, stream) in enumerate(probes)]  # noqa: E731
        if m_first:
            held = send_m()
            if not all(event.wait(300.0) for event in m_events):
                raise RuntimeError("a request of M did not start in 300 s")
            futures = send_probes()
        else:
            futures = send_probes()
            if not all(event.wait(300.0) for event in probe_events):
                raise RuntimeError("a request of a LoRA did not start in 300 s")
            held = send_m()
        replies, m_replies = [future.result() for future in futures], [future.result() for future in held]
    starts, ends = [reply["events"][0][0] for reply in replies], [reply["events"][-1][0] for reply in replies]
    full = all(reply["completion_tokens"] == tokens for reply in replies) and all(reply["completion_tokens"] == m_tokens for reply in m_replies)
    if m_first:  # M wrote from before the first probe to after the last
        inside = all(reply["events"][0][0] < min(starts) and reply["events"][-1][0] > max(ends) for reply in m_replies)
    else:  # M wrote only while all probes wrote
        inside = all(reply["events"][0][0] > max(starts) and reply["events"][-1][0] < min(ends) for reply in m_replies)
    m_from = min((reply["events"][0][0] for reply in m_replies), default=None)
    out = {}
    for (model, _), reply in zip(probes, replies):
        out[model] = {"ids": reply["ids"], "text": reply["text"], "ok": bool(full and inside and max(starts) < min(ends)),
                      "m_from_token": sum(1 for at, _ in reply["events"] if at < m_from) if m_from is not None and not m_first else 0}
    return out


def correctness(port: int, adapters: list[str], params: Params) -> dict[str, Any]:
    """For each k from 1 to the table: the first k LoRAs in flight at once, (a) with no request of M, (b) with requests of M.

    The reading under test (vLLM 0.32.0, ``--specialize-active-lora``): the recorded graph is selected by the count k of
    LoRAs in the step, and a request of M is not counted. But the kernel's id list has one more entry when the step has
    rows of M. So for a k with a graph of its own (1, 2, 4, ...) the LoRA of the highest slot gets no LoRA change in a
    step with M: its text follows M. Each of the k LoRAs is a probe, so the slot order cannot hide the fault. The other
    k (3, 5, 6 at a table of 6) and every case without M must give the same text in (a) and (b): they show that the
    test is sound. A server without the flag runs the same cases as the control. (b) runs in two orders, see ``run_case``.
    """
    for adapter in adapters:  # a slot is given at the first request of an adapter: this fixes the order (lora1 = slot 0, ...)
        post_json(port, "/v1/chat/completions", correct_body(adapter, 0, params, 1), 600.0)
    streams = {adapter: 100 + at for at, adapter in enumerate(adapters)}  # the prompt and seed of an adapter: the same in each case and on each server
    out: dict[str, Any] = {"tokens": CORRECT_TOKENS, "m_rows": CORRECT_M_ROWS, "cases": {}}
    out["M"] = {adapter: run_case(port, [(NULL_MODEL, streams[adapter])], 0, params)[NULL_MODEL] for adapter in adapters}  # (c): M's text of the same prompt
    for k in range(1, len(adapters) + 1):
        probes = [(adapter, streams[adapter]) for adapter in adapters[:k]]
        out["cases"][str(k)] = {"without_M": run_case(port, probes, 0, params), "without_M_again": run_case(port, probes, 0, params),
                                "M_inside": run_case(port, probes, CORRECT_M_ROWS, params, m_first=False), "M_first": run_case(port, probes, CORRECT_M_ROWS, params, m_first=True)}
    return out


def judge_correctness(results: dict[str, dict[str, Any]], table: int) -> tuple[list[str], dict[str, Any]]:
    """One line for each config and k: which LoRAs changed their text when M came into flight, and whether the text follows M."""
    lines, verdicts = [], {}
    for name, result in results.items():
        test = result.get("correctness") or {}
        if not test.get("cases"):
            continue
        has_flag = FLAG in (result.get("flags") or [])
        lines.append(f"{name} ({'flag ON' if has_flag else 'flag off: the control'})")
        faults = []
        for k, case in sorted(test["cases"].items(), key=lambda item: int(item[0])):
            changed: dict[str, list[str]] = {"without_M_again": [], "M_inside": [], "M_first": []}
            notes, sound, leaves = [], True, {}
            for adapter, without_m in case["without_M"].items():
                m_text = test["M"][adapter]
                leaves[adapter] = same(m_text, without_m)["first_difference"] if not same(m_text, without_m)["equal"] else None
                if leaves[adapter] is None or leaves[adapter] >= CORRECT_LEAVE_BY:
                    where = "is M's text" if leaves[adapter] is None else f"leaves M's text only at token {leaves[adapter]}"
                    notes.append(f"{adapter}: NOT VALID, its text without M {where}: too weak to show the fault")
                for run in changed:
                    reply = case[run][adapter]
                    sound = sound and reply["ok"] and without_m["ok"]
                    a_b = same(without_m, reply)
                    if not a_b["equal"]:
                        changed[run].append(adapter)
                        if run == "M_inside":
                            notes.append(f"{adapter}, M inside: its text changes at token {a_b['first_difference']}; M came into flight at its token {reply['m_from_token']}")
                        if run == "M_first":
                            b_c, a_c = same(m_text, reply), same(m_text, without_m)
                            notes.append(f"{adapter}, M first: its text changes at token {a_b['first_difference']}; against M's text of the prompt: {'THE SAME' if b_c['equal'] else 'the same up to token ' + str(b_c['first_difference'])} (without M: up to token {a_c['first_difference']})")
            expected = has_flag and int(k) in fault_counts(table)
            verdict = "FAIL" if changed["M_inside"] or (int(k) == 1 and changed["M_first"]) else "PASS, but a change in 'M first'" if changed["M_first"] else "PASS"
            verdicts[f"{name}/k{k}"] = {"verdict": verdict, "changed": changed, "flag": has_flag, "fault_reading_expects_fail": expected, "sound": sound,
                                         "valid": all(at is not None and at < CORRECT_LEAVE_BY for at in leaves.values()), "leaves_m_text_at_token": leaves}
            if verdict == "FAIL" and has_flag:
                faults.append(int(k))
            count = lambda run: f"{len(changed[run])} of {len(case['without_M'])}" + (f" ({', '.join(changed[run])})" if changed[run] else "")  # noqa: E731
            lines.append(f"  k = {k}: {verdict}. LoRAs with a changed text: M inside {count('M_inside')}; M first {count('M_first')}; no M, again {count('without_M_again')}"
                         + ("; the fault reading says FAIL here" if expected else "") + f"; without M the texts leave M's text at token {'/'.join(cell(at) for at in leaves.values())}" + ("" if sound else "  NOT SOUND: a request was short or the requests did not overlap as planned")
                         + "".join("\n      " + note for note in notes))
        if has_flag:
            lines.append(f"  {name}: FAIL at k = {faults}; the fault reading says {fault_counts(table)}. (A change in 'M first' only, at another k, is the same fault in a step that held fewer LoRAs.)" if faults else f"  {name}: no k fails")
    return lines, verdicts


def blind_cases(verdicts: dict[str, Any]) -> list[str]:
    """The cases with the flag, at a k where the fault reading says FAIL, that passed but could not have shown the fault."""
    return [f"{key} ({'not sound' if not verdict['sound'] else 'not valid: an adapter is too weak'})" for key, verdict in verdicts.items()
            if verdict["flag"] and verdict["fault_reading_expects_fail"] and verdict["verdict"] != "FAIL" and not (verdict["sound"] and verdict["valid"])]


def result_lines(results: dict[str, dict[str, Any]], verdicts: dict[str, Any], names: list[str]) -> list[str]:
    """The end lines: one for the test of the flag, one for each speed case of each config."""
    on = {key: verdict for key, verdict in verdicts.items() if verdict["flag"]}
    failed = {key: verdict for key, verdict in on.items() if verdict["verdict"] == "FAIL"}
    control = [key for key, verdict in verdicts.items() if not verdict["flag"] and verdict["verdict"] == "FAIL"]
    unsound = [key for key, verdict in verdicts.items() if not verdict["sound"]]
    slot = lambda adapter: f"{adapter} (slot {int(adapter.removeprefix('lora')) - 1})"  # noqa: E731
    if not on:
        test = "NOT RUN (no server with the flag ran the test)"
    elif failed and not control:
        test = "FAIL. With M in flight the text changed for: " + "; ".join(
            f"{key.replace('/k', ', k = ')}: {', '.join(slot(adapter) for adapter in (verdict['changed']['M_inside'] or verdict['changed']['M_first']))}" for key, verdict in failed.items())
    elif failed or control:
        test = f"NOT DECIDED. The control without the flag changed its text too ({', '.join(control)}); with the flag: {', '.join(failed) or 'no change'}"
    elif blind := blind_cases(verdicts):
        test = f"NOT DECIDED. No text changed, but these cases could not have shown the fault: {', '.join(blind)}. Run again with a larger --adapter-strength"
    else:
        test = f"PASS. No LoRA changed its text with M in flight, in {len(on)} cases with the flag" + ("" if any(not verdict["flag"] for verdict in verdicts.values()) else " (no control without the flag in this run)")
    lines = [f"RESULT: test of {FLAG}: {test}" + (f"  NOT SOUND in {', '.join(unsound)}" if unsound else "")]
    graphs = lambda result: f"{cell((result.get('start') or {}).get('graph_mode'))}, {cell((result.get('start') or {}).get('graph_capture_passes'))}, {cell((result.get('start') or {}).get('graph_capture_s'))} s, {cell((result.get('start') or {}).get('graph_capture_gib'))} GiB"  # noqa: E731
    lines.append("RESULT: recorded graphs (the flag acts only if a server with the flag records more than one without): " + "; ".join(f"{name}: {graphs(result)}" for name, result in results.items()))
    first: dict[str, float] = {}
    for name in names:
        for cases in (results.get(name, {}).get("phases") or {}).values():
            for case, got in cases.items():
                if not isinstance(got, dict) or "many_tok_s_sum" not in got:
                    continue
                step = got.get("many_ms_per_step")
                base = first.setdefault(case, step) if step else None
                against = f"; ms/step against {names[0]}: {100 * (step / base - 1):+.1f} %" if step and base and name != names[0] else ""
                lines.append(f"RESULT: speed {name}, {case}: {cell(step)} ms/step, {cell(got['many_tok_s_sum'])} tok/s in {got['many_streams']} streams, "
                             f"LoRA alone {cell(got.get('alone_lora_tok_s'))} tok/s, M alone {cell(got['alone_tok_s'][0])} tok/s{against}")
    return lines


# ---------------------------------------------------------------- one server


def start_facts(log: str) -> dict[str, Any]:
    """The facts of the start that the server log gives."""
    captured: dict[str, int] = {}
    passes: list[list[Any]] = []  # vLLM captures two times: first a sample to measure memory (2 FULL graphs), then all graphs
    for mode, total in re.findall(r"\((PIECEWISE|FULL)\)[^\r\n]*?\d+/(\d+)", log):
        captured[mode] = max(captured.get(mode, 0), int(total))
        if not passes or passes[-1] != [mode, int(total)]:
            passes.append([mode, int(total)])
    modes = re.findall(r"cudagraph_mode['\"]?\s*[=:]\s*<?CUDAGraphMode\.(\w+)", log)
    finished = re.search(r"Graph capturing finished in (\d+) secs, took ([-\d.]+) GiB", log)
    speculative = re.search(r"speculative_config=(None|\w+\([^)]{0,120})", log)
    kv_tokens = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
    kv_gib = re.findall(r"Available KV cache memory: ([-\d.]+) GiB", log)
    return {
        "graph_capture_gib": float(finished.group(2)) if finished else None,
        "kv_cache_tokens": int(kv_tokens[-1].replace(",", "")) if kv_tokens else None,
        "kv_cache_gib": float(kv_gib[-1]) if kv_gib else None,
        "graph_mode": modes[-1] if modes else None,
        "graph_sizes_captured": captured or None,
        "graph_capture_passes": passes[:8] or None,
        "graph_capture_s": int(finished.group(1)) if finished else None,
        "enforce_eager": (re.findall(r"enforce_eager=(\w+)", log) or [None])[-1],
        "attention_backends": sorted(set(re.findall(r"\b(FLASHINFER|FLASH_ATTN|TRITON_ATTN|FLEX_ATTENTION|XFORMERS)\b", log))) or None,
        "speculative_config": speculative.group(1) if speculative else None,
        "runner_v1_fallback": bool(re.search(r"[Ff]alling back to (the )?V1|V1 model runner", log)),
        "warnings_cudagraph": sorted({line.strip()[-160:] for line in re.findall(r"WARNING[^\r\n]*[Cc]udagraph[^\r\n]*", log)})[:3],
    }


def gpu_used_mib(gpu: int) -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60, check=True).stdout
        return next(int(line.split(",")[1]) for line in out.splitlines() if line.strip() and int(line.split(",")[0]) == gpu)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError, StopIteration):
        return None


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
        argv = [sys.executable, *config_argv(name, args.head, args.port, params, args.table)[1:]]
        out["flags"] = [flag for flag in argv[argv.index("--port") + 2 :]]
        started = time.monotonic()
        # vLLM 0.32.0 makes the route /v1/load_lora_adapter only with this variable. An older checkout of the box code does not set it.
        server = Child(argv, server_env(args.gpu, work) | {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"}, log_path)
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
        out["start"] = start_facts(read_log()) | {"gpu": args.gpu, "gpu_used_mib": gpu_used_mib(args.gpu)}
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
            # All adapters of the table are loaded in each case, as in a round: a later look has fewer LoRAs in flight, not a smaller table.
            mix = {f"M_and_{N_ADAPTERS}": [NULL_MODEL, *adapters], "M_and_2": [NULL_MODEL, *adapters[:2]], "M_and_1": [NULL_MODEL, adapters[0]]}
            phases = [(f"{N_ADAPTERS}_adapters", N_ADAPTERS, {case: mix[case] for case in args.cases.split(",") if case in mix})]
        loaded = 0
        for phase, wanted, targets in phases:
            for adapter in adapters[loaded:wanted]:
                post_json(args.port, "/v1/load_lora_adapter", {"lora_name": adapter, "lora_path": str(temp / "adapters" / adapter)}, 600.0)
            loaded = max(loaded, wanted)
            if lora and "correctness" not in out and not args.no_correctness:  # before any speed number
                try:
                    out["correctness"] = correctness(args.port, adapters, params)
                except Exception as exc:  # noqa: BLE001 - the speed part still runs
                    out["correctness"] = {"error": error_line(f"{type(exc).__name__}: {exc}", args.head)}
                (temp / f"{name}.correctness.json").write_text(json.dumps(out["correctness"]))
            out["phases"][phase] = {}
            for target, models in ({} if args.no_speed else targets).items():
                try:
                    out["phases"][phase][target] = measure_target(args.port, models, words, params, args.streams, args.decode_tokens, args.request_timeout_s)
                except Exception as exc:  # noqa: BLE001 - one failed target must not stop the others
                    out["phases"][phase][target] = {"error": error_line(f"{type(exc).__name__}: {exc}", args.head)}
                if not server.alive():
                    raise RuntimeError("the server stopped during the measurement: " + error_line(read_log(), args.head))
        out["start"]["gpu_used_mib_end"] = gpu_used_mib(args.gpu)
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
    rows = [("config", "case", "M alone", "LoRA alone", "1 of N", "sum of N", "ms/step", "window s", "ctx", "twice", "in N", "M vs base", "LoRA vs base")]
    failed = [f"{name}: FAILED: {result.get('error')}" for name, result in results.items() if not result.get("ok")]
    for name, result in results.items():
        for phase, targets in result["phases"].items():
            for target, got in targets.items():
                if "error" in got:
                    failed.append(f"{name} {phase} {target}: FAILED: {got['error']}")
                    continue
                equal = lambda item: "-" if item is None else "same" if item["equal"] else f"differ@{item['first_difference']}"  # noqa: E731
                rows.append((name, target, cell(got["alone_tok_s"][0]), cell(got.get("alone_lora_tok_s")), cell(got["many_tok_s_per_stream_median"]), cell(got["many_tok_s_sum"]), cell(got.get("many_ms_per_step")),
                             cell(got["many_window_s"]), cell(got["prompt_tokens_min"]), equal(got["same_alone_twice"]), equal(got["same_alone_and_among_many"]),
                             equal(compare.get(f"{name}/{phase}/{target}/M")), equal(compare.get(f"{name}/{phase}/{target}/lora"))))
    widths = [max(len(row[at]) for row in rows) for at in range(len(rows[0]))]
    return "\n".join(["  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() for row in rows] + failed)


def start_table(results: dict[str, dict[str, Any]]) -> str:
    rows = [("config", "gpu", "start s", "graph s", "graph GiB", "graphs", "KV tokens", "KV GiB", "GPU MiB", "flags changed")]
    for name, result in results.items():
        start = result.get("start") or {}
        rows.append((name, cell(start.get("gpu")), cell(result.get("start_s")), cell(start.get("graph_capture_s")), cell(start.get("graph_capture_gib")), cell(start.get("graph_sizes_captured")),
                     cell(start.get("kv_cache_tokens")), cell(start.get("kv_cache_gib")), cell(start.get("gpu_used_mib")), cell(result.get("change"))))
    widths = [max(len(row[at]) for row in rows) for at in range(len(rows[0]))]
    return "\n".join("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", help="directory of the head (default: found below SN120_WORK_DIR or /root/sn120-run/work)")
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS), help=f"configs to run, in order of importance (default: all but f2; known: {', '.join(CONFIGS)})")
    parser.add_argument("--gpus", help="GPU numbers to use, for example 0,1,2 (default: every GPU that is free)")
    parser.add_argument("--streams", type=int, default=STREAMS)
    parser.add_argument("--no-correctness", action="store_true", help="skip the correctness test of the flag")
    parser.add_argument("--no-speed", action="store_true", help="only the correctness test")
    parser.add_argument("--adapter-strength", type=float, default=ADAPTER_STRENGTH, help="the factor of the matrix B of the made-up adapters; a larger one makes a LoRA's text leave M's text sooner")
    parser.add_argument("--table", type=int, default=N_ADAPTERS, help="--max-loras of the servers: the entries of a round's ticket")
    parser.add_argument("--cases", default=f"M_and_{N_ADAPTERS},M_and_1,M_and_2", help="the models in flight, in order of importance")
    parser.add_argument("--context-tokens", type=int, default=CONTEXT_TOKENS)
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    parser.add_argument("--start-timeout-s", type=int, default=900)
    parser.add_argument("--request-timeout-s", type=int, default=REQUEST_TIMEOUT_S, help="time limit of one request, in seconds")
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
        config_argv(name, str(head), 0, params, args.table)  # stops when the box command lost a flag
    own_flag = FLAG in server_argv(str(head), 0, params)
    shapes = target_shapes(head, params.recipe.target_modules)
    if not shapes:
        fail("the head has no weight with the name of a target module of the recipe")
    body = chat_body(NULL_MODEL, [], 0, params)
    waves = [names[at : at + len(free)] for at in range(0, len(names), len(free))]
    print(f"decode microbench: vLLM {version}{'' if version == VLLM_VERSION else f' (WARNING: the box has {VLLM_VERSION})'}; {len(kinds)} GPUs ({kinds[0]}), {len(free)} free; "
          f"head with {len(shapes)} target layers; {len(names)} configs in {len(waves)} wave(s); table {args.table}; {args.streams} streams of {args.context_tokens} tokens; "
          f"the checkout's own server command has {FLAG}: {'yes' if own_flag else 'no'}", flush=True)

    temp = Path(tempfile.mkdtemp(prefix="decode-microbench-"))
    (temp / "logs").mkdir()
    children: list[subprocess.Popen] = []

    def stop(*_: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    results: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    try:
        size = sum(write_adapter(temp / "adapters" / f"lora{index}", shapes, params.recipe.rank, params.recipe.alpha, params.recipe.target_modules, SEED + index, args.adapter_strength) for index in range(1, N_ADAPTERS + 1))
        print(f"{N_ADAPTERS} made-up adapters written (rank {params.recipe.rank}, strength {args.adapter_strength:g}, {size / 2**20:.0f} MiB in all); temp directory {temp}", flush=True)
        for number, wave in enumerate(waves, 1):
            print(f"wave {number}: starting {', '.join(wave)} (one GPU each); the servers need some minutes to start", flush=True)
            children = []
            for at, name in enumerate(wave):
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", name, "--temp", str(temp), "--head", str(head), "--gpu", str(free[at]), "--port", str(args.first_port + at)]
                command += ["--no-correctness"] * args.no_correctness + ["--no-speed"] * args.no_speed + ["--table", str(args.table), "--cases", args.cases, "--streams", str(args.streams), "--context-tokens", str(args.context_tokens), "--decode-tokens", str(args.decode_tokens), "--start-timeout-s", str(args.start_timeout_s), "--request-timeout-s", str(args.request_timeout_s)]
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
    replies = {}
    for name, result in results.items():
        for phase, targets in result["phases"].items():
            for target, got in targets.items():
                for kind, field in (("M", "_reply"), ("lora", "_reply_lora")):
                    if got.get(field):
                        replies[f"{name}/{phase}/{target}/{kind}"] = got[field]
                        got[f"reply_{kind}_sha256"] = hashlib.sha256(json.dumps(got[field]["ids"] or got[field]["text"]).encode()).hexdigest()[:16]
                    got.pop(field, None)
    # The reference is the first config of the list (the base), in its first case. The request is the same in each case and config.
    first = next((key.rsplit("/", 1)[0] for key in replies if key.startswith(f"{names[0]}/")), None)
    compare = {key: same(replies[f"{first}/{key.rsplit('/', 1)[1]}"], reply) for key, reply in replies.items() if first and f"{first}/{key.rsplit('/', 1)[1]}" in replies and key.rsplit("/", 1)[0] != first}
    report = {
        "script": "decode_microbench", "vllm": version, "gpus": kinds, "gpus_free": len(free), "wall_s": round(time.monotonic() - started),
        "sampling": {key: body[key] for key in ("temperature", "top_p", "top_k", "seed") if key in body} | {"seed": f"{SEED} + stream number", "ignore_eos": True},
        "decode_tokens": args.decode_tokens, "context_tokens_wanted": args.context_tokens, "adapters": {"count": N_ADAPTERS, "rank": params.recipe.rank, "layers": len(shapes), "made_up": True, "strength": args.adapter_strength}, "table": args.table, "own_command_has_flag": own_flag,
        "proves": {name: CONFIGS[name][1] for name in names}, "same_as_base": compare, "configs": results,
    }
    (temp / "result.json").write_text(json.dumps(report, indent=1))
    lines, verdicts = judge_correctness(results, args.table)
    for result in results.values():  # the token lists stay out of the JSON file
        test = result.get("correctness") or {}
        for reply in [*(test.get("M") or {}).values(), *(reply for case in (test.get("cases") or {}).values() for run in case.values() for reply in run.values())]:
            reply["sha256"] = hashlib.sha256(json.dumps(reply.pop("ids") or reply["text"]).encode()).hexdigest()[:16]
            reply.pop("text")
    report["correctness_verdicts"] = verdicts
    (temp / "result.json").write_text(json.dumps(report, indent=1))
    print(f"\ncorrectness of {FLAG} (temperature 0, a fixed seed, {CORRECT_TOKENS} tokens a request, one server): the first k LoRAs in flight at once; a LoRA's text without M against its text with {CORRECT_M_ROWS} requests of M in flight.\n"
          f"'M inside' = M comes in when all k LoRAs write and ends before them: each step with M holds exactly k LoRAs; this run decides. 'M first' = M in flight for the whole reply; the LoRAs can come in one step apart, so for k above 1 a step can hold fewer LoRAs.")
    print("\n".join(lines + [f"{name}: the test FAILED to run: {result['correctness']['error']}" for name, result in results.items() if (result.get("correctness") or {}).get("error")]) or "not run")
    print("\nstart of each server ('KV tokens' = the room for stored context; 'GPU MiB' = memory in use on the GPU after the start)")
    print(start_table(results))
    print(f"\ntokens per second after the prompt pass; N = {args.streams} streams, stream i on model i mod (models of the case); 'ms/step' = wall time of the N-stream run over its decode steps;\n"
          f"'window s' = seconds in which all N streams decode (below 2: the N-stream speeds are not clean); 'twice' = the same request on M alone two times; 'in N' = alone against among N;\n"
          f"'M vs base', 'LoRA vs base' = the same seeded request alone, against the first case of {names[0]}")
    print(table(results, compare))
    print(f"\nJSON: {temp / 'result.json'}   server logs: {temp / 'logs'}   (paste the output and the JSON file)\n")
    print("\n".join(result_lines(results, verdicts, names)), flush=True)
    return 0 if results and all(result["ok"] for result in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
