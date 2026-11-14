#!/usr/bin/env python3
"""TP2 against TP1 for the tail of a look. Run it on a spare GPU box, never on the box of a live round.

    python v2/bench/tp2_tail_bench.py --gpus 5,6,7

It starts two vLLM servers with the server command and the request body of the box's own code
(``sn120_v2/gpu/evaluator.py``), the LoRA table of a round (6) and M plus 6 made-up LoRAs in flight:

    tp1   one GPU, the command as it is (prod's layout: one server for each GPU)
    tp2   two GPUs, the same command plus ``--tensor-parallel-size 2`` (the layout DP4 x TP2), 512 seats

Each server gets the same load for each of its GPUs. The points are the states of a look at the 64k cap:

    tail   few long rollouts are left: 1 stream alone, and 16 streams a GPU, at a context of 48,000 tokens
    load   the main part of a block: 128 and 256 streams a GPU at a context of 8,000 tokens

The speed of one stream in the tail is the speed of the last rollouts. It also tests whether a seeded request
gives the same tokens on the two servers.

It writes only below one temp directory. It reads no state, journal, store or chain. It opens no
connection except to its own servers on this machine. It stops its servers at the end and on Ctrl-C.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # decode_microbench.py lies beside this file; it puts the checkout's v2/ on the path
import decode_microbench as mb  # noqa: E402

LAYOUTS = {"tp1": 1, "tp2": 2}  # name -> GPUs of the server
TAIL_CONTEXT, LOAD_CONTEXT = 48000, 8000
TAIL_PER_GPU, LOAD_PER_GPU = (16,), (128, 256)
DECODE_TOKENS = 384


def layout_argv(name: str, head: str, port: int, params: Any, table: int, flag: bool) -> list[str]:
    argv = mb.with_flag(mb.config_argv("base", head, port, params, table), flag)
    if name == "tp2":
        argv = mb.set_flag(argv, "--max-num-seqs", str(2 * int(argv[argv.index("--max-num-seqs") + 1]))) + ["--tensor-parallel-size", "2"]
    return argv


def points(gpus: int, tail_context: int, load_context: int) -> list[tuple[str, int, int]]:
    """(name, streams, context tokens). The name gives the streams for each GPU, so the two layouts meet in one row."""
    out = [("tail_1_alone", 1, tail_context)]
    out += [(f"tail_{per_gpu}_a_gpu", per_gpu * gpus, tail_context) for per_gpu in TAIL_PER_GPU]
    out += [(f"load_{per_gpu}_a_gpu", per_gpu * gpus, load_context) for per_gpu in LOAD_PER_GPU]
    return out


def worker(args: argparse.Namespace) -> int:
    """Start one server, measure its points, write ``<temp>/<layout>.json``, stop the server."""
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    name, temp, params = args.worker, Path(args.temp), mb.Params()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)  # 512 streams: one open file for each, here and in the server
    wanted = 65536 if hard == resource.RLIM_INFINITY else min(hard, 65536)
    if wanted > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
    gpus = [int(item) for item in args.gpus.split(",")]
    work, log_path = temp / name, temp / "logs" / f"{name}.log"
    work.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"layout": name, "gpus": gpus, "ok": False, "points": {}}
    server = None
    read_log = lambda: log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""  # noqa: E731
    try:
        argv = [sys.executable, *layout_argv(name, args.head, args.port, params, args.table, args.flag)[1:]]
        out["flags"] = argv[argv.index("--port") + 2 :]
        env = mb.server_env(gpus[0], work) | {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1", "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
        started = time.monotonic()
        server = mb.Child(argv, env, log_path)
        (temp / f"{name}.pid").write_text(str(server.proc.pid))
        while True:
            try:
                mb.post_json(args.port, "/health", None, 10.0)
                break
            except (OSError, ValueError):
                if not server.alive():
                    raise RuntimeError("the server stopped at the start: " + mb.error_line(read_log(), args.head)) from None
                if time.monotonic() - started > args.start_timeout_s:
                    raise RuntimeError(f"the server was not ready after {args.start_timeout_s} s") from None
                time.sleep(3.0)
        out["start_s"] = round(time.monotonic() - started, 1)
        out["start"] = mb.start_facts(read_log()) | {"gpu_used_mib": [mb.gpu_used_mib(gpu) for gpu in gpus]}
        try:  # the tokens of one word, so the prompt has the wanted size
            per_word = int(mb.post_json(args.port, "/tokenize", {"model": mb.NULL_MODEL, "prompt": mb.prompt_words(0, 2000)}, 60.0)["count"]) / 2000
        except (OSError, KeyError, ValueError, TypeError):
            per_word = 1.6
        adapters = [f"lora{index}" for index in range(1, mb.N_ADAPTERS + 1)]
        for adapter in adapters:
            mb.post_json(args.port, "/v1/load_lora_adapter", {"lora_name": adapter, "lora_path": str(temp / "adapters" / adapter)}, 600.0)
        models = [mb.NULL_MODEL, *adapters]
        for point, streams, context in points(len(gpus), args.tail_context, args.load_context):
            began = time.monotonic()
            try:
                # The point 'alone' runs with two streams, so that the function also measures a LoRA alone (stream 1) after M alone (stream 0).
                got = mb.measure_target(args.port, models, int(context / per_word), params, max(streams, 2) if point == "tail_1_alone" else streams, args.decode_tokens, args.request_timeout_s)
                got["streams_per_gpu"] = streams / len(gpus)
            except Exception as exc:  # noqa: BLE001 - one failed point must not stop the others
                got = {"error": mb.error_line(f"{type(exc).__name__}: {exc}", args.head)}
            got["point_s"] = round(time.monotonic() - began, 1)
            out["points"][point] = got
            (temp / f"{name}.json").write_text(json.dumps(out))
            if not server.alive():
                raise RuntimeError("the server stopped during the measurement: " + mb.error_line(read_log(), args.head))
        out["ok"] = True
    except KeyboardInterrupt:
        out["error"] = "interrupted"
    except Exception as exc:  # noqa: BLE001 - the cause goes to the table in one line
        out["error"] = mb.error_line(f"{type(exc).__name__}: {exc}", args.head)
    finally:
        if server is not None:
            server.stop()
        shutil.rmtree(work, ignore_errors=True)
        (temp / f"{name}.json").write_text(json.dumps(out))
    return 0 if out["ok"] else 1


def text_table(rows: list[tuple[Any, ...]]) -> str:
    widths = [max(len(str(row[at])) for row in rows) for at in range(len(rows[0]))]
    return "\n".join("  ".join(str(value).ljust(width) for value, width in zip(row, widths)).rstrip() for row in rows)


def percent(new: float | None, old: float | None) -> str:
    return f"{100 * (new / old - 1):+.1f} %" if new and old else "-"


def result_lines(results: dict[str, dict[str, Any]], compare: dict[str, str]) -> list[str]:
    """The end lines: TP2 against TP1 at the same load for each GPU."""
    got = lambda layout, point: (results.get(layout, {}).get("points") or {}).get(point) or {}  # noqa: E731
    lines = []
    one, two = got("tp1", "tail_1_alone"), got("tp2", "tail_1_alone")
    for label, field in (("M", "alone_tok_s"), ("a LoRA", "alone_lora_tok_s")):
        a, b = (item.get(field)[0] if isinstance(item.get(field), list) else item.get(field) for item in (one, two))
        lines.append(f"RESULT: the last rollout alone, {label}, context {mb.cell(one.get('prompt_tokens_min'))}: tp1 {mb.cell(a)} tok/s, tp2 {mb.cell(b)} tok/s, tp2 against tp1 {percent(b, a)}")
    for point in [name for name in (results.get("tp1", {}).get("points") or {}) if name != "tail_1_alone"]:
        one, two = got("tp1", point), got("tp2", point)
        a, b = one.get("many_tok_s_per_stream_median"), two.get("many_tok_s_per_stream_median")
        sums = [item.get("many_tok_s_sum") for item in (one, two)]
        lines.append(f"RESULT: {point}, context {mb.cell(one.get('prompt_tokens_min'))}: one stream tp1 {mb.cell(a)} tok/s, tp2 {mb.cell(b)} tok/s, tp2 against tp1 {percent(b, a)}; "
                     f"a GPU tp1 {mb.cell(sums[0])} tok/s, tp2 {mb.cell(round(sums[1] / 2, 1) if sums[1] else None)} tok/s; ms/step tp1 {mb.cell(one.get('many_ms_per_step'))}, tp2 {mb.cell(two.get('many_ms_per_step'))}")
    lines.append("RESULT: the same seeded request on tp2 and on tp1: " + (", ".join(f"{key} {value}" for key, value in compare.items()) or "not compared"))
    lines.append("RESULT: start: " + ", ".join(f"{name} {mb.cell(result.get('start_s'))} s, room for {mb.cell((result.get('start') or {}).get('kv_cache_tokens'))} tokens of context" for name, result in results.items()))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", help="directory of the head (default: found below SN120_WORK_DIR or /root/sn120-run/work)")
    parser.add_argument("--gpus", required=True, help="three free GPUs, for example 5,6,7: the first for tp1, the next two for tp2")
    parser.add_argument("--layouts", default="tp1,tp2")
    parser.add_argument("--flag", action="store_true", help=f"with {mb.FLAG} on both servers (default: without it, as prod has it today)")
    parser.add_argument("--table", type=int, default=mb.N_ADAPTERS, help="--max-loras of the servers: the entries of a round's ticket")
    parser.add_argument("--tail-context", type=int, default=TAIL_CONTEXT)
    parser.add_argument("--load-context", type=int, default=LOAD_CONTEXT)
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    parser.add_argument("--start-timeout-s", type=int, default=1500)
    parser.add_argument("--request-timeout-s", type=int, default=mb.REQUEST_TIMEOUT_S)
    parser.add_argument("--first-port", type=int, default=8320)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--temp", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if mb.IMPORT_ERROR:
        mb.fail(f"the checkout or the install is not complete ({mb.IMPORT_ERROR}); run with the Python of the box install, from the checkout")
    if args.worker:
        return worker(args)

    params, names = mb.Params(), [name for name in args.layouts.split(",") if name]
    if unknown := [name for name in names if name not in LAYOUTS]:
        mb.fail(f"unknown layout {unknown}; known: {', '.join(LAYOUTS)}")
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        mb.fail("vLLM is not installed in this Python; run with the Python of the box install")
    try:
        free, kinds = mb.free_gpus(mb.ServeSettings().free_limit_mib)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        mb.fail(f"nvidia-smi gave no GPU list ({type(exc).__name__})")
    wanted = [int(item) for item in args.gpus.split(",")]
    if busy := [gpu for gpu in wanted if gpu not in free]:
        mb.fail(f"GPU {busy} is not free (a process is on it); give other GPUs with --gpus")
    if len(wanted) < sum(LAYOUTS[name] for name in names):
        mb.fail(f"{', '.join(names)} need {sum(LAYOUTS[name] for name in names)} GPUs; --gpus gives {len(wanted)}")
    head = mb.find_head(args.head)
    shapes = mb.target_shapes(head, params.recipe.target_modules)
    if not shapes:
        mb.fail("the head has no weight with the name of a target module of the recipe")
    given, at = {}, 0
    for name in names:
        given[name] = wanted[at : at + LAYOUTS[name]]
        at += LAYOUTS[name]
        layout_argv(name, str(head), 0, params, args.table, args.flag)  # stops when the box command lost a flag
    print(f"tp2 tail bench: vLLM {version}{'' if version == mb.VLLM_VERSION else f' (WARNING: the box has {mb.VLLM_VERSION})'}; {kinds[0]}; " + "; ".join(f"{name} on GPU {given[name]}" for name in names)
          + f"; table {args.table}, M and {mb.N_ADAPTERS} LoRAs in flight; {mb.FLAG}: {'on' if args.flag else 'off'}; tail context {args.tail_context}, load context {args.load_context}, {args.decode_tokens} tokens a stream", flush=True)

    temp = Path(tempfile.mkdtemp(prefix="tp2-tail-bench-"))
    (temp / "logs").mkdir()
    children: dict[str, subprocess.Popen] = {}

    def stop(*_: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    results: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    try:
        for index in range(1, mb.N_ADAPTERS + 1):
            mb.write_adapter(temp / "adapters" / f"lora{index}", shapes, params.recipe.rank, params.recipe.alpha, params.recipe.target_modules, mb.SEED + index)
        print(f"{mb.N_ADAPTERS} made-up adapters written (rank {params.recipe.rank}); temp directory {temp}; the servers need some minutes to start", flush=True)
        for at, name in enumerate(names):
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", name, "--temp", str(temp), "--head", str(head), "--gpus", ",".join(map(str, given[name])), "--port", str(args.first_port + at)]
            command += ["--flag"] * args.flag + ["--table", str(args.table), "--tail-context", str(args.tail_context), "--load-context", str(args.load_context), "--decode-tokens", str(args.decode_tokens)]
            command += ["--start-timeout-s", str(args.start_timeout_s), "--request-timeout-s", str(args.request_timeout_s)]
            children[name] = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=open(temp / "logs" / f"{name}.worker.log", "ab"))
        pending = dict(children)
        while pending:
            time.sleep(5.0)
            for name in [name for name, child in pending.items() if child.poll() is not None]:
                del pending[name]
                path = temp / f"{name}.json"
                results[name] = json.loads(path.read_text()) if path.exists() else {"layout": name, "ok": False, "points": {}, "error": "the worker stopped with no result"}
                print(f"  {name}: {'done' if results[name]['ok'] else 'FAILED: ' + str(results[name].get('error'))} ({time.monotonic() - started:.0f} s)", flush=True)
    except KeyboardInterrupt:
        print("interrupted: stopping the servers", flush=True)
    finally:
        for child in children.values():
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
        for child in children.values():
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
    for name in names:  # a layout that was interrupted still shows what it measured
        if name not in results and (temp / f"{name}.json").exists():
            results[name] = json.loads((temp / f"{name}.json").read_text())

    # The same seeded request alone (M, and a LoRA) on tp2 against tp1, at each context.
    replies: dict[str, dict[str, Any]] = {}
    for name, result in results.items():
        for point, got in result["points"].items():
            for kind, field in (("M", "_reply"), ("LoRA", "_reply_lora")):
                if got.get(field):
                    replies[f"{name}/{point}/{kind}"] = got[field]
                    got[f"reply_{kind}_sha256"] = hashlib.sha256(json.dumps(got[field]["ids"] or got[field]["text"]).encode()).hexdigest()[:16]
                got.pop(field, None)
    compare = {}
    for key, reply in replies.items():
        layout, rest = key.split("/", 1)
        if layout == "tp2" and f"tp1/{rest}" in replies:
            verdict = mb.same(replies[f"tp1/{rest}"], reply)
            compare[rest] = "same" if verdict["equal"] else f"differ at token {verdict['first_difference']}"
    report = {"script": "tp2_tail_bench", "vllm": version, "gpus": kinds, "wall_s": round(time.monotonic() - started, 1), "flag": args.flag, "table": args.table, "same_tp2_as_tp1": compare, "layouts": results}
    (temp / "result.json").write_text(json.dumps(report, indent=1))

    rows: list[tuple[Any, ...]] = [("layout", "point", "streams", "a GPU", "ctx", "M alone", "LoRA alone", "1 of N", "slowest", "sum of N", "sum a GPU", "ms/step", "window s", "twice", "in N", "point s")]
    failed: list[str] = []
    for name, result in results.items():
        for point, got in result["points"].items():
            if "error" in got:
                rows.append((name, point, "FAILED", *[""] * 13))
                failed.append(f"{name}, {point}: FAILED: {got['error'][:200]}")
                continue
            flag = lambda verdict: "same" if verdict["equal"] else f"differ@{verdict['first_difference']}"  # noqa: E731
            rows.append((name, point, got["many_streams"], mb.cell(got.get("streams_per_gpu")), got["prompt_tokens_min"], mb.cell(got["alone_tok_s"][0]), mb.cell(got.get("alone_lora_tok_s")), mb.cell(got["many_tok_s_per_stream_median"]),
                         mb.cell(got["many_tok_s_per_stream_min"]), mb.cell(got["many_tok_s_sum"]), mb.cell(round(got["many_tok_s_sum"] / LAYOUTS[name], 1) if got["many_tok_s_sum"] else None), mb.cell(got.get("many_ms_per_step")),
                         got["many_window_s"], flag(got["same_alone_twice"]), flag(got["same_alone_and_among_many"]), got["point_s"]))
    print("\ntokens per second after the prompt pass; stream i on model i mod 7 (M and 6 LoRAs); 'M alone', 'LoRA alone' = one stream with no other in flight; '1 of N', 'slowest' = one stream among N (median, lowest);\n"
          "'ms/step' = wall time of the N-stream run over its decode steps; 'window s' below 2: the N-stream speeds are not clean; 'twice' = the same request on M alone two times; 'in N' = alone against among N")
    print("\n".join([text_table(rows), *failed]))
    for name, result in results.items():
        start = result.get("start") or {}
        print(f"{name}: start {mb.cell(result.get('start_s'))} s, graphs {mb.cell(start.get('graph_capture_s'))} s, room for {mb.cell(start.get('kv_cache_tokens'))} tokens of context, GPU MiB {mb.cell(start.get('gpu_used_mib'))}"
              + (f", FAILED: {result['error']}" if result.get("error") else ""))
    print(f"\nJSON: {temp / 'result.json'}   server logs: {temp / 'logs'}   (paste the output and the JSON file)\n")
    print("\n".join(result_lines(results, compare)), flush=True)
    return 0 if results and all(result["ok"] for result in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
