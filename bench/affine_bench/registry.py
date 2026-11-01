"""Benchmark / model / serving registries (bench/config/*.toml)."""
import os, tomllib
CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


def _load(name):
    with open(os.path.join(CFG, name), "rb") as f:
        return tomllib.load(f)


def serving():
    return _load("serving.toml")


def benchmarks(enabled_only=False, runnable_only=False):
    d = _load("benchmarks.toml"); dflt = d.get("defaults", {})
    out = []
    for b in d["benchmark"]:
        b = {**dflt, **b}
        if enabled_only and not b.get("enabled"):
            continue
        if runnable_only and "env" not in b:
            continue
        out.append(b)
    return sorted(out, key=lambda b: b.get("order", 999))


def benchmark(bid):
    return next(b for b in benchmarks() if b["id"] == bid)


def models():
    return sorted(_load("models.toml")["model"], key=lambda m: m.get("priority", 999))


def bench_version(b):
    """Results are comparable only within (benchmark id, version, serving version)."""
    return f"{b['id']}/v{b.get('version', 1)}/{serving()['version']}"
