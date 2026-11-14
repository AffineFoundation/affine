"""Benchmark / serving / per-model sampling registries (bench/config/*.toml)."""
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


def model_settings():
    """Per-model native sampling + budgets (bench-v2). See config/model_settings.toml."""
    return _load("model_settings.toml")


def resolve_settings(model_id, kind=None):
    """Sampling + max_model_len for one model. Kings / Genesis use the qwen3.5-family profile.
    max_tokens is capped at global_max_tokens_ceiling; max_model_len = min(budget + headroom, native_context)."""
    ms = model_settings()
    profiles = ms["profiles"]
    mmap = ms.get("models", {})
    if kind in ("king", "genesis") or (model_id or "").startswith(("king-", "genesis-")):
        pname = ms.get("king_profile") or ms.get("default_profile")
    else:
        pname = mmap.get(model_id) or ms.get("default_profile")
    if pname not in profiles:
        raise KeyError(f"unknown settings profile {pname!r} for model {model_id!r}")
    p = dict(profiles[pname])
    ceiling = int(ms.get("global_max_tokens_ceiling", 65536))
    headroom = int(ms.get("prompt_headroom", 4096))
    budget = min(int(p["max_tokens"]), ceiling)
    native = int(p["native_context"])
    ml = min(budget + headroom, native)
    out = dict(
        profile=pname,
        source=p.get("source", ""),
        max_tokens=budget,
        budget=budget,
        native_context=native,
        max_model_len=ml,
        temperature=float(p["temperature"]),
        top_p=float(p["top_p"]),
        top_k=int(p.get("top_k", -1)),
        min_p=float(p.get("min_p", 0.0)),
        presence_penalty=float(p.get("presence_penalty", 0.0)),
        repetition_penalty=float(p.get("repetition_penalty", 1.0)),
        enable_thinking=bool(p.get("enable_thinking", True)),
        # Never extend RoPE past native: budget+headroom is clamped above.
        hf_overrides="",
    )
    return out


def bench_version(b):
    """Results are comparable only within (benchmark id, version, serving version)."""
    return f"{b['id']}/v{b.get('version', 1)}/{serving()['version']}"
