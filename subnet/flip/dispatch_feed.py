"""Auto-feed recent DISPATCH-selected task indices into probe bandit dispatch boost file.

Writes state/dispatch/{env}.json with {"indices":[...]} so bandit top-k can boost them.
Does NOT change LIVE weights. Empty file previously meant no boost.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def cr(ts=None):
    ts = time.time() if ts is None else ts
    return time.strftime("%Y-%m-%d %H:%M:%S CR (UTC-6)", time.gmtime(ts - 6 * 3600))


def _atomic(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    tmp = path.with_name("." + path.name + ".tmp%d" % os.getpid())
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def latest_epochs(epochs_root: Path, n: int) -> list[int]:
    eps = []
    for p in Path(epochs_root).iterdir():
        if p.is_dir() and p.name.isdigit():
            if (p / "coordinator" / "DISPATCH.json").exists():
                eps.append(int(p.name))
    eps.sort()
    return eps[-n:] if n else eps


def collect_indices(epochs_root: Path, epochs: list[int]) -> tuple[list[int], dict]:
    indices = set()
    meta = {}
    for e in epochs:
        p = Path(epochs_root) / str(e) / "coordinator" / "DISPATCH.json"
        try:
            d = json.loads(p.read_text())
            sel = (d.get("payload") or d).get("selected") or []
            idxs = []
            for s in sel:
                try:
                    idxs.append(int(s["index"]))
                except Exception:
                    pass
            indices.update(idxs)
            meta[str(e)] = dict(n_selected=len(sel), n_indices=len(idxs))
        except Exception as ex:
            meta[str(e)] = dict(error=str(ex))
    return sorted(indices), meta


def feed(cfg: dict) -> dict:
    df = cfg.get("dispatch_feed") or {}
    if not df.get("enabled", True):
        return dict(skipped=True, reason="dispatch_feed.enabled=false")
    paths = cfg["paths"]
    env = cfg.get("env") or "affine_math"
    lookback = int(df.get("lookback_epochs", 2))
    epochs_root = Path(paths["epochs_root"])
    out_path = Path(paths["dispatch_feed"])
    epochs = latest_epochs(epochs_root, lookback)
    indices, meta = collect_indices(epochs_root, epochs)
    doc = dict(
        indices=indices,
        env=env,
        source_epochs=epochs,
        n_indices=len(indices),
        per_epoch=meta,
        note="DISPATCH-selected task indices for bandit score boost (not FIFO). Auto-fed by flip-shadow dispatch_feed.py.",
        updated_at=time.time(),
        updated_at_cr=cr(),
    )
    _atomic(out_path, doc)
    return dict(path=str(out_path), n_indices=len(indices), source_epochs=epochs, per_epoch=meta)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    print(json.dumps(feed(cfg), indent=2))


if __name__ == "__main__":
    main()
