"""Entry point: feed DISPATCH → compute flip shadow scores → publish scores-shadow.json.

Honors config.enabled=false (still can dry-run compute to audit_out if --force-audit).
Never modifies LIVE scores.json / weights / chain.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import dispatch_feed
import flip_scorer



def _refresh_verify_ledger(cfg):
    """Rebuild sampling/TOPLOC verify ledger before each score (ops gate)."""
    import subprocess, sys
    paths = cfg.get("paths") or {}
    out = paths.get("verify_ledger")
    if not out:
        return
    builder = "/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/build_verify_ledger.py"
    denylist = "/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/ops-verify-denylist.json"
    train_out = paths.get("train_verify_fail_denylist") or "/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/train-verify-fail-denylist.json"
    epochs = paths.get("epochs_root")
    if not epochs or not Path(builder).exists():
        return
    cmd = [sys.executable, "-B", builder, "--epochs-root", epochs, "--denylist", denylist,
           "--min-epoch", str(int((cfg.get("formula") or {}).get("min_epoch") or 93)),
           "--out", out, "--train-denylist-out", train_out]
    try:
        subprocess.run(cmd, check=False, timeout=120)
    except Exception:
        pass

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.json"))
    ap.add_argument("--once", action="store_true", help="run one shot and exit (default)")
    ap.add_argument("--loop", action="store_true", help="loop every timer_seconds")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-dispatch-feed", action="store_true")
    args = ap.parse_args(argv)

    def tick():
        cfg = flip_scorer.load_config(args.config)
        result = dict(at_cr=flip_scorer.cr(), enabled=cfg.get("enabled"), live_weights=False)
        if not args.skip_dispatch_feed:
            try:
                result["dispatch_feed"] = dispatch_feed.feed(cfg)
            except Exception as e:
                result["dispatch_feed_error"] = repr(e)
        if not cfg.get("enabled", True) and not args.dry_run:
            # still write audit-only so Jacob can inspect, but skip website
            cfg = dict(cfg)
            pub = dict(cfg.get("publish") or {})
            pub["website_scores_shadow"] = False
            cfg["publish"] = pub
            result["note"] = "enabled=false: audit write only, website skipped"
        if args.dry_run:
            cfg = dict(cfg)
            cfg["publish"] = dict(cfg.get("publish") or {}, website_scores_shadow=False)
        _refresh_verify_ledger(cfg)
        doc, detail = flip_scorer.compute(cfg)
        info = flip_scorer.publish(cfg, doc, detail)
        result.update(
            score_sha256=doc.get("score_sha256"),
            n_miners=len(doc.get("miners") or []),
            n_pairs=(doc.get("window") or {}).get("n_pairs"),
            published_epochs=(doc.get("window") or {}).get("published_epochs"),
            gaps=doc.get("gaps"),
            top=[
                dict(
                    uid=m.get("uid"),
                    share=round(m.get("share_now") or 0, 6),
                    flip=round(m.get("flip_score") or 0, 4),
                    f2p=m.get("n_fail_to_pass"),
                    p2f=m.get("n_pass_to_fail"),
                )
                for m in (doc.get("miners") or [])[:10]
            ],
            publish=info,
        )
        # append run log
        audit = Path(cfg["paths"]["audit_out"])
        audit.mkdir(parents=True, exist_ok=True)
        with open(audit / "runs.jsonl", "a") as f:
            f.write(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2))
        return result

    if args.loop:
        cfg0 = flip_scorer.load_config(args.config)
        period = int(cfg0.get("timer_seconds") or 300)
        while True:
            try:
                tick()
            except Exception:
                traceback.print_exc()
            time.sleep(period)
    else:
        tick()


if __name__ == "__main__":
    main()
