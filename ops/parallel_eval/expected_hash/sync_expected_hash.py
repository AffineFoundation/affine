#!/usr/bin/env python3
"""Host-side sync: set pod HASH_FILE from published master of the current (or given) head.

Never weakens the check. If lookup fails, writes ALERT and leaves the previous hash file unchanged
(so the trainer keeps failing closed on mismatch rather than skipping the check).
Does NOT signal or restart the trainer; the live wrapper re-reads the file each turn.
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

BASE = Path("/home/const/subnet120-rewrite/state/root-audits/20261108-no-improvement-audit/fix-v2/expected-hash-dynamic")
sys.path.insert(0, str(BASE))
from lookup_expected_master import lookup, current_head, HEX64  # type: ignore

RSSH = "/home/const/subnet120-rewrite/state/root-audits/20261026-rollback-189de412/scripts/rssh.sh"
POD_OPS = "/workspace/affine-mimo-trainer-prod/ops"
POD_HASH = f"{POD_OPS}/fp32-expected-master.hash"
POD_ALERT = f"{POD_OPS}/fp32-expected-master.ALERT"
STATE = BASE / "state"
ALERT = STATE / "ALERT.json"
LAST = STATE / "LAST-SYNC.json"


def cr_now():
    return time.strftime("%Y-%m-%d %H:%M:%S CR", time.gmtime(time.time() - 6 * 3600))


def rssh(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", RSSH, cmd], capture_output=True, timeout=120)


def read_pod_hash() -> str | None:
    p = rssh(f"cat {POD_HASH} 2>/dev/null || true")
    v = (p.stdout or b"").decode().strip()
    return v if HEX64.match(v or "") else None


def write_pod_hash(digest: str) -> None:
    script = (
        f"umask 077; mkdir -p -m 700 {POD_OPS}; "
        f"tmp={POD_HASH}.tmp.$$; echo {digest} > $tmp; chmod 600 $tmp; mv -f $tmp {POD_HASH}; "
        f"rm -f {POD_ALERT}; echo OK"
    )
    p = rssh(script)
    if p.returncode != 0 or b"OK" not in (p.stdout or b""):
        raise RuntimeError(f"pod hash write failed rc={p.returncode}")


def write_pod_alert(msg: str) -> None:
    safe = "".join(c if c.isalnum() or c in "-_.:/" else "_" for c in msg)[:200]
    rssh(f"umask 077; mkdir -p -m 700 {POD_OPS}; echo {safe} > {POD_ALERT}")


def unfinished_training() -> list[int]:
    life = Path("/home/const/subnet120-rewrite/state/mimo-mainnet-v1")
    cur = json.loads((life / "CURRENT.json").read_text())
    e = int(cur["epoch_index"])
    bad = []
    for n in range(max(0, e - 1), e + 3):
        d = life / "epochs" / str(n)
        if (d / "TRAINING-JOB.json").exists() and not (d / "TRAINING-RESULT.json").exists():
            bad.append(n)
    return bad


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    force_head = None
    for a in argv[1:]:
        if a.startswith("--head="):
            force_head = a.split("=", 1)[1]
        elif a != "--dry-run" and HEX64.match(a):
            force_head = a
    head = force_head or current_head()
    res = lookup(head)
    prev = read_pod_hash()
    out = {
        "at_cr": cr_now(),
        "head": head,
        "lookup": res,
        "previous_pod_hash": prev,
        "dry_run": dry,
        "unfinished": unfinished_training(),
        "action": None,
    }
    if not res.get("ok"):
        out["action"] = "alert-fail-closed-keep-previous"
        ALERT.write_text(json.dumps(out, indent=2) + "\n")
        if not dry:
            write_pod_alert(res.get("alert") or "lookup-failed")
        print(json.dumps(out, indent=2))
        return 2
    digest = res["master_digest"]
    if prev == digest:
        out["action"] = "noop-already-current"
        LAST.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return 0
    out["action"] = "would-update" if dry else "updated"
    out["new_pod_hash"] = digest
    if not dry:
        if out["unfinished"]:
            out["action"] = "deferred-unfinished-training"
            ALERT.write_text(json.dumps(out, indent=2) + "\n")
            print(json.dumps(out, indent=2))
            return 3
        write_pod_hash(digest)
        LAST.write_text(json.dumps(out, indent=2) + "\n")
        if ALERT.exists():
            ALERT.unlink()
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
