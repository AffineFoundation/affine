#!/usr/bin/env python3
"""Idempotent source patches for the upstream checkouts on the datagen pods.

The pods run two upstream editable checkouts we do not own in this repo —
`/root/prime-pilot/verifiers` (PrimeIntellect verifiers) and
`/root/prime-pilot/research-environments` (their tasksets). When a fix has to
land in those trees, it goes here as an exact-string substitution with a
marker comment, and `ops/king-datagen/deploy_pods.sh` runs this script on
every pod after the file copies. Re-running is a no-op (marker present); an
upstream drift that removed the anchor text fails loud (exit 2) instead of
silently leaving the bug in.

2026-09-27 (datagen worker), from the reign-22 king-seat error audit
(27 % of king rollouts errored, 3 pods, 2 days):

1. swesmith_v1 grader `restore_eval_commit`: `git checkout HEAD~1` (restores the
   removed F2P tests) aborts when the agent re-created one of those test files
   as an UNTRACKED file ("untracked working tree files would be overwritten by
   checkout") -> the whole rollout errored instead of being scored. 1,290 of
   3,304 king errors (39 %). The scorer restores the official test files from
   the commit right after (`revert_test_files`), so the agent's copies are
   discarded either way: remove the files git names and retry once.

2. Debian 11 (bullseye) task images (swesmith js/ts): `apt-get update` succeeds
   but bullseye-security's pool no longer carries the .debs its index lists
   (LTS ended 2026-08-31) -> `apt-get install curl ca-certificates` 404s ->
   "failed to prepare uv script" / "Node.js install failed" / "Kimi Code install
   failed". 383 king errors (12 %). Reproduced in
   swebench/swesmith.x86_64.advplyr_1776_audiobookshelf.626596b1; dropping the
   `-security` source and retrying installs curl 7.74.0-1.3+deb11u13 from main
   (TLS to astral.sh verified). The retry is appended to every apt fallback the
   harness installers use.

    python3 apply_pod_patches.py            # apply (idempotent)
    python3 apply_pod_patches.py --check    # report only
"""
from __future__ import annotations

import sys
from pathlib import Path

VF = Path("/root/prime-pilot/verifiers/verifiers/v1")
RE = Path("/root/prime-pilot/research-environments/environments")

APT_RETRY = (
    "|| { sed -i -e '/-security/d' /etc/apt/sources.list 2>/dev/null; "
    "rm -f /etc/apt/sources.list.d/*security* 2>/dev/null; "
    "apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null; }"
)  # affine-pod-patch: bullseye-security pool 404s (2026-09-27)

PATCHES = [
    {
        "name": "swesmith_v1: retry HEAD~1 checkout after removing the untracked test files git names",
        "file": RE / "swe/swesmith_v1/swesmith_v1/taskset.py",
        "marker": "affine-pod-patch: untracked F2P copies",
        "old": (
            '        result = await runtime.run(["git", "checkout", "HEAD~1"], ENV)\n'
            '        if result.exit_code != 0:\n'
            '            raise RuntimeError(f"swesmith checkout HEAD~1 failed ({self.data.name}): {result.stderr.strip()[-500:]}")\n'
        ),
        "new": (
            '        result = await runtime.run(["git", "checkout", "HEAD~1"], ENV)\n'
            '        if result.exit_code != 0 and "untracked working tree files would be overwritten" in (result.stderr or ""):\n'
            '            # affine-pod-patch: untracked F2P copies (2026-09-27). The agent re-created, as\n'
            '            # untracked files, tests that HEAD~1 restores; git lists them tab-indented. The\n'
            '            # official copies come back from the commit and revert_test_files(), so drop\n'
            '            # the agent\'s and retry instead of erroring the rollout.\n'
            '            names = [ln.strip() for ln in (result.stderr or "").splitlines() if ln.startswith("\\t") and ln.strip()]\n'
            '            if names:\n'
            '                await runtime.run(["rm", "-rf", "--", *names], ENV)\n'
            '                result = await runtime.run(["git", "checkout", "HEAD~1"], ENV)\n'
            '        if result.exit_code != 0:\n'
            '            raise RuntimeError(f"swesmith checkout HEAD~1 failed ({self.data.name}): {result.stderr.strip()[-500:]}")\n'
        ),
    },
    {
        "name": "verifiers runtimes/base.py: apt fallback retries without bullseye-security",
        "file": VF / "runtimes/base.py",
        "marker": "affine-pod-patch: bullseye-security",
        "old": (
            '    "|| { apt-get update -qq && apt-get install -y -qq curl ca-certificates; } "\n'
        ),
        "new": (
            '    "|| { apt-get update -qq && apt-get install -y -qq curl ca-certificates; } "\n'
            '    # affine-pod-patch: bullseye-security pool 404s (2026-09-27) -- retry without it\n'
            '    "' + APT_RETRY.replace('"', '\\"') + ' "\n'
        ),
    },
    {
        "name": "verifiers harnesses/node.py: apt fallback retries without bullseye-security",
        "file": VF / "harnesses/node.py",
        "marker": "affine-pod-patch: bullseye-security",
        "old": (
            "        || { apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null; }\n"
        ),
        "new": (
            "        || { apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null; } \\\n"
            "        " + APT_RETRY + "\n"
        ),
    },
    {
        "name": "verifiers harnesses/kimi_code: apt fallback retries without bullseye-security",
        "file": VF / "harnesses/kimi_code/harness.py",
        "marker": "affine-pod-patch: bullseye-security",
        "old": (
            "command -v curl >/dev/null || { apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null; }\n"
        ),
        "new": (
            "command -v curl >/dev/null || { apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null; } \\\n"
            "  " + APT_RETRY + "\n"
        ),
    },
]


def main() -> int:
    check = "--check" in sys.argv
    rc = 0
    for p in PATCHES:
        f: Path = p["file"]
        if not f.exists():
            print(f"MISSING  {p['name']}: {f}")
            rc = 2
            continue
        text = f.read_text()
        if p["marker"] in text and p["old"] not in text:
            print(f"ok       {p['name']}")
            continue
        if p["old"] not in text:
            print(f"NO-ANCHOR {p['name']}: upstream text changed in {f}")
            rc = 2
            continue
        if check:
            print(f"PENDING  {p['name']}")
            rc = max(rc, 1)
            continue
        f.write_text(text.replace(p["old"], p["new"], 1))
        print(f"APPLIED  {p['name']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
