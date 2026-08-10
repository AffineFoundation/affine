"""Miner-side submission client: commit-reveal an HF checkpoint to Affine.

Standalone by design — this single file is the whole submit path. It is
published verbatim at code/scripts/submit.py on the subnet site; download it
(or clone https://github.com/AffineFoundation/affine) and run:

  pip install "bittensor>=11,<12" huggingface_hub
  python submit.py --repo you/Affine-{token}-mymodel \
      --wallet YOUR_WALLET --hotkey YOUR_HOTKEY [--revision <40hex>]

Your one eval slot burns at enqueue, so this client refuses to send a
submission the validator would reject at intake. Before touching the chain it
mirrors the validator's own metadata-only checks (affine/model_store.py):
repo naming + identity, anonymous readability of the pinned revision,
canonical safetensors layout, no *.py / no auto_map, and the size caps.
Run with --check to validate everything and print the payload without
submitting.

Requirements the validator enforces (see data/contract.json on the site):
  * repo name matches ^[^/]+/[Aa]ffine-.+$ AND embeds your identity: the
    first 5 and last 5 chars (lowercase) of your coldkey OR hotkey ss58 —
    the compact token (prefix+suffix) or the full ss58 both work
  * repo must be PUBLIC (the validator reads it without your credentials)
  * safetensors in canonical layout; no *.py; no auto_map; under size caps
  * one submission per hotkey, ever — a failed eval burns the slot
  * your hotkey must be registered on the subnet to receive emissions
    (btcli subnets register --netuid 120)

The constants below are the frozen chain contract (affine.toml [subnet] /
[submission]); changing any of them is a chain fork, so hardcoding them here
keeps this file dependency-free without risking drift.
"""

from __future__ import annotations

import argparse
import json
import re

import bittensor as bt
from huggingface_hub import HfApi

NETWORK = "finney"
NETUID = 120
REVEAL_PREFIX = "affine1"
ID_PREFIX_LEN = 5
ID_SUFFIX_LEN = 5

# Intake hygiene caps (affine.toml [submission], mirrored by the validator's
# validate_repo_hygiene in affine/model_store.py).
MAX_MODEL_SIZE_GB = 90.0
MAX_TOTAL_REPO_GB = 100.0
MAX_REPO_FILES = 5000
MAX_CONFIG_BYTES = 1 << 20

REPO_PATTERN = re.compile(r"^[^/]+/[Aa]ffine-.+$")
REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,50}$")
SAFETENSORS_SHARD_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")


def build_reveal(prefix: str, repo: str, revision: str, hotkey: str) -> str:
    """Reveal payload (chain contract): affine1|<repo>|<revision>|<hotkey>."""
    if not REPO_RE.match(repo):
        raise ValueError(f"invalid HF repo id: {repo!r}")
    if not REVISION_RE.match(revision):
        raise ValueError(f"invalid HF revision (need 40-hex commit sha): {revision!r}")
    if not SS58_RE.match(hotkey):
        raise ValueError(f"invalid hotkey ss58: {hotkey!r}")
    return f"{prefix}|{repo}|{revision}|{hotkey}"


def check_repo_name(repo: str, coldkey: str, hotkey: str) -> list[str]:
    """The two naming rules the validator rejects on (repo_name_rejected)."""
    problems = []
    if not REPO_PATTERN.match(repo):
        problems.append(
            f"repo {repo!r} does not match required pattern '^[^/]+/[Aa]ffine-.+$'")
    n, m = ID_PREFIX_LEN, ID_SUFFIX_LEN
    pairs = [(k[:n].lower(), k[-m:].lower()) for k in (coldkey, hotkey)]
    repo_l = repo.lower()
    if not any(p in repo_l and s in repo_l for p, s in pairs):
        ck_token = pairs[0][0] + pairs[0][1]
        problems.append(
            f"repo must embed your coldkey or hotkey identity "
            f"(e.g. you/Affine-{ck_token}-mymodel)")
    return problems


def preflight_repo(repo: str, revision: str) -> list[str]:
    """Metadata-only mirror of the validator's intake checks.

    Deliberately anonymous (token=False): the validator reads your repo
    without your credentials, so a private/gated repo fails there with
    revision_not_found no matter what your local token can see.
    """
    api = HfApi(token=False)
    try:
        tree = list(api.list_repo_tree(repo, revision=revision,
                                       recursive=True, expand=True))
    except Exception as e:
        return [f"cannot read {repo}@{revision[:12]} anonymously "
                f"({type(e).__name__}) — the validator would record "
                f"revision_not_found. Is the repo public and the revision pushed?"]

    problems: list[str] = []
    files = [getattr(t, "path", "") for t in tree]
    if len(files) > MAX_REPO_FILES:
        problems.append(f"repo lists {len(files)} files > {MAX_REPO_FILES} cap")

    py_files = sorted(f for f in files if f.endswith(".py"))
    if py_files:
        problems.append(f"repo ships *.py files (not allowed): {py_files[:3]}")

    if "config.json" not in files:
        problems.append("missing config.json")
    else:
        raw = api.hf_hub_download(repo, "config.json", revision=revision,
                                  token=False)
        with open(raw) as f:
            config = json.load(f)
        if "auto_map" in config:
            problems.append("auto_map present in config.json "
                            "(custom modeling code is not allowed)")
        if len(json.dumps(config)) > MAX_CONFIG_BYTES:
            problems.append(f"config.json exceeds {MAX_CONFIG_BYTES} bytes cap")

    st_files = [f for f in files if f.endswith(".safetensors")]
    if not st_files:
        problems.append("no .safetensors files in repo")
    else:
        has_single = "model.safetensors" in files
        has_index = "model.safetensors.index.json" in files
        has_shards = any(SAFETENSORS_SHARD_RE.match(f) for f in st_files)
        if not (has_single or (has_index and has_shards)):
            if has_shards and not has_index:
                problems.append("missing model.safetensors.index.json "
                                "for sharded layout")
            else:
                problems.append("safetensors present but not in canonical "
                                f"layout: {st_files[:3]}")

    total_st = sum(int(getattr(t, "size", 0) or 0) for t in tree
                   if getattr(t, "path", "").endswith(".safetensors"))
    total_all = sum(int(getattr(t, "size", 0) or 0) for t in tree)
    if total_st / 1e9 > MAX_MODEL_SIZE_GB:
        problems.append(f"oversized: {total_st / 1e9:.1f} GB of .safetensors "
                        f"> {MAX_MODEL_SIZE_GB:.0f} GB cap")
    if total_all / 1e9 > MAX_TOTAL_REPO_GB:
        problems.append(f"oversized: {total_all / 1e9:.1f} GB total repo "
                        f"> {MAX_TOTAL_REPO_GB:.0f} GB cap")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--repo", required=True, help="HF repo id (you/Affine-...)")
    ap.add_argument("--revision", default="",
                    help="commit sha / branch / tag (default: repo HEAD; "
                         "always resolved and pinned to a 40-hex sha)")
    ap.add_argument("--wallet", required=True)
    ap.add_argument("--hotkey", required=True)
    ap.add_argument("--network", default=NETWORK)
    ap.add_argument("--netuid", type=int, default=NETUID)
    ap.add_argument("--check", action="store_true",
                    help="run every pre-flight check and print the payload "
                         "without submitting anything on-chain")
    args = ap.parse_args()

    # Resolve whatever the miner passed (branch, tag, short/long sha) to the
    # immutable 40-hex commit the validator will pin.
    revision = HfApi().model_info(args.repo,
                                  revision=args.revision or None).sha
    wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    hotkey = wallet.hotkey.ss58_address
    coldkey = wallet.coldkeypub.ss58_address

    problems = check_repo_name(args.repo, coldkey, hotkey)
    problems += preflight_repo(args.repo, revision)
    if problems:
        print("pre-flight FAILED — the validator would reject this submission "
              "and burn your slot:")
        for p in problems:
            print(f"  * {p}")
        raise SystemExit(1)

    payload = build_reveal(REVEAL_PREFIX, args.repo, revision, hotkey)
    if args.check:
        print(f"pre-flight OK. Would commit: {payload}")
        return

    print(f"committing: {payload}")
    subtensor = bt.subtensor(network=args.network)
    # bittensor 11: timelock-encrypt the payload to a near-future drand round
    # and publish it via the raw Commitments call; the chain decrypts and
    # lands it in RevealedCommitments (~1 min).
    sealed = bt.timelock.encrypt(payload, reveal_in="60s")
    call = bt.calls.Commitments.set_commitment(
        args.netuid,
        {"fields": [{"TimelockEncrypted": {
            "encrypted": sealed.ciphertext,
            "reveal_round": sealed.reveal_round,
        }}]},
    )
    res = subtensor.submit_call(call, wallet, signer="hotkey")
    res.raise_for_failure()
    print(f"submitted in block {res.block_hash}; reveal opens at drand round "
          f"{sealed.reveal_round} (~1 min). Watch the queue on the dashboard.")


if __name__ == "__main__":
    main()
