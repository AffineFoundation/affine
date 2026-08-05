"""Miner-side submission client: commit-reveal an HF checkpoint to Affine.

Standalone by design — this single file is the whole submit path. It is
published verbatim at code/scripts/submit.py on the subnet site; download it
(or clone https://github.com/AffineFoundation/affine) and run:

  pip install "bittensor>=11,<12" huggingface_hub
  python submit.py --repo you/Affine-{token}-mymodel \
      --wallet YOUR_WALLET --hotkey YOUR_HOTKEY [--revision <40hex>]

Requirements the validator enforces (see data/contract.json on the site):
  * repo name matches ^[^/]+/[Aa]ffine-.+$ AND embeds your identity: the
    first 5 and last 5 chars (lowercase) of your coldkey OR hotkey ss58 —
    the compact token (prefix+suffix) or the full ss58 both work
  * safetensors in canonical layout; no *.py; no auto_map; under size cap
  * one submission per hotkey, ever — a failed eval burns the slot
  * your hotkey must be registered on the subnet to receive emissions
    (btcli subnet register --netuid 120)

The constants below are the frozen chain contract (affine.toml [subnet] /
[submission]); changing any of them is a chain fork, so hardcoding them here
keeps this file dependency-free without risking drift.
"""

from __future__ import annotations

import argparse
import re

import bittensor as bt
from huggingface_hub import HfApi

NETWORK = "finney"
NETUID = 120
REVEAL_PREFIX = "affine1"
ID_PREFIX_LEN = 5
ID_SUFFIX_LEN = 5

REPO_PATTERN = re.compile(r"^[^/]+/[Aa]ffine-.+$")
REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,50}$")


def build_reveal(prefix: str, repo: str, revision: str, hotkey: str) -> str:
    """Reveal payload (chain contract): affine1|<repo>|<revision>|<hotkey>."""
    if not REPO_RE.match(repo):
        raise ValueError(f"invalid HF repo id: {repo!r}")
    if not REVISION_RE.match(revision):
        raise ValueError(f"invalid HF revision (need 40-hex commit sha): {revision!r}")
    if not SS58_RE.match(hotkey):
        raise ValueError(f"invalid hotkey ss58: {hotkey!r}")
    return f"{prefix}|{repo}|{revision}|{hotkey}"


def check_repo_name(repo: str, coldkey: str, hotkey: str) -> None:
    """Fail fast on the two repo-naming rules the validator rejects on."""
    if not REPO_PATTERN.match(repo):
        raise SystemExit(
            f"repo {repo!r} does not match required pattern '^[^/]+/[Aa]ffine-.+$'")
    n, m = ID_PREFIX_LEN, ID_SUFFIX_LEN
    pairs = [(k[:n].lower(), k[-m:].lower()) for k in (coldkey, hotkey)]
    repo_l = repo.lower()
    if not any(p in repo_l and s in repo_l for p, s in pairs):
        ck_token = pairs[0][0] + pairs[0][1]
        raise SystemExit(
            f"repo must embed your coldkey or hotkey identity "
            f"(e.g. you/Affine-{ck_token}-mymodel) or the validator rejects it")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--repo", required=True, help="HF repo id (you/Affine-...)")
    ap.add_argument("--revision", default="", help="40-hex commit sha (default: repo HEAD)")
    ap.add_argument("--wallet", required=True)
    ap.add_argument("--hotkey", required=True)
    ap.add_argument("--network", default=NETWORK)
    ap.add_argument("--netuid", type=int, default=NETUID)
    args = ap.parse_args()

    revision = args.revision or HfApi().model_info(args.repo).sha
    wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    hotkey = wallet.hotkey.ss58_address
    coldkey = wallet.coldkeypub.ss58_address
    check_repo_name(args.repo, coldkey, hotkey)

    payload = build_reveal(REVEAL_PREFIX, args.repo, revision, hotkey)
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
