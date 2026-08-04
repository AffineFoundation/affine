"""Miner-side submission helper: commit-reveal an HF checkpoint to Affine.

Usage:
  python scripts/submit.py --repo you/Affine-mymodel \
      --wallet mywallet --hotkey myhotkey [--revision <40hex>]

Requirements the validator enforces (see data/contract.json on the site):
  * repo name matches the pattern AND embeds your identity: the first 5 and
    last 5 chars (lowercase) of your coldkey OR hotkey ss58 — the compact
    token (prefix+suffix) or the full ss58 both work
  * safetensors in canonical layout; no *.py; no auto_map; under size cap
  * one submission per hotkey, ever — a failed eval burns the slot
"""

from __future__ import annotations

import argparse

import bittensor as bt
from huggingface_hub import HfApi

from affine.chain import build_reveal
from affine.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id (you/Affine-...)")
    ap.add_argument("--revision", default="", help="40-hex commit sha (default: repo HEAD)")
    ap.add_argument("--wallet", required=True)
    ap.add_argument("--hotkey", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    revision = args.revision or HfApi().model_info(args.repo).sha
    wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    hotkey = wallet.hotkey.ss58_address
    coldkey = wallet.coldkeypub.ss58_address
    n, m = cfg.submission.coldkey_prefix_len, cfg.submission.coldkey_suffix_len
    pairs = [(k[:n].lower(), k[-m:].lower()) for k in (coldkey, hotkey)]
    repo_l = args.repo.lower()
    if not any(p in repo_l and s in repo_l for p, s in pairs):
        ck_token = pairs[0][0] + pairs[0][1]
        raise SystemExit(
            f"repo must embed your coldkey or hotkey identity "
            f"(e.g. you/Affine-{ck_token}-mymodel) or the validator rejects it")

    payload = build_reveal(cfg.submission.reveal_prefix, args.repo,
                           revision, hotkey)
    print(f"committing: {payload}")
    subtensor = bt.subtensor(network=cfg.network)
    # bittensor 11: no set_reveal_commitment. Timelock-encrypt the payload to
    # a near-future drand round and publish it via the raw Commitments call;
    # the chain decrypts and lands it in RevealedCommitments (~1 min).
    sealed = bt.timelock.encrypt(payload, reveal_in="60s")
    call = bt.calls.Commitments.set_commitment(
        cfg.netuid,
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
