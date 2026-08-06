"""Bittensor chain glue: reveal scanning, metagraph, rolling-king weights.

Reveal payload (chain contract, see affine.toml [submission]):

    affine1|<hf_repo>|<hf_revision_40hex>|<author_hotkey_ss58>

committed timelock-encrypted via `Commitments.set_commitment` (bittensor 11;
see scripts/submit.py). The
revision is the HF git commit sha the validator will pin (TOCTOU safety —
we evaluate exactly the snapshot the miner committed to, never a moving
branch tip). The author hotkey is a redundancy cross-check; the chain-side
commitment key is the source of truth.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import bittensor as bt
from bittensor.result import ChainError

log = logging.getLogger("affine.chain")

REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,50}$")

# Bittensor emits one block every 12 seconds.
BLOCKS_PER_HOUR = 300


@dataclass(frozen=True)
class Reveal:
    hotkey: str
    block: int
    repo: str
    revision: str


def build_reveal(prefix: str, repo: str, revision: str, hotkey: str) -> str:
    if not REPO_RE.match(repo):
        raise ValueError(f"invalid HF repo id: {repo!r}")
    if not REVISION_RE.match(revision):
        raise ValueError(f"invalid HF revision (need 40-hex commit sha): {revision!r}")
    if not SS58_RE.match(hotkey):
        raise ValueError(f"invalid hotkey ss58: {hotkey!r}")
    return f"{prefix}|{repo}|{revision}|{hotkey}"


def parse_reveal(prefix: str, payload: str) -> tuple[str, str, str]:
    """Return (repo, revision, author_hotkey) or raise ValueError."""
    parts = (payload or "").strip().split("|")
    if len(parts) != 4 or parts[0] != prefix:
        raise ValueError(f"expected {prefix}|repo|revision|hotkey reveal")
    repo, revision, hotkey = parts[1].strip(), parts[2].strip().lower(), parts[3].strip()
    if not REPO_RE.match(repo):
        raise ValueError(f"invalid repo in reveal: {repo!r}")
    if not REVISION_RE.match(revision):
        raise ValueError(f"invalid revision in reveal: {revision!r}")
    if not SS58_RE.match(hotkey):
        raise ValueError(f"invalid author hotkey in reveal: {hotkey!r}")
    return repo, revision, hotkey


def _decode_commitment_pair(pair) -> tuple[str, list[tuple[int, str]]]:
    """Return (hotkey_ss58, [(block, payload), ...]) for one RevealedCommitments row.

    Depending on the substrate client path the payload arrives either as a
    hex-serialized SCALE byte string ("0x...") or raw bytes wrapped in a str
    via latin-1. Normalize both, strip the SCALE compact-length prefix, and
    decode the rest as UTF-8. (Pattern proven in production on SN3; decoding
    ourselves avoids bittensor helpers that raise on any single bad row and
    poison the whole scan.)
    """
    key, data = pair
    if not isinstance(key, str):
        raise ValueError(f"unexpected commitment key type {type(key).__name__}")
    out = []
    for entry in data:
        text, block = entry
        if isinstance(text, (bytes, bytearray)):
            # bittensor 11 transport may hand the SCALE payload over as raw bytes.
            raw = bytes(text)
        elif isinstance(text, str):
            if text.startswith(("0x", "0X")):
                raw = bytes.fromhex(text[2:])
            else:
                raw = text.encode("latin-1")
        else:
            raise ValueError(f"unexpected commitment payload type {type(text).__name__}")
        if not raw:
            raise ValueError("empty commitment payload")
        mode = raw[0] & 0b11
        if mode == 0:
            offset = 1
        elif mode == 1:
            offset = 2
        elif mode == 2:
            offset = 4
        else:
            # SCALE big-integer mode: upper 6 bits give the byte length of the
            # length field itself (commitments never get this large, but a
            # malformed row must not silently mis-slice).
            offset = 1 + (raw[0] >> 2) + 4
        out.append((int(block), raw[offset:].decode("utf-8", errors="ignore")))
    return key, out


def scan_reveals(subtensor, netuid: int, prefix: str,
                 min_submission_block: int,
                 seen_hotkeys: set[str]) -> tuple[list[Reveal], list[dict]]:
    """Pull RevealedCommitments; return (parseable latest reveals, bad payloads).

    Returns the latest reveal per hotkey (oldest-first), including those the
    validator will later skip (seen slot / below min_submission_block) so
    intake can surface the decision. `min_submission_block` and `seen_hotkeys`
    are retained for call-site compat but no longer filter the reveal list —
    enqueue / intake owns those gates.

    `bad` entries are `{hotkey, block, detail}` for undecodable or unparseable
    payloads (so the dashboard can show rejected_bad_payload).
    """
    del min_submission_block, seen_hotkeys  # gates moved to State.enqueue
    try:
        query = subtensor.query_map(
            bt.storage.Commitments.RevealedCommitments, [netuid])
    except Exception:
        log.exception("query_map RevealedCommitments failed")
        return [], []

    all_reveals: dict[str, list[tuple[int, str]]] = {}
    bad_rows = 0
    for pair in query:
        try:
            hotkey, entries = _decode_commitment_pair(pair)
            all_reveals[hotkey] = entries
        except Exception:
            bad_rows += 1
    if bad_rows:
        log.warning("scan_reveals: skipped %d undecodable on-chain commitments",
                    bad_rows)

    out: list[Reveal] = []
    bad: list[dict] = []
    for hotkey, entries in all_reveals.items():
        if not entries:
            continue
        block, payload = max(entries, key=lambda e: e[0])
        try:
            repo, revision, author = parse_reveal(prefix, payload)
        except ValueError as e:
            bad.append({
                "hotkey": hotkey, "block": int(block),
                "detail": str(e)[:300],
            })
            continue
        if author != hotkey:
            # Chain key is the signer and therefore the source of truth.
            log.warning("reveal author %s mismatches chain key %s; trusting chain",
                        author[:16], hotkey[:16])
        out.append(Reveal(hotkey=hotkey, block=block, repo=repo, revision=revision))
    out.sort(key=lambda r: r.block)
    return out, bad


def safe_block(subtensor) -> int:
    try:
        return int(subtensor.block)
    except Exception:
        log.warning("could not read current block", exc_info=True)
        return 0


class BlockHashUnavailable(Exception):
    """Chain hiccup while fetching the seed block hash. Fail CLOSED: a
    predictable fallback seed would let miners precompute the duel slice."""


def block_hash_at(subtensor, block: int) -> str:
    """Reveal-block hash — the externally auditable seed material for the
    duel slice. Raises BlockHashUnavailable on chain failure; the validator
    requeues the challenge (no burn) rather than dueling on a predictable
    slice."""
    try:
        h = subtensor.block_info(block).hash
    except Exception as e:
        raise BlockHashUnavailable(f"block_info({block}) failed: {e}") from e
    if not h:
        raise BlockHashUnavailable(f"block_info({block}) returned empty hash")
    return str(h)


class Metagraph:
    """Thin cache over the subnet metagraph: hotkey→uid and hotkey→coldkey.
    Tracks refresh age so callers can refuse to act on a stale uid map."""

    def __init__(self):
        self.uid_of: dict[str, int] = {}
        self.coldkey_of: dict[str, str] = {}
        self.n = 0
        self._refreshed_at: float | None = None

    def refresh(self, subtensor, netuid: int) -> None:
        try:
            meta = subtensor.subnets.metagraph(netuid)
        except Exception:
            log.exception("metagraph refresh failed (keeping stale map)")
            return
        self.uid_of = {hk: int(uid) for uid, hk in enumerate(meta.hotkeys)}
        self.coldkey_of = dict(zip(meta.hotkeys, meta.coldkeys))
        self.n = len(meta.hotkeys)
        self._refreshed_at = time.monotonic()

    def age_s(self) -> float:
        """Seconds since the last successful refresh (inf if never)."""
        if self._refreshed_at is None:
            return float("inf")
        return time.monotonic() - self._refreshed_at

    def identity_token_pairs(self, hotkey: str, prefix_len: int,
                             suffix_len: int) -> list[tuple[str, str]]:
        """Anti-impersonation (prefix, suffix) pairs: the repo id must contain
        the first `prefix_len` AND last `suffix_len` chars (lowercase) of the
        submitter's coldkey OR hotkey ss58. Substring matching on the pair
        (not a concatenated token) accepts both the compact token style and a
        full embedded ss58. The hotkey pair is always available, so a
        deregistered submitter is judged rather than retried forever."""
        keys = [hotkey]
        ck = self.coldkey_of.get(hotkey)
        if ck:
            keys.append(ck)
        return [(k[:prefix_len].lower(), k[-suffix_len:].lower()) for k in keys]


def set_rolling_weights(subtensor, wallet, netuid: int,
                        king_chain_hotkeys: list[str],
                        metagraph: Metagraph, burn_uid: int,
                        max_metagraph_age_s: float = 1800.0,
                        version_key: int = 0) -> bool:
    """Equal-share weights across registered members of the rolling king
    chain (current king first, then prior distinct kings). Falls back to the
    burn UID when nobody in the chain is registered. Refuses to act on a
    stale metagraph — a rotated uid map would pay the wrong neurons."""
    age = metagraph.age_s()
    if age > max_metagraph_age_s:
        log.error("metagraph is %.0fs stale (> %.0fs); refusing to set weights",
                  age, max_metagraph_age_s)
        return False
    uids: list[int] = []
    for hk in king_chain_hotkeys:
        uid = metagraph.uid_of.get(hk)
        if uid is not None and uid not in uids:
            uids.append(uid)
    if not uids:
        uids = [burn_uid]
    w = 1.0 / len(uids)
    weights = [w] * len(uids)
    try:
        # bittensor 11 intent path; execute() waits for inclusion and returns
        # an ExtrinsicResult (raise_for_failure covers dispatch errors).
        res = subtensor.execute(
            bt.SetWeights(netuid=netuid, uids=uids, weights=weights,
                          version_key=version_key),
            wallet,
        )
        res.raise_for_failure()
        log.info("set_weights uids=%s weights=%.3f each block=%s", uids, w,
                 res.block_hash)
        return True
    except ChainError as e:
        # The chain allows one set_weights per rate-limit window (100 blocks);
        # our loop retries faster than that by design, so preflight rejections
        # inside the window are expected cadence, not faults. One quiet line
        # instead of a 30-line traceback every attempt.
        if "rate limit" in str(e):
            log.info("set_weights rate-limited: %s", e)
        else:
            log.exception("set_weights failed")
        return False
    except Exception:
        log.exception("set_weights failed")
        return False
