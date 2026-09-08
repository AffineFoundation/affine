"""Affine (SN120) miner client — private model submission over Cloudflare R2.

Standalone by design: this single file is the whole miner path. It is
published verbatim at https://affine.io/code/scripts/submit.py; download it
(or clone https://github.com/AffineFoundation/affine) and run:

  pip install "bittensor>=11,<12" boto3 httpx
  python submit.py submit --wallet W --hotkey H --model-dir ./my-checkpoint

`submit` runs the four steps below in order and resumes where it left off.
Each step is also its own subcommand so an agent can drive them one by one.

  hotkey    create (or regenerate) an ED25519 hotkey in your wallet. Affine
            requires Ed25519 hotkeys: the validator seals your upload
            credentials to the hotkey and an sr25519 key cannot open them.
            Then register it: btcli subnets register --netuid 120 ...
  check     offline pre-flight of --model-dir (layout, size caps, genesis-
            family architecture pin). Nothing touches the chain.
  register  commit `affine2|activate|<hotkey>|<sig>` on-chain.
  auth      poll the public mailbox until the validator posts your sealed
            credentials; decrypt with the hotkey → upload-auth.json (0600).
  upload    sha256 + upload every checkpoint file into your private prefix,
            sign and upload manifest.json last.
  ready     commit `affine2|ready|<registration_id>|<sha256(manifest.json)>`
            → the validator revokes the credential, verifies the manifest,
            and queues the duel. Deletes upload-auth.json.
  status    what the validator did with your submission (intake decision,
            registration state, queue position, verdict).

Your model stays private: nobody but the validator can read the private
bucket. Only a crowned model is copied to the public bucket
(https://models.affine.io/models/sha256/<model_digest>/).

Rules the validator enforces (see https://affine.io/llms.txt):
  * Ed25519 hotkey, registered on netuid 120
  * one submission per hotkey, ever — a failed eval burns the slot
  * checkpoint = bare directory of allowed files (config.json, tokenizer
    files, model.safetensors or model-XXXXX-of-YYYYY.safetensors + index);
    no *.py, no auto_map, ≤ 90 GB of safetensors, ≤ 100 GB total
  * config.json must match the genesis-family architecture pin
    (a Qwen/Qwen3.6-35B-A3B fine-tune; dtype / rope / token ids free)
  * commit `ready` only after `upload` printed OK; the validator lists your
    prefix and rejects any drift from the signed manifest

The constants below mirror the frozen chain contract (affine.toml). The
protocol helpers are a verbatim copy of affine/r2protocol.py.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import bittensor as bt
from bittensor.wallet import Keypair
import boto3
import httpx
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig

NETWORK = "finney"
NETUID = 120
SITE = "https://affine.io"
MAILBOX_BASE_URL = "https://dash.affine.io"
PUBLIC_MODELS_BASE_URL = "https://models.affine.io"
ED25519 = 0  # bittensor KeypairType.ED25519

# Intake hygiene caps (affine.toml [submission]).
MAX_MODEL_SIZE_GB = 90.0
MAX_TOTAL_REPO_GB = 100.0
MAX_REPO_FILES = 5000
MAX_CONFIG_BYTES = 1 << 20

# Architecture pin (affine.toml [submission.pinned_arch] + [[pinned_arch_alt]]):
# config.json must match one of these nested subsets exactly. Keys not listed
# stay free. The second profile is the text-only extraction of the genesis
# (vision tower dropped, text_config flattened to the root; 2026-09-04).
PINNED_ARCH: dict = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "vocab_size": 248320,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
    },
}
PINNED_ARCH_ALT: list[dict] = [{
    "architectures": ["Qwen3_5MoeForCausalLM"],
    "model_type": "qwen3_5_moe_text",
    "tie_word_embeddings": False,
    **{k: v for k, v in PINNED_ARCH["text_config"].items() if k != "model_type"},
}]

# ---------------------------------------------------------------------------
# Protocol (verbatim mirror of affine/r2protocol.py — keep in sync)
# ---------------------------------------------------------------------------
PROTOCOL_VERSION = 1
REVEAL_PREFIX = "affine2"
REGISTRATION_DOMAIN = b"affine-registration-v1\0"
ACTIVATE_DOMAIN = "affine-activate|v1"
ENVELOPE_DOMAIN = b"affine-mailbox-envelope-v1\0"
MANIFEST_DOMAIN = b"affine-manifest-v1\0"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,50}$")
PRIVATE_PREFIX_RE = re.compile(r"^models/registrations/[0-9a-f]{64}/$")
ALLOWED_FILE_RE = re.compile(
    r"^(config\.json|generation_config\.json|tokenizer\.json|tokenizer_config\.json|"
    r"special_tokens_map\.json|vocab\.json|merges\.txt|tokenizer\.model|"
    r"chat_template\.jinja|chat_template\.json|added_tokens\.json|preprocessor_config\.json|"
    r"video_preprocessor_config\.json|model\.safetensors|model\.safetensors\.index\.json|"
    r"model-\d{5}-of-\d{5}\.safetensors|README\.md|LICENSE(\.[a-z]+)?|\.gitattributes)$")
SAFETENSORS_SHARD_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")
ENVELOPE_FIELDS = (
    "protocol_version", "validator_identity", "netuid", "hotkey",
    "registration_id", "credential_generation", "r2_endpoint",
    "private_model_bucket", "allowed_prefix", "credential_scope",
    "revocation_event", "submission_policy", "access_key_id",
    "secret_access_key", "session_token", "expires_at", "issued_at",
    "signature_scheme", "validator_signature")


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def registration_id(netuid: int, hotkey: str) -> str:
    return sha256_hex(REGISTRATION_DOMAIN
                      + canonical_json({"hotkey": hotkey, "netuid": int(netuid)}))


def private_prefix(reg_id: str) -> str:
    return f"models/registrations/{reg_id}/"


def mailbox_key(reg_id: str, generation: int) -> str:
    return f"mailbox/v1/{reg_id}/generations/{int(generation):020d}.bin"


def activate_message(netuid: int, hotkey: str, reg_id: str) -> bytes:
    return f"{ACTIVATE_DOMAIN}|{int(netuid)}|{hotkey}|{reg_id}".encode()


def build_activate_payload(hotkey: str, signature: bytes) -> str:
    return f"{REVEAL_PREFIX}|activate|{hotkey}|{b64url(signature)}"


def build_ready_payload(reg_id: str, manifest_sha256: str) -> str:
    return f"{REVEAL_PREFIX}|ready|{reg_id}|{manifest_sha256}"


def model_digest_from_inventory(files: list[dict]) -> str:
    rows = sorted((str(f["path"]), int(f["size"]), str(f["sha256"]).lower())
                  for f in files)
    return sha256_hex(MANIFEST_DOMAIN + canonical_json(rows))


def manifest_signing_bytes(manifest: dict) -> bytes:
    return canonical_json({k: v for k, v in manifest.items() if k != "signature"})


def envelope_signing_bytes(envelope: dict) -> bytes:
    body = {k: v for k, v in envelope.items() if k != "validator_signature"}
    return ENVELOPE_DOMAIN + canonical_json(body)


def build_manifest(*, reg_id: str, hotkey: str, model_name: str,
                   files: list[dict]) -> dict:
    files = sorted(({"path": str(f["path"]), "size": int(f["size"]),
                     "sha256": str(f["sha256"]).lower()} for f in files),
                   key=lambda f: f["path"])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "signature_scheme": "ed25519",
        "registration_id": reg_id,
        "hotkey": hotkey,
        "model_name": model_name,
        "files": files,
        "model_digest": model_digest_from_inventory(files),
        "signature": "",
    }


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def scale_encode_vec_u8(data: bytes) -> bytes:
    """SCALE compact-length prefix the validator strips before parsing."""
    n = len(data)
    if n < 64:
        return bytes([(n << 2) & 0xFF]) + data
    if n < 16384:
        return ((n << 2) | 0b01).to_bytes(2, "little") + data
    if n < (1 << 30):
        return ((n << 2) | 0b10).to_bytes(4, "little") + data
    raise ValueError(f"payload too large for SCALE compact: {n}")


def commit_payload(wallet, payload: str, *, network: str, netuid: int) -> str:
    """Timelock-encrypt and publish a reveal payload. Returns the block hash."""
    subtensor = bt.subtensor(network=network)
    sealed = bt.timelock.encrypt(
        scale_encode_vec_u8(payload.encode("utf-8")), reveal_in="180s")
    call = bt.calls.Commitments.set_commitment(
        netuid,
        {"fields": [{"TimelockEncrypted": {
            "encrypted": sealed.ciphertext,
            "reveal_round": sealed.reveal_round,
        }}]},
    )
    res = subtensor.submit_call(call, wallet, signer="hotkey")
    try:
        res.raise_for_failure()
    except Exception as e:
        msg = str(e)
        if "SpaceLimit" in msg or "RateLimit" in msg:
            die("the chain refused the commitment: this hotkey used up its "
                "commitment space for the current window (each activate/ready "
                "is ~400 bytes of a 3100-byte budget). Wait for the window to "
                f"roll over (~1 h) and retry. Chain error: {msg[:200]}")
        die(f"commitment extrinsic failed: {msg[:300]}")
    return str(res.block_hash)


def open_wallet(args) -> "bt.Wallet":
    kw = {"path": args.wallet_path} if getattr(args, "wallet_path", None) else {}
    return bt.Wallet(name=args.wallet, hotkey=args.hotkey, **kw)


def load_wallet(args) -> "bt.Wallet":
    wallet = open_wallet(args)
    if not wallet.hotkey_file.exists_on_device():
        die(f"hotkey {args.wallet}/{args.hotkey} does not exist — run: "
            f"python submit.py hotkey --wallet {args.wallet} --hotkey {args.hotkey}")
    kp = wallet.hotkey
    if getattr(kp, "crypto_type", None) != ED25519:
        die("this hotkey is not Ed25519 (probably sr25519, the btcli default). "
            "Affine seals your upload credentials to an Ed25519 hotkey.\n"
            f"  create one:  python submit.py hotkey --wallet {args.wallet} "
            f"--hotkey {args.hotkey}-ed\n"
            "  register it: btcli subnets register --netuid 120 "
            f"--wallet.name {args.wallet} --wallet.hotkey {args.hotkey}-ed")
    return wallet


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def say(msg: str) -> None:
    print(msg, flush=True)


def fetch_json(url: str, timeout: float = 30.0) -> dict | None:
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True)
        if r.status_code == 200:
            return r.json()
    except (httpx.HTTPError, ValueError):
        pass
    return None


def validator_identity(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    c = fetch_json(f"{SITE}/api/v1/contract") or fetch_json(f"{SITE}/data/contract.json")
    if c:
        return (c.get("submission_r2") or {}).get("validator_identity")
    return None


def s3_from_auth(auth: dict):
    return boto3.client(
        "s3", endpoint_url=auth["r2_endpoint"],
        aws_access_key_id=auth["access_key_id"],
        aws_secret_access_key=auth["secret_access_key"],
        aws_session_token=auth["session_token"],
        region_name="auto",
        config=BotoConfig(signature_version="s3v4",
                          request_checksum_calculation="when_required",
                          response_checksum_validation="when_required",
                          connect_timeout=15, read_timeout=300,
                          retries={"max_attempts": 6, "mode": "standard"}))


def check_pinned_arch(config: dict, pinned: dict, path: str = "") -> str | None:
    if not isinstance(config, dict):
        return f"config.json {path or '<root>'} is not a table"
    for key, expect in pinned.items():
        where = f"{path}.{key}" if path else key
        if key not in config:
            return f"config.json missing pinned key {where}"
        got = config[key]
        if isinstance(expect, dict):
            fault = check_pinned_arch(got, expect, where)
            if fault:
                return fault
        elif got != expect:
            return f"architecture mismatch at {where}: {got!r} != pinned {expect!r}"
    return None


def scan_model_dir(model_dir: Path) -> tuple[list[dict], list[str]]:
    """Inventory of the checkpoint dir + the problems the validator would
    reject on. sizes only here; sha256 happens at upload."""
    problems: list[str] = []
    if not model_dir.is_dir():
        return [], [f"{model_dir} is not a directory"]
    files: list[dict] = []
    for p in sorted(model_dir.iterdir()):
        if p.name == "manifest.json" or p.name.startswith(".") and p.name != ".gitattributes":
            continue
        if p.is_dir():
            problems.append(f"subdirectory {p.name!r} not allowed (flat checkpoint dir only)")
            continue
        if not ALLOWED_FILE_RE.match(p.name):
            problems.append(f"file {p.name!r} is not an allowed checkpoint file "
                            f"(delete it or move it out)")
            continue
        files.append({"path": p.name, "size": p.stat().st_size})
    names = {f["path"] for f in files}
    if len(files) > MAX_REPO_FILES:
        problems.append(f"{len(files)} files > {MAX_REPO_FILES} cap")
    if "config.json" not in names:
        problems.append("missing config.json")
    else:
        raw = (model_dir / "config.json").read_bytes()
        if len(raw) > MAX_CONFIG_BYTES:
            problems.append(f"config.json is {len(raw)} bytes > {MAX_CONFIG_BYTES} cap")
        try:
            config = json.loads(raw)
        except ValueError as e:
            config = None
            problems.append(f"config.json is not JSON: {e}")
        if isinstance(config, dict):
            if "auto_map" in config:
                problems.append("auto_map present in config.json (custom code not allowed)")
            fault = check_pinned_arch(config, PINNED_ARCH)
            if fault and any(check_pinned_arch(config, alt) is None
                             for alt in PINNED_ARCH_ALT):
                fault = None
            if fault:
                problems.append("arch not pinned to the genesis family (must be a "
                                f"Qwen/Qwen3.6-35B-A3B fine-tune): {fault}")
    st = [n for n in names if n.endswith(".safetensors")]
    if not st:
        problems.append("no .safetensors files")
    else:
        has_single = "model.safetensors" in names
        has_index = "model.safetensors.index.json" in names
        has_shards = any(SAFETENSORS_SHARD_RE.match(n) for n in st)
        if not (has_single or (has_index and has_shards)):
            problems.append("safetensors not in canonical layout (model.safetensors, or "
                            "model-XXXXX-of-YYYYY.safetensors + model.safetensors.index.json)")
    total_st = sum(f["size"] for f in files if f["path"].endswith(".safetensors"))
    total = sum(f["size"] for f in files)
    if total_st / 1e9 > MAX_MODEL_SIZE_GB:
        problems.append(f"{total_st / 1e9:.1f} GB of safetensors > {MAX_MODEL_SIZE_GB:.0f} GB cap")
    if total / 1e9 > MAX_TOTAL_REPO_GB:
        problems.append(f"{total / 1e9:.1f} GB total > {MAX_TOTAL_REPO_GB:.0f} GB cap")
    tok = {"tokenizer.json", "tokenizer.model", "vocab.json"} & names
    if not tok:
        problems.append("no tokenizer files (tokenizer.json / tokenizer_config.json) — "
                        "vLLM cannot serve the checkpoint")
    return files, problems


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_auth(path: Path) -> dict:
    if not path.is_file():
        die(f"{path} not found — run `auth` first")
    auth = json.loads(path.read_text())
    if time.time() > _parse_iso(auth["expires_at"]):
        die(f"credentials in {path} expired at {auth['expires_at']}; run "
            "`register` again for a fresh set")
    return auth


def _parse_iso(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_hotkey(args) -> None:
    wallet = open_wallet(args)
    if wallet.hotkey_file.exists_on_device() and not args.overwrite:
        kp = wallet.hotkey
        kind = "Ed25519" if kp.crypto_type == ED25519 else "NOT Ed25519"
        say(f"hotkey exists: {kp.ss58_address} ({kind})")
        if kp.crypto_type != ED25519:
            die("pass --overwrite to replace it with an Ed25519 key, or use a "
                "new --hotkey name")
        return
    if args.mnemonic:
        wallet.regenerate_hotkey(mnemonic=args.mnemonic, use_password=False,
                                 overwrite=args.overwrite, suppress=True,
                                 crypto_type=ED25519)
    else:
        wallet.create_new_hotkey(n_words=12, use_password=False,
                                 overwrite=args.overwrite, suppress=True,
                                 crypto_type=ED25519)
    kp = wallet.hotkey
    assert kp.crypto_type == ED25519
    say(f"created Ed25519 hotkey {args.wallet}/{args.hotkey}: {kp.ss58_address}")
    say("next: register it on the subnet (costs the registration fee):")
    say(f"  btcli subnets register --netuid {NETUID} --network {NETWORK} "
        f"--wallet.name {args.wallet} --wallet.hotkey {args.hotkey}")


def cmd_check(args) -> None:
    files, problems = scan_model_dir(Path(args.model_dir))
    total = sum(f["size"] for f in files)
    say(f"{len(files)} files, {total / 1e9:.2f} GB in {args.model_dir}")
    if problems:
        say("pre-flight FAILED — the validator would reject this checkpoint:")
        for p in problems:
            say(f"  * {p}")
        raise SystemExit(1)
    say("pre-flight OK")


def cmd_register(args) -> None:
    wallet = load_wallet(args)
    hotkey = wallet.hotkey.ss58_address
    reg_id = registration_id(args.netuid, hotkey)
    sig = wallet.hotkey.sign(activate_message(args.netuid, hotkey, reg_id))
    payload = build_activate_payload(hotkey, sig)
    say(f"hotkey:          {hotkey}")
    say(f"registration_id: {reg_id}")
    say(f"mailbox:         {MAILBOX_BASE_URL}/{mailbox_key(reg_id, 1)}")
    if args.dry_run:
        say(f"would commit: {payload}")
        return
    block_hash = commit_payload(wallet, payload, network=args.network,
                                netuid=args.netuid)
    say(f"committed activate in block {block_hash}; reveals in ~3 min. "
        f"Run `auth` next (it waits for the validator).")


def _poll_mailbox(reg_id: str, timeout_s: float, min_generation: int = 1) -> tuple[int, bytes]:
    deadline = time.time() + timeout_s
    attempt = 0
    while True:
        attempt += 1
        for gen in range(min_generation + 8, min_generation - 1, -1):
            url = f"{MAILBOX_BASE_URL}/{mailbox_key(reg_id, gen)}?poll={attempt}-{uuid.uuid4().hex[:8]}"
            try:
                r = httpx.get(url, timeout=20.0)
            except httpx.HTTPError:
                continue
            if r.status_code == 200 and r.content:
                return gen, r.content
        if time.time() > deadline:
            die("no mailbox blob yet — is the activate reveal on-chain "
                "(≈3 min) and the validator scanning? Check `status`.")
        if attempt == 1 or attempt % 6 == 0:
            say(f"  waiting for the validator to post credentials "
                f"({int(deadline - time.time())}s left)...")
        time.sleep(10)


def _mailbox_exists(reg_id: str, min_generation: int = 1) -> bool:
    """One pass over the plausible generations — is a blob already posted?"""
    for gen in range(min_generation + 8, min_generation - 1, -1):
        try:
            r = httpx.get(f"{MAILBOX_BASE_URL}/{mailbox_key(reg_id, gen)}"
                          f"?probe={uuid.uuid4().hex[:8]}", timeout=20.0)
        except httpx.HTTPError:
            continue
        if r.status_code == 200 and r.content:
            return True
    return False


def cmd_auth(args) -> None:
    wallet = load_wallet(args)
    hotkey = wallet.hotkey.ss58_address
    reg_id = registration_id(args.netuid, hotkey)
    out = Path(args.out)
    gen, blob = _poll_mailbox(reg_id, args.timeout, args.min_generation)
    try:
        plain = wallet.hotkey.decrypt(blob)
    except Exception as e:
        die(f"could not decrypt the mailbox blob with this hotkey ({e}) — is "
            "this the same Ed25519 hotkey that committed activate?")
    env = json.loads(plain)
    missing = set(ENVELOPE_FIELDS) - set(env)
    extra = set(env) - set(ENVELOPE_FIELDS)
    if missing or extra:
        die(f"envelope fields mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    if env["hotkey"] != hotkey or env["registration_id"] != reg_id:
        die("envelope is for a different hotkey/registration")
    if time.time() >= _parse_iso(env["expires_at"]) - 3600:
        die(f"these credentials expire(d) at {env['expires_at']}; run "
            "`register` again for a fresh set (the old prefix contents stay)")
    if not PRIVATE_PREFIX_RE.match(env["allowed_prefix"]) or \
            env["allowed_prefix"] != private_prefix(reg_id):
        die(f"envelope prefix {env['allowed_prefix']!r} is not yours")
    ident = validator_identity(args.validator_identity)
    if ident:
        if env["validator_identity"] != ident:
            die(f"envelope signed by {env['validator_identity']}, expected "
                f"validator {ident}")
        vk = Keypair(ss58_address=ident, crypto_type=ED25519)
        if not vk.verify(envelope_signing_bytes(env),
                         base64.b64decode(env["validator_signature"])):
            die("validator signature on the envelope does not verify")
        say(f"envelope signature OK (validator {ident})")
    else:
        say("WARNING: could not fetch the validator identity from the site; "
            "envelope signature not checked (pass --validator-identity)")
    out.write_text(json.dumps(env, indent=1))
    os.chmod(out, 0o600)
    say(f"credentials (generation {gen}) saved to {out} (mode 0600); "
        f"expire {env['expires_at']}")
    say(f"upload prefix: s3://{env['private_model_bucket']}/{env['allowed_prefix']}")


def cmd_upload(args) -> None:
    wallet = load_wallet(args)
    hotkey = wallet.hotkey.ss58_address
    reg_id = registration_id(args.netuid, hotkey)
    auth = load_auth(Path(args.auth))
    if auth["registration_id"] != reg_id:
        die("upload-auth.json belongs to another registration")
    model_dir = Path(args.model_dir)
    files, problems = scan_model_dir(model_dir)
    if problems:
        say("pre-flight FAILED — fix before uploading (a rejected upload burns your slot):")
        for p in problems:
            say(f"  * {p}")
        raise SystemExit(1)
    bucket, prefix = auth["private_model_bucket"], auth["allowed_prefix"]
    s3 = s3_from_auth(auth)
    say(f"hashing {len(files)} files...")
    for f in files:
        f["sha256"] = sha256_file(model_dir / f["path"])
    # Skip objects already uploaded with the same size + sha256 (resume).
    existing: dict[str, dict] = {}
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        res = s3.list_objects_v2(**kw)
        for o in res.get("Contents", []):
            existing[o["Key"][len(prefix):]] = {"size": int(o["Size"])}
        if not res.get("IsTruncated"):
            break
        token = res.get("NextContinuationToken")
    tcfg = TransferConfig(multipart_threshold=256 << 20, multipart_chunksize=256 << 20,
                          max_concurrency=args.concurrency)
    total = sum(f["size"] for f in files)
    done = 0
    t0 = time.time()
    for f in files:
        key = prefix + f["path"]
        prev = existing.get(f["path"])
        if prev and prev["size"] == f["size"]:
            head = s3.head_object(Bucket=bucket, Key=key)
            if (head.get("Metadata") or {}).get("sha256") == f["sha256"]:
                done += f["size"]
                say(f"  = {f['path']} (already uploaded)")
                continue
        say(f"  ↑ {f['path']} ({f['size'] / 1e9:.2f} GB)")
        s3.upload_file(str(model_dir / f["path"]), bucket, key, Config=tcfg,
                       ExtraArgs={"Metadata": {"sha256": f["sha256"]}})
        done += f["size"]
        rate = done / max(time.time() - t0, 1e-6) / 1e6
        say(f"    {done / 1e9:.1f}/{total / 1e9:.1f} GB, {rate:.0f} MB/s")
    stale = sorted(set(existing) - {f["path"] for f in files} - {"manifest.json"})
    if stale:
        say(f"removing {len(stale)} stale object(s) not in this checkpoint: {stale[:5]}")
        s3.delete_objects(Bucket=bucket, Delete={
            "Objects": [{"Key": prefix + k} for k in stale], "Quiet": True})
    manifest = build_manifest(reg_id=reg_id, hotkey=hotkey,
                              model_name=args.name or model_dir.resolve().name,
                              files=files)
    manifest["signature"] = base64.b64encode(
        wallet.hotkey.sign(manifest_signing_bytes(manifest))).decode("ascii")
    raw = json.dumps(manifest, indent=1).encode()
    s3.put_object(Bucket=bucket, Key=prefix + "manifest.json", Body=raw,
                  ContentType="application/json",
                  Metadata={"sha256": sha256_hex(raw)})
    Path(args.manifest_out).write_bytes(raw)
    # Verify the bucket matches the manifest before letting `ready` go.
    listed = {}
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        res = s3.list_objects_v2(**kw)
        for o in res.get("Contents", []):
            listed[o["Key"][len(prefix):]] = int(o["Size"])
        if not res.get("IsTruncated"):
            break
        token = res.get("NextContinuationToken")
    want = {f["path"]: f["size"] for f in files}
    want["manifest.json"] = len(raw)
    if listed != want:
        die(f"bucket inventory differs from manifest after upload: "
            f"missing={sorted(set(want) - set(listed))[:5]} "
            f"extra={sorted(set(listed) - set(want))[:5]}")
    say(f"upload OK: {len(files)} files, {total / 1e9:.2f} GB")
    say(f"model_digest:    {manifest['model_digest']}")
    say(f"manifest_sha256: {sha256_hex(raw)}  (saved {args.manifest_out})")
    say("next: python submit.py ready")


def cmd_ready(args) -> None:
    wallet = load_wallet(args)
    hotkey = wallet.hotkey.ss58_address
    reg_id = registration_id(args.netuid, hotkey)
    mpath = Path(args.manifest)
    if not mpath.is_file():
        die(f"{mpath} not found — run `upload` first")
    raw = mpath.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("registration_id") != reg_id or manifest.get("hotkey") != hotkey:
        die("manifest.json belongs to another hotkey/registration")
    msha = sha256_hex(raw)
    payload = build_ready_payload(reg_id, msha)
    if args.dry_run:
        say(f"would commit: {payload}")
        return
    block_hash = commit_payload(wallet, payload, network=args.network,
                                netuid=args.netuid)
    auth = Path(args.auth)
    if auth.exists():
        auth.unlink()
        say(f"deleted {auth} (credentials are revoked once ready reveals)")
    say(f"committed ready in block {block_hash}; reveals in ~3 min.")
    say(f"The validator will revoke your upload credential, verify the manifest "
        f"and queue the duel. Track it: python submit.py status --wallet "
        f"{args.wallet} --hotkey {args.hotkey}")


def cmd_status(args) -> None:
    wallet = open_wallet(args)
    hotkey = wallet.hotkey.ss58_address
    reg_id = registration_id(args.netuid, hotkey)
    snap = fetch_json(f"{SITE}/api/v1/snapshot")
    if not snap:
        die(f"could not fetch {SITE}/api/v1/snapshot")
    say(f"hotkey {hotkey}  registration {reg_id[:16]}…")
    kind = "Ed25519" if wallet.hotkey.crypto_type == ED25519 else "NOT Ed25519 (required!)"
    say(f"hotkey type: {kind}")
    r2 = snap.get("submission_r2") or {}
    say(f"R2 intake enabled: {r2.get('enabled')}  hf_cutover_block: {r2.get('hf_cutover_block')}")
    regs = [r for r in snap.get("registrations") or [] if r.get("hotkey") == hotkey]
    for r in regs:
        say(f"registration: state={r.get('state')} gen={r.get('generation')} "
            f"activate_block={r.get('activate_block')} ready_block={r.get('ready_block')}")
        if r.get("detail"):
            say(f"  {r['detail']}")
        if r.get("mailbox_url"):
            say(f"  mailbox: {r['mailbox_url']}")
        if r.get("public_url"):
            say(f"  public:  {r['public_url']}")
    if not regs:
        say("registration: none seen by the validator yet")
    for e in snap.get("intake") or []:
        if e.get("hotkey") == hotkey:
            say(f"intake @{e.get('block')}: {e.get('decision')} — {e.get('detail')}")
    for i, q in enumerate(snap.get("queue") or []):
        if q.get("hotkey") == hotkey:
            say(f"queue position {i + 1}: {q.get('challenge_id')}")
    cur = snap.get("current_eval") or {}
    if cur.get("hotkey") == hotkey:
        say(f"DUELING NOW: {cur.get('challenge_id')} stage={cur.get('stage')}")
    hist = fetch_json(f"{SITE}/api/v1/history") or {}
    rows = hist.get("rows") if isinstance(hist, dict) else hist
    for row in rows or []:
        if row.get("hotkey") == hotkey:
            say(f"history {row.get('at')}: {row.get('event')} "
                f"{row.get('challenge_id')} accepted={row.get('accepted')} "
                f"z={row.get('z')} error={row.get('error_code') or ''} "
                f"{row.get('rejection_reason') or ''}")
    king = snap.get("king") or {}
    if king.get("hotkey") == hotkey:
        say(f"YOU ARE THE KING (reign #{king.get('reign_number')})")


def cmd_submit(args) -> None:
    """register → auth → upload → ready, resuming from local artifacts."""
    wallet = load_wallet(args)
    files, problems = scan_model_dir(Path(args.model_dir))
    if problems:
        say("pre-flight FAILED:")
        for p in problems:
            say(f"  * {p}")
        raise SystemExit(1)
    say(f"[1/4] pre-flight OK ({len(files)} files)")
    auth = Path(args.auth)
    if auth.is_file() and time.time() < _parse_iso(json.loads(auth.read_text())["expires_at"]):
        say(f"[2/4] using existing {auth}")
    else:
        reg_id = registration_id(args.netuid, wallet.hotkey.ss58_address)
        if _mailbox_exists(reg_id, args.min_generation):
            # A previous run already activated; a second activate would
            # revoke those credentials and spend commitment space for nothing.
            say("[2/4] credentials already posted for this hotkey; skipping register")
        else:
            cmd_register(args)
            say("[2/4] waiting for credentials...")
        cmd_auth(args)
    say("[3/4] uploading...")
    cmd_upload(args)
    say("[4/4] signalling ready...")
    cmd_ready(args)
    del wallet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--network", default=NETWORK)
    ap.add_argument("--netuid", type=int, default=NETUID)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def wallet_args(p):
        p.add_argument("--wallet", required=True, help="wallet (coldkey) name")
        p.add_argument("--hotkey", required=True, help="hotkey name (must be Ed25519)")
        p.add_argument("--wallet-path", default=None,
                       help="wallet directory (default ~/.bittensor/wallets)")

    p = sub.add_parser("hotkey", help="create an Ed25519 hotkey")
    wallet_args(p)
    p.add_argument("--mnemonic", default="", help="regenerate from a mnemonic")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_hotkey)

    p = sub.add_parser("check", help="offline pre-flight of a checkpoint dir")
    p.add_argument("--model-dir", required=True)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("register", help="commit activate on-chain")
    wallet_args(p)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_register)

    p = sub.add_parser("auth", help="fetch + decrypt upload credentials")
    wallet_args(p)
    p.add_argument("--out", default="upload-auth.json")
    p.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait")
    p.add_argument("--min-generation", type=int, default=1)
    p.add_argument("--validator-identity", default="",
                   help="validator Ed25519 ss58 (default: fetched from the site)")
    p.set_defaults(fn=cmd_auth)

    p = sub.add_parser("upload", help="upload checkpoint + signed manifest")
    wallet_args(p)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--auth", default="upload-auth.json")
    p.add_argument("--name", default="", help="model_name in the manifest")
    p.add_argument("--manifest-out", default="manifest.json")
    p.add_argument("--concurrency", type=int, default=8)
    p.set_defaults(fn=cmd_upload)

    p = sub.add_parser("ready", help="commit ready on-chain")
    wallet_args(p)
    p.add_argument("--manifest", default="manifest.json")
    p.add_argument("--auth", default="upload-auth.json")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_ready)

    p = sub.add_parser("status", help="what the validator did with your submission")
    wallet_args(p)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("submit", help="register → auth → upload → ready")
    wallet_args(p)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--name", default="")
    p.add_argument("--auth", default="upload-auth.json")
    p.add_argument("--out", default="upload-auth.json")
    p.add_argument("--manifest", default="manifest.json")
    p.add_argument("--manifest-out", default="manifest.json")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--min-generation", type=int, default=1)
    p.add_argument("--validator-identity", default="")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.set_defaults(fn=cmd_submit)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
