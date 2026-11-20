#!/usr/bin/env python3
"""Opt-in Generator rehearsal on real store/scorer infrastructure, with scratch output.

Only the fixed private rehearsal R2 object and /root/gen/jobs/<epoch> on the pod
may be created. Nothing is deleted outside scratch and no public mirror is used.
The clock is an explicit test-only simulation; outputs are not production data.
Run from v2/: python ../tools/generator_rehearsal.py --config ~/sn120-generator/config.json --scratch DIR
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import secrets
import shlex
import sys
import time
import traceback
import re
import urllib.request
from pathlib import Path

from nacl.signing import SigningKey

from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.errors import ConfigError, V2Error
from sn120_v2.generator import control
from sn120_v2.generator.common import atomic_json, open_store, read_json, read_jsonl, utc_text
from sn120_v2.generator.config import Config
from sn120_v2.generator.epoch import epoch_deadline, epoch_id, epoch_start, submission_object
from sn120_v2.generator.links import GeneratorPresigner, link_files
from sn120_v2.generator.pool import PriorPool
from sn120_v2.generator.remote import REMOTE_ROOT, ScorerRemote
from sn120_v2.links import activation_message

EPOCH = "20261119T1600Z"
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
AUDIT = "rehearsal-audit.json"
REAL_CLOCK = time.time


def ss58(key: SigningKey) -> str:
    body = bytes([42]) + bytes(key.verify_key)
    number = int.from_bytes(body + hashlib.blake2b(b"SS58PRE" + body, digest_size=64).digest()[:2], "big")
    text = ""
    while number:
        number, digit = divmod(number, 58)
        text = B58[digit] + text
    return text


def load_kit():
    source = Path(__file__).resolve().parents[1] / "v2/miner/generator_miner.py"
    spec = importlib.util.spec_from_file_location("generator_rehearsal_miner", source)
    if spec is None or spec.loader is None:
        raise ConfigError("cannot load the Generator miner kit")
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)
    return kit


def validate_paths(config: Config, scratch: Path) -> None:
    scratch = scratch.resolve()
    original_site = config.site_dir.resolve()
    original_work = config.work_dir.resolve()
    for original in (original_site, original_work):
        if scratch == original or original in scratch.parents:
            raise ConfigError("scratch must not be the production work/site directory or contain it")
    scratch.mkdir(parents=True, mode=0o700)
    scratch.chmod(0o700)
    site, work = scratch / "site", scratch / "work"
    for directory in (site, work):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
    if scratch.stat().st_mode & 0o077:
        raise ConfigError("scratch parent must not be group/world accessible")


def sanitized_error(exc: BaseException) -> dict:
    """Diagnose a failed run without echoing capabilities or secret key material."""
    message = re.sub(r"https?://[^\s]+", "[URL]", str(exc))
    message = re.sub(r"(?i)(?:secret|token|signature|credential|private[_ -]?key|seed)\s*[=:]\s*\S+", "[redacted]", message)
    message = re.sub(r"\b[0-9a-fA-F]{64,}\b", "[redacted-hex]", message)
    frames = [f"{Path(frame.filename).name}:{frame.lineno} {frame.name}" for frame in traceback.extract_tb(exc.__traceback__)[-4:]]
    return {"type": type(exc).__name__, "message": message[:2000], "frames": frames}


def isolated_config(config: Config, scratch: Path) -> Config:
    return Config(
        site_dir=scratch / "site", work_dir=scratch / "work", pool_dir=config.pool_dir,
        network=config.network, public_url=config.public_url, credentials_file=config.credentials_file,
        secret_file=config.secret_file, ssh=config.ssh, head_url=config.head_url, first_epoch=EPOCH,
    )


def upload_simulated(store, hotkey: str, key: SigningKey, task_ids: list[str], simulated_start: float) -> str:
    """Time-shift only the link/miner/store clock, not protocol evidence or production."""
    kit = load_kit()
    presigner = GeneratorPresigner(store)
    files = link_files([hotkey], presigner, simulated_start + 60)
    outer = json.loads(files[f"generator/links/{hotkey}.json"])
    opened = kit.open_sealed_link(outer, hotkey, key, int(simulated_start + 61))
    document = kit.batch_document(EPOCH, hotkey, task_ids, key)
    raw = canonical_json(document)
    kit.check_document(raw)
    kit.upload(opened, raw, replace=True)
    return opened["object_key"]


class RehearsalClock:
    """Injected into store metadata only for this process's past-epoch rehearsal."""
    def __init__(self, store, when: float):
        self.store = store
        self.when = when

    def describe(self, key):
        row = dict(self.store._head(key))
        row["LastModified"] = self.when
        return row

    def __getattr__(self, name):
        return getattr(self.store, name)


def read_public_head(url: str) -> tuple[str, dict]:
    request = urllib.request.Request(url, headers={"User-Agent": "sn120-generator-rehearsal/1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as reply:  # noqa: S310 - operator-configured public URL
            raw = reply.read(1024 * 1024 + 1)
    except OSError as exc:
        raise ConfigError(f"cannot read current public head: {type(exc).__name__}") from exc
    if len(raw) > 1024 * 1024:
        raise ConfigError("public head document is too large")
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise ConfigError("current public head is not valid JSON") from exc
    from sn120_v2.generator.snapshot import head_version
    return head_version(document), document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)
    original = Config.load(args.config.expanduser())
    scratch = args.scratch.expanduser().resolve()
    validate_paths(original, scratch)
    if not original.ssh:
        raise ConfigError("rehearsal requires the configured scorer SSH endpoint")
    if original.first_epoch <= EPOCH:
        raise ConfigError("the production first_epoch must be later than the rehearsal epoch")
    started = REAL_CLOCK()
    simulated = epoch_deadline(EPOCH).timestamp() + 60
    audit_path = scratch / AUDIT
    if audit_path.exists():
        raise ConfigError("rehearsal audit already exists; preserve it and choose fresh scratch")
    atomic_json(audit_path, {
        "rehearsal": True, "epoch": EPOCH, "clock": "test-only simulated epoch time",
        "started_at": utc_text(started), "simulated_epoch_time": utc_text(simulated),
        "store_metadata_clock": "test-only injected LastModified for snapshot admission",
        "public_mirroring": False,
        "production_config_used_for": ["pool_dir", "credentials_file", "ssh", "head_url"],
        "note": "Not a production epoch record. No chain reads/writes or payout writer are used.",
    }, private=True)
    store = open_store(original.credentials_file)
    pool = PriorPool(original.pool_dir)
    key = SigningKey(secrets.token_bytes(32))
    hotkey = ss58(key)
    object_key = submission_object(EPOCH, hotkey)
    pod_job = f"{REMOTE_ROOT}/jobs/{EPOCH}"
    audit = {
        "epoch": EPOCH, "rehearsal": True, "hotkey": hotkey,
        "r2_object": object_key, "pod_job": pod_job,
        "proposed_cleanup_paths": [object_key, pod_job],
        "deletion_performed": False,
    }
    remote = None
    stage = "store-preflight"
    try:
        if store.exists(object_key):
            raise ConfigError("refusing to overwrite an existing rehearsal submission")
        # A throwaway hotkey has no chain identity. The rehearsal supplies this exact fake registration.
        signature = key.sign(activation_message(hotkey, 120)).signature
        activation = "affine2|activate|" + hotkey + "|" + base64.urlsafe_b64encode(signature).decode().rstrip("=")
        from sn120_v2.links import valid_activation
        assert valid_activation(activation, hotkey, 120)
        control.read_registrations = lambda client, clock: ({1: hotkey}, {hotkey})
        ids = [row["id"] for _, row in zip(range(6), pool.iter_rows(0))]
        print(f"Rehearsal epoch {EPOCH}: six real prior-v1 tasks; throwaway identity only.")
        stage = "upload"
        object_key = upload_simulated(store, hotkey, key, ids, epoch_start(EPOCH).timestamp())
        audit["uploaded"] = True
        if not store.exists(object_key):
            raise ConfigError("rehearsal submission was not found after upload")
        config = isolated_config(original, scratch)
        original_sink = control.PublicSink
        control.PublicSink = lambda site_dir, store, public_url, **kwargs: original_sink(site_dir, None, public_url, mirror=False)
        control.write_seed(config.work_dir, EPOCH)
        control.write_seed(config.work_dir, epoch_id(simulated))
        control.write_seed(config.work_dir, EPOCH)
        # Actual public head and actual capture time, not a fabricated past capture.
        stage = "read-public-head"
        head, _ = read_public_head(config.head_url)
        audit["actual_head_observed_at"] = utc_text(REAL_CLOCK())
        audit["simulated_king_capture_at"] = utc_text(simulated)
        atomic_json(config.work_dir / "king" / f"{EPOCH}.json", {
            "version": head, "epoch": EPOCH, "captured_at": utc_text(simulated),
            "captured_at_epoch": simulated,
        }, private=True)
        clocked_store = RehearsalClock(store, epoch_start(EPOCH).timestamp() + 120)
        print("Snapshotting the real signed upload into isolated scratch work.")
        stage = "snapshot"
        control.snapshot_epoch(config, epoch=EPOCH, now=simulated, client=None, store=clocked_store, pool=pool)
        if not (config.work_dir / "jobs" / EPOCH / "job.json").is_file():
            raise ConfigError("rehearsal snapshot did not produce a scoring job")
        remote = ScorerRemote.from_config(original.ssh, config.work_dir)
        if remote.runner.ssh(f"if [ -e {shlex.quote(pod_job)} ]; then printf occupied; fi", timeout=30).stdout.strip():
            raise ConfigError("refusing to overwrite an existing pod rehearsal job")
        stage = "dispatch"
        remote.dispatch(EPOCH)
        print("Scorer dispatched. Polling its durable completion marker (no secrets printed).")
        stage = "collect"
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            status = remote.collect(EPOCH)
            if status.complete:
                break
            time.sleep(10)
        else:
            raise TimeoutError("rehearsal scorer did not complete within the timeout; evidence is preserved")
        output = read_json(config.work_dir / "epochs" / EPOCH / "out" / "results.json")
        stage = "publish"
        control.publish_epoch(config, epoch=EPOCH, now=simulated, client={1: hotkey}, store=None, pool=pool)
        public = config.site_dir / "generator"
        results = read_json(public / "epochs" / EPOCH / "results.json")
        intake_path = public / "epochs" / EPOCH / "intake.jsonl"
        raw_intake = intake_path.read_bytes()
        intake = read_jsonl(raw_intake)
        expected = f"{sha256_hex(raw_intake)}  intake.jsonl\n".encode()
        actual = (public / "epochs" / EPOCH / "intake.sha256").read_bytes()
        if actual != expected:
            raise ConfigError("rehearsal intake checksum does not match")
        audit.update({
            "status": "complete", "decisions": {row["id"]: row["decision"] for row in results["tasks"]},
            "accepted": results["counts"]["accepted"], "intake_lines": len(intake),
            "intake_sha256": sha256_hex(raw_intake),
            "scorer_epoch": output["epoch"], "scratch_site": str(public),
        })
        print(json.dumps({
            "rehearsal": True, "epoch": EPOCH, "decisions": audit["decisions"],
            "accepted": audit["accepted"], "intake_lines": audit["intake_lines"],
            "intake_sha256_ok": True, "scratch": str(scratch),
            "preserved_for_review": audit["proposed_cleanup_paths"],
        }, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        audit["status"] = "failed"
        audit["stage"] = stage
        audit["error"] = sanitized_error(exc)
        print(f"Rehearsal failed during {stage}: {audit['error']['message']}", file=sys.stderr)
        for frame in audit["error"]["frames"]:
            print(f"  {frame}", file=sys.stderr)
        print("Evidence is preserved for review; no capability URLs are printed.", file=sys.stderr)
        return 1
    finally:
        audit["deletion_performed"] = False
        atomic_json(audit_path, audit, private=True)
        print(f"Rehearsal audit: {audit_path}")
        print("Proposed cleanup only; nothing outside scratch was deleted:")
        for path in audit["proposed_cleanup_paths"]:
            print(f"  {path}")


if __name__ == "__main__":
    raise SystemExit(main())
