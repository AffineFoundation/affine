"""Generate public Generator fixtures by running PR #297's publication functions.

Usage, from the page worktree:
  PYTHONPATH=/path/to/pr-297-worktree/v2 uv run --with pynacl --no-project \
    python -I v2/tests/generate_generator_fixtures.py \
    --control-plane /path/to/pr-297-worktree/v2

Add ``--out`` only for the default fixture directory. All inputs are fake. No chain,
network, credentials, or production host is used. This development tool imports
the control plane; the checked-in fixtures are its public-file outputs, not
hand-written protocol examples. The normal Node test only reads those outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from nacl.signing import SigningKey

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--control-plane", required=True, type=Path, help="PR #297 worktree's v2 directory")
parser.add_argument("--out", type=Path, default=Path(__file__).parent / "fixtures" / "generator")
arguments = parser.parse_args()
CONTROL, OUT = arguments.control_plane, arguments.out
sys.path.insert(0, str(CONTROL.resolve()))

from sn120_v2.canonical import sha256_hex
from sn120_v2.generator.common import PublicSink, atomic_json, json_bytes
from sn120_v2.generator.control import update_window_state
from sn120_v2.generator.decisions import decision
from sn120_v2.generator.epoch import epoch_deadline, epoch_start
from sn120_v2.generator.links import link_files
from sn120_v2.generator.protocol import sign_submission
from sn120_v2.generator.publish import publish
from sn120_v2.generator.snapshot import build_snapshot

EPOCH = "20261119T1600Z"
LIVE = "20261119T2000Z"
KING = "fixture-king-version"
SECRET = "fixture-only deterministic public test secret"
COMMITMENT = sha256_hex(SECRET.encode())


def fake_hotkey(fill: bytes) -> str:
    raw = b"\x2a" + SigningKey(fill * 32).verify_key.encode()
    suffix = hashlib.blake2b(b"SS58PRE" + raw, digest_size=64).digest()[:2]
    number = int.from_bytes(raw + suffix, "big")
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    result = ""
    while number:
        number, digit = divmod(number, 58)
        result = alphabet[digit] + result
    return result


ALPHA = fake_hotkey(b"\x01")
BETA = fake_hotkey(b"\x02")
IDS = ["00000000000000000001", "00000000000000000002", "00000000000000000003"]
UPLOADS = {
    ALPHA: epoch_start(EPOCH).timestamp() + 7200,
    BETA: epoch_start(EPOCH).timestamp() + 8100,
}


class Store:
    def __init__(self, documents):
        self.documents = documents
        self.uploads = {
            f"private/generator/sub/{EPOCH}/{hotkey}.json": {"size": len(raw), "uploaded_at": UPLOADS[hotkey]}
            for hotkey, raw in documents.items()
        }

    def list(self, prefix):
        return sorted(key for key in self.uploads if key.startswith(prefix))

    def describe(self, key):
        row = self.uploads[key]
        return {"ContentLength": row["size"], "LastModified": row["uploaded_at"]}

    def sha256(self, key):
        return sha256_hex(self.documents[key.rsplit("/", 1)[-1][:-5]])

    def get(self, key):
        return self.documents[key.rsplit("/", 1)[-1][:-5]]


class Pool:
    def row(self, task_id):
        return {"problem": f"fixture question {task_id}", "answer": "42", "source_id": f"fixture-{task_id}"}


class LinksPresigner:
    def presign_get(self, key, ttl_s):
        return "fixture sealed-box-only placeholder, not a URL"

    def presign_put(self, key, ttl_s):
        return "fixture sealed-box-only placeholder, not a URL", {"Content-Type": "application/octet-stream"}


def main():
    # Fixture-only encryption seam; no sealed grant is written to fixtures.
    import sn120_v2.generator.links as links_module
    links_module.seal = lambda link, hotkey: "fixture-redacted-sealed-box"
    out = Path(OUT).resolve()
    expected = Path(__file__).resolve().parent / "fixtures" / "generator"
    if out != expected:
        raise ValueError(f"refusing to replace anything except the fixture directory {expected}")
    out.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="affine-generator-fixtures-")
    work = Path(temporary.name)
    site = work / "site"
    site.mkdir(parents=True)
    sink = PublicSink(site, public_url="https://affine.example")

    ids_a, ids_b = IDS[:2], IDS[2:]
    documents = {
        hotkey: sign_submission(EPOCH, hotkey, ids, lambda message: SigningKey(fill * 32).sign(message).signature, require_activation=False).canonical_json()
        for hotkey, ids, fill in ((ALPHA, ids_a, b"\x01"), (BETA, ids_b, b"\x02"))
    }
    # The real snapshot checks every schema and signature; only private keys and
    # grants are fixture-local and excluded from the public outputs.
    store = Store(documents)
    manifest, job = build_snapshot(
        store, epoch=EPOCH, registered={17: ALPHA, 23: BETA},
        activations={ALPHA, BETA}, known_task_ids=set(IDS), previously_accepted=set(),
        king=KING, seed_commitment=COMMITMENT,
    )
    sink.put(f"epochs/{EPOCH}/manifest.json", json_bytes(manifest))
    # The next live window has a separate manifest built by the same real function.
    store.uploads = {key.replace(EPOCH, LIVE): dict(row, uploaded_at=epoch_start(LIVE).timestamp() + i)
                     for i, (key, row) in enumerate(store.uploads.items(), 1)}
    live_ids = ["00000000000000000004", "00000000000000000005", "00000000000000000006"]
    live_documents = {
        hotkey: sign_submission(LIVE, hotkey, ids, lambda message: SigningKey(fill * 32).sign(message).signature, require_activation=False).canonical_json()
        for hotkey, ids, fill in ((ALPHA, live_ids[:2], b"\x01"), (BETA, live_ids[2:], b"\x02"))
    }
    store.documents.update(live_documents)
    live_manifest, _ = build_snapshot(
        store, epoch=LIVE, registered={17: ALPHA, 23: BETA},
        activations={ALPHA, BETA}, known_task_ids=set(IDS + live_ids), previously_accepted=set(IDS),
        king=KING, seed_commitment=COMMITMENT,
    )
    sink.put(f"epochs/{LIVE}/manifest.json", json_bytes(live_manifest))
    assert job["counts"][ALPHA]["submitted"] == 2
    assert len(job["tasks"]) == 3

    counts = [(4, 8), (1, 8), (6, 16)]
    results = []
    rollouts = {}
    for task, (passes, total) in zip(job["tasks"], counts):
        verdict = decision(passes, total)
        results.append({"task_id": task["task_id"], "s": passes, "n": total, "q": verdict.q, "decision": verdict.action})
        # Every scored task has rollouts; the publisher enforces full coverage.
        rollouts[task["task_id"]] = [
            {"text": f"fixture rollout {j}", "seed": j, "pass": j < passes}
            for j in range(total)
        ]
    output = {"epoch": EPOCH, "king": KING, "seed_commitment": COMMITMENT, "tasks": results, "rollouts": rollouts}
    now = epoch_deadline(LIVE).timestamp() - 2400
    publish(
        sink=sink, job=job, output=output, out_dir=work, secret=SECRET,
        registered={17: ALPHA, 23: BETA}, now=now, pool_rows=Pool(),
        frozen_at=epoch_deadline(EPOCH).timestamp(),
    )

    # Use the real public link-index producer. Its sealed private per-hotkey
    # documents are deliberately not copied into public fixtures.
    issued_at = int(epoch_start(LIVE).timestamp()) + 60
    files = link_files([ALPHA, BETA], LinksPresigner(), issued_at)
    sink.put("links.json", files["generator/links.json"])

    # update_window_state preserves published_epochs and last_epoch while opening LIVE.
    atomic_json(work / "seeds" / f"{LIVE}.json", {"epoch": LIVE, "secret": SECRET, "seed_commitment": COMMITMENT})
    config = SimpleNamespace(work_dir=work, site_dir=site, public_url="https://affine.example")
    update_window_state(config, epoch=LIVE, now=now, store=None)
    state = json.loads(sink.read("state.json"))
    assert state["current_epoch"] == LIVE and state["last_epoch"] == EPOCH
    assert [row["epoch"] for row in state["published_epochs"]] == [EPOCH]
    assert state["window_open"] == "2026-11-19T16:00:00Z"
    assert state["deadline"] == "2026-11-19T20:00:00Z"
    # Redact only test evidence generated by the real publisher. No production
    # values, credentials, URLs, private inputs, or rollout texts enter fixtures.
    for path in (site / "generator").rglob("*.json"):
        document = json.loads(path.read_text())
        if "seed_secret" in document:
            document["seed_secret"] = "fixture-redacted-before-commit"
            path.write_bytes(json_bytes(document))
    for relative in ("state.json", "links.json", "payout.json", "accepted-all.jsonl", f"epochs/{EPOCH}/manifest.json", f"epochs/{EPOCH}/results.json", f"epochs/{EPOCH}/accepted.jsonl", f"epochs/{LIVE}/manifest.json"):
        source = site / "generator" / relative
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    empty_work = work / "empty"
    empty_site = empty_work / "site"
    empty_site.mkdir(parents=True)
    empty_sink = PublicSink(empty_site, public_url="https://affine.example")
    empty_config = SimpleNamespace(work_dir=empty_work, site_dir=empty_site, public_url="https://affine.example")
    atomic_json(empty_work / "seeds" / f"{LIVE}.json", {"epoch": LIVE, "secret": SECRET, "seed_commitment": COMMITMENT})
    update_window_state(empty_config, epoch=LIVE, now=now, store=None)
    shutil.copyfile(empty_sink.path("state.json"), out / "empty-state.json")
    temporary.cleanup()
    print(f"Generated public fixtures from control-plane publication functions in {out}")


if __name__ == "__main__":
    main()
