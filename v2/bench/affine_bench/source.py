"""The one place where the benchmark service touches v2 state. It reads public objects of the store, with `get` and
`presign_get`. Its only write is the result table at `layout.BENCHMARKS`.

Rows come from the public chain of heads: the head pointer, and for each head the head it came from. A stacked head
names that head as its parent. A folded head names its round, and the public round file names the head the round opened
on. No journal is read and the controller is not called, so the service runs on any host that can read the store.

The node serves plain weights. So a king is benchmarked from the FOLDED head of its round (its LoRA merged into the
base), and the launch model from the genesis head.
"""
import functools, importlib, json, os
from sn120_v2 import layout
from sn120_v2.errors import VerifyError
from sn120_v2.params import GENESIS_BASE_HASH, GENESIS_MODEL_REPO
from sn120_v2.types import WINNER, HeadManifest
from . import config as C

URL_TTL_S = 7200  # the node gets links that expire, never a store credential
# A head version is content-addressed and the file of a closed round is written once, so a cached one never changes.
_manifests = {}  # version_id -> HeadManifest
_rounds = {}  # round_id -> (version_id of the head the round opened on, uid of the winner, hotkey of the winner)
_served = {}  # base_hash -> HeadManifest of a listed row


@functools.cache
def store():
    module, _, factory = C.STORE.partition(":")  # "module:function" or "module:Class.method"
    return functools.reduce(getattr, factory.split("."), importlib.import_module(module))()


def _manifest(version_id):
    if version_id not in _manifests:
        m = HeadManifest.from_json(store().get(layout.head_manifest(version_id)))
        if m.head.version_id != version_id:
            raise VerifyError(f"the manifest stored for head {version_id[:12]} is of another head")
        _manifests[version_id] = m
    return _manifests[version_id]


def _round(round_id):
    if round_id not in _rounds:
        report = json.loads(store().get(layout.round_file(round_id)))
        if report.get("round_id") != round_id or not report.get("closed_at") or not report.get("head_version_id"):
            raise VerifyError(f"the public file of round {round_id} is not the file of that closed round")
        winner = next((e for e in report.get("entries", []) if e.get("status") == WINNER), {})
        _rounds[round_id] = (report["head_version_id"], winner.get("uid"), winner.get("hotkey"))
    return _rounds[round_id]


def lineage():
    """Manifests of the folded heads from the current head back to genesis, newest first.

    A public file that is not there yet cuts the chain at that point: the older heads are listed on a later poll.
    Any other store failure is raised, so a step never works from a list it could not read."""
    try:
        m = HeadManifest.from_json(store().get(layout.CURRENT))
    except KeyError:
        return []  # no head is published: the validator has not started
    out, seen = [], set()
    while True:
        head = m.head
        if head.version_id in seen:
            raise VerifyError(f"the public chain of heads comes back to head {head.version_id[:12]}")
        seen.add(head.version_id)
        if head.folded:
            out.append(m)
        if head.parent_version_id is None:
            return out
        try:
            m = _manifest(_round(head.round_id)[0] if head.folded else head.parent_version_id)
        except KeyError:
            return out


def models():
    """One row per folded head, oldest first: the genesis head (the launch model), then each fold of a winner.

    `base_hash` is the hash of the plain weight files the node serves. A head with a stacked LoRA has no row."""
    rows = []
    for m in lineage():
        head, k = m.head, m.head.base_hash
        _served[k] = m
        row = dict(base_hash=k, reasoning_parser="qwen3", round_id=head.round_id, head_version_id=head.version_id, crowned_at=head.applied_at)
        if head.parent_version_id is None:
            name = GENESIS_MODEL_REPO.rsplit("/", 1)[-1] if k == GENESIS_BASE_HASH else k[:8]
            row |= dict(model_id="genesis-" + k[:16], kind="genesis", label=f"genesis {name}", uid=None, hotkey=None)
        else:
            try:
                _, uid, hotkey = _round(head.round_id)
            except KeyError:
                uid = hotkey = None  # the round file is not published yet
            row |= dict(model_id="king-" + k[:16], kind="king", label=f"{head.round_id} {k[:8]}", uid=uid, hotkey=hotkey)
        rows.append(row)
    return sorted(rows, key=lambda r: r["crowned_at"])


def presigned(base_hash):
    """What the node needs to fetch one row's weights: the file hashes, and one expiring GET link per file."""
    if base_hash not in _served:
        models()
    files = dict(_served[base_hash].base_files)  # layout refuses a file name that is not one plain path segment
    urls = {n: store().presign_get(layout.public_base_file(base_hash, n), URL_TTL_S) for n in files}
    return dict(descriptor=dict(id=base_hash, files=files), urls=urls)


def upload_table(path):
    """Write the result table to the bucket. The bucket is the durable record; the local file is a cache."""
    store().put(layout.BENCHMARKS, open(path, "rb").read())


def sync_table(path):
    """Replace the local table by the published one. Returns False, and leaves the local file, when none is published."""
    try:
        data = store().get(layout.BENCHMARKS)
    except KeyError:
        return False
    with open(path + ".tmp", "wb") as f: f.write(data)
    os.replace(path + ".tmp", path)
    return True
