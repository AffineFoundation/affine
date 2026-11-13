"""The one place where the benchmark service reads v2 state. It only reads: the journal file, and `get` and `presign_get`
on the store.

Applied kings come from `sn120_v2.state.load`. It replays the controller's journal read-only, so it is safe while the
controller writes and when the controller is down. There is no call into the controller.

The node serves plain weights. So a king is benchmarked from the FOLDED head of its round (its LoRA merged into the
base). The file hashes of that head come from its manifest in the store.
"""
import functools, importlib, re
from sn120_v2 import layout, state
from sn120_v2.errors import VerifyError
from sn120_v2.types import HeadManifest
from . import config as C

URL_TTL_S = 7200  # the node gets links that expire, never a store credential
_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_manifests = {}  # version_id -> HeadManifest. A head version is content-addressed, so a cached one never changes.


@functools.cache
def store():
    module, _, factory = C.STORE.partition(":")  # "module:function" or "module:Class.method"
    return functools.reduce(getattr, factory.split("."), importlib.import_module(module))()


def folded():
    """{base_hash: manifest} of each applied round's folded head. Genesis and a head with a stacked LoRA are left out."""
    out = {}
    for head in state.load(C.JOURNAL).heads:
        if head.folded and head.parent_version_id and head.version_id not in _manifests:
            try:
                m = HeadManifest.from_json(store().get(layout.head_manifest(head.version_id)))
            except KeyError:
                continue  # the controller has not published this head yet; it is listed on a later poll
            if m.head != head:
                raise VerifyError(f"the published manifest of head {head.version_id[:12]} does not match the journal")
            _manifests[head.version_id] = m
        if head.version_id in _manifests:
            out[head.base_hash] = _manifests[head.version_id]
    return out


def kings():
    """One row per applied king that can be served, oldest first. `king` is the hash of its folded base files."""
    rows = [dict(model_id="king-" + k[:16], king=k, kind="king", green=True, reasoning_parser="qwen3",
                 label=f"{m.head.round_id} {k[:8]}", round_id=m.head.round_id, head_version_id=m.head.version_id,
                 crowned_at=m.head.applied_at) for k, m in folded().items()]
    return sorted(rows, key=lambda r: r["crowned_at"])


def presigned(king):
    """What the node needs to fetch one king: its file hashes, and one expiring GET link per file."""
    files = dict(folded()[king].base_files)
    bad = [n for n in files if not _FILE.fullmatch(n)]
    if bad:
        raise VerifyError(f"unsafe file name in the manifest of king {king[:12]}: {bad[0]!r}")
    prefix = layout.public_base_prefix(king)
    return dict(descriptor=dict(id=king, files=files), urls={n: store().presign_get(f"{prefix}/{n}", URL_TTL_S) for n in files})
