"""The one place where the benchmark service reads v2 state. It only reads: `get` and `presign_get`, nothing else.

Applied kings come from the public head log. The walk starts at the current-head pointer and follows the parent links.
Every head version after genesis exists because a round winner was applied, so the walk gives the applied kings with
no call into the controller. When `sn120_v2.public` lands, `heads()` is the only function to change.

The node serves plain weights. So a king is benchmarked from the FOLDED head of its round (its LoRA merged into the
base). A round whose head still has a stacked LoRA is not listed; it appears when its fold is published.
"""
import functools, importlib, json, re
from sn120_v2 import layout
from sn120_v2.errors import VerifyError
from sn120_v2.types import HeadManifest
from . import config as C

URL_TTL_S = 7200  # the node gets links that expire, never a store credential
_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_manifests = {}  # version_id -> HeadManifest. A head version is content-addressed, so a cached one never changes.


@functools.cache
def store():
    module, _, factory = C.STORE.partition(":")
    return getattr(importlib.import_module(module), factory)()


def heads():
    """Verified head manifests from the current head back to genesis, newest first. Genesis is left out."""
    try:
        raw = store().get(layout.CURRENT)
    except KeyError:
        return []  # v2 has not published a head yet
    current = json.loads(raw)
    version_id, out = current.get("version_id") or current["head"]["version_id"], []
    while version_id:
        if version_id not in _manifests:
            _manifests[version_id] = HeadManifest.from_json(store().get(layout.head_manifest(version_id)))
        m = _manifests[version_id]
        if m.head.version_id != version_id:
            raise VerifyError(f"head manifest {version_id[:12]} holds another head")
        if m.head.parent_version_id is None:
            break
        out.append(m); version_id = m.head.parent_version_id
    return out


def folded():
    """{base_hash: manifest} of each applied round's folded head."""
    return {m.head.base_hash: m for m in heads() if m.head.folded}


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
