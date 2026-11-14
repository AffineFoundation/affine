"""The one place where the benchmark service touches v2 state. It reads public objects of the store, with `get`,
`exists` and `presign_get`. Its only write is the result table at `layout.BENCHMARKS`.
The one row that is not of v2, the last king of v1, reads its weights from a store prefix that the operator sets.

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
# The last king of v1 (public record: https://affine.io/v1-data.json, kings[0], the green turn of v1 epoch 1211). It is a
# full checkpoint of v1's training line, of the model family of the v2 launch model: the node serves it as it serves a king.
V1_KING = dict(checkpoint="b6d592c3bb1b439365e82147a47e65d93c083a13444aceae078f8a2053cd5ce3", v1_epoch=1211, uid=920,
               hotkey="5FaTFWogz3AiQbEFRnL9CZcqR1keeypEYkd7f1L5YLbWH1cP")
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


def extra_rows():
    """The rows that are not heads of v2: the last king of v1. Its kind is its own; it has no time of a v2 head, so it is
    the last row of the queue and changes no king row. `source` is the operator's setting BENCH_V1_KING_SOURCE: a prefix
    in the service's store that holds `descriptor.json` ({"id": <the checkpoint>, "files": {<name>: <sha256>}}, the form
    of v1's publication record) and each file under its name. Empty: the row waits and nothing is rented for it."""
    k = V1_KING["checkpoint"]
    return [dict(model_id="v1king-" + k[:16], kind="v1-king", label=f"v1 king E{V1_KING['v1_epoch']} {k[:8]}", base_hash=k, reasoning_parser="qwen3",
                 round_id=None, head_version_id=None, crowned_at=None, uid=V1_KING["uid"], hotkey=V1_KING["hotkey"], v1_epoch=V1_KING["v1_epoch"],
                 source=os.environ.get("BENCH_V1_KING_SOURCE", "").strip().strip("/"))]


def _own_files(m):
    """(file name -> sha256, store key of each file) of a row with a source of its own. Raises KeyError when the
    descriptor is not there, VerifyError when it is not the descriptor of the row's checkpoint."""
    d = json.loads(store().get(m["source"] + "/descriptor.json"))
    files = d.get("files") if isinstance(d, dict) else None
    if not isinstance(files, dict) or not files or d.get("id") != m["base_hash"]:
        raise VerifyError(f"the descriptor at the source of {m['model_id']} is not of checkpoint {m['base_hash'][:12]}")
    for n, h in files.items():
        if not isinstance(n, str) or not n or n.startswith(".") or "/" in n or "\\" in n or not isinstance(h, str) or len(h) != 64:
            raise VerifyError(f"the descriptor at the source of {m['model_id']} has a file name or a sha256 that cannot be used")
    return files, {n: f"{m['source']}/{n}" for n in files}


def absent(m):
    """The name of a weight file of a listed row that the store does not hold, or None when it holds them all.

    The head of a king can be public before its weight files: the box sends them after the next round has opened. An
    object is at its name only when its upload is complete. The answer is never kept: the store is asked each time.
    A store failure other than "not there" is raised. For a row with a source of its own the answer also says that the
    source is not set, or that its descriptor is not there or not the row's."""
    if m.get("source") is not None:
        if not m["source"]:
            return "the source is not set (BENCH_V1_KING_SOURCE)"
        try:
            files, keys = _own_files(m)
        except KeyError:
            return "descriptor.json is not at the source"
        except (VerifyError, ValueError) as e:
            return str(e)
        return next((n for n in sorted(files) if not store().exists(keys[n])), None)
    base_hash = m["base_hash"]
    if base_hash not in _served:
        models()
    return next((n for n in sorted(_served[base_hash].base_files) if not store().exists(layout.public_base_file(base_hash, n))), None)


def presigned(m):
    """What the node needs to fetch one row's weights: the file hashes, and one expiring GET link per file."""
    base_hash = m["base_hash"]
    if m.get("source"):
        files, keys = _own_files(m)
        return dict(descriptor=dict(id=base_hash, files=files), urls={n: store().presign_get(keys[n], URL_TTL_S) for n in files})
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


def table_results():
    """The result rows of the published table, read from the store now (never from the local copy). No table: none."""
    try:
        return json.loads(store().get(layout.BENCHMARKS)).get("results") or []
    except KeyError:
        return []
