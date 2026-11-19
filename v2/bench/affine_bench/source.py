"""The one place where the benchmark service touches v2 state. It reads public objects of the store, with `get`,
`exists` and `presign_get`. Its only writes are the result table at `layout.BENCHMARKS` and public rollout objects made by the row's worker.
The one row that is not of v2, the last king of v1, reads its weight files from a store prefix that the operator sets.

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
# Its files: name -> sha256, from v1's signed opening manifest. The sha256 of the canonical JSON of this map is the
# checkpoint id above (the rule of sn120_v2.canonical.manifest_digest; a test proves it), so the map needs no other record.
# The node checks each file against it after the download.
V1_KING_FILES = {
    "chat_template.jinja": "59a64ebb4df6d1489d09a91267cf3ceb106162d4a893c4f84833cfb8c897ff63",
    "config.json": "44a011dc9cc7203c8b13867f643ef12b7056b2eb8bbf9410704e30a2cec27928",
    "generation_config.json": "cde05246e60160302c49ac904fbc45365d4a98b16e003b4a1ee15cfc7315bd92",
    "model-00001-of-00010.safetensors": "ff7603c5b1b9f3af737de319653abfa9f36640ea86171b2a7f2f1d8637ec3c1e",
    "model-00002-of-00010.safetensors": "e0d3dbbb773813f99ee5c6755a3d5d5c2dfc6cbf5af6ee71507b2efb6ae44975",
    "model-00003-of-00010.safetensors": "22f7d3306cdf77e245c807d66d56dfd6d69c3a7bc09c5517f98dbfb7537e768a",
    "model-00004-of-00010.safetensors": "babfdcac6be9d009a841071c7b283f5dd45ad662695add64ca1e88b946ad1c54",
    "model-00005-of-00010.safetensors": "fe17918dab3b558fb08765ebf2ffd65b808c2e26e244c01b49fd2b75ac275a53",
    "model-00006-of-00010.safetensors": "a6dedc20c92228f14a0bcc28a09cf0998d33f98bb7a35c75c8a1006fd5d4bb58",
    "model-00007-of-00010.safetensors": "3774f926187202c81fa6c467d1dbb2aa22d17b93a3544e6074b2fe59ee63fa54",
    "model-00008-of-00010.safetensors": "642aa3b1d241112ad21a53d4bf2bc9a8a40b6bdc6e1fafc2e6217ba760ec106c",
    "model-00009-of-00010.safetensors": "d387c109dc1de422c5662ce05a0fc2cca82f34e45c43c60fe0673d35dcc7bf28",
    "model-00010-of-00010.safetensors": "19f5f1f58d5932c21aae05b74f8f99a05627c66ad3ac3b9513aa21a3ac45becf",
    "model.safetensors.index.json": "e6b1277a7962e812d1a3f6773ac6c55cc32f203b9af71e2d320e482b744d7608",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
}
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
    the last row of the queue and changes no king row. `source` is the operator's setting BENCH_V1_KING_SOURCE: the prefix
    in the service's store under which the files of V1_KING_FILES are, each under its name (v1 wrote them to
    `<v1 namespace>/checkpoints/<checkpoint>/model`). Empty: the row waits and nothing is rented for it."""
    k = V1_KING["checkpoint"]
    return [dict(model_id="v1king-" + k[:16], kind="v1-king", label=f"v1 king E{V1_KING['v1_epoch']} {k[:8]}", base_hash=k, reasoning_parser="qwen3",
                 round_id=None, head_version_id=None, crowned_at=None, uid=V1_KING["uid"], hotkey=V1_KING["hotkey"], v1_epoch=V1_KING["v1_epoch"],
                 source=os.environ.get("BENCH_V1_KING_SOURCE", "").strip().strip("/"))]


def absent(m):
    """The name of a weight file of a listed row that the store does not hold, or None when it holds them all.

    The head of a king can be public before its weight files: the box sends them after the next round has opened. An
    object is at its name only when its upload is complete. The answer is never kept: the store is asked each time.
    A store failure other than "not there" is raised. For a row with a source of its own the answer also says that the
    source is not set."""
    if m.get("source") is not None:
        if not m["source"]:
            return "the source is not set (BENCH_V1_KING_SOURCE)"
        return next((n for n in sorted(V1_KING_FILES) if not store().exists(f"{m['source']}/{n}")), None)
    base_hash = m["base_hash"]
    if base_hash not in _served:
        models()
    return next((n for n in sorted(_served[base_hash].base_files) if not store().exists(layout.public_base_file(base_hash, n))), None)


def presigned(m):
    """What the node needs to fetch one row's weights: the file hashes, and one expiring GET link per file."""
    base_hash = m["base_hash"]
    if m.get("source"):
        return dict(descriptor=dict(id=base_hash, files=dict(V1_KING_FILES)), urls={n: store().presign_get(f"{m['source']}/{n}", URL_TTL_S) for n in V1_KING_FILES})
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
