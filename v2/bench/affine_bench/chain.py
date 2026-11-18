"""Host side: the weights of a king whose full files are not in the store (the head chain).

The box can upload the full files of a head only for some kings. Such a head is a checkpoint. For each other king the
store has the LoRA of its fold, the small files of its base, the sha256 of every folded file (the head manifest), and
the chain record: one step for each fold, from the genesis head to the current one. The node of a row then makes the
king's base itself: it downloads the newest checkpoint before the king and replays each step after it, with the
rebuild tool of the chain code (`python -m sn120_v2.headfold rebuild`). deploy/chain_rebuild.py runs it on the node.

This module decides, for one row, between three answers:
  fetch    the store holds every base file of the king: the node downloads them, as before (fetch.fetch)
  rebuild  the chain record leads from a whole checkpoint to the king, and every file of that way is in the store
  wait     neither is true yet: the row gets no fail, no worker and no pod (daemon.waiting)

It reads the store with `get`, `exists`, `size` and `presign_get`, and it writes nothing.

Settings (environment of the host, each one optional):
  BENCH_CHAIN_REF            full git commit id (40 hex digits) of the chain code that the node runs. Not set: no
                             rebuild, a king without its full files waits, as before. Set and not a full commit id:
                             no rebuild, and the wait says so.
  BENCH_REBUILD_DEVICE       where the node computes a step: `cpu` (default) or `cuda:N`
  BENCH_CHECKPOINT_GRACE_S   a new head is not rebuilt before this age (default 300): the first file of a checkpoint
                             upload can still be on its way. Not used when the chain record has the flag PLANNED.
  BENCH_CHECKPOINT_WAIT_S    a checkpoint upload that is not whole this long after the head was applied (default 3600)
                             is taken as stalled: the row rebuilds, and the table gets an alarm
  BENCH_REBUILD_MAX_STEPS    a king more steps than this after its checkpoint waits for a newer checkpoint (default 12)
"""
import json, os, re, time
from sn120_v2 import layout
from sn120_v2.errors import VerifyError
from . import source

HEAD_CHAIN = "public/head_chain.json"  # `layout.HEAD_CHAIN` of the chain code
SCHEMA = "sn120-head-chain-v1"
# The kernel of a step names the arithmetic of its LoRA product. `fixed-order-v1` gives the same bytes on every machine.
# `torch-matmul-v1` (a fold of `full` mode) depends on the CPU and on the torch build: only a machine like the box
# repeats it. So the node never replays such a step: the row waits for a checkpoint after it.
KERNELS = ("fixed-order-v1",)
WEIGHT_SUFFIX = ".safetensors"  # a base file with this suffix is a weight file; the other ones are small files
# A flag that the box can put into a step of the chain record: true = this head gets a checkpoint (its upload is on its
# way or will start), false = it gets none. The record of today has no such flag; then GRACE_S stands in for it.
PLANNED = "checkpoint_planned"
REF = os.environ.get("BENCH_CHAIN_REF", "").strip()
DEVICE = os.environ.get("BENCH_REBUILD_DEVICE", "").strip() or "cpu"
GRACE_S = float(os.environ.get("BENCH_CHECKPOINT_GRACE_S") or 300)
UPLOAD_WAIT_S = float(os.environ.get("BENCH_CHECKPOINT_WAIT_S") or 3600)
MAX_STEPS = int(os.environ.get("BENCH_REBUILD_MAX_STEPS") or 12)
BASE_S = 3600  # the time limit of a rebuild on the node: this (the download and the cast, as a fetch has) ...
STEP_S = 900  # ... and this for each step (measured for one step on 4 CPU cores: 313 s)
LINK_SLACK_S = 600  # the links live this much longer than the limit
_IDS = ("version_id", "base_hash", "base_fp32_master_digest")
_sizes = {}  # store key -> size. Only an object that was seen is kept: a public object is written once


class Wait(Exception):
    """The row cannot start yet. The text says why."""


def on():
    """True when the node may rebuild: BENCH_CHAIN_REF is one full commit id, and the device is a known one."""
    return bool(re.fullmatch(r"[0-9a-f]{40}", REF)) and bool(re.fullmatch(r"cpu|cuda:\d+", DEVICE))


def size(key):
    """The size of an object of the store. KeyError: it is not there (now)."""
    if key not in _sizes:
        _sizes[key] = int(source.store().size(key))
    return _sizes[key]


def there(key):
    try:
        size(key); return True
    except KeyError:
        return False


def manifest_of(m):
    if m["base_hash"] not in source._served:
        source.models()
    return source._served[m["base_hash"]]


def heads_of(doc):
    """The folded heads of a chain record, oldest first, after a check that fails closed: the steps link up, the first
    head is a checkpoint, and each step names its ids, its kernel and its LoRAs. The tool on the node checks the record
    again, and that the files of each LoRA hash to its id."""
    try:
        if doc["schema"] != SCHEMA:
            raise VerifyError(f"the head chain has the schema {doc['schema']!r}; this code reads {SCHEMA!r}")
        heads = [doc["genesis"] if "genesis" in doc else doc["start"], *doc["steps"]]
        if heads[0]["checkpoint"] is not True:
            raise VerifyError("the head chain does not start at a checkpoint")
        for before, step in zip(heads, heads[1:]):
            if step["from_version_id"] != before["version_id"]:
                raise VerifyError(f"the step to head {step['version_id'][:12]} does not start at the head before it")
            for lora in step["loras"]:
                if not re.fullmatch(r"[0-9a-f]{64}", str(lora["lora_hash"])) or not lora["files"] or int(lora["rank"]) < 1:
                    raise VerifyError(f"a LoRA of the step to head {step['version_id'][:12]} has no id, no file or no rank")
            if not all(re.fullmatch(r"[0-9a-f]{64}", str(step[key])) for key in _IDS) or not isinstance(step["kernel"], str):
                raise VerifyError("a step of the head chain has an id that is not a sha256")
        return heads
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        raise VerifyError(f"the head chain is not complete: {type(exc).__name__}: {exc}") from None


def plan(m):
    """The way to rebuild the king of row `m`: dict(start=<the checkpoint head>, steps=[<step>, ...], stalled=<bool>).

    Call it only for a king with a base file that the store does not hold. Wait: the row cannot start yet.
    The start is the newest checkpoint before the king (way_from). `stalled`: a checkpoint upload of this king was seen
    or announced, and it is not whole UPLOAD_WAIT_S after the head was applied. The way was made from the store as it is
    now; nothing of it is kept."""
    if not on():
        raise Wait("BENCH_CHAIN_REF is not a full commit id, or BENCH_REBUILD_DEVICE is not cpu or cuda:N: no rebuild")
    target, manifest = m["head_version_id"], manifest_of(m)
    try:
        heads = heads_of(json.loads(source.store().get(HEAD_CHAIN)))
    except KeyError:
        raise Wait("the chain record is not in the store") from None
    at = next((i for i, head in enumerate(heads) if head["version_id"] == target), None)
    if at is None:
        raise Wait("the chain record does not name this head yet")
    king = heads[at]
    if (king["base_hash"], king["base_fp32_master_digest"]) != (m["base_hash"], manifest.head.base_fp32_master_digest):
        raise VerifyError(f"the chain record and the manifest of head {target[:12]} name other files")
    if at == 0 or king["checkpoint"] is True:
        raise Wait("the chain record says that this head is a checkpoint, and the store does not hold each of its files")
    # A checkpoint upload puts each file straight at its final name, the base first. The small files of a base are in
    # the store for every king, so only a weight file or a master file says that an upload is on its way.
    weights = [layout.public_base_file(m["base_hash"], n) for n in sorted(manifest.base_files) if n.endswith(WEIGHT_SUFFIX)]
    master = [layout.public_master_file(king["base_fp32_master_digest"], n) for n in sorted(manifest.master_files)]
    age, planned, stalled = time.time() - (m.get("crowned_at") or 0), king.get(PLANNED), False
    if planned is True or any(there(key) for key in weights + master):
        if age <= UPLOAD_WAIT_S:
            raise Wait("the checkpoint of this head is in upload")
        stalled = True
    elif planned is None and age < GRACE_S:
        raise Wait("the head is new: a checkpoint upload may still start")
    first = next(i for i in range(at - 1, -1, -1) if heads[i]["checkpoint"] is True)
    return way_from(heads, first, at) | dict(stalled=stalled)


def way_from(heads, first, at):
    """The way from the checkpoint heads[first] to heads[at]: dict(start, steps, point=<the manifest of the start>).
    Wait: a step that the node does not replay, too many steps, or a file of the way that the store does not hold."""
    start, steps = heads[first], heads[first + 1 : at + 1]
    if start["checkpoint"] is not True or not steps:
        raise Wait(f"head {start['version_id'][:12]} is not a checkpoint before the head to make")
    if len(steps) > MAX_STEPS:
        raise Wait(f"the head is {len(steps)} steps after its checkpoint; the bound is {MAX_STEPS}")
    for step in steps:
        if step["kernel"] not in KERNELS:
            raise Wait(f"the step to head {step['version_id'][:12]} has the kernel {step['kernel']}: the node does not replay it")
    try:
        point = source._manifest(start["version_id"])
    except KeyError:
        raise Wait(f"the manifest of checkpoint {start['version_id'][:12]} is not in the store") from None
    if (point.head.base_hash, point.head.base_fp32_master_digest) != (start["base_hash"], start["base_fp32_master_digest"]):
        raise VerifyError(f"the chain record and the manifest of checkpoint {start['version_id'][:12]} name other files")
    keys = [layout.public_base_file(start["base_hash"], n) for n in sorted(point.base_files)]
    keys += [layout.public_master_file(start["base_fp32_master_digest"], n) for n in sorted(point.master_files)]
    keys += [layout.public_lora_file(lora["lora_hash"], n) for step in steps for lora in step["loras"] for n in sorted(lora["files"])]
    for key in keys:
        if not there(key):
            raise Wait(f"{key} is not in the store")
    return dict(start=start, steps=steps, point=point)


def blocked(m):
    """Why the weights of a listed row cannot be brought to a node now, or None when they can.

    None: the store holds every base file, or the node can rebuild the king from the chain. A store failure other than
    "not there" is raised, as source.absent raises it. A chain record that is wrong is a reason to wait, not a fault of
    the service: the row starts when the record is right, or when the full files are there."""
    name = source.absent(m)
    if not name or m.get("source") is not None or not REF:
        return name
    try:
        plan(m)
    except (Wait, VerifyError) as why:
        return f"{name}; no rebuild: {why}"
    return None


def order(m):
    """What the node of a row needs to rebuild the row's king, or None when the node downloads the full files as before:
    pack() of the way from the newest checkpoint, and `stalled` (plan)."""
    if m.get("source") is not None or not REF:
        return None
    try:
        if not source.absent(m):
            return None
        way = plan(m)
    except (Wait, VerifyError):
        return None  # the download of the full files decides, and its failure is a failed try
    return pack(m, way) | dict(stalled=way["stalled"])


def pack(m, way, device=None):
    """The order of one rebuild on a node: dict(blob=<the input of deploy/chain_rebuild.py>, steps, start, limit_s, ref, device).

    The blob has the file hashes of the king's base (from its head manifest), the part of the chain record from the
    checkpoint to the king, and one expiring GET link for each file of the checkpoint and of each LoRA. A link is a
    secret until it expires."""
    start, steps, point = way["start"], way["steps"], way["point"]
    limit = BASE_S + STEP_S * len(steps); ttl = limit + LINK_SLACK_S; store = source.store()

    def file_set(set_id, files, key_of):
        rows = [dict(name=n, sha256=files[n], size=size(key_of(set_id, n)), url=store.presign_get(key_of(set_id, n), ttl)) for n in sorted(files)]
        return dict(id=set_id, bytes=sum(r["size"] for r in rows), files=rows)

    links = dict(expires_at=int(time.time() + ttl),
                 checkpoints=[dict(version_id=start["version_id"], base=file_set(start["base_hash"], point.base_files, layout.public_base_file),
                                   master=file_set(start["base_fp32_master_digest"], point.master_files, layout.public_master_file))],
                 chain_loras=[file_set(lora["lora_hash"], lora["files"], layout.public_lora_file) | dict(round_id=lora.get("round_id"))
                              for step in steps for lora in step["loras"]])
    tail = dict(schema=SCHEMA, head_version_id=m["head_version_id"], stacked=None, steps=steps,
                start={key: start[key] for key in _IDS} | dict(checkpoint=True, steps_to_head=len(steps)))
    blob = dict(descriptor=dict(id=m["base_hash"], files=dict(manifest_of(m).base_files)), head=m["head_version_id"], chain=tail, links=links)
    return dict(blob=blob, steps=len(steps), start=start["version_id"], limit_s=limit, ref=REF, device=device or DEVICE)
