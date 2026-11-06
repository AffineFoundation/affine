"""Trainer-native fp32 metadata and durable, non-expiring pending-eval pins.

All writes require an isolated Stage. install_gc_guard must be included in an
admitted trainer release before live use; legacy GC ignores PE-PINS.json.
"""
import contextlib
import fcntl
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from .durable import Stage, atomic, digest, file_sha, fsync_dir, read

VERSION = 'v833f-fp32-head-master-v1'
HEX = re.compile('[0-9a-f]{64}')


def tensor_sha(tensor):
    import torch
    t = tensor.detach().cpu().contiguous()
    raw = t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32) if t.dtype == torch.float32 else t
    import hashlib
    return hashlib.sha256(memoryview(raw.numpy()).cast('B')).hexdigest()


def tensor_map(root):
    from safetensors import safe_open
    result = {}
    for p in sorted(Path(root).glob('*.safetensors')):
        if p.is_symlink():
            raise ValueError('symlink shard')
        with safe_open(str(p), 'pt') as f:
            for name in f.keys():
                if name in result:
                    raise ValueError('duplicate tensor')
                result[name] = p
    if not result:
        raise ValueError('empty tensor inventory')
    return result


def load_tensor(mapping, name):
    from safetensors import safe_open
    with safe_open(str(mapping[name]), 'pt') as f:
        return f.get_tensor(name)


def metadata(model, master, *, optimizer_step, lineage):
    """Digest scheme is exactly native _digest_map, NOT checkpoint file-map id."""
    import torch
    mm, fm = tensor_map(model), tensor_map(master)
    if not set(fm).issubset(mm):
        raise ValueError('master absent from model')
    bf, fp = {}, {}
    for name in fm:
        m, w = load_tensor(fm, name), load_tensor(mm, name)
        if m.dtype != torch.float32 or m.shape != w.shape or not torch.isfinite(m).all():
            raise ValueError('invalid fp32 master')
        if not torch.equal(m.to(w.dtype), w):
            raise ValueError('master projection mismatch')
        fp[name], bf[name] = tensor_sha(m), tensor_sha(w)
    if type(optimizer_step) is not int or optimizer_step < 0:
        raise ValueError('optimizer step')
    files = {p.name: dict(sha256=file_sha(p), bytes=p.stat().st_size) for p in sorted(set(fm.values()))}
    return dict(version=VERSION, bf16_sha256=digest(bf), master_digest=digest(fp),
                tensor_sha256=fp, files=files, tensor_file={k: v.name for k, v in fm.items()},
                optimizer_step=optimizer_step, source='pe-fp32-delta', lineage=lineage,
                shard_bytes=max(row['bytes'] for row in files.values()))


class MasterStore:
    def __init__(self, stage):
        self.stage = stage if isinstance(stage, Stage) else Stage(stage)
        self.root = self.stage.root / 'fp32-master'
        self.root.mkdir(exist_ok=True)

    @contextlib.contextmanager
    def locked(self):
        with (self.root / '.pe-pin.lock').open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    @staticmethod
    def entry_id(entry):
        if type(entry) is not str or not HEX.fullmatch(entry):
            raise ValueError('invalid native master entry id')
        return entry

    def install(self, model, master, *, optimizer_step, lineage):
        if any(not Path(p).resolve().is_relative_to(self.stage.root) for p in (model, master)):
            raise ValueError('master inputs must remain inside staging root')
        meta = metadata(model, master, optimizer_step=optimizer_step, lineage=lineage)
        entry = self.entry_id(meta['bf16_sha256'])
        with self.locked():
            dst = self.root / entry
            if dst.is_symlink():
                raise ValueError('symlink master entry')
            if dst.exists():
                old = read(dst / 'meta.json')
                # Equal bf16 need not imply equal fp32 residuals. Never silently
                # replace a native bf16-key collision with another master.
                if old['master_digest'] != meta['master_digest'] or old['tensor_file'] != meta['tensor_file']:
                    raise ValueError('native bf16-key/master collision')
                self.verify(entry, model)
                return entry, old
            tmp = Path(tempfile.mkdtemp(prefix='.pe-install-', dir=self.root))
            try:
                for fn in meta['files']:
                    with (Path(master) / fn).open('rb') as src, (tmp / fn).open('wb') as out:
                        shutil.copyfileobj(src, out, 1 << 24)
                        out.flush()
                        os.fsync(out.fileno())
                atomic(tmp / 'meta.json', meta)
                os.rename(tmp, dst)
                fsync_dir(self.root)
            finally:
                if tmp.exists():
                    shutil.rmtree(tmp)
        self.verify(entry, model)
        return entry, meta

    def verify(self, entry, model):
        p = self.root / self.entry_id(entry)
        meta = read(p / 'meta.json')
        for fn, row in meta['files'].items():
            if Path(fn).name != fn or (p / fn).is_symlink() or file_sha(p / fn) != row['sha256'] or (p / fn).stat().st_size != row['bytes']:
                raise ValueError('native master shard integrity')
        actual = metadata(model, p, optimizer_step=meta['optimizer_step'], lineage=meta['lineage'])
        if actual != meta or meta['bf16_sha256'] != entry:
            raise ValueError('native metadata integrity')
        return meta

    def pin(self, entry, owner):
        if not isinstance(owner, str) or not owner or len(owner) > 200:
            raise ValueError('pin owner')
        with self.locked():
            p = self.root / self.entry_id(entry)
            meta = read(p / 'meta.json')
            pin = p / 'PE-PINS.json'
            value = read(pin) if pin.exists() else dict(version='pe-durable-pins-v1', entry=entry,
                                                        master_digest=meta['master_digest'], owners=[])
            if value['entry'] != entry or value['master_digest'] != meta['master_digest']:
                raise ValueError('pin/master binding')
            value['owners'] = sorted(set(value['owners']) | {owner})
            atomic(pin, value)

    def unpin(self, entry, owner):
        with self.locked():
            pin = self.root / self.entry_id(entry) / 'PE-PINS.json'
            if not pin.exists():
                return
            value = read(pin)
            value['owners'] = sorted(set(value['owners']) - {owner})
            if value['owners']:
                atomic(pin, value)
            else:
                pin.unlink()
                fsync_dir(pin.parent)


def install_gc_guard(trainer, store):
    """Staged release hook. Original GC runs under the SAME lock as pin writers.

    Marker existence protects even malformed metadata (fail closed). No TTL,
    so a controller outage exceeding the old PIN's three hours cannot prune a
    dependent master. Explicit owner ACK/unpin is the only release mechanism.
    """
    if Path(trainer.FP32_STORE).resolve() != store.root.resolve():
        raise ValueError('GC hook/store mismatch')
    if getattr(trainer._fp32_gc, '_pe_durable', False):
        return
    original = trainer._fp32_gc

    def guarded(protect, emit):
        with store.locked():
            pinned = {p.name for p in store.root.iterdir() if p.is_dir() and (p / 'PE-PINS.json').exists()}
            return original(set(protect) | pinned, emit)
    guarded._pe_durable = True
    trainer._fp32_gc = guarded
