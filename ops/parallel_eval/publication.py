"""Private staged signed merge publication/readback/restorable generations.

Uses production Ed25519 envelope verification and ReleasePolicy handover. The
PE generation is EXPLICITLY a distinct type, not a forged gradient-training
Transaction. A production trainer needs an admitted PE-generation restore
adapter and real object-store publisher; this filesystem adapter cannot go live.
"""
import os
import fcntl
import tempfile
import uuid
import shutil
from pathlib import Path
from .durable import Stage, atomic, digest, file_sha, read
from .masters import MasterStore


def checkpoint(root):
    root = Path(root)
    files = {}
    for p in sorted(root.iterdir()):
        if p.is_symlink() or not p.is_file() or p.name.startswith('.') or p.suffix not in ('.json', '.txt', '.safetensors', '.model', '.jinja', '.tiktoken'):
            raise ValueError('model-only ordinary file inventory required')
        if 'optimizer' in p.name.lower():
            raise ValueError('optimizer must not be published')
        files[p.name] = file_sha(p)
    if 'config.json' not in files or not any(n.endswith('.safetensors') for n in files) or len(files) > 32:
        raise ValueError('complete model inventory required')
    return dict(id=digest(files), files=files)


def copy_immutable(source, target):
    source, target = Path(source), Path(target)
    if target.exists():
        if checkpoint(source) != checkpoint(target):
            raise ValueError('immutable publication collision')
        return
    tmp = target.with_name('.' + target.name + '.copy')
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(source, tmp)
    for p in tmp.iterdir():
        with p.open('rb') as f:
            os.fsync(f.fileno())
    os.rename(tmp, target)


def copy_master_immutable(source, target):
    """Resume only this generation's copy; incomplete bytes are never admitted."""
    source, target = Path(source), Path(target)
    files = read(source / 'meta.json')['files']
    if any(Path(n).name != n for n in files):
        raise ValueError('ordinary native master filenames')
    if target.exists():
        observed = list(target.iterdir())
        if target.is_symlink() or any(p.is_symlink() or not p.is_file() or p.name not in files for p in observed):
            raise ValueError('unsafe incomplete master inventory')
        complete = {p.name for p in observed} == set(files) and all(
            (target/n).stat().st_size == row['bytes'] and file_sha(target/n) == row['sha256']
            for n,row in files.items())
        if complete:
            return
        # Retain the incomplete evidence; never silently use it or remove an
        # unrelated directory. Caller holds this generation's publication lock.
        os.rename(target, target.with_name('.incomplete-master-' + uuid.uuid4().hex))
    tmp = Path(tempfile.mkdtemp(prefix='.master-copy-', dir=target.parent))
    for name,row in files.items():
        with (source/name).open('rb') as src, (tmp/name).open('wb') as dst:
            shutil.copyfileobj(src,dst,1 << 24);dst.flush();os.fsync(dst.fileno())
        if (tmp/name).stat().st_size != row['bytes'] or file_sha(tmp/name) != row['sha256']:
            raise ValueError('copied native master SHA/size')
    os.rename(tmp,target)
    from .durable import fsync_dir
    fsync_dir(target.parent)


class PrivatePublisher:
    def __init__(self, root, authority, publisher, policy, current, object_store=None):
        self.stage = Stage(root)
        self.authority, self.publisher, self.policy = authority, publisher, policy
        self.current = current
        self.object_store = object_store
        if policy.mode != 'private' or authority.id != policy.authority:
            raise ValueError('isolated nonpayable replay policy required')
        self.masters = MasterStore(self.stage)
        registry = self.stage.root / 'ARTIFACTS.json'
        self.registry = read(registry) if registry.exists() else {}
        for name in ('objects', 'generations', 'work'):
            (self.stage.root / name).mkdir(exist_ok=True)

    def register(self, model, master, *, optimizer_step=0, lineage=None, generation_id=None):
        model, master = Path(model).resolve(), Path(master).resolve()
        if not model.is_relative_to(self.stage.root) or not master.is_relative_to(self.stage.root):
            raise ValueError('replay inputs must stay inside marked staging root')
        cp = checkpoint(model)
        entry, meta = self.masters.install(model, master, optimizer_step=optimizer_step, lineage=lineage or {}, generation_id=generation_id)
        return self._remember_verified(cp, entry, meta, model)

    def _remember_verified(self, cp, entry, meta, model):
        """Record this call's fully verified install; never re-project the same bytes."""
        head = dict(version='pe-head-v1', checkpoint=cp['id'], master_digest=meta['master_digest'], generation_id=meta['generation_id'])
        row = dict(id=digest(head), head=head, generation_id=meta['generation_id'], checkpoint=cp, master_entry=entry, master_digest=meta['master_digest'], model=str(model))
        old = self.registry.get(row['id'])
        if old:
            if any(old[k] != row[k] for k in ('checkpoint', 'master_entry', 'master_digest', 'generation_id')):
                raise ValueError('checkpoint/master registry collision')
            return old
        with self.stage.locked():
            path = self.stage.root / 'ARTIFACTS.json'
            registry = read(path) if path.exists() else {}
            if row['id'] in registry and any(registry[row['id']][k] != row[k] for k in ('checkpoint','master_entry','master_digest','generation_id')):
                raise ValueError('concurrent checkpoint/master collision')
            registry.setdefault(row['id'], row)
            atomic(path, registry)
            self.registry = registry
        return self.registry[row['id']]

    def request(self, command_id, base, candidate, training, optimizer_step):
        from subnet.mimo_release_policy import sign
        payload = dict(version='pe-merge-request-v1', command_id=command_id,
                       policy_sha256=self.policy.policy_sha256, optimizer_step=optimizer_step,
                       base=self.registry[base], candidate=self.registry[candidate], training=self.registry[training])
        return sign(self.authority, payload)

    def publish(self, request, fault=lambda phase: None):
        """Persist authorization BEFORE side effects; serialize one command id."""
        from subnet.backend_jobs import signed
        payload = signed(request, self.authority.id)
        if payload['version'] != 'pe-merge-request-v1' or payload['policy_sha256'] != self.policy.policy_sha256:
            raise ValueError('merge request policy')
        cmd = payload['command_id']
        if not isinstance(cmd,str) or len(cmd) != 64 or any(c not in '0123456789abcdef' for c in cmd):
            raise ValueError('command id')
        gd = self.stage.root / 'generations' / cmd
        gd.mkdir(exist_ok=True)
        if gd.resolve() != gd or (gd/'.publish.lock').is_symlink():
            raise ValueError('generation publication containment')
        with (gd/'.publish.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            p = gd/'REQUEST.json'
            if p.exists():
                if read(p) != request:
                    raise ValueError('immutable signed publication request conflict')
            else:
                atomic(p,request)
            fault('request-durable')
            return self._publish(request,fault)

    def _publish(self, request, fault):
        from subnet.backend_jobs import signed
        from subnet.mimo_release_policy import sign
        from .merge import merge
        payload = signed(request, self.authority.id)
        if payload['version'] != 'pe-merge-request-v1' or payload['policy_sha256'] != self.policy.policy_sha256:
            raise ValueError('merge request policy')
        cmd = payload['command_id']
        if len(cmd) != 64 or any(c not in '0123456789abcdef' for c in cmd):
            raise ValueError('command id')
        base = payload['base']['id']
        base_checkpoint = payload['base']['checkpoint']['id']
        if self.current() != base:
            raise ValueError('actual current head changed before publication')
        gd = self.stage.root / 'generations' / cmd
        gd.mkdir(exist_ok=True)
        done = gd / 'PUBLICATION.json'
        if done.exists():
            out = read(done)
            self.verify(out, base)
            if signed(out['receipt'], self.publisher.id)['request_sha256'] != digest(request):
                raise ValueError('command-id/request replay conflict')
            return out
        verified = set()
        for name in ('base', 'candidate', 'training'):
            row = payload[name]
            if self.registry.get(row['id']) != row:
                raise ValueError('authenticated merge input inventory')
            if row['id'] in verified:
                continue
            if checkpoint(row['model']) != row['checkpoint']:
                raise ValueError('authenticated merge input inventory')
            meta = self.masters.verify(row['master_entry'], row['model'])
            if meta['master_digest'] != row['master_digest']:
                raise ValueError('authenticated master digest')
            verified.add(row['id'])
        model, master = gd / 'model', gd / 'master'
        if base == payload['training']['id']:
            copy_immutable(payload['candidate']['model'], model)
            source = self.masters.root / payload['candidate']['master_entry']
            copy_master_immutable(source,master)
        else:
            args = [payload[n]['model'] for n in ('base', 'candidate', 'training')]
            args += [self.masters.root / payload[n]['master_entry'] for n in ('base', 'candidate', 'training')]
            merge(*args, model, master)
        fault('model-and-master-durable')
        cp = checkpoint(model)
        entry, meta = self.masters.install(model, master, optimizer_step=payload['optimizer_step'],
            generation_id=cmd, lineage=dict(request_sha256=digest(request), base=base,
                         candidate=payload['candidate']['checkpoint']['id'], training=payload['training']['checkpoint']['id']))
        row = self._remember_verified(cp, entry, meta, model)
        self.masters.pin(entry, 'generation:' + cmd)
        obj = self.stage.root / 'objects' / cp['id']
        copy_immutable(model, obj)
        if checkpoint(obj) != cp:
            raise ValueError('full stored model readback')
        if self.current() != base:
            raise ValueError('actual current head changed during publication')
        remote = self.object_store.publish(cp, obj, entry, self.masters.root / entry) if self.object_store else None
        fault('remote-readback-durable')
        receipt = sign(self.publisher, dict(version='pe-model-publication-v1', command_id=cmd,
            request_sha256=digest(request), input_head=base, input_checkpoint=base_checkpoint, descriptor=cp, head=row['head'], head_id=row['id'],
            optimizer_step=payload['optimizer_step'], model_full_get_verified=True, optimizer_state_uploaded=False, object_readback=remote))
        authorization = self.policy.authorize_checkpoint(cp, input_checkpoint=base_checkpoint,
            optimizer_step=payload['optimizer_step'], publication_receipt_sha256=digest(receipt), identity=self.authority)
        # No fake native gradient report: this is an explicitly admitted-new-type
        # requirement for production. Its restoration is tested through restore().
        marker = dict(version='pe-fresh-moments-marker-v1', checkpoint=cp['id'], input_checkpoint=base_checkpoint, input_head=base,
                      optimizer_step=payload['optimizer_step'], persisted=False, recovery='fresh-moments',
                      native_master_entry=entry, native_master_digest=meta['master_digest'])
        atomic(gd / 'optimizer' / 'NOT-PERSISTED.json', marker)
        tx = dict(version='pe-merged-generation-v1', state='published', command_id=cmd,
                  request=request, publication=receipt, checkpoint_authorization=authorization,
                  output_checkpoint=cp['id'], native_master_entry=entry,
                  marker_sha256=digest(marker), authority_committed=False, automatic_retraining=False)
        atomic(gd / 'TRANSACTION.json', tx)
        fault('transaction-durable')
        out = dict(checkpoint=cp, head=row['head'], head_id=row['id'], optimizer_step=payload['optimizer_step'], checkpoint_authorization=authorization,
                   publication_receipt_sha256=digest(receipt), receipt=receipt, generation=cmd,
                   transaction_sha256=digest(tx), master_entry=entry)
        atomic(done, out)
        self.verify(out, base)
        return out

    def verify(self, publication, base):
        from subnet.backend_jobs import signed
        value = signed(publication['receipt'], self.publisher.id)
        auth = signed(publication['checkpoint_authorization'], self.authority.id)
        if value['version'] != 'pe-model-publication-v1' or value['input_head'] != base or value['input_checkpoint'] != self.registry[base]['checkpoint']['id'] or value['model_full_get_verified'] is not True or value['command_id'] != publication['generation'] or value['optimizer_step'] != publication['optimizer_step']:
            raise ValueError('publication binding')
        cp = publication['checkpoint']
        expected_head = dict(version='pe-head-v1', checkpoint=cp['id'], master_digest=self.registry[publication['head_id']]['master_digest'], generation_id=publication['generation'])
        if publication['head'] != expected_head or publication['head_id'] != digest(expected_head) or value['head'] != expected_head or value['head_id'] != publication['head_id']:
            raise ValueError('signed generation-aware head identity')
        if value['descriptor'] != cp or checkpoint(self.stage.root / 'objects' / cp['id']) != cp:
            raise ValueError('stored model full readback')
        if publication['publication_receipt_sha256'] != digest(publication['receipt']):
            raise ValueError('receipt digest')
        if auth['version'] != 'mimo-checkpoint-authorization-v1' or auth['run_id'] != self.policy.run_id or auth['checkpoint'] != cp or auth['input_checkpoint'] != self.registry[base]['checkpoint']['id'] or auth['optimizer_step'] != publication['optimizer_step'] or auth['publication_receipt_sha256'] != digest(publication['receipt']) or auth['policy_sha256'] != self.policy.policy_sha256:
            raise ValueError('handover authorization')
        if self.object_store:
            self.object_store.verify(value['object_readback'], cp, publication['master_entry'])
        elif value.get('object_readback') is not None:
            raise ValueError('remote publication requires its actual readback adapter')
        return True

    def restore(self, publication, base):
        from subnet.backend_jobs import signed
        self.verify(publication, base)
        gd = self.stage.root / 'generations' / publication['generation']
        tx = read(gd / 'TRANSACTION.json')
        marker = read(gd / 'optimizer' / 'NOT-PERSISTED.json')
        if digest(tx) != publication['transaction_sha256'] or digest(marker) != tx['marker_sha256']:
            raise ValueError('restorable generation integrity')
        if (tx['publication'] != publication['receipt'] or tx['checkpoint_authorization'] != publication['checkpoint_authorization']
                or tx['command_id'] != publication['generation'] or tx['output_checkpoint'] != publication['checkpoint']['id']
                or marker['checkpoint'] != publication['checkpoint']['id'] or marker['optimizer_step'] != publication['optimizer_step']
                or checkpoint(gd / 'model') != publication['checkpoint'] or marker['native_master_entry'] != publication['master_entry']):
            raise ValueError('restorable generation model/master binding')
        meta = self.masters.verify(publication['master_entry'], gd / 'model')
        row = self.registry[publication['head_id']]
        if row['master_entry'] != publication['master_entry'] or meta['master_digest'] != publication['head']['master_digest'] or meta['generation_id'] != publication['generation']:
            raise ValueError('restorable signed master/generation selection')
        request = signed(tx['request'], self.authority.id)
        receipt = signed(publication['receipt'], self.publisher.id)
        if (tx.get('version') != 'pe-merged-generation-v1' or tx.get('state') != 'published'
                or request != dict(version='pe-merge-request-v1', command_id=publication['generation'], policy_sha256=self.policy.policy_sha256, optimizer_step=publication['optimizer_step'], base=self.registry[base], candidate=self.registry[request['candidate']['id']], training=self.registry[request['training']['id']])
                or digest(tx['request']) != receipt['request_sha256'] or meta['lineage']['request_sha256'] != receipt['request_sha256']
                or marker.get('version') != 'pe-fresh-moments-marker-v1' or marker.get('persisted') is not False or marker.get('recovery') != 'fresh-moments'):
            raise ValueError('signed restore lineage/fresh-moments contract')
        return dict(checkpoint=publication['checkpoint'], head=publication['head'], head_id=publication['head_id'], optimizer_step=publication['optimizer_step'],
                    model=str(gd / 'model'), master=str(self.masters.root / publication['master_entry']),
                    optimizer_recovery='fresh-moments')
