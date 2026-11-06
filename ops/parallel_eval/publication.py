"""Private staged signed merge publication/readback/restorable generations.

Uses production Ed25519 envelope verification and ReleasePolicy handover. The
PE generation is EXPLICITLY a distinct type, not a forged gradient-training
Transaction. A production trainer needs an admitted PE-generation restore
adapter and real object-store publisher; this filesystem adapter cannot go live.
"""
import os
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


class PrivatePublisher:
    def __init__(self, root, authority, publisher, policy, current):
        self.stage = Stage(root)
        self.authority, self.publisher, self.policy = authority, publisher, policy
        self.current = current
        if policy.mode != 'private' or authority.id != policy.authority:
            raise ValueError('isolated nonpayable replay policy required')
        self.masters = MasterStore(self.stage)
        registry = self.stage.root / 'ARTIFACTS.json'
        self.registry = read(registry) if registry.exists() else {}
        for name in ('objects', 'generations', 'work'):
            (self.stage.root / name).mkdir(exist_ok=True)

    def register(self, model, master, *, optimizer_step=0, lineage=None):
        model, master = Path(model).resolve(), Path(master).resolve()
        if not model.is_relative_to(self.stage.root) or not master.is_relative_to(self.stage.root):
            raise ValueError('replay inputs must stay inside marked staging root')
        cp = checkpoint(model)
        entry, meta = self.masters.install(model, master, optimizer_step=optimizer_step, lineage=lineage or {})
        row = dict(checkpoint=cp, master_entry=entry, master_digest=meta['master_digest'], model=str(model))
        old = self.registry.get(cp['id'])
        if old:
            if any(old[k] != row[k] for k in ('checkpoint', 'master_entry', 'master_digest')):
                raise ValueError('checkpoint/master registry collision')
            return old
        with self.stage.locked():
            path = self.stage.root / 'ARTIFACTS.json'
            registry = read(path) if path.exists() else {}
            if cp['id'] in registry and any(registry[cp['id']][k] != row[k] for k in ('checkpoint','master_entry','master_digest')):
                raise ValueError('concurrent checkpoint/master collision')
            registry.setdefault(cp['id'], row)
            atomic(path, registry)
            self.registry = registry
        return self.registry[cp['id']]

    def request(self, command_id, base, candidate, training, optimizer_step):
        from subnet.mimo_release_policy import sign
        payload = dict(version='pe-merge-request-v1', command_id=command_id,
                       policy_sha256=self.policy.policy_sha256, optimizer_step=optimizer_step,
                       base=self.registry[base], candidate=self.registry[candidate], training=self.registry[training])
        return sign(self.authority, payload)

    def publish(self, request):
        from subnet.backend_jobs import signed
        from subnet.mimo_release_policy import sign
        from .merge import merge
        payload = signed(request, self.authority.id)
        if payload['version'] != 'pe-merge-request-v1' or payload['policy_sha256'] != self.policy.policy_sha256:
            raise ValueError('merge request policy')
        cmd = payload['command_id']
        if len(cmd) != 64 or any(c not in '0123456789abcdef' for c in cmd):
            raise ValueError('command id')
        base = payload['base']['checkpoint']['id']
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
        for name in ('base', 'candidate', 'training'):
            row = payload[name]
            if self.registry.get(row['checkpoint']['id']) != row or checkpoint(row['model']) != row['checkpoint']:
                raise ValueError('authenticated merge input inventory')
            meta = self.masters.verify(row['master_entry'], row['model'])
            if meta['master_digest'] != row['master_digest']:
                raise ValueError('authenticated master digest')
        model, master = gd / 'model', gd / 'master'
        if base == payload['training']['checkpoint']['id']:
            copy_immutable(payload['candidate']['model'], model)
            source = self.masters.root / payload['candidate']['master_entry']
            if not master.exists():
                master.mkdir()
                for fn in read(source / 'meta.json')['files']:
                    shutil.copyfile(source / fn, master / fn)
        else:
            args = [payload[n]['model'] for n in ('base', 'candidate', 'training')]
            args += [self.masters.root / payload[n]['master_entry'] for n in ('base', 'candidate', 'training')]
            merge(*args, model, master)
        cp = checkpoint(model)
        entry, meta = self.masters.install(model, master, optimizer_step=payload['optimizer_step'],
            lineage=dict(request_sha256=digest(request), base=base,
                         candidate=payload['candidate']['checkpoint']['id'], training=payload['training']['checkpoint']['id']))
        self.masters.pin(entry, 'generation:' + cmd)
        obj = self.stage.root / 'objects' / cp['id']
        copy_immutable(model, obj)
        if checkpoint(obj) != cp:
            raise ValueError('full stored model readback')
        if self.current() != base:
            raise ValueError('actual current head changed during publication')
        receipt = sign(self.publisher, dict(version='pe-model-publication-v1', command_id=cmd,
            request_sha256=digest(request), input_checkpoint=base, descriptor=cp,
            optimizer_step=payload['optimizer_step'], model_full_get_verified=True, optimizer_state_uploaded=False))
        authorization = self.policy.authorize_checkpoint(cp, input_checkpoint=base,
            optimizer_step=payload['optimizer_step'], publication_receipt_sha256=digest(receipt), identity=self.authority)
        # No fake native gradient report: this is an explicitly admitted-new-type
        # requirement for production. Its restoration is tested through restore().
        marker = dict(version='pe-fresh-moments-marker-v1', checkpoint=cp['id'], input_checkpoint=base,
                      optimizer_step=payload['optimizer_step'], persisted=False, recovery='fresh-moments',
                      native_master_entry=entry, native_master_digest=meta['master_digest'])
        atomic(gd / 'optimizer' / 'NOT-PERSISTED.json', marker)
        tx = dict(version='pe-merged-generation-v1', state='published', command_id=cmd,
                  request=request, publication=receipt, checkpoint_authorization=authorization,
                  output_checkpoint=cp['id'], native_master_entry=entry,
                  marker_sha256=digest(marker), authority_committed=False, automatic_retraining=False)
        atomic(gd / 'TRANSACTION.json', tx)
        out = dict(checkpoint=cp, optimizer_step=payload['optimizer_step'], checkpoint_authorization=authorization,
                   publication_receipt_sha256=digest(receipt), receipt=receipt, generation=cmd,
                   transaction_sha256=digest(tx), master_entry=entry)
        atomic(done, out)
        self.verify(out, base)
        return out

    def verify(self, publication, base):
        from subnet.backend_jobs import signed
        value = signed(publication['receipt'], self.publisher.id)
        auth = signed(publication['checkpoint_authorization'], self.authority.id)
        if value['version'] != 'pe-model-publication-v1' or value['input_checkpoint'] != base or value['model_full_get_verified'] is not True or value['command_id'] != publication['generation'] or value['optimizer_step'] != publication['optimizer_step']:
            raise ValueError('publication binding')
        cp = publication['checkpoint']
        if value['descriptor'] != cp or checkpoint(self.stage.root / 'objects' / cp['id']) != cp:
            raise ValueError('stored model full readback')
        if publication['publication_receipt_sha256'] != digest(publication['receipt']):
            raise ValueError('receipt digest')
        if auth['version'] != 'mimo-checkpoint-authorization-v1' or auth['run_id'] != self.policy.run_id or auth['checkpoint'] != cp or auth['input_checkpoint'] != base or auth['optimizer_step'] != publication['optimizer_step'] or auth['publication_receipt_sha256'] != digest(publication['receipt']) or auth['policy_sha256'] != self.policy.policy_sha256:
            raise ValueError('handover authorization')
        return True

    def restore(self, publication, base):
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
        self.masters.verify(publication['master_entry'], gd / 'model')
        return dict(checkpoint=publication['checkpoint'], optimizer_step=publication['optimizer_step'],
                    model=str(gd / 'model'), master=str(self.masters.root / publication['master_entry']),
                    optimizer_recovery='fresh-moments')
