"""Real R2 upload + full GET readback, private isolated replay prefix only.

The host supplies its existing native Bucket instance. No credential discovery,
raw credential transport, presigned URL logging, or public namespace writes.
"""
import hashlib
from pathlib import Path
from .durable import Stage, digest, file_sha, read
from .masters import MasterStore


class R2Objects:
    def __init__(self, root, bucket):
        self.stage = Stage(root)
        self.bucket = bucket
        self.prefix = 'private/sn120-parallel-eval/replay/' + digest(str(self.stage.root)) + '/'

    def _key(self, relative):
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or str(path) != relative:
            raise ValueError('object key containment')
        return self.prefix + relative

    def _get(self, key):
        if not key.startswith(self.prefix):
            raise ValueError('private replay namespace required')
        body = self.bucket.client.get_object(Bucket=self.bucket.name, Key=key)['Body']
        h, length = hashlib.sha256(), 0
        try:
            for chunk in iter(lambda: body.read(1 << 24), b''):
                h.update(chunk)
                length += len(chunk)
        finally:
            body.close()
        return dict(sha256=h.hexdigest(), bytes=length)

    def publish(self, checkpoint, model, master_entry, master):
        MasterStore.entry_id(master_entry)
        rows = {}
        for category, directory, files in (
            ('models/' + checkpoint['id'], Path(model), checkpoint['files']),
            ('masters/' + master_entry, Path(master), dict(read(Path(master) / 'meta.json')['files'], **{'meta.json':{}})),
        ):
            if not directory.resolve().is_relative_to(self.stage.root):
                raise ValueError('isolated R2 source required')
            for name in sorted(files):
                if Path(name).name != name or (directory / name).is_symlink():
                    raise ValueError('ordinary object filename required')
                path = directory / name
                expected = dict(sha256=file_sha(path), bytes=path.stat().st_size)
                if category.startswith('models/') and expected['sha256'] != files[name]:
                    raise ValueError('model publication source hash')
                key = self._key(category + '/' + name)
                # Content-addressed model keys and generation/master keys. Every
                # upload is followed by a complete GET, not HEAD/ETag shortcuts.
                self.bucket.client.upload_file(str(path), self.bucket.name, key)
                if self._get(key) != expected:
                    raise ValueError('actual R2 full GET mismatch')
                rows[key] = expected
        return dict(version='pe-r2-full-readback-v1', prefix=self.prefix,
                    checkpoint=checkpoint['id'], master_entry=master_entry, objects=rows)

    def verify(self, proof, checkpoint, master_entry):
        if not isinstance(proof, dict) or proof.get('version') != 'pe-r2-full-readback-v1' or proof.get('prefix') != self.prefix or proof.get('checkpoint') != checkpoint['id'] or proof.get('master_entry') != master_entry:
            raise ValueError('R2 proof namespace/checkpoint/master binding')
        model_keys = {self._key('models/' + checkpoint['id'] + '/' + fn): sha for fn, sha in checkpoint['files'].items()}
        meta = read(self.stage.root / 'fp32-master' / MasterStore.entry_id(master_entry) / 'meta.json')
        master_keys = {self._key('masters/' + master_entry + '/' + fn): row['sha256'] for fn, row in meta['files'].items()}
        master_keys[self._key('masters/' + master_entry + '/meta.json')] = file_sha(self.stage.root / 'fp32-master' / master_entry / 'meta.json')
        expected = dict(model_keys, **master_keys)
        if set(proof['objects']) != set(expected):
            raise ValueError('R2 exact model/master inventory')
        for key, sha in expected.items():
            row = proof['objects'][key]
            if row['sha256'] != sha or self._get(key) != row:
                raise ValueError('R2 signed object/full GET integrity')
        return True
