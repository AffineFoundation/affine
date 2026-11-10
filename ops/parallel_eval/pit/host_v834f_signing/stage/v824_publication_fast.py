"""v8.2.4 candidate: model-only R2 publication without the trainer's own full re-download.

Differences from R2ModelPublisher (v1 receipts):
* Every local model file is read ONCE, sequentially, and hashed while its parts
  are dispatched; the streamed SHA-256 must equal the staged inventory digest
  before the multipart upload is completed (replaces the separate local rehash).
* Multipart parts upload in parallel (bounded global in-flight parts) with a
  per-part Content-MD5 that R2/S3 verifies server-side; the part ETag returned
  must equal that MD5 (transport integrity for every byte).
* After completion a HEAD checks the stored length. The trainer no longer GETs
  all bytes back. The authoritative full-object SHA-256 readback stays on Arbos
  (Lifecycle readback_publication) BEFORE any checkpoint authorization.
* The receipt says exactly that: version v2, model_full_get_verified=False,
  model_upload_verification='streamed-sha256+part-content-md5+head-length'.
* An object that already exists (retry after a lost acknowledgement) is still
  fully GET-verified as in v1; it is never trusted from metadata.
"""
import base64
import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from botocore.exceptions import ClientError

from subnet.storage import canonical
from subnet.mimo_grpo.publication import R2ModelPublisher, _missing, _precondition, VERSION as V1
from subnet.mimo_runtime import validate_descriptor
from subnet.mimo_grpo.data import digest, sha

VERSION = 'affine-mimo-model-publication-v2'
UPLOAD_VERIFICATION = 'streamed-sha256+part-content-md5+head-length'


class FastR2ModelPublisher(R2ModelPublisher):
    def __init__(self, bucket, signer, *, part_concurrency=16, **kwargs):
        kwargs.setdefault('shard_concurrency', 8)
        super().__init__(bucket, signer, **kwargs)
        if type(part_concurrency) is not int or not 1 <= part_concurrency <= 64:
            raise ValueError('bounded part concurrency')
        self.part_concurrency = part_concurrency
        self._parts = ThreadPoolExecutor(max_workers=part_concurrency, thread_name_prefix='r2-part')
        # Bounds buffered part bytes: in-flight <= part_concurrency + shard_concurrency parts.
        self._slots = threading.BoundedSemaphore(part_concurrency + self.shard_concurrency)
        # (key, sha256, size) completed by THIS process with streamed verification. In-memory only,
        # never persisted: a restarted worker falls back to the full-GET path for existing objects.
        self._uploaded = set()
        self._uploaded_lock = threading.Lock()

    def preupload(self, model_directory, model_files, output_checkpoint):
        """Upload immutable model objects while the optimizer state is still being written.

        Issues no descriptor or receipt; the transaction later calls __call__ after `staged`.
        Objects live under the content-addressed checkpoint prefix, so an interrupted
        preupload leaves only unreferenced immutable bytes (state stays `updating`).
        """
        from subnet.mimo_runtime import validate_descriptor as _vd
        descriptor = _vd(dict(id=output_checkpoint, files={n: r['sha256'] for n, r in model_files.items()}))
        directory = Path(model_directory).absolute()
        prefix = self.namespace + '/checkpoints/' + descriptor['id']
        def one(item):
            name, expected = item
            key = prefix + '/model/' + name
            if not self._exists(key) and self._stream_upload(directory / name, key, expected):
                self._head_length(key, expected)
                with self._uploaded_lock:
                    self._uploaded.add((key, expected['sha256'], expected['size']))
        largest_first = sorted(model_files.items(), key=lambda kv: (-kv[1]['size'], kv[0]))
        with ThreadPoolExecutor(max_workers=self.shard_concurrency) as executor:
            list(executor.map(one, largest_first))

    def _exists(self, key):
        self.guard()
        try:
            self.bucket.client.head_object(Bucket=self.bucket.name, Key=key)
            return True
        except ClientError as error:
            if _missing(error):
                return False
            raise

    def _part(self, key, upload_id, index, data):
        try:
            self.guard()
            md5 = hashlib.md5(data)
            response = self.bucket.client.upload_part(Bucket=self.bucket.name, Key=key,
                UploadId=upload_id, PartNumber=index, Body=data, ContentLength=len(data),
                ContentMD5=base64.b64encode(md5.digest()).decode())
            if response['ETag'].strip('"') != md5.hexdigest():
                raise ValueError('part ETag differs from verified Content-MD5: %s#%d' % (key, index))
            return dict(PartNumber=index, ETag=response['ETag'])
        finally:
            self._slots.release()

    def _stream_upload(self, path, key, expected):
        client, bucket = self.bucket.client, self.bucket.name
        size = expected['size']
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            before = os.fstat(fd)
            if before.st_size != size:
                raise ValueError('original local model file changed: ' + str(path))
            checksum = hashlib.sha256()
            if size < self.threshold:
                data = os.pread(fd, size + 1, 0)
                checksum.update(data)
                if len(data) != size or checksum.hexdigest() != expected['sha256']:
                    raise ValueError('original local model file changed: ' + str(path))
                self.guard()
                try:
                    client.put_object(Bucket=bucket, Key=key, Body=data, ContentLength=size,
                        ContentMD5=base64.b64encode(hashlib.md5(data).digest()).decode(),
                        ContentType='application/octet-stream', IfNoneMatch='*')
                except ClientError as error:
                    if not _precondition(error):
                        raise
                    return False  # created concurrently by an original retry: full GET below
                return True
            upload_id = client.create_multipart_upload(Bucket=bucket, Key=key,
                ContentType='application/octet-stream')['UploadId']
            completed, futures = False, []
            try:
                offset, index = 0, 1
                while offset < size:
                    self.guard()
                    if index > 10000:
                        raise ValueError('bounded multipart count')
                    self._slots.acquire()
                    try:
                        data = os.pread(fd, min(self.part_bytes, size - offset), offset)
                        if not data:
                            raise ValueError('local model file truncated: ' + str(path))
                        checksum.update(data)  # in file order, before dispatch
                        futures.append(self._parts.submit(self._part, key, upload_id, index, data))
                    except BaseException:
                        self._slots.release()
                        raise
                    offset += len(data); index += 1
                parts = [f.result() for f in futures]
                after = os.fstat(fd)
                if (checksum.hexdigest() != expected['sha256'] or offset != size or
                        (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) !=
                        (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)):
                    raise ValueError('streamed bytes differ from staged inventory: ' + str(path))
                self.guard()
                try:
                    client.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id,
                        MultipartUpload={'Parts': parts}, IfNoneMatch='*')
                    completed = True
                except ClientError as error:
                    if not _precondition(error):
                        raise
                    return False
                return True
            finally:
                for f in futures:
                    f.cancel()
                if not completed:
                    for f in futures:
                        try: f.result()
                        except BaseException: pass
                    try:
                        client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
                    except ClientError as error:
                        if str(error.response.get('Error', {}).get('Code')) != 'NoSuchUpload':
                            raise
        finally:
            os.close(fd)

    def _head_length(self, key, expected):
        self.guard()
        response = self.bucket.client.head_object(Bucket=self.bucket.name, Key=key)
        if response['ContentLength'] != expected['size']:
            raise ValueError('immutable model object length differs: ' + key)

    def __call__(self, request):
        # Same request validation as v1 (copied semantics via a dry descriptor build).
        self.guard()
        required = ('input_checkpoint', 'output_checkpoint', 'report_sha256')
        if (any(not sha(request.get(k)) for k in required) or
                request.get('optimizer_state_uploaded') is not False or
                request.get('optimizer_state_location') != 'trainer-local' or
                request.get('authority_committed') is not False or
                type(request.get('optimizer_step')) is not int or request['optimizer_step'] < 1 or
                not isinstance(request.get('job_id'), str) or not request['job_id'] or
                not isinstance(request.get('epoch'), str) or not request['epoch']):
            raise ValueError('original model-only transaction publication request')
        files = request['model_files']
        if type(files) is not dict or not files:
            raise ValueError('actual model inventory required')
        for name, row in files.items():
            if (type(row) is not dict or set(row) != {'size', 'sha256'} or
                    type(row['size']) is not int or not 0 < row['size'] <= 64 * 1024**3 or
                    not sha(row['sha256']) or 'optimizer' in name.lower() or
                    Path(name).suffix not in ('.json', '.safetensors', '.txt', '.model', '.jinja', '.tiktoken')):
                raise ValueError('safe model-only inventory')
        descriptor = validate_descriptor(dict(id=request['output_checkpoint'],
            files={name: row['sha256'] for name, row in files.items()}))
        directory = Path(request['model_directory']).absolute()
        if directory.resolve() != directory or not directory.is_dir():
            raise ValueError('original ordinary model directory')
        if set(p.name for p in directory.iterdir()) != set(files):
            raise ValueError('exact local model-only physical inventory')
        prefix = self.namespace + '/checkpoints/' + descriptor['id']
        get_verified = {}

        def publish_file(item):
            name, expected = item
            path = directory/name
            if path.is_symlink():
                raise ValueError('original local model file changed: ' + name)
            key = prefix + '/model/' + name
            with self._uploaded_lock:
                mine = (key, expected['sha256'], expected['size']) in self._uploaded
            if mine:
                self._head_length(key, expected)
                get_verified[name] = False
                return name, dict(key=key, **expected)
            fresh = (not self._exists(key)) and self._stream_upload(path, key, expected)
            if fresh:
                self._head_length(key, expected)
                get_verified[name] = False
            else:
                # Pre-existing object (lost ack / concurrent original retry): full v1 GET proof.
                self._verify(key, expected)
                get_verified[name] = True
            return name, dict(key=key, **expected)

        largest_first = sorted(files.items(), key=lambda kv: (-kv[1]['size'], kv[0]))
        with ThreadPoolExecutor(max_workers=self.shard_concurrency) as executor:
            objects = dict(sorted(executor.map(publish_file, largest_first)))
        self._immutable_json(prefix + '/descriptor.json', descriptor)
        # Deterministic payload (idempotent immutable receipt on lost-ack retries): v2 never
        # asserts a trainer full GET, even when a retry happened to perform one.
        full_get = False
        payload = dict(version=VERSION, job_id=request['job_id'], epoch=request['epoch'],
            input_checkpoint=request['input_checkpoint'], output_checkpoint=descriptor['id'],
            optimizer_step=request['optimizer_step'], report_sha256=request['report_sha256'],
            source_closure_sha256=self.source, descriptor=descriptor, objects=objects,
            model_full_get_verified=full_get, model_upload_verification=UPLOAD_VERIFICATION,
            optimizer_state_uploaded=False, authority_committed=False)
        envelope = dict(payload=payload, signer=self.signer.id,
            signature=base64.b64encode(self.signer.key.sign(canonical(payload)).signature).decode())
        key = self.namespace + '/publications/' + digest(dict(job_id=request['job_id'])) + '.json'
        self._immutable_json(key, envelope)
        self.last_receipt = envelope
        return dict(job_id=request['job_id'], output_checkpoint=descriptor['id'],
            model_full_get_verified=full_get, model_upload_verification=UPLOAD_VERIFICATION,
            optimizer_state_uploaded=False, receipt_sha256=digest(envelope))

    def receipt(self, job_id):
        """Recover the same small signed v2 receipt after a worker restart."""
        import json
        from subnet.backend_jobs import signed
        self.guard()
        key = self.namespace + '/publications/' + digest(dict(job_id=job_id)) + '.json'
        response = self.bucket.client.get_object(Bucket=self.bucket.name, Key=key)
        body = response['Body']
        try:
            if not 0 < response['ContentLength'] <= 4 * 1024**2:
                raise ValueError('bounded signed publication receipt')
            data = body.read(response['ContentLength'] + 1)
            if len(data) != response['ContentLength']:
                raise ValueError('publication receipt length')
        finally:
            body.close()
        envelope = json.loads(data)
        value = signed(envelope, self.signer.id)
        if (canonical(envelope) != data or value.get('version') not in (VERSION, V1) or
                value.get('job_id') != job_id or value.get('source_closure_sha256') != self.source):
            raise ValueError('original publication receipt source/job binding')
        self.last_receipt = envelope
        return envelope
