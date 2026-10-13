"""Immutable model-only R2 publication with streamed readback and signed receipts.

No optimizer or discovery credentials are needed by this adapter. The supplied
bucket and role signer are authorized by the host; the sole-trainer lease is
checked throughout publication. An unsupported conditional operation fails
closed rather than falling back to an overwrite.
"""
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath

from botocore.exceptions import ClientError

from ..backend_jobs import signed
from ..storage import canonical
from ..mimo_runtime import file_hash, validate_descriptor
from .data import digest, sha

VERSION = 'affine-mimo-model-publication-v1'


def _missing(error):
    return (isinstance(error, ClientError) and
            str(error.response.get('Error', {}).get('Code')) in ('404', 'NoSuchKey', 'NotFound'))


def _precondition(error):
    return (isinstance(error, ClientError) and
            (str(error.response.get('Error', {}).get('Code')) in ('412', 'PreconditionFailed') or
             error.response.get('ResponseMetadata', {}).get('HTTPStatusCode') == 412))


def _namespace(value):
    p = PurePosixPath(value)
    if (not value or p.is_absolute() or '..' in p.parts or str(p) != value or
            any(not part or part.startswith('.') for part in p.parts)):
        raise ValueError('explicit ordinary bucket namespace')
    return value


class R2ModelPublisher:
    def __init__(self, bucket, signer, *, namespace, source_closure_sha256,
                 assert_trainer_lease, part_bytes=64 * 1024**2,
                 multipart_threshold=64 * 1024**2, read_chunk_bytes=4 * 1024**2):
        if (not sha(source_closure_sha256) or not callable(assert_trainer_lease) or
                type(part_bytes) is not int or not 5 * 1024**2 <= part_bytes <= 256 * 1024**2 or
                type(multipart_threshold) is not int or not 1 <= multipart_threshold <= 5 * 1024**3 or
                type(read_chunk_bytes) is not int or not 1 <= read_chunk_bytes <= 8 * 1024**2):
            raise ValueError('authenticated source and bounded streaming configuration')
        self.bucket, self.signer = bucket, signer
        self.namespace = _namespace(namespace)
        self.source = source_closure_sha256
        self.guard = assert_trainer_lease
        self.part_bytes, self.threshold, self.read_chunk_bytes = part_bytes, multipart_threshold, read_chunk_bytes
        self.last_receipt = None

    def _verify(self, key, expected, *, missing_ok=False):
        self.guard()
        try:
            response = self.bucket.client.get_object(Bucket=self.bucket.name, Key=key)
        except ClientError as error:
            if missing_ok and _missing(error):
                return False
            raise
        body = response['Body']
        try:
            if response['ContentLength'] != expected['size']:
                raise ValueError('immutable model object length differs: ' + key)
            checksum, count = hashlib.sha256(), 0
            while True:
                self.guard()
                chunk = body.read(self.read_chunk_bytes)
                if not chunk:
                    break
                count += len(chunk)
                if count > expected['size']:
                    raise ValueError('remote model exceeded signed size')
                checksum.update(chunk)
            if count != expected['size'] or checksum.hexdigest() != expected['sha256']:
                raise ValueError('full model GET hash differs: ' + key)
        finally:
            body.close()
        return True

    def _upload(self, path, key, size):
        client, bucket = self.bucket.client, self.bucket.name
        self.guard()
        if size < self.threshold:
            try:
                with path.open('rb') as stream:
                    client.put_object(Bucket=bucket, Key=key, Body=stream,
                        ContentLength=size, ContentType='application/octet-stream', IfNoneMatch='*')
            except ClientError as error:
                if not _precondition(error):
                    raise
            return
        upload = client.create_multipart_upload(Bucket=bucket, Key=key,
                                                 ContentType='application/octet-stream')
        upload_id = upload['UploadId']
        completed = False
        try:
            parts = []
            with path.open('rb') as stream:
                index = 1
                while True:
                    self.guard()
                    data = stream.read(self.part_bytes)
                    if not data:
                        break
                    if index > 10000:
                        raise ValueError('bounded multipart count')
                    part = client.upload_part(Bucket=bucket, Key=key, UploadId=upload_id,
                        PartNumber=index, Body=data, ContentLength=len(data))
                    parts.append(dict(PartNumber=index, ETag=part['ETag']))
                    index += 1
            self.guard()
            try:
                client.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id,
                    MultipartUpload={'Parts': parts}, IfNoneMatch='*')
                completed = True
            except ClientError as error:
                if not _precondition(error):
                    raise
                # Another original retry completed the immutable object. The
                # caller still verifies every byte, never trusts this status.
        finally:
            if not completed:
                try:
                    client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
                except ClientError as error:
                    if str(error.response.get('Error', {}).get('Code')) != 'NoSuchUpload':
                        raise

    def _immutable_json(self, key, value):
        data = canonical(value)
        self.guard()
        try:
            self.bucket.client.put_object(Bucket=self.bucket.name, Key=key, Body=data,
                ContentLength=len(data), ContentType='application/json', IfNoneMatch='*')
        except ClientError as error:
            if not _precondition(error):
                raise
        self._verify(key, dict(size=len(data), sha256=hashlib.sha256(data).hexdigest()))

    def __call__(self, request):
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
        objects = {}
        for name, expected in sorted(files.items()):
            path = directory/name
            if path.is_symlink() or path.stat().st_size != expected['size'] or file_hash(path) != expected['sha256']:
                raise ValueError('original local model file changed: ' + name)
            key = prefix + '/model/' + name
            if not self._verify(key, expected, missing_ok=True):
                self._upload(path, key, expected['size'])
                self._verify(key, expected)
            objects[name] = dict(key=key, **expected)
        self._immutable_json(prefix + '/descriptor.json', descriptor)
        payload = dict(version=VERSION, job_id=request['job_id'], epoch=request['epoch'],
            input_checkpoint=request['input_checkpoint'], output_checkpoint=descriptor['id'],
            optimizer_step=request['optimizer_step'], report_sha256=request['report_sha256'],
            source_closure_sha256=self.source, descriptor=descriptor, objects=objects,
            model_full_get_verified=True, optimizer_state_uploaded=False,
            authority_committed=False)
        envelope = dict(payload=payload, signer=self.signer.id,
            signature=base64.b64encode(self.signer.key.sign(canonical(payload)).signature).decode())
        key = self.namespace + '/publications/' + digest(dict(job_id=request['job_id'])) + '.json'
        self._immutable_json(key, envelope)
        self.last_receipt = envelope
        return dict(job_id=request['job_id'], output_checkpoint=descriptor['id'],
            model_full_get_verified=True, optimizer_state_uploaded=False,
            receipt_sha256=digest(envelope))

    def receipt(self, job_id):
        """Recover the same small signed receipt after a worker restart."""
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
        if (canonical(envelope) != data or value.get('version') != VERSION or
                value.get('job_id') != job_id or value.get('source_closure_sha256') != self.source):
            raise ValueError('original publication receipt source/job binding')
        self.last_receipt = envelope
        return envelope


def validate_publication(envelope, *, authority, transaction, source_closure_sha256):
    """Host-side binding before discovery or local CURRENT can advance."""
    value = signed(envelope, authority)
    if (transaction.get('state') != 'published' or value.get('version') != VERSION or
            value.get('job_id') != transaction['job']['job_id'] or
            value.get('epoch') != transaction['job']['epoch'] or
            value.get('input_checkpoint') != transaction['job']['input_checkpoint'] or
            value.get('output_checkpoint') != transaction['output_checkpoint'] or
            value.get('report_sha256') != transaction['report_sha256'] or
            value.get('optimizer_step') != transaction['report']['optimizer_step'] or
            value.get('source_closure_sha256') != source_closure_sha256 or
            transaction['job']['source_closure_sha256'] != source_closure_sha256 or
            value.get('model_full_get_verified') is not True or
            value.get('optimizer_state_uploaded') is not False or
            value.get('authority_committed') is not False or
            digest(envelope) != transaction['publication']['receipt_sha256']):
        raise ValueError('signed publication does not bind original completed update')
    descriptor = validate_descriptor(value['descriptor'])
    model_files = {n.removeprefix('model/'): row['sha256'] for n, row in transaction['artifacts'].items()
                   if n.startswith('model/')}
    if descriptor != dict(id=transaction['output_checkpoint'], files=model_files):
        raise ValueError('publication descriptor differs from original staged model')
    objects = value.get('objects')
    if type(objects) is not dict or set(objects) != set(model_files):
        raise ValueError('complete published object inventory')
    for name, row in objects.items():
        original = transaction['artifacts']['model/' + name]
        if (row.get('sha256') != original['sha256'] or row.get('size') != original['size'] or
                not isinstance(row.get('key'), str) or
                not row['key'].endswith('/checkpoints/' + descriptor['id'] + '/model/' + name)):
            raise ValueError('published object identity differs from original model')
    return value
