"""Parallel multipart transfer of large artifacts to R2 and inside R2, with no download to the host.

A part is a byte range of an object. R2 (the S3 API) takes the parts of one object in parallel and joins them.

  upload  ``plan_upload`` signs one URL for each part of each file. The box tool (pe_r2_box_upload.py) sends
          the parts and returns a manifest: the sha256 of every file and the MD5 of every part.
          ``finish_upload`` joins the parts and checks every object.
  check   The size and the ETag of the object in R2 against the manifest. The ETag is a function of the MD5
          of the parts, so an equal ETag shows that R2 holds the bytes that the box hashed. When R2 gives an
          ETag that does not have this form, the host hashes the object as a stream (slow, and still fail
          closed).
  copy    ``server_copy`` copies an object inside R2 in parts (UploadPartCopy), in parallel. One CopyObject
          call on a 4 GB object runs longer than the read timeout of the client; a part of 256 MiB does not.
          With the part size of the upload, the copy has the ETag of its source.

Settings (environment, all optional):

  PIT_R2_PART_BYTES      bytes in one part; 256 MiB. From 5 MiB (the R2 minimum) to 5 GiB (the R2 maximum).
  PIT_R2_WORKERS         parts in flight at one time; 16.
  PIT_R2_READ_TIMEOUT_S  seconds that one request can wait for data; 900.
"""
from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from ops.parallel_eval.live import pe_r2_box_upload as BOX

PART_BYTES = 256 * 1024 * 1024
MAX_PART_BYTES = 5 * 1024 * 1024 * 1024
WORKERS = 16
READ_TIMEOUT_S = 900
CONNECT_TIMEOUT_S = 30
ATTEMPTS = 5                # tries of one request, in the client and in the box tool
BACKOFF_S = 1.0             # wait before the second try; each later wait is twice as long, up to 30 s
URL_TTL_S = 3 * 3600        # a signed URL of the upload stays valid this long
CONTENT_TYPE = 'application/octet-stream'
NOT_FOUND = ('404', 'NoSuchKey', 'NotFound')
PRECONDITION = ('412', 'PreconditionFailed')


class R2TransferError(RuntimeError):
    """A transfer or its check failed. Nothing was accepted."""


def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise R2TransferError('%s must be an integer: %r' % (name, raw)) from exc
    if not low <= value <= high:
        raise R2TransferError('%s must be from %d to %d: %d' % (name, low, high, value))
    return value


def part_bytes() -> int:
    return _env_int('PIT_R2_PART_BYTES', PART_BYTES, BOX.MIN_PART_BYTES, MAX_PART_BYTES)


def workers() -> int:
    return _env_int('PIT_R2_WORKERS', WORKERS, 1, 256)


def read_timeout_s() -> int:
    return _env_int('PIT_R2_READ_TIMEOUT_S', READ_TIMEOUT_S, 1, 7200)


def transfer_client(bucket: Any) -> Any:
    """A client of the same bucket for large transfers: long read timeout, retries, a connection for each worker."""
    options = getattr(bucket, '_client_options', None)
    if not isinstance(options, Mapping):
        return bucket.client
    config = bucket.client.meta.config.merge(Config(
        connect_timeout=CONNECT_TIMEOUT_S, read_timeout=read_timeout_s(),
        retries={'mode': 'standard', 'total_max_attempts': ATTEMPTS},
        max_pool_connections=max(10, 2 * workers())))
    return boto3.client('s3', **options, config=config)


def _code(exc: ClientError) -> str:
    return str((exc.response.get('Error') or {}).get('Code') or '')


def clean_etag(value: Any) -> str:
    return str(value or '').strip().strip('"')


def head(client: Any, bucket: str, key: str) -> Optional[Dict[str, Any]]:
    """Size and ETag of an object, or None when R2 has no object with this key."""
    try:
        response = client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if _code(exc) in NOT_FOUND:
            return None
        raise
    return dict(bytes=int(response['ContentLength']), etag=clean_etag(response.get('ETag')), raw_etag=response.get('ETag'))


def sha256_of_object(client: Any, bucket: str, key: str) -> str:
    """sha256 of an object, read as a stream. The slow check: one full download, nothing kept."""
    body = client.get_object(Bucket=bucket, Key=key)['Body']
    digest = hashlib.sha256()
    try:
        for chunk in iter(lambda: body.read(1 << 20), b''):
            digest.update(chunk)
    finally:
        body.close()
    return digest.hexdigest()


def _abort(client: Any, bucket: str, key: str, upload_id: str) -> None:
    try:
        client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
    except (ClientError, BotoCoreError):
        pass        # R2 also drops an unfinished upload after its own time limit


# ------------------------------------------------------------------ upload from a box
def plan_upload(bucket: Any, files: Mapping[str, Mapping[str, Any]], *, client: Any = None,
                part: Optional[int] = None, expires: int = URL_TTL_S) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Sign the upload of ``files``: {id: {key, path, bytes}}. -> (job for the box tool, record for ``finish_upload``).

    A file of one part or less goes as one PUT. A larger file is a multipart upload. The box queues the parts
    in the order of ``files``.
    """
    client = client or transfer_client(bucket)
    part = int(part or part_bytes())
    job_files: List[Dict[str, Any]] = []
    record: Dict[str, Dict[str, Any]] = {}
    try:
        for name in files:
            key, size = str(files[name]['key']), int(files[name]['bytes'])
            ranges = BOX.part_ranges(size, part)
            multipart = len(ranges) > 1
            entry = dict(key=key, bytes=size, multipart=multipart, part_bytes=part, parts=len(ranges), upload_id=None)
            record[name] = entry
            box = dict(id=name, path=str(files[name]['path']), bytes=size, multipart=multipart, part_bytes=part)
            existing = head(client, bucket.name, key)
            if existing is not None and existing['bytes'] == size:
                box['have_etag'] = existing['etag']
            if multipart:
                entry['upload_id'] = client.create_multipart_upload(
                    Bucket=bucket.name, Key=key, ContentType=CONTENT_TYPE)['UploadId']
                box['parts'] = [dict(n=number, url=client.generate_presigned_url(
                    'upload_part', ExpiresIn=expires,
                    Params=dict(Bucket=bucket.name, Key=key, UploadId=entry['upload_id'], PartNumber=number)))
                    for number in range(1, len(ranges) + 1)]
            else:
                box['headers'] = {'Content-Type': CONTENT_TYPE}
                box['parts'] = [dict(n=1, url=client.generate_presigned_url(
                    'put_object', ExpiresIn=expires,
                    Params=dict(Bucket=bucket.name, Key=key, ContentType=CONTENT_TYPE)))]
            job_files.append(box)
    except BaseException:
        abort_upload(bucket, record, client=client)
        raise
    job = dict(version=BOX.VERSION, workers=workers(), attempts=ATTEMPTS, timeout_s=read_timeout_s(), files=job_files)
    return job, record


def abort_upload(bucket: Any, record: Mapping[str, Mapping[str, Any]], *, client: Any = None) -> None:
    """Drop every multipart upload of ``record`` that is not joined yet."""
    client = client or transfer_client(bucket)
    for entry in record.values():
        if entry.get('upload_id') and not entry.get('joined'):
            _abort(client, bucket.name, str(entry['key']), str(entry['upload_id']))


def _is_hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in '0123456789abcdef' for c in value)


def _check_manifest_file(name: str, plan: Mapping[str, Any], got: Any) -> Mapping[str, Any]:
    if not isinstance(got, Mapping):
        raise R2TransferError('the box manifest has no file %s' % name)
    if got.get('bytes') != plan['bytes']:
        raise R2TransferError('%s: the box sent %s bytes, the plan has %d' % (name, got.get('bytes'), plan['bytes']))
    hashed = list(got.get('hashed') or [])
    if not _is_hex(got.get('sha256'), 64):
        raise R2TransferError('%s: the box manifest has no sha256' % name)
    if len(hashed) != plan['parts'] or not all(_is_hex(value, 32) for value in hashed):
        raise R2TransferError('%s: the box hashed %d parts, the plan has %d' % (name, len(hashed), plan['parts']))
    if not got.get('reused'):
        if list(got.get('sent') or []) != hashed:
            raise R2TransferError('%s: the bytes sent are not the bytes hashed (the file changed on the box)' % name)
        if len(list(got.get('etags') or [])) != plan['parts']:
            raise R2TransferError('%s: the box manifest has no ETag for every part' % name)
    return got


def finish_upload(bucket: Any, record: Dict[str, Dict[str, Any]], manifest: Any, *, client: Any = None) -> Dict[str, Dict[str, Any]]:
    """Join the parts and check every object against the box manifest. -> {id: {key, bytes, sha256, etag, ...}}.

    Fail closed: on a failure every unfinished multipart upload is dropped and nothing is returned.
    """
    client = client or transfer_client(bucket)
    try:
        if not isinstance(manifest, Mapping) or manifest.get('version') != BOX.VERSION:
            raise R2TransferError('the box returned no upload manifest')
        if not manifest.get('ok'):
            raise R2TransferError('the box upload failed: %s' % '; '.join(map(str, manifest.get('errors') or []))[:400])
        got = manifest.get('files') or {}
        checked = {name: _check_manifest_file(name, plan, got.get(name)) for name, plan in record.items()}
        out: Dict[str, Dict[str, Any]] = {}
        for name, plan in record.items():
            box = checked[name]
            if plan['upload_id'] and box.get('reused'):
                _abort(client, bucket.name, plan['key'], plan['upload_id'])
                plan['joined'] = True
            elif plan['upload_id']:
                parts = [dict(PartNumber=number, ETag=str(etag)) for number, etag in enumerate(box['etags'], 1)]
                _retry(lambda: client.complete_multipart_upload(
                    Bucket=bucket.name, Key=plan['key'], UploadId=plan['upload_id'],
                    MultipartUpload={'Parts': parts}), 'join of %s' % plan['key'])
                plan['joined'] = True
            out[name] = dict(
                check_object(client, bucket.name, plan['key'], size=plan['bytes'], sha256=box['sha256'],
                             etag=BOX.object_etag(box['hashed'], plan['multipart'])),
                key=plan['key'], sha256=box['sha256'], multipart=plan['multipart'], part_bytes=plan['part_bytes'],
                parts=plan['parts'], reused=bool(box.get('reused')))
        return out
    except (ClientError, BotoCoreError, BOX.UploadError) as exc:
        abort_upload(bucket, record, client=client)
        raise R2TransferError('R2 upload could not be finished: %s' % str(exc)[:300]) from exc
    except BaseException:
        abort_upload(bucket, record, client=client)
        raise


def check_object(client: Any, bucket: str, key: str, *, size: int, sha256: str, etag: Optional[str]) -> Dict[str, Any]:
    """Check an object in R2 against a size, an expected ETag and a sha256. -> {bytes, etag, verify}.

    ``verify`` is 'etag' when the ETag of R2 is the expected one (no download), and 'download' when the host
    had to hash the object. Raises when the object is absent, has another size, or has another sha256.
    """
    found = head(client, bucket, key)
    if found is None:
        raise R2TransferError('R2 has no object %s' % key)
    if found['bytes'] != int(size):
        raise R2TransferError('%s has %d bytes in R2, expected %d' % (key, found['bytes'], int(size)))
    if etag and found['etag'] == etag:
        return dict(bytes=found['bytes'], etag=found['etag'], verify='etag')
    if len(str(sha256 or '')) != 64:
        raise R2TransferError('%s: R2 ETag %s is not the expected %s, and there is no sha256 to check' % (
            key, found['etag'], etag))
    got = sha256_of_object(client, bucket, key)
    if got != sha256:
        raise R2TransferError('%s: sha256 in R2 is %s, expected %s' % (key, got[:12], str(sha256)[:12]))
    return dict(bytes=found['bytes'], etag=found['etag'], verify='download')


# ------------------------------------------------------------------ copy inside R2
def _retry(call: Callable[[], Any], what: str) -> Any:
    """Run one R2 request again after a transport fault or a server fault. The client also retries on its own."""
    for attempt in range(ATTEMPTS):
        try:
            return call()
        except ClientError as exc:
            status = int((exc.response.get('ResponseMetadata') or {}).get('HTTPStatusCode') or 0)
            if _code(exc) in PRECONDITION or (status < 500 and status != 429) or attempt + 1 == ATTEMPTS:
                raise R2TransferError('%s failed: %s' % (what, str(exc)[:300])) from exc
        except BotoCoreError as exc:
            if attempt + 1 == ATTEMPTS:
                raise R2TransferError('%s failed: %s' % (what, str(exc)[:300])) from exc
        time.sleep(min(30.0, BACKOFF_S * 2.0 ** attempt))
    raise R2TransferError('%s failed' % what)


def server_copy(bucket: Any, source: str, destination: str, *, size: Optional[int] = None, etag: Optional[str] = None,
                part: Optional[int] = None, client: Any = None) -> Dict[str, Any]:
    """Copy ``source`` to ``destination`` inside R2, in parallel parts. -> {bytes, source_etag, etag, parts}.

    ``size`` and ``etag`` are what the upload recorded for the source; a source that differs is refused. Every
    part copy has the condition that the source still has the ETag read at the start.
    """
    client = client or transfer_client(bucket)
    found = head(client, bucket.name, source)
    if found is None:
        raise R2TransferError('source object missing: %s' % source)
    if size is not None and found['bytes'] != int(size):
        raise R2TransferError('%s has %d bytes, the record has %d' % (source, found['bytes'], int(size)))
    if etag and found['etag'] != clean_etag(etag):
        raise R2TransferError('%s changed after its upload: ETag %s, the record has %s' % (source, found['etag'], etag))
    ranges = BOX.part_ranges(found['bytes'], int(part or part_bytes()))
    origin = {'Bucket': bucket.name, 'Key': source}
    if len(ranges) == 1:
        _retry(lambda: client.copy_object(Bucket=bucket.name, Key=destination, CopySource=origin,
                                          CopySourceIfMatch=found['raw_etag']), 'copy of %s' % source)
    else:
        upload_id = _retry(lambda: client.create_multipart_upload(
            Bucket=bucket.name, Key=destination, ContentType=CONTENT_TYPE), 'copy of %s' % source)['UploadId']

        def copy_part(number: int, offset: int, length: int) -> Dict[str, Any]:
            result = _retry(lambda: client.upload_part_copy(
                Bucket=bucket.name, Key=destination, UploadId=upload_id, PartNumber=number, CopySource=origin,
                CopySourceRange='bytes=%d-%d' % (offset, offset + length - 1), CopySourceIfMatch=found['raw_etag']),
                'copy of part %d of %s' % (number, source))
            return dict(PartNumber=number, ETag=result['CopyPartResult']['ETag'])

        try:
            with ThreadPoolExecutor(workers()) as pool:
                futures = [pool.submit(copy_part, number, offset, length)
                           for number, (offset, length) in enumerate(ranges, 1)]
                parts = [future.result() for future in futures]
            _retry(lambda: client.complete_multipart_upload(
                Bucket=bucket.name, Key=destination, UploadId=upload_id, MultipartUpload={'Parts': parts}),
                'copy of %s' % source)
        except BaseException:
            _abort(client, bucket.name, destination, upload_id)
            raise
    copied = head(client, bucket.name, destination)
    if copied is None or copied['bytes'] != found['bytes']:
        raise R2TransferError('copy of %s is incomplete at %s' % (source, destination))
    return dict(bytes=copied['bytes'], source_etag=found['etag'], etag=copied['etag'], parts=len(ranges))


def abort_stale_uploads(bucket: Any, prefix: str, *, client: Any = None) -> int:
    """Drop every unfinished multipart upload below ``prefix``. -> how many. Their parts cost storage until then."""
    client = client or transfer_client(bucket)
    dropped, marker = 0, {}
    while True:
        page = client.list_multipart_uploads(Bucket=bucket.name, Prefix=prefix, **marker)
        for upload in page.get('Uploads') or []:
            _abort(client, bucket.name, upload['Key'], upload['UploadId'])
            dropped += 1
        if not page.get('IsTruncated') or not page.get('NextKeyMarker'):
            return dropped
        marker = dict(KeyMarker=page['NextKeyMarker'], UploadIdMarker=page.get('NextUploadIdMarker') or '')
