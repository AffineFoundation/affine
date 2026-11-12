"""The R2 adapter of the object store (the S3 API of Cloudflare R2).

What the controller needs from R2, and how this adapter gives it:

* Compare-and-swap: ``put`` sends ``If-None-Match: *`` or ``If-Match: <etag>``.
  R2 refuses the write with HTTP 412 when the condition does not hold. A
  conditional write is always one request.
* Large objects: an unconditional ``put`` above ``put_multipart_above`` and
  every ``put_file`` of a large file is a multipart upload. A ``copy`` of an
  object above ``copy_multipart_above`` is a multipart copy
  (``UploadPartCopy`` with byte ranges). So a 4 GiB shard never depends on
  one long request, and an object is not limited to the 5 GiB of one request.
* ``digest`` reads the object back and hashes it. The read is tied to one
  version of the object (``If-Match``), and it goes on after a broken
  connection (``Range``).

Error rule: only the first answer of R2 to a request can be "definite"
(``NotFound``, ``PreconditionFailed``, ``TooLarge``). The client must be built
without automatic retries (``make_client`` does that). With retries inside
the SDK, a conditional write whose answer was lost would be sent again, find
its own write, and report "refused, nothing written" for a write that did
happen.

``R2Presigner`` makes the short-lived URLs that a box uses. A box holds no
R2 credentials.
"""
from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import boto3
import urllib3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from .interfaces import (
    Blob, Digest, NotFound, ObjectInfo, PreconditionFailed, StoreError, TooLarge,
)

MIB = 1 << 20
# S3 and R2: every part but the last has at least 5 MiB, and at most 10,000
# parts make one object. R2 also wants all parts but the last to have one
# size; ``_plan`` always cuts that way.
PART_MIN_BYTES = 5 * MIB
MAX_PARTS = 10000
# A single request (PutObject, CopyObject, one part) stays at or below this.
SINGLE_MAX_BYTES = 4 * 1024 * MIB
# The longest life of a presigned URL (signature version 4).
PRESIGN_MAX_S = 7 * 24 * 3600
CONTENT_TYPE = 'application/octet-stream'
# Errors below the S3 protocol: no answer, or a broken one. Outcome unknown.
TRANSPORT = (BotoCoreError, OSError, urllib3.exceptions.HTTPError)


@dataclass(frozen=True)
class R2Config:
    put_part_bytes: int = 64 * MIB
    put_multipart_above: int = 64 * MIB
    copy_part_bytes: int = 512 * MIB
    copy_multipart_above: int = 1024 * MIB
    workers: int = 8
    read_chunk_bytes: int = 4 * MIB
    # Tries of one part, and of one resumed read. Both are safe to repeat.
    attempts: int = 4

    def __post_init__(self) -> None:
        for size in (self.put_part_bytes, self.copy_part_bytes):
            if not PART_MIN_BYTES <= size <= SINGLE_MAX_BYTES:
                raise ValueError('a part must have between 5 MiB and 4 GiB')
        for limit in (self.put_multipart_above, self.copy_multipart_above):
            if not 0 < limit <= SINGLE_MAX_BYTES:
                raise ValueError('a single request must not be above 4 GiB')
        if self.workers < 1 or self.attempts < 1 or self.read_chunk_bytes < 1:
            raise ValueError('workers, attempts and read_chunk_bytes must be above zero')


def _status(exc: ClientError) -> Tuple[str, int]:
    error = exc.response.get('Error') or {}
    meta = exc.response.get('ResponseMetadata') or {}
    try:
        status = int(meta.get('HTTPStatusCode') or 0)
    except (TypeError, ValueError):
        status = 0
    return str(error.get('Code') or ''), status


def _missing_key(exc: ClientError) -> bool:
    code, status = _status(exc)
    # A missing bucket or a missing multipart upload is not a missing key.
    return code in ('NoSuchKey', '404', 'NotFound') and status in (0, 404)


def _refused(exc: ClientError) -> bool:
    code, status = _status(exc)
    return status == 412 or code == 'PreconditionFailed'


def _unknown(what: str, key: str, exc: BaseException) -> StoreError:
    if isinstance(exc, ClientError):
        return StoreError('%s %s: %s %d' % ((what, key) + _status(exc)))
    return StoreError('%s %s: %s' % (what, key, type(exc).__name__))


def _plan(size: int, part_bytes: int) -> Tuple[int, int]:
    """(part size, number of parts) for an object of ``size`` bytes."""
    part = max(part_bytes, -(-size // MAX_PARTS))
    part = -(-part // MIB) * MIB
    if part > SINGLE_MAX_BYTES:
        raise ValueError('an object of %d bytes needs more than %d parts' % (size, MAX_PARTS))
    return part, max(1, -(-size // part))


class R2Store:
    """``ObjectStore`` on one bucket. Thread-safe: boto3 clients are."""

    def __init__(self, client: Any, bucket: str, config: Optional[R2Config] = None):
        self._c = client
        self.bucket = bucket
        self.cfg = config or R2Config()

    # -- reads ---------------------------------------------------------------

    def head(self, key: str) -> Optional[ObjectInfo]:
        try:
            resp = self._c.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if _missing_key(exc):
                return None
            raise _unknown('head', key, exc) from None
        except TRANSPORT as exc:
            raise _unknown('head', key, exc) from None
        return ObjectInfo(key, int(resp['ContentLength']), resp['ETag'])

    def _open(self, key: str, **extra: Any) -> Mapping[str, Any]:
        try:
            return self._c.get_object(Bucket=self.bucket, Key=key, **extra)
        except ClientError as exc:
            if _missing_key(exc):
                raise NotFound(key) from None
            if _refused(exc):
                raise PreconditionFailed(key) from None
            raise _unknown('get', key, exc) from None
        except TRANSPORT as exc:
            raise _unknown('get', key, exc) from None

    def get(self, key: str, *, max_bytes: int) -> Blob:
        resp = self._open(key)
        body = resp['Body']
        try:
            size = int(resp['ContentLength'])
            if size > max_bytes:
                raise TooLarge(key)
            chunks: List[bytes] = []
            got = 0
            while got < size:
                chunk = body.read(size - got)
                if not chunk:
                    break
                chunks.append(chunk)
                got += len(chunk)
        except TRANSPORT as exc:
            raise _unknown('get', key, exc) from None
        finally:
            body.close()
        if got != size:
            raise StoreError('get %s: short read' % key)
        return Blob(b''.join(chunks), resp['ETag'])

    def digest(self, key: str) -> Digest:
        resp = self._open(key)
        etag, size = resp['ETag'], int(resp['ContentLength'])
        sha = hashlib.sha256()
        offset = 0
        for attempt in range(self.cfg.attempts):
            if attempt:
                # The connection broke. Go on from the same byte of the same
                # version. If the object changed, R2 refuses, so the hash of
                # a mix of two versions is never returned.
                try:
                    resp = self._open(key, IfMatch=etag, Range='bytes=%d-' % offset)
                except (NotFound, PreconditionFailed):
                    raise StoreError('digest %s: the object changed during the read' % key) from None
                status = (resp.get('ResponseMetadata') or {}).get('HTTPStatusCode')
                if status != 206 or not str(resp.get('ContentRange', '')).startswith('bytes %d-' % offset):
                    resp['Body'].close()
                    raise StoreError('digest %s: the store did not honour the byte range' % key)
            body = resp['Body']
            try:
                while True:
                    chunk = body.read(self.cfg.read_chunk_bytes)
                    if not chunk:
                        break
                    sha.update(chunk)
                    offset += len(chunk)
            except TRANSPORT:
                pass
            finally:
                body.close()
            if offset >= size:
                break
        if offset != size:
            raise StoreError('digest %s: the read did not finish' % key)
        return Digest(size, sha.hexdigest(), etag)

    def list(self, prefix: str) -> List[ObjectInfo]:
        found: List[ObjectInfo] = []
        token: Optional[str] = None
        while True:
            extra = {'ContinuationToken': token} if token else {}
            try:
                resp = self._c.list_objects_v2(Bucket=self.bucket, Prefix=prefix, **extra)
            except (ClientError,) + TRANSPORT as exc:
                raise _unknown('list', prefix, exc) from None
            for item in resp.get('Contents') or ():
                found.append(ObjectInfo(item['Key'], int(item['Size']), item['ETag']))
            if not resp.get('IsTruncated'):
                return found
            token = resp.get('NextContinuationToken')
            if not token:
                raise StoreError('list %s: a truncated page without a token' % prefix)

    # -- writes --------------------------------------------------------------

    def put(
        self, key: str, data: bytes, *, if_none_match: bool = False, if_match: Optional[str] = None,
    ) -> str:
        if if_none_match and if_match is not None:
            raise ValueError('if_none_match and if_match exclude each other')
        cond: Dict[str, str] = {}
        if if_none_match:
            cond['IfNoneMatch'] = '*'
        if if_match is not None:
            cond['IfMatch'] = if_match
        if not cond and len(data) > self.cfg.put_multipart_above:
            return self._multipart_put(key, len(data), lambda first, last: data[first:last])
        if len(data) > SINGLE_MAX_BYTES:
            raise ValueError('a conditional put is one request; %d bytes are too many' % len(data))
        try:
            resp = self._c.put_object(Bucket=self.bucket, Key=key, Body=data, **cond)
        except ClientError as exc:
            # A put with ``If-Match`` on a key that does not exist: some
            # stores answer 404, some 412. Both mean: nothing was written.
            if cond and (_refused(exc) or (if_match is not None and _missing_key(exc))):
                raise PreconditionFailed(key) from None
            raise _unknown('put', key, exc) from None
        except TRANSPORT as exc:
            raise _unknown('put', key, exc) from None
        return resp['ETag']

    def put_file(self, key: str, path: Path) -> str:
        """Upload a local file of any size. No condition: use it for keys
        that the caller verifies afterwards (``digest``). Memory use is
        bounded by ``workers`` x ``put_part_bytes``."""
        path = Path(path)
        size = path.stat().st_size
        if size <= self.cfg.put_multipart_above:
            return self.put(key, path.read_bytes())

        def read(first: int, last: int) -> bytes:
            with path.open('rb') as handle:
                handle.seek(first)
                data = handle.read(last - first)
            if len(data) != last - first:
                raise OSError('the file changed during the upload')
            return data

        return self._multipart_put(key, size, read)

    def _parts(self, count: int, one: Callable[[int], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run ``one(part number)`` for every part. A part is tried again
        after an error that is not definite: the part number makes it safe."""
        first_error: List[BaseException] = []
        lock = threading.Lock()

        def tried(number: int) -> Optional[Dict[str, Any]]:
            last: Optional[BaseException] = None
            for _attempt in range(self.cfg.attempts):
                if first_error:
                    return None
                try:
                    return one(number)
                except ClientError as exc:
                    last = exc
                    if _refused(exc) or _status(exc)[1] == 404:
                        break
                except TRANSPORT as exc:
                    last = exc
            with lock:
                if last is not None and not first_error:
                    first_error.append(last)
            return None

        with ThreadPoolExecutor(max_workers=min(self.cfg.workers, count)) as pool:
            parts = list(pool.map(tried, range(1, count + 1)))
        if first_error:
            raise first_error[0]
        return [part for part in parts if part is not None]

    def _abort(self, key: str, upload_id: str) -> None:
        """Best effort. A bucket rule must remove what this cannot."""
        try:
            self._c.abort_multipart_upload(Bucket=self.bucket, Key=key, UploadId=upload_id)
        except (ClientError,) + TRANSPORT:
            pass

    def _start(self, key: str) -> str:
        try:
            return self._c.create_multipart_upload(Bucket=self.bucket, Key=key)['UploadId']
        except (ClientError,) + TRANSPORT as exc:
            raise _unknown('multipart start', key, exc) from None

    def _finish(self, key: str, upload_id: str, parts: Sequence[Dict[str, Any]]) -> str:
        try:
            resp = self._c.complete_multipart_upload(
                Bucket=self.bucket, Key=key, UploadId=upload_id,
                MultipartUpload={'Parts': list(parts)},
            )
        except (ClientError,) + TRANSPORT as exc:
            # Nobody knows if the object exists now. If it does, the upload
            # is gone and this abort does nothing.
            self._abort(key, upload_id)
            raise _unknown('multipart finish', key, exc) from None
        return resp['ETag']

    def _multipart_put(self, key: str, size: int, read: Callable[[int, int], bytes]) -> str:
        part, count = _plan(size, self.cfg.put_part_bytes)
        upload_id = self._start(key)

        def one(number: int) -> Dict[str, Any]:
            body = read((number - 1) * part, min(number * part, size))
            resp = self._c.upload_part(
                Bucket=self.bucket, Key=key, UploadId=upload_id, PartNumber=number, Body=body,
            )
            return {'ETag': resp['ETag'], 'PartNumber': number}

        try:
            parts = self._parts(count, one)
        except (ClientError,) + TRANSPORT as exc:
            self._abort(key, upload_id)
            raise _unknown('multipart put', key, exc) from None
        return self._finish(key, upload_id, parts)

    def copy(self, src: str, dst: str, *, src_etag: str, src_bucket: Optional[str] = None) -> None:
        """Server-side copy. ``src_bucket``: read the source from another
        bucket (the import of a first state). The default is this bucket."""
        source_store = self if src_bucket in (None, self.bucket) else R2Store(self._c, src_bucket, self.cfg)
        # R2 checks the etag of the source itself (``CopySourceIfMatch``).
        # This first look also covers a store that ignores that header, and
        # it gives the size, which decides between one request and many.
        info = source_store.head(src)
        if info is None:
            raise NotFound(src)
        if info.etag != src_etag:
            raise PreconditionFailed(src)
        source = {'Bucket': source_store.bucket, 'Key': src}
        if info.size > self.cfg.copy_multipart_above:
            self._multipart_copy(source_store, source, dst, info)
            return
        try:
            self._c.copy_object(
                Bucket=self.bucket, Key=dst, CopySource=source, CopySourceIfMatch=src_etag,
            )
        except ClientError as exc:
            if _refused(exc):
                raise PreconditionFailed(src) from None
            if _missing_key(exc):
                raise NotFound(src) from None
            raise _unknown('copy', dst, exc) from None
        except TRANSPORT as exc:
            raise _unknown('copy', dst, exc) from None

    def _multipart_copy(
        self, source_store: 'R2Store', source: Mapping[str, str], dst: str, info: ObjectInfo,
    ) -> None:
        part, count = _plan(info.size, self.cfg.copy_part_bytes)
        upload_id = self._start(dst)

        def one(number: int) -> Dict[str, Any]:
            first = (number - 1) * part
            last = min(number * part, info.size) - 1
            resp = self._c.upload_part_copy(
                Bucket=self.bucket, Key=dst, UploadId=upload_id, PartNumber=number,
                CopySource=dict(source), CopySourceRange='bytes=%d-%d' % (first, last),
                CopySourceIfMatch=info.etag,
            )
            return {'ETag': resp['CopyPartResult']['ETag'], 'PartNumber': number}

        try:
            parts = self._parts(count, one)
            # All parts must come from one version of the source.
            again = source_store.head(info.key)
        except ClientError as exc:
            self._abort(dst, upload_id)
            if _refused(exc):
                raise PreconditionFailed(info.key) from None
            raise _unknown('multipart copy', dst, exc) from None
        except TRANSPORT + (StoreError,) as exc:
            self._abort(dst, upload_id)
            raise _unknown('multipart copy', dst, exc) from None
        if again is None or (again.etag, again.size) != (info.etag, info.size):
            self._abort(dst, upload_id)
            raise PreconditionFailed(info.key)
        self._finish(dst, upload_id, parts)
        made = self.head(dst)
        if made is None or made.size != info.size:
            raise StoreError('multipart copy %s: the copy has another size than its source' % dst)

    def delete(self, key: str) -> None:
        try:
            self._c.delete_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if _missing_key(exc):
                return
            raise _unknown('delete', key, exc) from None
        except TRANSPORT as exc:
            raise _unknown('delete', key, exc) from None

    # -- start-up check ------------------------------------------------------

    def check_bucket(self) -> None:
        """Fail early on a wrong bucket name or wrong credentials. A HEAD of
        a key in a bucket that does not exist looks like a missing key."""
        try:
            self._c.head_bucket(Bucket=self.bucket)
        except (ClientError,) + TRANSPORT as exc:
            raise _unknown('bucket', self.bucket, exc) from None


class ReadOnlyStore:
    """The read calls of a store, and nothing else.

    Used for the live side: the batch documents and the first HEAD are read
    from live keys. This object has no call that writes, copies or deletes.
    """

    def __init__(self, store: R2Store):
        self._head = store.head
        self._get = store.get
        self._digest = store.digest
        self.bucket = store.bucket

    def head(self, key: str) -> Optional[ObjectInfo]:
        return self._head(key)

    def get(self, key: str, *, max_bytes: int) -> Blob:
        return self._get(key, max_bytes=max_bytes)

    def digest(self, key: str) -> Digest:
        return self._digest(key)


class R2Presigner:
    """Short-lived URLs for one object each. Signing is local: no request."""

    def __init__(self, client: Any, bucket: str):
        self._c = client
        self.bucket = bucket

    def _url(self, method: str, params: Dict[str, Any], expires_s: float) -> str:
        seconds = int(expires_s)
        if not 0 < seconds <= PRESIGN_MAX_S:
            raise ValueError('a presigned URL lives between one second and seven days')
        try:
            return self._c.generate_presigned_url(
                method, Params=dict(params, Bucket=self.bucket), ExpiresIn=seconds,
            )
        except (ClientError, BotoCoreError) as exc:
            raise StoreError('presign: %s' % type(exc).__name__) from None

    def get_url(self, key: str, expires_s: float) -> str:
        return self._url('get_object', {'Key': key}, expires_s)

    def put_url(self, key: str, expires_s: float) -> str:
        """The upload must send ``Content-Type: application/octet-stream``."""
        return self._url('put_object', {'Key': key, 'ContentType': CONTENT_TYPE}, expires_s)


def make_client(
    *, endpoint: str, access_key_id: str, secret_access_key: str,
    connect_timeout_s: float = 15.0, read_timeout_s: float = 120.0, pool: int = 64,
) -> Any:
    """An S3 client for R2 that never repeats a request by itself."""
    return boto3.client(
        's3', endpoint_url=endpoint, region_name='auto',
        aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key,
        config=BotoConfig(
            signature_version='s3v4',
            retries={'total_max_attempts': 1, 'mode': 'standard'},
            connect_timeout=connect_timeout_s, read_timeout=read_timeout_s,
            max_pool_connections=pool,
            # Newer SDKs add a CRC trailer to every upload by default. Keep
            # the plain requests that R2 has always taken.
            request_checksum_calculation='when_required',
            response_checksum_validation='when_required',
        ),
    )


def read_bucket_config(path: Path) -> Dict[str, str]:
    """Read the bucket document that the host already has.

    Shapes: ``{"bucket": <name>, "endpoint": ..., "credentials_file": ...}``,
    or that object under the key ``"bucket"`` (the lifecycle config), with or
    without a ``"payload"`` wrapper. The credentials file has
    ``R2_ACCESS_KEY_ID=`` and ``R2_SECRET_ACCESS_KEY=`` lines. The values
    are returned to the caller only; nothing here logs them.
    """
    doc = json.loads(Path(path).read_text())
    if isinstance(doc.get('payload'), dict):
        doc = doc['payload']
    if isinstance(doc.get('bucket'), dict):
        doc = doc['bucket']
    values: Dict[str, str] = {}
    for line in Path(doc['credentials_file']).read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            name, value = line.split('=', 1)
            values[name.strip()] = value.strip().strip('"\'')
    return {
        'bucket': doc['bucket'], 'endpoint': doc['endpoint'],
        'access_key_id': values['R2_ACCESS_KEY_ID'],
        'secret_access_key': values['R2_SECRET_ACCESS_KEY'],
    }


def client_from_bucket_config(path: Path) -> Tuple[Any, str]:
    values = read_bucket_config(path)
    client = make_client(
        endpoint=values['endpoint'], access_key_id=values['access_key_id'],
        secret_access_key=values['secret_access_key'],
    )
    return client, values['bucket']


__all__ = [
    'CONTENT_TYPE', 'R2Config', 'R2Presigner', 'R2Store', 'ReadOnlyStore',
    'client_from_bucket_config', 'make_client', 'read_bucket_config',
]
