#!/usr/bin/env python3
"""Box side of the parallel R2 upload. Standard library only: the host ships this file to the box as it is.

    python3 pe_r2_box_upload.py stat REQUEST.json     print the size of every file to upload
    python3 pe_r2_box_upload.py upload JOB.json       send the files; the last line of output is the manifest

A part is a byte range of a file. R2 joins the parts of a file into one object. The host signs one URL for
each part. This tool sends the parts of all files in parallel, straight from disk, and it reads every file
a second time to hash it. No file is held in memory.

The manifest has, for every file, the sha256 and three lists with one entry for each part:

  hashed   MD5 of the part as the hash pass read it (the pass that gives the sha256)
  sent     MD5 of the bytes that went to R2
  etags    the ETag that R2 returned for the part

The host needs no download to check R2: the ETag of an object is a function of the MD5 of its parts (see
``object_etag``). Equal ``hashed`` and ``sent`` lists show that the bytes sent are the bytes hashed.
"""
import hashlib
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

VERSION = 'pit-r2-upload-v1'
MIN_PART_BYTES = 5 * 1024 * 1024        # S3 and R2 refuse a smaller part, but the last part can be smaller
MAX_PARTS = 10000                       # S3 and R2 refuse more parts in one object
CHUNK_BYTES = 4 * 1024 * 1024           # bytes read from disk and sent in one step


class UploadError(RuntimeError):
    """The upload of a file cannot be completed."""


def part_ranges(size, part_bytes):
    """(offset, length) of every part of a file of ``size`` bytes. All parts but the last have ``part_bytes``."""
    size, part_bytes = int(size), int(part_bytes)
    if size < 0 or part_bytes < 1:
        raise UploadError('bad part plan: %d bytes in parts of %d' % (size, part_bytes))
    if size <= part_bytes:
        return [(0, size)]
    ranges = [(offset, min(part_bytes, size - offset)) for offset in range(0, size, part_bytes)]
    if len(ranges) > MAX_PARTS:
        raise UploadError('%d parts; the limit is %d' % (len(ranges), MAX_PARTS))
    return ranges


def object_etag(part_md5, multipart):
    """The ETag that S3 and R2 give an object from the MD5 (hex) of its parts.

    One PUT: the MD5 of the body. Multipart: the MD5 of the joined binary MD5 values, then '-' and the part count.
    """
    part_md5 = list(part_md5)
    if not multipart:
        if len(part_md5) != 1:
            raise UploadError('an object from one PUT has one part')
        return part_md5[0]
    joined = b''.join(bytes.fromhex(value) for value in part_md5)
    return '%s-%d' % (hashlib.md5(joined).hexdigest(), len(part_md5))


def hash_file(path, ranges, chunk=CHUNK_BYTES):
    """One pass over a file: its sha256 and the MD5 of every part. Fails when the file is not as long as the plan."""
    total = hashlib.sha256()
    parts = []
    with open(path, 'rb') as handle:
        for _offset, length in ranges:
            part, left = hashlib.md5(), length
            while left:
                data = handle.read(min(chunk, left))
                if not data:
                    raise UploadError('%s is shorter than planned' % path)
                total.update(data)
                part.update(data)
                left -= len(data)
            parts.append(part.hexdigest())
        if handle.read(1):
            raise UploadError('%s is longer than planned' % path)
    return dict(sha256=total.hexdigest(), parts=parts)


def send_part(url, path, offset, length, headers=None, timeout=900.0, chunk=CHUNK_BYTES, stop=None):
    """PUT one byte range of a file to a signed URL. -> (MD5 of the bytes sent, ETag from the response)."""
    target = urllib.parse.urlsplit(url)
    connection = (http.client.HTTPSConnection if target.scheme == 'https' else http.client.HTTPConnection)(
        target.netloc, timeout=timeout)
    try:
        # urllib would add a Content-Type that the signature does not cover; http.client sends what is given.
        connection.putrequest('PUT', target.path + ('?' + target.query if target.query else ''))
        connection.putheader('Content-Length', str(length))
        for name, value in sorted((headers or {}).items()):
            connection.putheader(name, value)
        connection.endheaders()
        digest, left = hashlib.md5(), length
        with open(path, 'rb') as handle:
            handle.seek(offset)
            while left:
                if stop is not None and stop.is_set():
                    raise UploadError('stopped: another part failed')
                data = handle.read(min(chunk, left))
                if not data:
                    raise UploadError('%s is shorter than planned' % path)
                try:
                    connection.send(data)
                except OSError as broken:
                    # The store can answer and close before it has the full body (a refused signature, for example).
                    try:
                        early = connection.getresponse()
                    except (OSError, http.client.HTTPException):
                        raise broken
                    raise UploadError('HTTP %d: %s' % (early.status, early.read()[:200].decode('utf-8', 'replace')))
                digest.update(data)
                left -= len(data)
        response = connection.getresponse()
        body = response.read()
        if response.status not in (200, 201):
            raise UploadError('HTTP %d: %s' % (response.status, body[:200].decode('utf-8', 'replace')))
        return digest.hexdigest(), (response.getheader('ETag') or '').strip().strip('"')
    finally:
        connection.close()


def stat_trees(request):
    """Sizes of the files of every tree. ``names`` null: every file below ``dir``. Else: exactly those names."""
    trees, missing = {}, []
    for label, tree in sorted(request['trees'].items()):
        root, names, sizes = tree['dir'], tree.get('names'), {}
        if names is None:
            for folder, _dirs, files in os.walk(root):
                for name in files:
                    path = os.path.join(folder, name)
                    sizes[os.path.relpath(path, root).replace(os.sep, '/')] = os.path.getsize(path)
        else:
            for name in names:
                path = os.path.join(root, name)
                if os.path.isfile(path):
                    sizes[name] = os.path.getsize(path)
                else:
                    missing.append('%s/%s' % (label, name))
        trees[label] = sizes
    return dict(version=VERSION, ok=not missing, trees=trees, missing=missing)


def upload(job):
    """Send every file of the job. -> the manifest (``ok`` false and ``errors`` when a file failed)."""
    started = time.time()
    attempts = max(1, int(job.get('attempts') or 5))
    timeout = float(job.get('timeout_s') or 900)
    cores = os.cpu_count() or 2
    stop = threading.Event()
    part_pool = ThreadPoolExecutor(max(1, int(job.get('workers') or 16)))
    hash_pool = ThreadPoolExecutor(max(1, int(job.get('hash_workers') or min(8, max(2, cores // 2)))))

    def send(entry, part, offset, length):
        last = None
        for attempt in range(attempts):
            if stop.is_set():
                raise UploadError('stopped: another part failed')
            try:
                return send_part(part['url'], entry['path'], offset, length, headers=entry.get('headers'),
                                 timeout=timeout, stop=stop)
            except (OSError, http.client.HTTPException, UploadError) as exc:
                last = exc
                if attempt + 1 < attempts:
                    stop.wait(min(30.0, 2.0 ** attempt))
        stop.set()
        raise UploadError('part %s of %s failed %d times: %s' % (part.get('n'), entry['id'], attempts, str(last)[:300]))

    def begin(entry):
        size, multipart = int(entry['bytes']), bool(entry['multipart'])
        if os.path.getsize(entry['path']) != size:
            raise UploadError('%s is not %d bytes any more' % (entry['id'], size))
        ranges = part_ranges(size, entry['part_bytes']) if multipart else [(0, size)]
        if len(ranges) != len(entry['parts']):
            raise UploadError('%s: %d signed parts for %d planned' % (entry['id'], len(entry['parts']), len(ranges)))
        return ranges, hash_pool.submit(hash_file, entry['path'], ranges)

    def queue(entry, ranges):
        return [part_pool.submit(send, entry, part, offset, length)
                for part, (offset, length) in zip(entry['parts'], ranges)]

    def finish(entry, ranges, hashing, sending):
        multipart = bool(entry['multipart'])
        record = dict(bytes=int(entry['bytes']), multipart=multipart, part_bytes=int(entry['part_bytes']), reused=False)
        if sending is None:
            # R2 has an object with this key and this size. Equal ETag: R2 has these bytes; send nothing.
            hashed = hashing.result()
            if object_etag(hashed['parts'], multipart) == entry['have_etag']:
                return dict(record, sha256=hashed['sha256'], hashed=hashed['parts'], sent=[], etags=[], reused=True)
            sending = queue(entry, ranges)
        sent = [future.result() for future in sending]
        hashed = hashing.result()
        return dict(record, sha256=hashed['sha256'], hashed=hashed['parts'],
                    sent=[md5 for md5, _etag in sent], etags=[etag for _md5, etag in sent])

    files, errors, waiting = {}, [], []

    def failed(name, exc):                      # every failure goes into the manifest; the host fails closed on it
        stop.set()
        errors.append('%s: %s' % (name, str(exc)[:300]))

    with ThreadPoolExecutor(max(1, len(job['files']))) as file_pool:
        for entry in job['files']:              # the parts go into the queue in the order of the job
            try:
                ranges, hashing = begin(entry)
                sending = None if entry.get('have_etag') else queue(entry, ranges)
                waiting.append((entry['id'], file_pool.submit(finish, entry, ranges, hashing, sending)))
            except Exception as exc:
                failed(entry['id'], exc)
        for name, future in waiting:
            try:
                files[name] = future.result()
            except Exception as exc:
                failed(name, exc)
    part_pool.shutdown(wait=True)
    hash_pool.shutdown(wait=True)
    return dict(
        version=VERSION, ok=not errors, files=files, errors=errors[:20], seconds=round(time.time() - started, 3),
        bytes_sent=sum(record['bytes'] for record in files.values() if not record['reused']),
        bytes_reused=sum(record['bytes'] for record in files.values() if record['reused']),
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2 or argv[0] not in ('stat', 'upload'):
        sys.stderr.write(__doc__)
        return 2
    command, path = argv
    with open(path) as handle:
        document = json.load(handle)
    if command == 'stat':
        result = stat_trees(document)
    else:
        try:
            result = upload(document)
        finally:
            os.unlink(path)                     # the job holds signed URLs; it must not stay on the box
    sys.stdout.write('\n' + json.dumps(result, sort_keys=True) + '\n')
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
