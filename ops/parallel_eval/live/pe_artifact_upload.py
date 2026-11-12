"""Box→R2 artifact upload helpers (bf16 + fp32 stream; private pending until APPLY).

Never stage a full fp32 master tree on the host, and never download an artifact
to the host to check it. The box sends every file to R2 in parallel parts
(``upload_trees_from_box``); the host checks each object by its size and ETag
against the sha256 manifest of the box. APPLY copies pending → live inside R2 in
parallel parts and checks the copy the same way. RED artifacts stay under a
private pending prefix/index until APPLY promotes them (or GC on release).

``PIT_R2_LOCAL_MIRROR`` (tests, dry runs) and injected PUT URLs keep the serial
transport: one PUT for each file, and the host hashes what it reads back.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ops.parallel_eval.live import pe_r2_multipart as R2MP

BOX_TOOL = Path(__file__).with_name('pe_r2_box_upload.py')
BOX_UPLOAD_TIMEOUT_S = 7200
# Fields of one file that the parallel upload records for the R2 copy and its check at APPLY.
TRANSFER_FIELDS = ('etag', 'part_bytes', 'multipart', 'parts')


class ArtifactUploadError(RuntimeError):
    """Fail-closed upload / verify error."""


EARLY_UPLOAD_RETRIES = 3


def _env_path(*names: str) -> Optional[Path]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return Path(v)
    return None


def r2_local_mirror() -> Optional[Path]:
    return _env_path('PIT_R2_LOCAL_MIRROR')


def box_multipart_transport() -> bool:
    """True when the box sends its artifacts to R2 in parallel parts (the default).

    False in the two modes that keep the serial transport: a local directory in place of R2
    (``PIT_R2_LOCAL_MIRROR``), and PUT URLs that the operator injected.
    """
    if r2_local_mirror() is not None:
        return False
    injected = (_env_path('PIT_PRESIGN_MASTER_PUT_JSON'), _env_path('PIT_PRESIGN_PUT_JSON'))
    return not any(path is not None and path.is_file() for path in injected)


def open_life_bucket() -> Any:
    """The R2 bucket of the life config: the one place where the artifact code gets its R2 client."""
    from subnet.storage import Bucket  # type: ignore
    life_cfg = _env_path('PIT_LIFE_CFG', 'AFFINE_LIFE_CFG')
    if life_cfg is None or not life_cfg.is_file():
        raise ArtifactUploadError('PIT_LIFE_CFG required for R2 object ops')
    return Bucket(json.loads(life_cfg.read_text())['bucket'])


def pending_r2_prefix(candidate_id: str) -> str:
    """Isolated private prefix — never the live website/k1 path."""
    base = (os.environ.get('PIT_R2_PENDING_PREFIX') or 'pit-pending').rstrip('/')
    return '%s/%s' % (base, candidate_id)


def live_fp32_r2_prefix(candidate_id: str) -> str:
    base = (os.environ.get('PIT_R2_FP32_PREFIX') or 'fp32-master').rstrip('/')
    return '%s/%s' % (base, candidate_id)


def pending_master_index_dir() -> Path:
    """Private pending index (not website/k1/fp32-master)."""
    p = _env_path('PIT_FP32_PENDING_INDEX')
    if p is not None:
        return p
    pe = _env_path('PIT_STATE', 'PE_STATE')
    if pe is None:
        raise ArtifactUploadError(
            'PIT_FP32_PENDING_INDEX or PIT_STATE/PE_STATE required for pending master index'
        )
    if pe.name == 'pit':
        return pe / 'fp32-pending'
    return pe / 'pit' / 'fp32-pending'


def public_master_index_dir() -> Path:
    p = _env_path('PIT_FP32_MASTER_INDEX', 'AFFINE_FP32_MASTER_INDEX')
    if p is None:
        raise ArtifactUploadError('PIT_FP32_MASTER_INDEX required for public promote')
    return p


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _wj(path: Path, obj: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def assert_master_tree_complete(files_meta: Mapping[str, Any]) -> None:
    if not files_meta:
        raise ArtifactUploadError('fp32 master empty')
    shards = [
        n for n in files_meta
        if str(n).endswith('.safetensors') or str(n).endswith('.bin')
    ]
    if not shards:
        raise ArtifactUploadError(
            'fp32 master has no weight shards (digest-only refused)'
        )
    if 'meta.json' not in files_meta:
        raise ArtifactUploadError(
            'fp32 master missing meta.json (required for trainer tensor_file)'
        )


def meta_json_r2_ptr(files_meta: Mapping[str, Any], *, prefix: str) -> Dict[str, str]:
    """Public/pending index pointer at the real uploaded meta.json (never inline).

    Live HEAD indexes use ``{"r2_key": "<prefix>/meta.json"}``. A synthetic
    ``{"version": "pit-fp32-master-v1", ...}`` would be written over the real
    meta.json by ``fetch_master_on_box`` and break ``_fp32_load`` (no tensor_file).
    """
    meta = files_meta.get('meta.json')
    if not isinstance(meta, Mapping):
        raise ArtifactUploadError('files_meta missing meta.json for index pointer')
    key = str(meta.get('r2_key') or ('%s/meta.json' % prefix.rstrip('/')))
    if not key.endswith('meta.json'):
        raise ArtifactUploadError('meta_json r2_key must end with meta.json: %s' % key)
    return {'r2_key': key}


def inventory_box_master(P: Any, box_dir: str) -> Dict[str, Dict[str, Any]]:
    inv_raw = P.run(
        "python3 - <<'PY'\n"
        "import hashlib, json, os\n"
        "root = %r\n"
        "out = {}\n"
        "for dirpath, _, files in os.walk(root):\n"
        "    for name in files:\n"
        "        path = os.path.join(dirpath, name)\n"
        "        rel = os.path.relpath(path, root)\n"
        "        h = hashlib.sha256()\n"
        "        with open(path, 'rb') as f:\n"
        "            for b in iter(lambda: f.read(1 << 20), b''):\n"
        "                h.update(b)\n"
        "        out[rel.replace(os.sep, '/')] = dict("
        "sha256=h.hexdigest(), bytes=os.path.getsize(path))\n"
        "print(json.dumps(out, sort_keys=True))\n"
        "PY" % box_dir,
        timeout=1800, check=True,
    )
    try:
        inv = json.loads(str(inv_raw).strip().splitlines()[-1])
    except Exception as exc:
        raise ArtifactUploadError('box master inventory parse failed: %s' % str(exc)[:200]) from exc
    if not isinstance(inv, dict) or not inv:
        raise ArtifactUploadError('box master inventory empty at %s' % box_dir)
    assert_master_tree_complete(inv)
    return inv


def _presign_master_puts(
    candidate_id: str,
    inv: Mapping[str, Mapping[str, Any]],
    *,
    pending: bool = True,
) -> Dict[str, str]:
    """Presigned PUT URLs (or file:// mirror) for each master shard."""
    inj = _env_path('PIT_PRESIGN_MASTER_PUT_JSON')
    if inj is not None and inj.is_file():
        doc = json.loads(inj.read_text())
        urls = doc.get('urls') if isinstance(doc, Mapping) else None
        if isinstance(urls, Mapping):
            return {n: str(urls[n]) for n in inv if n in urls}
    mirror = r2_local_mirror()
    prefix = pending_r2_prefix(candidate_id) if pending else live_fp32_r2_prefix(candidate_id)
    if mirror is not None:
        out: Dict[str, str] = {}
        for name in inv:
            dest = mirror / prefix / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            out[name] = 'file://' + str(dest)
        return out
    try:
        bucket = open_life_bucket()
        return {
            n: bucket.presign('%s/%s' % (prefix, n), 'put_object', 7200)
            for n in sorted(inv)
        }
    except ArtifactUploadError:
        raise
    except Exception as exc:
        raise ArtifactUploadError('master presign failed: %s' % str(exc)[:300]) from exc


def box_put_files(
    P: Any,
    *,
    box_dir: str,
    urls: Mapping[str, str],
    put: Any,
) -> None:
    """Box streams each https PUT; file:// mirror is host-materialized via P.run cat.

    Never stage a full live master on the host for https; file:// is test-mirror only.
    """
    file_urls = {n: u for n, u in urls.items() if str(u).startswith('file://')}
    https_urls = {n: u for n, u in urls.items() if not str(u).startswith('file://')}
    for name, url in file_urls.items():
        dest = Path(str(url)[7:])
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Host reads from box (Fake SSH rewrites box path, not host dest).
        raw = P.run(
            "python3 -c \"import sys;sys.stdout.buffer.write(open(%r,'rb').read())\""
            % ('%s/%s' % (box_dir.rstrip('/'), name)),
            timeout=7200,
        )
        data = raw if isinstance(raw, (bytes, bytearray)) else str(raw).encode()
        # Some FakePod.run returns text; prefer binary get when available.
        get = getattr(P, 'get', None)
        if callable(get):
            try:
                data = get('%s/%s' % (box_dir.rstrip('/'), name))
            except Exception:
                pass
        if isinstance(data, str):
            data = data.encode()
        dest.write_bytes(bytes(data))
    if not https_urls:
        return
    remote_urls = '/tmp/pit-master-put-%s.urls.private.json' % hashlib.sha256(
        json.dumps(sorted(https_urls)).encode()
    ).hexdigest()[:12]
    put(
        P,
        remote_urls,
        json.dumps(dict(urls=dict(https_urls), headers={'Content-Type': 'application/octet-stream'})).encode(),
        '600',
    )
    script = r"""
set -e
python3 - <<'PY'
import json, os, sys, urllib.request
doc = json.load(open(%r))
urls = doc['urls']; headers = doc.get('headers') or {}
box = %r
bad = []
for name, url in urls.items():
    src = os.path.join(box, name)
    if not os.path.isfile(src):
        bad.append(('missing', name)); continue
    with open(src, 'rb') as fh:
        data = fh.read()
    req = urllib.request.Request(url, data=data, method='PUT')
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=7200) as resp:
        if getattr(resp, 'status', 200) not in (200, 201):
            bad.append(('http', name, getattr(resp, 'status', None)))
print('PUT_OK', len(urls), 'BAD', len(bad))
if bad:
    sys.exit(1)
PY
rm -f %s
""".strip() % (remote_urls, box_dir, remote_urls)
    out = P.run(script, timeout=7200)
    if 'PUT_OK' not in str(out):
        raise ArtifactUploadError('box master R2 PUT failed: %s' % str(out)[-400:])


def verify_master_from_r2(
    *,
    candidate_id: str,
    inv: Mapping[str, Mapping[str, Any]],
    pending: bool = True,
) -> Dict[str, str]:
    """Serial transport only: the host hashes every shard that it reads back from the mirror or from R2.

    The default transport (``upload_trees_from_box``) checks size and ETag in place of this download.
    """
    mirror = r2_local_mirror()
    prefix = pending_r2_prefix(candidate_id) if pending else live_fp32_r2_prefix(candidate_id)
    verified: Dict[str, str] = {}
    for name, meta in inv.items():
        expected = str(meta.get('sha256') or '')
        if len(expected) != 64:
            raise ArtifactUploadError('inventory missing sha for %s' % name)
        if mirror is not None:
            path = mirror / prefix / name
            if not path.is_file():
                raise ArtifactUploadError('R2 mirror missing master shard: %s' % name)
            got = _sha256_file(path)
            if got != expected:
                raise ArtifactUploadError(
                    'master sha mismatch %s: got %s want %s' % (name, got[:12], expected[:12])
                )
            verified[name] = got
            continue
        # Live: short-lived GET, hash, discard (never keep 35 GB on host).
        tmp_dir = Path(tempfile.mkdtemp(prefix='pit-master-verify-'))
        try:
            local = tmp_dir / Path(name).name
            try:
                bucket = open_life_bucket()
                url = bucket.presign('%s/%s' % (prefix, name), 'get_object', 3600)
                with urllib.request.urlopen(
                    urllib.request.Request(url, headers={'User-Agent': 'affine-pit-master-verify/1'}),
                    timeout=7200,
                ) as resp:
                    h = hashlib.sha256()
                    with local.open('wb') as out:
                        while True:
                            chunk = resp.read(1 << 20)
                            if not chunk:
                                break
                            h.update(chunk)
                            out.write(chunk)
                    got = h.hexdigest()
            except ArtifactUploadError:
                raise
            except Exception as exc:
                raise ArtifactUploadError(
                    'master GET verify failed for %s: %s' % (name, str(exc)[:200])
                ) from exc
            if got != expected:
                raise ArtifactUploadError(
                    'master sha mismatch %s: got %s want %s' % (name, got[:12], expected[:12])
                )
            verified[name] = got
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return verified


def write_pending_master_index(
    *,
    candidate_id: str,
    bf16: str,
    digest: str,
    inv: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Private pending index — not the public website index."""
    files_meta = {
        n: dict(
            {k: m[k] for k in TRANSFER_FIELDS if m.get(k) is not None},
            sha256=m['sha256'],
            bytes=int(m.get('bytes') or 0),
            r2_key='%s/%s' % (pending_r2_prefix(candidate_id), n),
        )
        for n, m in inv.items()
    }
    assert_master_tree_complete(files_meta)
    idx_dir = pending_master_index_dir()
    idx_dir.mkdir(parents=True, exist_ok=True)
    idx_path = idx_dir / ('%s.json' % candidate_id)
    prefix = pending_r2_prefix(candidate_id)
    doc = {
        'bf16_sha256': bf16,
        'master_digest': digest,  # host-recorded; never overwritten from box meta.json
        'files': files_meta,
        'pending': True,
        'private': True,
        'source': 'pit-pending',
        'at': time.time(),
        # Pointer only — real meta.json (with tensor_file) lives at r2_key.
        'meta_json': meta_json_r2_ptr(files_meta, prefix=prefix),
    }
    _wj(idx_path, doc, 0o600)
    return idx_path


def _last_json(output: Any, what: str) -> Dict[str, Any]:
    """The JSON document in the last line of the output of a box command."""
    lines = [line for line in str(output or '').strip().splitlines() if line.strip()]
    try:
        doc = json.loads(lines[-1])
    except (IndexError, ValueError) as exc:
        raise ArtifactUploadError('%s: the box returned no result: %s' % (what, str(output)[-300:])) from exc
    if not isinstance(doc, dict):
        raise ArtifactUploadError('%s: the box result is not an object' % what)
    return doc


def upload_trees_from_box(
    P: Any,
    *,
    candidate_id: str,
    trees: Mapping[str, Mapping[str, Any]],
    put: Any,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """One parallel job: the box sends file trees to R2 in parts, the host checks every object.

    ``trees``: {label: {dir, prefix, names}}. ``dir`` is the directory on the box, ``prefix`` the R2 prefix,
    ``names`` the files to send (None: every file below ``dir``). All trees share one pool of part uploads;
    the parts of the first tree are first in its queue.

    -> {label: {name: {sha256, bytes, r2_key, etag, part_bytes, multipart, parts, verify, reused}}}

    The host never reads an artifact back: it compares the size and the ETag of each object with the sha256
    manifest of the box (see pe_r2_multipart). Fail closed: any difference raises and nothing is recorded.
    """
    if not isinstance(candidate_id, str) or len(candidate_id) != 64:
        raise ArtifactUploadError('candidate_id must be hex64')
    work = '%s/pit-r2-upload-%s' % ((os.environ.get('PIT_R2_BOX_TMP') or '/tmp').rstrip('/'), candidate_id[:12])
    tool, stat_path, job_path = (work + '/' + name for name in ('pe_r2_box_upload.py', 'stat.json', 'job.private.json'))
    try:
        bucket = open_life_bucket()
        client = R2MP.transfer_client(bucket)
        put(P, tool, BOX_TOOL.read_bytes(), '700')
        put(P, stat_path, json.dumps({'trees': {
            label: dict(dir=str(tree['dir']), names=None if tree.get('names') is None else sorted(tree['names']))
            for label, tree in trees.items()}}).encode(), '600')
        stat = _last_json(P.run('python3 %s stat %s' % (shlex.quote(tool), shlex.quote(stat_path)),
                                timeout=600, check=False), 'file sizes')
        if not stat.get('ok'):
            raise ArtifactUploadError('files missing on the box: %s' % list(stat.get('missing') or [])[:5])
        files: Dict[str, Dict[str, Any]] = {}
        for label, tree in trees.items():
            sizes = (stat.get('trees') or {}).get(label) or {}
            if not sizes:
                raise ArtifactUploadError('box tree %s is empty at %s' % (label, tree['dir']))
            for name, size in sorted(sizes.items()):
                files['%s/%s' % (label, name)] = dict(
                    key='%s/%s' % (str(tree['prefix']).rstrip('/'), name),
                    path='%s/%s' % (str(tree['dir']).rstrip('/'), name), bytes=int(size))
        job, record = R2MP.plan_upload(bucket, files, client=client)
        try:
            put(P, job_path, json.dumps(job, sort_keys=True).encode(), '600')
            manifest = _last_json(P.run('python3 %s upload %s' % (shlex.quote(tool), shlex.quote(job_path)),
                                        timeout=BOX_UPLOAD_TIMEOUT_S, check=False), 'upload')
        except BaseException:
            R2MP.abort_upload(bucket, record, client=client)
            raise
        done = R2MP.finish_upload(bucket, record, manifest, client=client)
    except R2MP.R2TransferError as exc:
        raise ArtifactUploadError(str(exc)) from exc
    out: Dict[str, Dict[str, Dict[str, Any]]] = {label: {} for label in trees}
    for ident, entry in done.items():
        label, name = ident.split('/', 1)
        out[label][name] = dict(
            sha256=entry['sha256'], bytes=entry['bytes'], r2_key=entry['key'], etag=entry['etag'],
            part_bytes=entry['part_bytes'], multipart=entry['multipart'], parts=entry['parts'],
            verify=entry['verify'], reused=entry['reused'])
    return out


def _mirror_copy(mirror: Path, src_key: str, dst_key: str) -> None:
    src, dst = mirror / src_key, mirror / dst_key
    if not src.is_file():
        raise ArtifactUploadError('pending object missing for promote: %s' % src_key)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _mirror_sha256(mirror: Path, key: str) -> str:
    path = mirror / key
    if not path.is_file():
        raise ArtifactUploadError('live object missing after promote: %s' % key)
    return _sha256_file(path)


def _promote_object(bucket: Any, client: Any, name: str, meta: Mapping[str, Any], src_key: str, dst_key: str) -> str:
    """Copy one pending object to its live key inside R2 and check the copy. -> how it was checked.

    The copy is checked without a download when the upload recorded the ETag of the pending object and the
    live object has that ETag: the upload tied that ETag to the sha256 of the box. Else the host hashes the
    live object as a stream. A difference in size, ETag-and-sha256, or a missing object raises.
    """
    sha, etag = str(meta.get('sha256') or ''), str(meta.get('etag') or '')
    if len(sha) != 64:
        raise ArtifactUploadError('promote refused: pending index has no sha256 for %s' % name)
    try:
        size = int(meta['bytes']) if etag else None     # an index of the serial transport is not exact on size
        if src_key != dst_key:
            copied = R2MP.server_copy(bucket, src_key, dst_key, size=size, etag=etag or None,
                                      part=meta.get('part_bytes'), client=client)
            size = copied['bytes']
        elif size is None:
            found = R2MP.head(client, bucket.name, dst_key)
            if found is None:
                raise ArtifactUploadError('pending object missing for promote: %s' % src_key)
            size = found['bytes']
        return R2MP.check_object(client, bucket.name, dst_key, size=size, sha256=sha, etag=etag or None)['verify']
    except R2MP.R2TransferError as exc:
        text = str(exc)
        if text.startswith('source object missing'):
            raise ArtifactUploadError('pending object missing for promote: %s' % src_key) from exc
        raise ArtifactUploadError('promote verify mismatch %s: %s' % (name, text[:300])) from exc
    except ArtifactUploadError:
        raise
    except Exception as exc:
        raise ArtifactUploadError('R2 server-side copy failed: %s' % str(exc)[:200]) from exc


def promote_pending_master_to_live(
    *,
    candidate_id: str,
    digest: str,
) -> Path:
    """Server-side copy pending→live objects, verify, THEN write public index.

    Never rewrite the public index to live keys before objects exist there —
    the next box's master download reads index r2_key and would stall.

    The copy runs inside R2 in parallel parts (one CopyObject on a 4 GB shard
    runs into the read timeout). The host downloads nothing unless R2 gives an
    ETag that it cannot tie to the recorded sha256.
    """
    pending_path = pending_master_index_dir() / ('%s.json' % candidate_id)
    if not pending_path.is_file():
        raise ArtifactUploadError('pending master index missing for %s' % candidate_id[:12])
    doc = json.loads(pending_path.read_text())
    if str(doc.get('master_digest') or '') != str(digest):
        raise ArtifactUploadError(
            'pending master digest mismatch: index=%s item=%s'
            % (str(doc.get('master_digest'))[:12], str(digest)[:12])
        )
    files = dict(doc.get('files') or {})
    if not files:
        raise ArtifactUploadError('pending master has no files for %s' % candidate_id[:12])
    src_prefix = pending_r2_prefix(candidate_id)
    dst_prefix = live_fp32_r2_prefix(candidate_id)
    mirror = r2_local_mirror()
    bucket = client = None
    if mirror is None:
        try:
            bucket = open_life_bucket()
            client = R2MP.transfer_client(bucket)
        except ArtifactUploadError:
            raise
        except Exception as exc:
            raise ArtifactUploadError('R2 server-side copy failed: %s' % str(exc)[:200]) from exc
    live_files: Dict[str, Any] = {}
    for n, meta in files.items():
        meta = dict(meta or {})
        src_key = str(meta.get('r2_key') or ('%s/%s' % (src_prefix, n)))
        dst_key = '%s/%s' % (dst_prefix, n)
        # Verify live object before committing the public index.
        if mirror is not None:
            _mirror_copy(mirror, src_key, dst_key)
            expected = str(meta.get('sha256') or '')
            if expected:
                got = _mirror_sha256(mirror, dst_key)
                if got != expected:
                    raise ArtifactUploadError(
                        'promote verify mismatch %s: got %s want %s'
                        % (n, got[:12], expected[:12])
                    )
        else:
            _promote_object(bucket, client, n, meta, src_key, dst_key)
        live_files[n] = dict(meta, r2_key=dst_key)
    if 'meta.json' not in live_files:
        raise ArtifactUploadError(
            'promote refused: meta.json missing after pending→live copy'
        )
    public = public_master_index_dir()
    public.mkdir(parents=True, exist_ok=True)
    out_path = public / ('%s.json' % candidate_id)
    out_doc = {
        'bf16_sha256': doc.get('bf16_sha256') or candidate_id,
        'master_digest': digest,
        'files': live_files,
        'pending': False,
        'source': 'pit',
        'promoted_from_pending': True,
        'at': time.time(),
        # Same shape as live HEAD index: pointer to uploaded meta.json only.
        'meta_json': meta_json_r2_ptr(live_files, prefix=dst_prefix),
    }
    _wj(out_path, out_doc, 0o600)
    return out_path


def gc_pending_artifacts(candidate_id: str) -> Dict[str, Any]:
    """Remove private pending index AND R2/mirror objects (~55 GB), not just the index."""
    removed: list = []
    pending_path = pending_master_index_dir() / ('%s.json' % candidate_id)
    keys: list = []
    if pending_path.is_file():
        try:
            doc = json.loads(pending_path.read_text())
            for n, meta in (doc.get('files') or {}).items():
                k = (meta or {}).get('r2_key') or ('%s/%s' % (pending_r2_prefix(candidate_id), n))
                keys.append(str(k))
        except Exception:
            pass
        pending_path.unlink()
        removed.append(str(pending_path))
    # Always include the whole pending prefix (bf16 model/ + master shards).
    prefix = pending_r2_prefix(candidate_id)
    mirror = r2_local_mirror()
    if mirror is not None:
        root = mirror / prefix
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
            removed.append(str(root))
    else:
        try:
            bucket = open_life_bucket()
            client = getattr(bucket, 'client', None)
            # Parts of an upload that never finished cost storage and are in no object listing.
            if client is not None:
                try:
                    dropped = R2MP.abort_stale_uploads(bucket, prefix + '/', client=client)
                    if dropped:
                        removed.append('unfinished-uploads:%d' % dropped)
                except Exception as exc:
                    removed.append('r2-gc-upload-abort-error:%s' % str(exc)[:120])
            # Prefer boto3 list+delete so ~55 GB objects leave with the index.
            if client is not None:
                token = None
                while True:
                    kwargs = dict(Bucket=bucket.name, Prefix=prefix + '/', MaxKeys=1000)
                    if token:
                        kwargs['ContinuationToken'] = token
                    page = client.list_objects_v2(**kwargs)
                    for entry in page.get('Contents') or []:
                        key = entry.get('Key')
                        if key:
                            keys.append(str(key))
                    if not page.get('IsTruncated'):
                        break
                    token = page.get('NextContinuationToken')
                    if not token:
                        break
            delete_fn = getattr(bucket, 'delete_object', None) or getattr(bucket, 'delete', None)
            for key in sorted(set(keys)):
                try:
                    if callable(delete_fn):
                        delete_fn(key)
                    elif client is not None:
                        client.delete_object(Bucket=bucket.name, Key=key)
                    else:
                        continue
                    removed.append(key)
                except Exception:
                    pass
        except Exception as exc:
            removed.append('r2-gc-error:%s' % str(exc)[:120])
    return {'ok': True, 'removed': removed, 'candidate_id': candidate_id}


def stream_upload_master_from_box(
    P: Any,
    *,
    candidate_id: str,
    master_digest: str,
    bf16_sha256: str = '',
    box_master_dir: str,
    put: Any,
    pending: bool = True,
    uploaded: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Box→R2 stream for every fp32 shard; host verifies; write pending index.

    ``master_digest`` is the host-recorded digest (from train report) — never
    replaced by box meta.json.

    ``uploaded`` is the master tree of an ``upload_trees_from_box`` job that
    also carried other trees; without it this call runs the transfer itself.
    """
    if not isinstance(candidate_id, str) or len(candidate_id) != 64:
        raise ArtifactUploadError('candidate_id must be hex64')
    digest = str(master_digest or '').strip()
    if len(digest) != 64:
        raise ArtifactUploadError('master_digest must be hex64 (host-recorded)')
    if uploaded is None and box_multipart_transport():
        if not pending:
            # The public index is written by promote alone, after its own check of the live objects.
            raise ArtifactUploadError('fp32 master upload is pending-only; APPLY promotes it to live')
        uploaded = upload_trees_from_box(P, candidate_id=candidate_id, put=put, trees={
            'master': dict(dir=box_master_dir, prefix=pending_r2_prefix(candidate_id), names=None)})['master']
    if uploaded is not None:
        inv: Dict[str, Any] = {n: dict(m) for n, m in uploaded.items()}
        assert_master_tree_complete(inv)
        verified = {n: m['sha256'] for n, m in inv.items()}
        transport = 'box-r2-multipart'
    else:
        inv = inventory_box_master(P, box_master_dir)
        urls = _presign_master_puts(candidate_id, inv, pending=pending)
        missing = [n for n in inv if n not in urls]
        if missing:
            raise ArtifactUploadError('presign PUT missing for %s' % missing[:5])
        box_put_files(P, box_dir=box_master_dir, urls=urls, put=put)
        verified = verify_master_from_r2(candidate_id=candidate_id, inv=inv, pending=pending)
        transport = 'box-r2-stream'
    bf16 = str(bf16_sha256 or candidate_id)
    idx = write_pending_master_index(
        candidate_id=candidate_id, bf16=bf16, digest=digest, inv=inv,
    )
    return dict(
        ok=True,
        candidate_id=candidate_id,
        master_digest=digest,
        index=str(idx),
        pending=True,
        published=False,
        box_master_dir=box_master_dir,
        shard_count=sum(
            1 for n in inv
            if str(n).endswith('.safetensors') or str(n).endswith('.bin')
        ),
        complete=True,
        verified=len(verified),
        transport=transport,
        downloads=sum(1 for m in inv.values() if m.get('verify') == 'download'),
    )
