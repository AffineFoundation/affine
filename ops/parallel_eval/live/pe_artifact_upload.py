"""Box→R2 artifact upload helpers (bf16 + fp32 stream; private pending until APPLY).

Never stage a full fp32 master tree on the host. Box PUTs directly to R2
(presigned); host verifies digests from R2/mirror and may download only enough
to hash, then deletes any temp. RED artifacts stay under a private pending
prefix/index until APPLY promotes them (or GC on release).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


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
        from subnet.storage import Bucket  # type: ignore
        life_cfg = _env_path('PIT_LIFE_CFG', 'AFFINE_LIFE_CFG')
        if life_cfg is None or not life_cfg.is_file():
            raise ArtifactUploadError(
                'PIT_LIFE_CFG required for master put_object presign '
                '(or PIT_R2_LOCAL_MIRROR / PIT_PRESIGN_MASTER_PUT_JSON)'
            )
        bucket = Bucket(json.loads(life_cfg.read_text())['bucket'])
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
    """Host verifies shard sha from R2/mirror. Temp downloads deleted immediately."""
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
                from subnet.storage import Bucket  # type: ignore
                life_cfg = _env_path('PIT_LIFE_CFG', 'AFFINE_LIFE_CFG')
                if life_cfg is None or not life_cfg.is_file():
                    raise ArtifactUploadError('PIT_LIFE_CFG required to GET master for verify')
                bucket = Bucket(json.loads(life_cfg.read_text())['bucket'])
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
        n: {
            'sha256': m['sha256'],
            'bytes': int(m.get('bytes') or 0),
            'r2_key': '%s/%s' % (pending_r2_prefix(candidate_id), n),
        }
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


def _open_life_bucket() -> Any:
    from subnet.storage import Bucket  # type: ignore
    life_cfg = _env_path('PIT_LIFE_CFG', 'AFFINE_LIFE_CFG')
    if life_cfg is None or not life_cfg.is_file():
        raise ArtifactUploadError('PIT_LIFE_CFG required for R2 object ops')
    return Bucket(json.loads(life_cfg.read_text())['bucket'])


def _sha256_live_object(key: str) -> str:
    """Hash an object from local mirror or live R2 (fail closed)."""
    mirror = r2_local_mirror()
    if mirror is not None:
        path = mirror / key
        if not path.is_file():
            raise ArtifactUploadError('live object missing after promote: %s' % key)
        return _sha256_file(path)
    try:
        bucket = _open_life_bucket()
        data = bucket.get(key)
    except ArtifactUploadError:
        raise
    except Exception as exc:
        raise ArtifactUploadError('R2 GET for promote verify failed: %s' % str(exc)[:200]) from exc
    return hashlib.sha256(data).hexdigest()


def _copy_object_pending_to_live(src_key: str, dst_key: str) -> None:
    """Server-side copy pending → live (mirror: shutil; live: Bucket.copy)."""
    mirror = r2_local_mirror()
    if mirror is not None:
        src = mirror / src_key
        dst = mirror / dst_key
        if not src.is_file():
            raise ArtifactUploadError('pending object missing for promote: %s' % src_key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    try:
        bucket = _open_life_bucket()
        copy_fn = getattr(bucket, 'copy_object', None) or getattr(bucket, 'copy', None)
        if callable(copy_fn):
            copy_fn(src_key, dst_key)
            return
        # Fallback: GET+PUT (streaming) when Bucket has no copy helper.
        data = bucket.get(src_key)
        bucket.put(dst_key, data)
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
    live_files: Dict[str, Any] = {}
    for n, meta in files.items():
        src_key = str((meta or {}).get('r2_key') or ('%s/%s' % (src_prefix, n)))
        dst_key = '%s/%s' % (dst_prefix, n)
        _copy_object_pending_to_live(src_key, dst_key)
        # Verify live object before committing the public index.
        expected = str((meta or {}).get('sha256') or '')
        if expected:
            got = _sha256_live_object(dst_key)
            if got != expected:
                raise ArtifactUploadError(
                    'promote verify mismatch %s: got %s want %s'
                    % (n, got[:12], expected[:12])
                )
        live_files[n] = dict(meta or {}, r2_key=dst_key)
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
            bucket = _open_life_bucket()
            client = getattr(bucket, 'client', None)
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
) -> Dict[str, Any]:
    """Box→R2 stream for every fp32 shard; host verifies; write pending index.

    ``master_digest`` is the host-recorded digest (from train report) — never
    replaced by box meta.json.
    """
    if not isinstance(candidate_id, str) or len(candidate_id) != 64:
        raise ArtifactUploadError('candidate_id must be hex64')
    digest = str(master_digest or '').strip()
    if len(digest) != 64:
        raise ArtifactUploadError('master_digest must be hex64 (host-recorded)')
    inv = inventory_box_master(P, box_master_dir)
    urls = _presign_master_puts(candidate_id, inv, pending=pending)
    missing = [n for n in inv if n not in urls]
    if missing:
        raise ArtifactUploadError('presign PUT missing for %s' % missing[:5])
    box_put_files(P, box_dir=box_master_dir, urls=urls, put=put)
    verified = verify_master_from_r2(candidate_id=candidate_id, inv=inv, pending=pending)
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
        transport='box-r2-stream',
    )
