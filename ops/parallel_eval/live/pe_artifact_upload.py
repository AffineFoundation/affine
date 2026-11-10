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
    """Box streams each file to its PUT URL (https or file://). No host scp."""
    remote_urls = '/tmp/pit-master-put-%s.urls.private.json' % hashlib.sha256(
        json.dumps(sorted(urls)).encode()
    ).hexdigest()[:12]
    put(
        P,
        remote_urls,
        json.dumps(dict(urls=dict(urls), headers={'Content-Type': 'application/octet-stream'})).encode(),
        '600',
    )
    script = r"""
set -e
python3 - <<'PY'
import json, os, subprocess, sys, urllib.request
doc = json.load(open(%r))
urls = doc['urls']; headers = doc.get('headers') or {}
box = %r
bad = []
for name, url in urls.items():
    src = os.path.join(box, name)
    if not os.path.isfile(src):
        bad.append(('missing', name)); continue
    if url.startswith('file://'):
        dest = url[7:]
        os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
        subprocess.check_call(['cp', '-f', src, dest])
    else:
        # Stream without loading whole shard into memory when possible.
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
    doc = {
        'bf16_sha256': bf16,
        'master_digest': digest,  # host-recorded; never overwritten from box meta.json
        'files': files_meta,
        'pending': True,
        'private': True,
        'source': 'pit-pending',
        'at': time.time(),
        'meta_json': {
            'version': 'pit-fp32-pending-v1',
            'bf16_sha256': bf16,
            'master_digest': digest,
            'pending': True,
            'shard_count': sum(
                1 for n in files_meta
                if str(n).endswith('.safetensors') or str(n).endswith('.bin')
            ),
        },
    }
    _wj(idx_path, doc, 0o600)
    return idx_path


def promote_pending_master_to_live(
    *,
    candidate_id: str,
    digest: str,
) -> Path:
    """Copy pending index → public index; rewrite r2_keys to live prefix.

    Object copy on R2 is best-effort (mirror: rename/copy; live: leave pending
    readable or copy when Bucket supports it). Index promote is authoritative
    for APPLY.
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
    live_files = {}
    for n, meta in files.items():
        live_files[n] = dict(
            meta,
            r2_key='%s/%s' % (live_fp32_r2_prefix(candidate_id), n),
        )
    # Mirror: copy pending objects to live prefix.
    mirror = r2_local_mirror()
    if mirror is not None:
        src_prefix = pending_r2_prefix(candidate_id)
        dst_prefix = live_fp32_r2_prefix(candidate_id)
        for n in live_files:
            src = mirror / src_prefix / n
            dst = mirror / dst_prefix / n
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_file():
                shutil.copy2(src, dst)
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
        'meta_json': {
            'version': 'pit-fp32-master-v1',
            'bf16_sha256': doc.get('bf16_sha256') or candidate_id,
            'master_digest': digest,
            'source': 'pit',
            'shard_count': sum(
                1 for n in live_files
                if str(n).endswith('.safetensors') or str(n).endswith('.bin')
            ),
        },
    }
    _wj(out_path, out_doc, 0o600)
    return out_path


def gc_pending_artifacts(candidate_id: str) -> Dict[str, Any]:
    """Remove private pending index + mirror objects for a RED / released item."""
    removed = []
    pending_path = pending_master_index_dir() / ('%s.json' % candidate_id)
    if pending_path.is_file():
        pending_path.unlink()
        removed.append(str(pending_path))
    mirror = r2_local_mirror()
    if mirror is not None:
        prefix = mirror / pending_r2_prefix(candidate_id)
        if prefix.exists():
            shutil.rmtree(prefix, ignore_errors=True)
            removed.append(str(prefix))
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
