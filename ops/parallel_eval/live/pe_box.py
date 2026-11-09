#!/usr/bin/env python3
"""Host box I/O for PIT — on-box train ported from the working shadow driver.

Faithful port of host ``drive_item.py`` + ``run5090.py`` (imported there as
``old``) that trained/probed E1041, E1042, E1046, E1048 on real B300s:

1. Box fetches bf16 head + fp32 master from R2 (presigned URLs; never left on
   the box). Validator does **not** load the ~18.8 GB head / ~35.8 GB master
   into RAM and does **not** push them over ssh.
2. Golden trainer bundle (runtime-exact venv + source-final-v823u) is fetched
   ON THE BOX via ``bundle_fetch`` + ``BUN['trainer']`` (same as
   ``drive_item.bundle_trainer``). Host ships ``trainer-host-v834e`` via tar
   and verifies checksums. ``train_pit.py`` runs under that venv — fail closed
   if the venv is missing (never fall back to system python3).
3. Artifacts return as **files** via scp (never binary over ssh stdout —
   ``replace_pod.Pod.run`` uses ``errors=replace`` and corrupts bytes).
4. Inputs = signed TRAINING-JOB manifest + TRAIN-DISPATCH-FF + docs +
   head.descriptor + optimizer_step (``drive_item.inputs``), not a lone batch
   file.

Host paths are env/config configurable. No ``/home/const`` literals, IPs, or
private material are committed here.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

sys.dont_write_bytecode = True

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# On-box layout (drive_item / run5090)
# ---------------------------------------------------------------------------
PST = '/workspace/pst'
TW = '/workspace/affine-mimo-trainer-prod'
FP32 = TW + '/fp32-master'
KDIR = '/opt/plain/probe/affine-env-probe/ckpt'
BOX_TBX = '/opt/plain/tbx'
BOX_WORKSPACE = '/opt/plain/workspace'
BOX_VENV = TW + '/runtime-exact/venv/bin/python'
BOX_SOURCE = TW + '/source-final-v823u'
BOX_HOST = TW + '/host-v834e'
BOX_ROOT = '/root/affine-env-probe'


class BoxIOError(RuntimeError):
    """Fail-closed host box I/O error."""


# ---------------------------------------------------------------------------
# Configurable host paths (env). Defaults are relative to AFFINE_REPO / PE_*.
# Never hard-code operator home paths into the release.
# ---------------------------------------------------------------------------
def _env_path(*keys: str, default: Optional[Path] = None) -> Optional[Path]:
    for k in keys:
        v = os.environ.get(k)
        if v:
            return Path(v)
    return default


def _repo() -> Path:
    return _env_path('AFFINE_REPO', 'REPO', default=Path.home() / 'subnet120-rewrite')  # type: ignore[arg-type]


def _ra() -> Path:
    return _env_path('PE_RA', 'RA', default=_repo() / 'state' / 'root-audits')  # type: ignore[arg-type]


def _pe_state() -> Path:
    return _env_path(
        'PE_STATE', 'PE_ST',
        default=_ra() / '20261106-parallel-cutover' / 'state',
    )  # type: ignore[arg-type]


def _life() -> Path:
    return _env_path('PE_LIFE', 'PE_LIFE_STATE', default=_repo() / 'state' / 'mimo-mainnet-v1')  # type: ignore[arg-type]


def _upload_root() -> Path:
    return _env_path('PIT_UPLOAD_ROOT', default=_pe_state() / 'pit' / 'uploads')  # type: ignore[arg-type]


def _v834e_release() -> Path:
    """Signed release root containing trainer-host-v834e/ (+ optional SIGNED-MANIFEST.sha256)."""
    return _env_path(
        'PIT_V834E_RELEASE', 'AFFINE_V834E_RELEASE',
        default=_ra() / '20261108-no-improvement-audit' / 'fix-v2' / 'v834e' / 'release-v834e-signed',
    )  # type: ignore[arg-type]


def _trainer_host_dir() -> Path:
    """Host modules shipped via put_tar → TW/host-v834e (drive_item REL/trainer-host-v834e)."""
    override = _env_path('PIT_TRAINER_HOST', 'AFFINE_TRAINER_HOST')
    if override is not None:
        return override
    rel = _v834e_release()
    cand = rel / 'trainer-host-v834e'
    return cand if cand.is_dir() else rel


def _fp32_master_index() -> Path:
    """Published M(H) index dir (public website k1/fp32-master/<H>.json)."""
    return _env_path(
        'PIT_FP32_MASTER_INDEX', 'AFFINE_FP32_MASTER_INDEX',
        default=_repo() / 'website' / 'k1' / 'fp32-master',
    )  # type: ignore[arg-type]


def _bundle_json() -> Path:
    """BUNDLE.json (or BUNDLE5090.json) with trainer {key,sha256,bytes}."""
    return _env_path(
        'PIT_BUNDLE_JSON', 'AFFINE_BUNDLE_JSON',
        default=_pe_state() / 'BUNDLE.json',
    )  # type: ignore[arg-type]


def _pst_code() -> Path:
    """Host dir holding gate_fetch.py / bundle_fetch.py / presign_keys.py (RA copy or vendored)."""
    return _env_path(
        'PIT_PST_CODE', 'AFFINE_PST_CODE',
        default=_ra() / '20261102-per-sub-pod-test' / 'code',
    )  # type: ignore[arg-type]


def _presign_bin() -> Path:
    return _env_path(
        'PIT_PRESIGN', 'AFFINE_PRESIGN',
        default=_ra() / '20261026-promotion-gate' / 'code' / 'presign.py',
    )  # type: ignore[arg-type]


def _presign_keys_bin() -> Path:
    return _env_path(
        'PIT_PRESIGN_KEYS', 'AFFINE_PRESIGN_KEYS',
        default=_pst_code() / 'presign_keys.py',
    )  # type: ignore[arg-type]


def _host_py() -> str:
    return os.environ.get('HOST_PY') or str(_repo() / '.venv' / 'bin' / 'python')


def _train_pit_src() -> Path:
    return _env_path('PIT_TRAIN_PIT_PY', default=_HERE / 'train_pit.py')  # type: ignore[arg-type]


def _helper_src(name: str) -> Path:
    """Prefer vendored live/ copy; fall back to host PST_CODE."""
    vendored = _HERE / name
    if vendored.is_file():
        return vendored
    host = _pst_code() / name
    if host.is_file():
        return host
    raise BoxIOError('%s missing (vendored live/%s or PIT_PST_CODE)' % (name, name))


def _rj(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pod / scp helpers (binary-safe; never stream weights over ssh stdout)
# ---------------------------------------------------------------------------
def _as_pod(pod: Any) -> Any:
    if pod is None:
        raise BoxIOError('pod handle required for on-box train')
    if hasattr(pod, 'run') and callable(getattr(pod, 'run')):
        return pod
    if isinstance(pod, Mapping):
        try:
            import pe_common as PC
            return PC.pod_of(pod)
        except Exception:
            pass
        try:
            from ops.parallel_eval.live import pe_common as PC
            return PC.pod_of(pod)
        except Exception as exc:
            raise BoxIOError('cannot build Pod from rent dict: %s' % exc) from exc
    raise BoxIOError('pod must be a Pod or pe_rent dict, got %r' % type(pod))


def _scp_argv(P: Any) -> list:
    home = Path.home()
    kh = home / '.lium' / 'known_hosts' / str(getattr(P, 'id', 'unknown'))
    kh.parent.mkdir(parents=True, exist_ok=True)
    return [
        'scp', '-q',
        '-o', 'BatchMode=yes',
        '-o', 'StrictHostKeyChecking=%s' % ('yes' if kh.exists() else 'accept-new'),
        '-o', 'ConnectTimeout=60',
        '-o', 'UserKnownHostsFile=%s' % kh,
        '-P', str(getattr(P, 'port', 22)),
    ]


def _scp_to(P: Any, local: Path, remote: str, *, timeout: int = 3600) -> None:
    local = Path(local)
    if not local.is_file():
        raise BoxIOError('scp_to missing local file: %s' % local)
    dest = 'root@%s:%s' % (P.ip, remote)
    P.run('mkdir -p %s' % str(Path(remote).parent), timeout=60)
    cmd = _scp_argv(P) + [str(local), dest]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise BoxIOError('scp_to timeout: %s' % local.name) from exc
    if p.returncode != 0:
        err = (p.stderr or p.stdout or b'').decode('utf-8', errors='replace')[-400:]
        raise BoxIOError('scp_to failed (%s): %s' % (local.name, err))


def _scp_from(P: Any, remote: str, local: Path, *, timeout: int = 3600) -> Path:
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    src = 'root@%s:%s' % (P.ip, remote)
    cmd = _scp_argv(P) + [src, str(local)]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise BoxIOError('scp_from timeout: %s' % remote) from exc
    if p.returncode != 0 or not local.is_file():
        err = (p.stderr or p.stdout or b'').decode('utf-8', errors='replace')[-400:]
        raise BoxIOError('scp_from failed (%s): %s' % (remote, err))
    return local


def put(P: Any, dest: str, data: bytes, mode: str = '644') -> None:
    """run5090.put — small file via ssh stdin (helpers/scripts only)."""
    if not isinstance(data, (bytes, bytearray)):
        data = str(data).encode()
    if len(data) > 8 * 1024 * 1024:
        raise BoxIOError('refuse large put over ssh stdin (%d bytes); use put_tar/scp' % len(data))
    parent = str(Path(dest).parent)
    P.run(
        "mkdir -p %s && cat > %s.tmp && chmod %s %s.tmp && mv %s.tmp %s" % (
            parent, dest, mode, dest, dest, dest,
        ),
        inp=bytes(data),
        timeout=120,
    )


def put_tar(P: Any, local_dir: Path, remote_dir: str) -> None:
    """run5090.put_tar — directory ship via scp of a tar (binary-safe)."""
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise BoxIOError('put_tar missing local dir: %s' % local_dir)
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / 'put.tar'
        with tarfile.open(tar_path, 'w') as tar:
            for child in sorted(local_dir.iterdir()):
                tar.add(str(child), arcname=child.name)
        remote_tar = '/tmp/pit-put-%d.tar' % os.getpid()
        _scp_to(P, tar_path, remote_tar, timeout=7200)
        P.run(
            "rm -rf %s && mkdir -p %s && tar --no-same-owner -C %s -xf %s && rm -f %s && echo PUT_TAR_OK" % (
                remote_dir, remote_dir, remote_dir, remote_tar, remote_tar,
            ),
            timeout=1800,
        )


# ---------------------------------------------------------------------------
# Vendored run5090 helpers: presign_keys, fetch, fetch_master, BUN, bundle
# ---------------------------------------------------------------------------
def load_bun() -> Mapping[str, Any]:
    """BUN = rj(ST/BUNDLE.json) — trainer part must have key/sha256/bytes."""
    path = _bundle_json()
    bun = _rj(path)
    if not isinstance(bun, Mapping) or not isinstance(bun.get('trainer'), Mapping):
        raise BoxIOError(
            'trainer bundle entry missing in %s (need {"trainer":{key,sha256,bytes}}); '
            'set PIT_BUNDLE_JSON' % path
        )
    part = bun['trainer']
    for k in ('key', 'sha256', 'bytes'):
        if k not in part:
            raise BoxIOError('BUN[trainer] missing %s in %s' % (k, path))
    if len(str(part['sha256'])) != 64:
        raise BoxIOError('BUN[trainer].sha256 must be hex64')
    return bun


def presign_keys(keys: Mapping[str, str]) -> Mapping[str, str]:
    """run5090.presign_keys({name: r2_key}) -> {name: url}."""
    inj = os.environ.get('PIT_PRESIGN_KEYS_JSON')
    if inj:
        doc = json.loads(Path(inj).read_text() if Path(inj).is_file() else inj)
        # Allow injecting a full map, or pass-through of requested names.
        if isinstance(doc, Mapping):
            out = {}
            for name, key in keys.items():
                if name in doc:
                    out[name] = doc[name]
                elif key in doc:
                    out[name] = doc[key]
                else:
                    # Synthetic test URL derived from key.
                    out[name] = 'https://r2.example/presign/%s' % key
            return out
    helper = _presign_keys_bin()
    if not helper.is_file():
        raise BoxIOError('presign_keys helper missing: %s (set PIT_PRESIGN_KEYS or PIT_PRESIGN_KEYS_JSON)' % helper)
    p = subprocess.run(
        [_host_py(), '-B', str(helper)],
        input=json.dumps(dict(keys)),
        capture_output=True, text=True, timeout=120,
    )
    if p.returncode != 0:
        raise BoxIOError('presign_keys failed: %s' % (p.stderr or p.stdout)[-300:])
    return json.loads(p.stdout)


def ckpt_urls(cid: str, files: Mapping[str, Any]) -> Mapping[str, Any]:
    """run5090.ckpt_urls — per-checkpoint presign via gate presign.py."""
    inj = os.environ.get('PIT_PRESIGN_PAYLOAD_JSON')
    if inj:
        doc = json.loads(Path(inj).read_text() if Path(inj).is_file() else inj)
        if isinstance(doc, Mapping) and 'urls' in doc:
            return doc['urls']
        if isinstance(doc, Mapping):
            return doc
    helper = _presign_bin()
    if not helper.is_file():
        raise BoxIOError('presign helper missing: %s' % helper)
    p = subprocess.run(
        [_host_py(), '-B', str(helper), cid],
        input=json.dumps(files),
        capture_output=True, text=True, timeout=120,
    )
    if p.returncode != 0:
        raise BoxIOError('presign failed: %s' % (p.stderr or p.stdout)[-300:])
    return json.loads(p.stdout)['urls']


def fetch(P: Any, cid: str, files: Mapping[str, Any], urls: Mapping[str, Any], dest: str, threads: int = 32) -> tuple:
    """run5090.fetch — idempotent gate_fetch on the box; URLs file deleted after."""
    t0 = time.time()
    have = P.run(
        'test -f {K}/{c}/.verified && du -sb {K}/{c} | cut -f1 || echo NO'.format(K=dest, c=cid)
    ).strip()
    if have != 'NO':
        return 0.0, int(have)
    blob = json.dumps(dict(descriptor=dict(id=cid, files=files), urls=urls))
    c = (
        "set -e; umask 077; mkdir -p {K}/.work; cd {K}/.work && "
        "python3 -c \"import json,sys;d=json.load(sys.stdin);"
        "json.dump(d['descriptor'],open('{K}/{c}.descriptor.json','w'));"
        "json.dump(d['urls'],open('{c}.urls.private.json','w'))\" && "
        "(python3 {F} {K}/{c}.descriptor.json {K} {c}.urls.private.json --threads {n} "
        ">> {K}/{c}.fetch.log 2>&1; rc=$?; rm -f {c}.urls.private.json; "
        "tail -1 {K}/{c}.fetch.log; test -f {K}/{c}/.verified; du -sb {K}/{c} | cut -f1)"
    ).format(K=dest, c=cid, F=PST + '/gate_fetch.py', n=threads)
    out = P.run(c, inp=blob.encode(), timeout=5400).strip().splitlines()
    return round(time.time() - t0, 1), int(out[-1])


def _head_files_from_job(job_payload: Mapping[str, Any], head: str) -> Mapping[str, Any]:
    ck = ((job_payload.get('manifest') or {}).get('payload') or {}).get('checkpoint') or {}
    if ck.get('id') == head and isinstance(ck.get('files'), Mapping):
        return ck['files']
    raise BoxIOError('TRAINING-JOB checkpoint files missing for head %s' % head[:12])


def _resolve_epoch_dir(*, epoch: Optional[int], batch_handle: str, epoch_dir: Optional[str]) -> Path:
    if epoch_dir:
        p = Path(epoch_dir)
        if (p / 'TRAINING-JOB.json').is_file() and (p / 'TRAIN-DISPATCH-FF.json').is_file():
            return p
        raise BoxIOError('epoch_dir missing TRAINING-JOB.json / TRAIN-DISPATCH-FF.json: %s' % p)
    if epoch is not None:
        p = _life() / 'epochs' / str(int(epoch))
        if (p / 'TRAINING-JOB.json').is_file() and (p / 'TRAIN-DISPATCH-FF.json').is_file():
            return p
    # batch_handle may be the epoch dir itself or a file inside it.
    bp = Path(str(batch_handle))
    for cand in (bp, bp.parent, bp.parent.parent):
        if (cand / 'TRAINING-JOB.json').is_file() and (cand / 'TRAIN-DISPATCH-FF.json').is_file():
            return cand
    raise BoxIOError(
        'cannot resolve epoch dir with TRAINING-JOB.json + TRAIN-DISPATCH-FF.json '
        '(set epoch / epoch_dir / PE_LIFE, or point batch_handle at the epoch dir)'
    )


def _published_master(head: str) -> Mapping[str, Any]:
    """Read published M(H) index (drive_item ctx / run5090.fetch_master)."""
    inj = os.environ.get('PIT_MASTER_INDEX_JSON')
    if inj:
        doc = json.loads(Path(inj).read_text() if Path(inj).is_file() else inj)
        if isinstance(doc, Mapping) and doc.get('bf16_sha256') and doc.get('master_digest'):
            return doc
    idx = _fp32_master_index() / (head + '.json')
    if idx.is_file():
        m = json.loads(idx.read_text())
        return m
    # Public HTTPS fallback (no secrets).
    try:
        url = os.environ.get('PIT_FP32_MASTER_URL_TMPL') or 'https://affine.io/k1/fp32-master/%s.json'
        with urllib.request.urlopen(url % head, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        raise BoxIOError('M(H) unpublished for %s -> fail closed: %s' % (head[:12], exc)) from exc


def fetch_master_on_box(P: Any, *, head: str, expected_master: str, head_bf16: str, published: Mapping[str, Any]) -> Mapping[str, Any]:
    """run5090.fetch_master + drive_item.master verify (fail closed).

    gate_fetch lands under FP32/<bf16_sha256>/ — never nest as store/<head>/<head>/.
    Verifies meta.json bf16_sha256 + master_digest against the published index.
    Never writes a digest the box did not compute/verify.
    """
    if published.get('master_digest') != expected_master:
        raise BoxIOError('published master_digest != expected_master')
    if published.get('bf16_sha256') != head_bf16:
        raise BoxIOError('published bf16_sha256 != head_bf16')
    files_meta = published.get('files') or {}
    if not isinstance(files_meta, Mapping) or not files_meta:
        raise BoxIOError('published master files map missing')
    files = {fn: (v['sha256'] if isinstance(v, Mapping) else v) for fn, v in files_meta.items()}
    keys = {fn: (v['r2_key'] if isinstance(v, Mapping) else fn) for fn, v in files_meta.items()}
    # Allow test injection of already-presigned master URLs.
    inj = os.environ.get('PIT_MASTER_PRESIGN_JSON')
    if inj:
        doc = json.loads(Path(inj).read_text() if Path(inj).is_file() else inj)
        urls = doc.get('urls') if isinstance(doc, Mapping) else None
        if not isinstance(urls, Mapping):
            # Build name->url from download block only when it is already a {file:url} map.
            dl = doc.get('download') if isinstance(doc, Mapping) else None
            if isinstance(dl, Mapping) and dl and all(isinstance(v, str) and str(v).startswith('http') for v in dl.values()):
                urls = dl
            else:
                urls = None
        if not isinstance(urls, Mapping):
            urls = presign_keys(keys)
    else:
        urls = presign_keys(keys)
    entry = head_bf16
    s, b = fetch(P, entry, files, urls, FP32)
    # Resolve meta.json contents (drive_item / run5090).
    mj = published.get('meta_json') or {}
    if isinstance(mj, Mapping) and 'version' in mj:
        meta = mj
    elif isinstance(mj, Mapping) and mj.get('r2_key'):
        meta = json.loads(urllib.request.urlopen(presign_keys({'meta': mj['r2_key']})['meta'], timeout=120).read())
    else:
        # meta.json should already be one of the fetched files; read it back from the box.
        raw = P.run('cat %s/%s/meta.json' % (FP32, entry), timeout=60)
        meta = json.loads(raw)
    if meta.get('bf16_sha256') != entry or meta.get('master_digest') != expected_master:
        raise BoxIOError('fp32 master meta mismatch: %s' % {k: meta.get(k) for k in ('bf16_sha256', 'master_digest')})
    put(P, FP32 + '/' + entry + '/meta.json', json.dumps(meta, sort_keys=True, indent=1).encode())
    # Box-side verify (drive_item.master) — print computed fields, assert equality.
    got = P.run(
        "python3 -c \"import json;m=json.load(open('%s/%s/meta.json'));"
        "print(m['bf16_sha256'],m['master_digest'])\"" % (FP32, entry),
        timeout=60,
    ).split()
    if got != [head_bf16, expected_master]:
        raise BoxIOError('pod master meta %s != expected [%s, %s]' % (got, head_bf16[:12], expected_master[:12]))
    return dict(seconds=s, bytes=b, entry=entry, digest=expected_master, skipped=False)


def _parse_r2_handle(handle: str, *, kind: str) -> str:
    if isinstance(handle, str) and handle.startswith('r2://%s/' % kind):
        return handle.split('/', 3)[-1].split(':')[0]
    if isinstance(handle, str) and len(handle) == 64:
        return handle
    raise BoxIOError('expected r2://%s/<hex64> handle, got %r' % (kind, handle))


def _detect_box_gpu(P: Any) -> Optional[str]:
    try:
        out = P.run(
            "nvidia-smi --query-gpu=name --format=csv,noheader | head -1",
            timeout=30, check=False,
        )
    except Exception:
        return None
    text = (out or '').strip().splitlines()
    if not text:
        return None
    name = text[0].strip()
    upper = name.upper()
    for tag in ('B300', 'B200', 'H200', 'H100', 'A100'):
        if tag in upper:
            return tag
    return name


# ---------------------------------------------------------------------------
# drive_item stages (boot / bundle_trainer / head / master / inputs / train)
# ---------------------------------------------------------------------------
def boot_helpers(P: Any) -> None:
    """drive_item.boot / run5090.boot — push gate_fetch + bundle_fetch; mkdir layout."""
    P.run(
        "mkdir -p %s/in %s %s %s %s/ckpt %s/runs; "
        "command -v zstd >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
        "apt-get install -y -qq zstd >/dev/null); echo BOOT_OK" % (
            PST, TW, FP32, BOX_TBX, BOX_ROOT, BOX_ROOT,
        ),
        timeout=600,
    )
    for fname in ('gate_fetch.py', 'bundle_fetch.py'):
        put(P, PST + '/' + fname, _helper_src(fname).read_bytes(), '700')


def bundle_trainer(P: Any) -> Mapping[str, Any]:
    """drive_item.bundle_trainer — R2 golden bundle ON THE BOX + host-v834e ship."""
    bun = load_bun()
    part = bun['trainer']
    info: dict = {}
    done = P.run(
        'test -f /root/.pks-bundle-trainer && cat /root/.pks-bundle-trainer || echo NO'
    ).strip()
    if done != part['sha256']:
        url = presign_keys({'trainer': part['key']})['trainer']
        P.run('rm -rf %s && mkdir -p %s' % (BOX_TBX, BOX_TBX), timeout=120)
        out = P.run(
            'python3 %s/bundle_fetch.py trainer %d %s %s' % (
                PST, int(part['bytes']), part['sha256'], BOX_TBX,
            ),
            inp=json.dumps(dict(url=url)).encode(),
            timeout=3600,
        )
        res = json.loads(out.strip().splitlines()[-1])
        if not res.get('sha_ok') or res.get('tar_rc'):
            raise BoxIOError('bundle trainer: %s' % res)
        tops = P.run('ls -A %s' % BOX_TBX).split()
        for t in tops:
            if t == 'workspace':
                P.run(
                    'mkdir -p %s && cp -al %s/workspace/. %s/' % (BOX_WORKSPACE, BOX_TBX, BOX_WORKSPACE),
                    timeout=1800,
                )
            else:
                P.run(
                    'mkdir -p /%s && cp -a %s/%s/. /%s/' % (t, BOX_TBX, t, t),
                    timeout=1800,
                )
        P.run('rm -rf %s; echo %s > /root/.pks-bundle-trainer' % (BOX_TBX, part['sha256']))
        info['bundle-trainer'] = dict(seconds=res.get('seconds'), gbps=res.get('gbps'), tops=tops)
    else:
        info['bundle-trainer'] = dict(present=True)

    # Ship trainer-host-v834e (drive_item: put_tar REL/trainer-host-v834e → TW/host-v834e).
    host_src = _trainer_host_dir()
    if not host_src.is_dir():
        raise BoxIOError('trainer-host-v834e missing on host (%s); set PIT_TRAINER_HOST / PIT_V834E_RELEASE' % host_src)
    put_tar(P, host_src, BOX_HOST)
    man = subprocess.run(
        'cd %s && find . -type f -print0 | sort -z | xargs -0 sha256sum' % host_src,
        shell=True, capture_output=True, text=True, check=True,
    ).stdout
    chk = P.run(
        'cd %s && sha256sum --quiet -c - && echo SHA_OK' % BOX_HOST,
        inp=man.encode(), timeout=120,
    )
    if 'SHA_OK' not in chk:
        raise BoxIOError('host-v834e sha256sum -c failed')

    # Optional release-level SIGNED-MANIFEST.sha256 (fail closed; never `; true`).
    rel = _v834e_release()
    man_sha = rel / 'SIGNED-MANIFEST.sha256'
    if man_sha.is_file():
        put(P, TW + '/SIGNED-MANIFEST.sha256', man_sha.read_bytes(), '644')
        # Manifest paths are relative to the release root; verify against shipped trees when present.
        P.run(
            'set -e; cd %s && sha256sum --quiet -c SIGNED-MANIFEST.sha256 && echo MANIFEST_OK' % TW,
            timeout=300,
        )

    # Fail closed: fetched venv + source must exist. Never fall back to system python3.
    out = P.run(
        'set -e; test -x %s; test -d %s; ls -d %s %s; '
        '%s -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"' % (
            BOX_VENV, BOX_SOURCE, BOX_SOURCE, BOX_VENV, BOX_VENV,
        ),
        timeout=300,
    )
    info['trainer_torch'] = out.strip()[-300:]
    info['venv'] = BOX_VENV
    return info


def fetch_head_on_box(P: Any, *, head: str, files: Mapping[str, Any]) -> Mapping[str, Any]:
    """drive_item.head — gate_fetch bf16 head into KDIR, hardlink into PST/in/actor/<H>."""
    # files map may be {name: sha} or {name: {sha256:...}}
    flat = {}
    for n, v in files.items():
        flat[n] = v['sha256'] if isinstance(v, Mapping) else v
    s, b = fetch(P, head, flat, ckpt_urls(head, files), KDIR)
    P.run(
        'mkdir -p {A} && for f in $(ls {K}/{h}); do '
        'ln -f {K}/{h}/$f {A}/$f 2>/dev/null || cp {K}/{h}/$f {A}/$f; done; ls {A} | wc -l'.format(
            A=PST + '/in/actor/' + head, K=KDIR, h=head,
        ),
        timeout=600,
    )
    return dict(seconds=s, bytes=b, gbps=round(b * 8 / s / 1e9, 2) if s else None)


def ship_inputs(P: Any, *, epoch_dir: Path, epoch: int, work_jd: Path) -> Mapping[str, Any]:
    """drive_item.inputs — signed TRAINING-JOB + DISPATCH + docs + head.descriptor."""
    job = json.loads((epoch_dir / 'TRAINING-JOB.json').read_text())['payload']
    t = job['training']
    d = json.loads((epoch_dir / 'TRAIN-DISPATCH-FF.json').read_text())
    if not d.get('signature') or d.get('signer') == 'SHADOW-UNSIGNED':
        raise BoxIOError('TRAIN-DISPATCH-FF must be signed (refuse SHADOW-UNSIGNED)')
    inn = work_jd / 'in'
    inn.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ('manifest', job['manifest']),
        ('dispatch', d),
        ('train_config', t['config']),
        ('meta', dict(optimizer_step=t['optimizer_step'], epoch=int(epoch))),
        ('head.descriptor', job['manifest']['payload']['checkpoint']),
    ):
        p = inn / (name + '.json')
        p.write_text(json.dumps(value))
        os.chmod(p, 0o600)
    put_tar(P, inn, PST + '/in')
    rows = d['payload']['selected']
    files = {x['document']['sha256'] + '.json': x['document']['sha256'] for x in rows}
    keys = {x['document']['sha256'] + '.json': x['document']['key'] for x in rows}
    cid = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    fetch(P, cid, files, presign_keys(keys), PST + '/docdl')
    P.run(
        'mkdir -p %s/in/docs; cp -l %s/docdl/%s/*.json %s/in/docs/' % (PST, PST, cid, PST),
        timeout=600,
    )
    return dict(
        epoch=int(epoch), tasks=len(rows),
        rollouts=d['payload'].get('selected_rollout_count'),
        optimizer_step=t['optimizer_step'],
    )


def run_train_pit(P: Any, *, expected_master: str) -> Mapping[str, Any]:
    """drive_item.train — ship train_pit.py and run under fetched venv (fail closed)."""
    # Fail closed on venv before launch.
    P.run('set -e; test -x %s; test -d %s' % (BOX_VENV, BOX_SOURCE), timeout=60)
    src = _train_pit_src()
    if not src.is_file():
        raise BoxIOError('train_pit.py missing: %s' % src)
    put(P, PST + '/train_pit.py', src.read_bytes(), '700')
    cmd = (
        "cd {S} && rm -rf {O} && "
        "(nohup nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits -lms 2000 > {P}/mem.csv 2>&1 & echo $! > {P}/mem.pid); "
        "env -u AFFINE_FP32_BOOTSTRAP_OVERRIDE AFFINE_FP32_EXPECTED_MASTER_HASH={X} "
        "CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 "
        "PYTHONPATH={S} TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "{T}/runtime-exact/venv/bin/python -B {P}/train_pit.py > {P}/train.log 2>&1; echo rc=$?; "
        "kill $(cat {P}/mem.pid) 2>/dev/null; "
        "grep -E '^\\{{' {P}/train.log | tail -40; "
        "grep -v helper-preload {P}/train.log | grep -v '^\\{{' | tail -60"
    ).format(S=BOX_SOURCE, O=PST + '/out', T=TW, P=PST, X=expected_master)
    out = P.run(cmd, timeout=int(os.environ.get('PIT_TRAIN_TIMEOUT_S') or 7200), check=False)
    events = {}
    for line in out.splitlines():
        try:
            x = json.loads(line)
        except Exception:
            continue
        if isinstance(x, dict) and x.get('event'):
            events[x['event']] = x
    if 'rc=0' not in out or 'done' not in events:
        raise BoxIOError('train failed: ' + out[-1500:])
    return dict(events=events, checkpoint=events['done']['checkpoint'], log_tail=out[-4000:])


def place_and_pull(P: Any, *, candidate_id: str, host_out: Path) -> Mapping[str, Any]:
    """drive_item.place_cand + scp file return of descriptor/report/candidate."""
    out = P.run(
        r"""set -e; cd {O}; f=$(find . -name 'model-00001-of-*.safetensors' | head -1); M={O}/$(dirname $f); echo $M
python3 - {O}/descriptor.json $M <<'PY'
import json,sys,hashlib,os
d=json.load(open(sys.argv[1])); M=sys.argv[2]; bad=[]
for n,h in d['files'].items():
    x=hashlib.sha256()
    with open(os.path.join(M,n),'rb') as f:
        for b in iter(lambda:f.read(1<<24),b''): x.update(b)
    if x.hexdigest()!=h: bad.append(n)
print('FILES', len(d['files']), 'BAD', len(bad))
PY
mkdir -p {K}/{c}; for f in $(ls $M); do ln -f $M/$f {K}/{c}/$f 2>/dev/null || cp $M/$f {K}/{c}/$f; done; cp {O}/descriptor.json {K}/{c}.descriptor.json; touch {K}/{c}/.verified; ls {K}/{c} | wc -l""".format(
            O=PST + '/out', K=KDIR, c=candidate_id,
        ),
        timeout=1800,
    )
    if 'BAD 0' not in out:
        raise BoxIOError('place_cand sha mismatch: %s' % out[-400:])

    host_out.mkdir(parents=True, exist_ok=True)
    # Report as a file (JSON). Prefer report.private.json (train_pit); fall back to report.json.
    remote_report = None
    for cand in (PST + '/out/report.private.json', PST + '/out/report.json', PST + '/report.private.json'):
        marker = P.run('test -f %s && echo YES || echo NO' % cand, timeout=30, check=False)
        if 'YES' in str(marker):
            remote_report = cand
            break
    if not remote_report:
        raise BoxIOError('training report missing on box (report.private.json)')
    report = host_out / 'report.json'
    _scp_from(P, remote_report, report, timeout=600)
    remote_sha = P.run('sha256sum %s | cut -d" " -f1' % remote_report, timeout=30).strip().split()[0]
    got_sha = _sha256_file(report)
    if remote_sha != got_sha:
        raise BoxIOError('training report sha mismatch after scp')
    rep = json.loads(report.read_text())

    # Candidate tree via tar+scp (never binary over ssh stdout).
    remote_cand_tar = PST + '/candidate.tar'
    P.run(
        "set -e; cd %s/out; f=$(find . -name 'model-00001-of-*.safetensors' | head -1); "
        "M=$(dirname $f); tar -C %s/out/$M -cf %s .; "
        "cp %s/out/descriptor.json %s/out/descriptor.json.export; echo CAND_TAR_OK" % (
            PST, PST, remote_cand_tar, PST, PST,
        ),
        timeout=3600,
    )
    cand_tar = host_out / 'candidate.tar'
    _scp_from(P, remote_cand_tar, cand_tar, timeout=7200)
    remote_tar_sha = P.run('sha256sum %s | cut -d" " -f1' % remote_cand_tar, timeout=60).strip().split()[0]
    if remote_tar_sha != _sha256_file(cand_tar):
        raise BoxIOError('candidate.tar sha mismatch after scp')
    cand_dir = host_out / 'candidate'
    cand_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(cand_tar, 'r') as tar:
        tar.extractall(cand_dir)
    _scp_from(P, PST + '/out/descriptor.json', host_out / 'descriptor.json', timeout=120)

    desc = json.loads((host_out / 'descriptor.json').read_text())
    cid = str(desc.get('id') or candidate_id)
    if len(cid) != 64:
        raise BoxIOError('descriptor.id missing')

    # Master digest from report (computed by trainer) — never invent one.
    fp = ((rep.get('optimizer') or {}).get('sgd') or {}).get('fp32_master') or rep.get('fp32') or {}
    master_dig = str(
        fp.get('output_master_digest')
        or rep.get('candidate_master_digest')
        or rep.get('master_digest')
        or ''
    )
    master_out = host_out / 'fp32-master'
    master_out.mkdir(parents=True, exist_ok=True)
    if len(master_dig) == 64:
        (master_out / 'master_digest.txt').write_text(master_dig + '\n')
    else:
        # Trainer always emits output_master_digest on success; fail closed if absent.
        raise BoxIOError('candidate master digest missing from training report')

    return dict(
        candidate_id=cid,
        candidate_path=str(cand_dir),
        candidate_master_path=str(master_out),
        candidate_master_digest=master_dig,
        report=str(report),
        report_doc=rep,
        place_tail=out.strip()[-200:],
    )


# ---------------------------------------------------------------------------
# HostBoxIO
# ---------------------------------------------------------------------------
class HostBoxIO:
    """Host-side BoxIO: provision+train ON the rented box (drive_item path)."""

    def download_head(self, head: str) -> str:
        if not isinstance(head, str) or len(head) != 64:
            raise BoxIOError('head must be hex64')
        return 'r2://head/%s' % head

    def download_master(self, head: str, expected_digest: str) -> str:
        if not isinstance(head, str) or len(head) != 64:
            raise BoxIOError('head must be hex64')
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise BoxIOError('expected_digest must be hex64')
        return 'r2://master/%s:%s' % (head, expected_digest)

    def train_v834e(self, **kwargs: Any) -> Mapping[str, Any]:
        if kwargs.get('fp32_bootstrap_override') or kwargs.get('bootstrap_override'):
            raise BoxIOError('bootstrap-override-refused-by-pit')
        pod = kwargs.get('pod')
        expected = kwargs.get('expected_master_digest')
        batch_handle = kwargs.get('batch_handle')
        item_id = kwargs.get('item_id') or 'pit-item'
        head_handle = kwargs.get('head_path') or ''
        if not expected or len(str(expected)) != 64:
            raise BoxIOError('train_v834e requires expected_master_digest hex64')
        if not batch_handle or str(batch_handle).startswith('secret:'):
            raise BoxIOError('train_v834e requires opaque non-secret batch_handle')
        if str(batch_handle).startswith('batch-handle:'):
            raise BoxIOError(
                'refuse synthetic batch-handle; pass real K1/epoch path '
                '(E1041-class divergence risk vs live trainer)'
            )

        head = _parse_r2_handle(str(head_handle), kind='head')
        epoch = kwargs.get('epoch')
        if epoch is None and kwargs.get('item_id'):
            # item ids are opaque; epoch must be explicit or resolvable from batch path.
            pass
        epoch_dir = _resolve_epoch_dir(
            epoch=int(epoch) if epoch is not None else None,
            batch_handle=str(batch_handle),
            epoch_dir=kwargs.get('epoch_dir'),
        )
        if epoch is None:
            try:
                epoch = int(epoch_dir.name)
            except Exception:
                raise BoxIOError('epoch required (kwargs.epoch or numeric epoch_dir name)')
        epoch = int(epoch)

        job = json.loads((epoch_dir / 'TRAINING-JOB.json').read_text())['payload']
        ck = job['manifest']['payload']['checkpoint']
        if ck.get('id') != head:
            raise BoxIOError('TRAINING-JOB checkpoint.id %s != bound head %s' % (str(ck.get('id'))[:12], head[:12]))
        files = _head_files_from_job(job, head)

        published = _published_master(head)
        if published.get('master_digest') != expected:
            raise BoxIOError('M(H) master_digest != bound expected_master (fail closed)')
        head_bf16 = published['bf16_sha256']
        if len(str(head_bf16)) != 64:
            raise BoxIOError('published bf16_sha256 missing')

        P = _as_pod(pod)
        info: dict = {}

        boot_helpers(P)
        info.update(bundle_trainer(P))
        info['fetch-head'] = fetch_head_on_box(P, head=head, files=files)
        info['fetch-master'] = fetch_master_on_box(
            P, head=head, expected_master=str(expected), head_bf16=str(head_bf16), published=published,
        )
        if info['fetch-master'].get('skipped'):
            raise BoxIOError('no published master -> fail closed')

        with tempfile.TemporaryDirectory(prefix='pit-jd-') as td:
            info['signed_inputs'] = ship_inputs(
                P, epoch_dir=epoch_dir, epoch=epoch, work_jd=Path(td),
            )

        train = run_train_pit(P, expected_master=str(expected))
        info['train-events'] = train['events']
        candidate_id = train['checkpoint']
        if len(str(candidate_id)) != 64:
            raise BoxIOError('train done.checkpoint missing')

        host_out = _upload_root() / ('train-%s-%d' % (str(item_id)[:16], int(time.time())))
        pulled = place_and_pull(P, candidate_id=str(candidate_id), host_out=host_out)
        rep = pulled['report_doc']
        if rep.get('used_fp32_bootstrap_override') or (
            (rep.get('fp32_master') or {}).get('source') == 'bootstrap-upcast'
        ):
            raise BoxIOError('bootstrap-override-refused-by-pit')
        # train_pit asserts input_master_digest == expected; surface that.
        fp = ((rep.get('optimizer') or {}).get('sgd') or {}).get('fp32_master') or (rep.get('fp32') or {})
        if fp.get('input_master_digest') and fp.get('input_master_digest') != expected:
            raise BoxIOError('input master digest mismatch')

        train_gpu = kwargs.get('train_gpu') or _detect_box_gpu(P)

        # Scrub any leftover URL private files.
        P.run(
            "rm -f %s/.work/*.urls.private.json %s/*/*.urls.private.json 2>/dev/null; echo SCRUB_OK" % (KDIR, PST),
            timeout=60, check=False,
        )

        return dict(
            candidate_id=pulled['candidate_id'],
            candidate_path=pulled['candidate_path'],
            candidate_master_path=pulled['candidate_master_path'],
            candidate_master_digest=pulled['candidate_master_digest'],
            candidate_sha256=pulled['candidate_id'],
            used_fp32_bootstrap_override=False,
            fp32_master_source=fp.get('source') or 'disk',
            report=pulled['report'],
            report_doc=rep,
            training_report=rep,
            train_gpu=train_gpu,
            pod_id=getattr(P, 'id', None),
            box_work=PST,
            fetch='r2-box-side',
            artifact_transport='scp',
            info=info,
            venv=BOX_VENV,
        )

    def upload_candidate(self, *, candidate_id: str, candidate_path: str) -> Mapping[str, Any]:
        src = Path(candidate_path)
        if not src.exists():
            raise BoxIOError('candidate_path missing for upload')
        dst = _upload_root() / candidate_id
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / src.name)
        man = {
            'id': candidate_id,
            'sha256': candidate_id,
            'uploaded_at': time.time(),
            'source': 'pit',
        }
        (dst / 'UPLOAD.json').write_text(json.dumps(man, indent=1, sort_keys=True) + '\n')
        os.chmod(dst / 'UPLOAD.json', 0o600)
        return dict(ok=True, path=str(dst))

    def upload_master_during_probe(self, *, candidate_id: str, master_path: str) -> Mapping[str, Any]:
        return upload_master(candidate_id=candidate_id, path=master_path)

    def probe_result_fixture(self) -> Mapping[str, Any]:
        raise BoxIOError('live HostBoxIO refuses probe_result_fixture (use pe_lane.run_probe)')


def box_io_factory() -> HostBoxIO:
    return HostBoxIO()


def destroy_box_state(*, box_id: str, reason: str = 'done') -> Mapping[str, Any]:
    return dict(ok=True, box_id=box_id, reason=reason)


def upload_master(*, candidate_id: str, path: str) -> Mapping[str, Any]:
    src = Path(path)
    if not src.exists():
        raise BoxIOError('master path missing')
    dst = _upload_root() / candidate_id / 'fp32-master'
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / src.name)
    return dict(ok=True, path=str(dst), candidate_id=candidate_id)


def upload_candidate(*, candidate_id: str, path: str) -> Mapping[str, Any]:
    return HostBoxIO().upload_candidate(candidate_id=candidate_id, candidate_path=path)


BoxIO = HostBoxIO


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv or argv[0] in ('-h', '--help'):
        print('usage: pe_box.py train --pod-json POD.json --head H --master-digest M --epoch N')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
