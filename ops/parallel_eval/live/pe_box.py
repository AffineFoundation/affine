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

# drive_item.GRADER_FIX — stale grader uv symlinks from the probe bundle (zero-sample fix).
GRADER_FIX = (
    'find /opt/plain/probe/.cache/uv/environments-v2 -maxdepth 4 -type l '
    '-lname "/opt/affine-grader-uv/*" -delete 2>/dev/null; true'
)


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
    """Published M(H) index dir — required (host sets website k1/fp32-master). No default."""
    p = _env_path('PIT_FP32_MASTER_INDEX', 'AFFINE_FP32_MASTER_INDEX')
    if p is None:
        raise BoxIOError(
            'PIT_FP32_MASTER_INDEX required (host path to public k1/fp32-master index dir); '
            'no default — refuse PE_STATE guess'
        )
    return p


def _bundle_json() -> Path:
    """BUNDLE5090.json with trainer {key,sha256,bytes} — required. No PE_STATE/BUNDLE.json default."""
    p = _env_path('PIT_BUNDLE_JSON', 'AFFINE_BUNDLE_JSON')
    if p is None:
        raise BoxIOError(
            'PIT_BUNDLE_JSON required (host path to train-cap-64k BUNDLE5090.json with trainer entry); '
            'no default — PE_STATE/BUNDLE.json has no trainer'
        )
    return p


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
    """common5090.put_tar semantics — never ``rm -rf`` the dest (preserves in/actor/<H>).

    Binary ship via scp of a tar (safer than ssh stdin for large trees). Remote:
    ``mkdir -p D && tar --no-same-owner -C D -xf T && rm -f T``.
    Callers that need a clean dir (host-v834e) must ``rm -rf`` that path first.
    """
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
            "mkdir -p %s && tar --no-same-owner -C %s -xf %s && rm -f %s && echo PUT_TAR_OK" % (
                remote_dir, remote_dir, remote_tar, remote_tar,
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
    # Public HTTPS fallback (no secrets). Send User-Agent — bare urllib gets 403.
    try:
        url = os.environ.get('PIT_FP32_MASTER_URL_TMPL') or 'https://affine.io/k1/fp32-master/%s.json'
        req = urllib.request.Request(
            url % head,
            headers={'User-Agent': 'affine-pit-pe-box/1'},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
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
    # A PIT index lists meta.json among the files: the trainer's own meta.json came
    # down with the shards and gate_fetch checked its sha256. Keep it — an inline
    # index body has no tensor_file and the trainer rejects the master without it.
    meta_on_box = 'meta.json' in files
    if meta_on_box:
        meta = json.loads(P.run('cat %s/%s/meta.json' % (FP32, entry), timeout=60))
    elif isinstance(mj, Mapping) and 'version' in mj:
        meta = mj
    elif isinstance(mj, Mapping) and mj.get('r2_key'):
        meta = json.loads(urllib.request.urlopen(presign_keys({'meta': mj['r2_key']})['meta'], timeout=120).read())
    else:
        # meta.json should already be one of the fetched files; read it back from the box.
        raw = P.run('cat %s/%s/meta.json' % (FP32, entry), timeout=60)
        meta = json.loads(raw)
    if meta.get('bf16_sha256') != entry or meta.get('master_digest') != expected_master:
        raise BoxIOError('fp32 master meta mismatch: %s' % {k: meta.get(k) for k in ('bf16_sha256', 'master_digest')})
    if not meta_on_box:
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
def _pe_lane():
    try:
        import pe_lane as L
        return L
    except ImportError:
        from ops.parallel_eval.live import pe_lane as L
        return L


def grader_fix(P: Any) -> None:
    """drive_item GRADER_FIX (boot + before run_probe)."""
    P.run(GRADER_FIX, check=False, timeout=120)


def boot_helpers(P: Any) -> None:
    """drive_item.boot: pe_lane.boot FIRST, then helpers + GRADER_FIX.

    pe_lane.boot symlinks ``/workspace`` → ``/opt/plain/workspace`` and
    ``/root/affine-env-probe``. Creating real ``/workspace/pst`` or
    ``/workspace/affine-mimo-trainer-prod`` *before* that breaks the venv
    check (TW lands on the wrong mount).
    """
    L = _pe_lane()
    L.boot(P)
    # Now mkdir under the symlink targets (never create real /workspace first).
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
    grader_fix(P)


def push_probe_bundle(P: Any) -> None:
    """drive_item.bundle_probe — pe_lane.push_bundle (podctl.py + vLLM env)."""
    _pe_lane().push_bundle(P)


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
    # Clean only BOX_HOST — put_tar itself must not wipe (preserves pst/in/actor/<H>).
    host_src = _trainer_host_dir()
    if not host_src.is_dir():
        raise BoxIOError('trainer-host-v834e missing on host (%s); set PIT_TRAINER_HOST / PIT_V834E_RELEASE' % host_src)
    P.run('rm -rf %s' % BOX_HOST, timeout=120)
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

    # SIGNED-MANIFEST paths are trainer-host-v834e/...; files ship to TW/host-v834e.
    # Symlink then fail-closed check (never `; true`).
    rel = _v834e_release()
    man_sha = rel / 'SIGNED-MANIFEST.sha256'
    if man_sha.is_file():
        put(P, TW + '/SIGNED-MANIFEST.sha256', man_sha.read_bytes(), '644')
        P.run(
            'set -e; cd %s && ln -sfn host-v834e trainer-host-v834e && '
            'sha256sum --quiet -c SIGNED-MANIFEST.sha256 && echo MANIFEST_OK' % TW,
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


def _train_pit_inner_cmd(*, expected_master: str) -> str:
    """Foreground train body (venv python + mem monitor). Used by sync and detached paths.

    ``rc=$?`` is tee'd onto train.log so a detached job can parse success without
    relying on the host SSH stdout capture.
    """
    return (
        "cd {S} && rm -rf {O} && "
        "(nohup nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits -lms 2000 > {P}/mem.csv 2>&1 & echo $! > {P}/mem.pid); "
        "env -u AFFINE_FP32_BOOTSTRAP_OVERRIDE AFFINE_FP32_EXPECTED_MASTER_HASH={X} "
        "CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 "
        "PYTHONPATH={S} TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "{T}/runtime-exact/venv/bin/python -B {P}/train_pit.py > {P}/train.log 2>&1; "
        "echo rc=$? | tee -a {P}/train.log; "
        "kill $(cat {P}/mem.pid) 2>/dev/null; "
        "grep -E '^\\{{' {P}/train.log | tail -40; "
        "grep -v helper-preload {P}/train.log | grep -v '^\\{{' | tail -60"
    ).format(S=BOX_SOURCE, O=PST + '/out', T=TW, P=PST, X=expected_master)


def _parse_train_pit_output(out: str) -> Mapping[str, Any]:
    events = {}
    for line in (out or '').splitlines():
        try:
            x = json.loads(line)
        except Exception:
            continue
        if isinstance(x, dict) and x.get('event'):
            events[x['event']] = x
    if 'rc=0' not in (out or '') or 'done' not in events:
        raise BoxIOError('train failed: ' + (out or '')[-1500:])
    return dict(events=events, checkpoint=events['done']['checkpoint'], log_tail=(out or '')[-4000:])


def run_train_pit(P: Any, *, expected_master: str) -> Mapping[str, Any]:
    """drive_item.train — ship train_pit.py and run under fetched venv (fail closed).

    Prefer ``run_train_pit_detached`` from HostBoxIO so host restart cannot cut
    the train process mid-flight. This sync helper remains for unit tests.
    """
    # Fail closed on venv before launch.
    P.run('set -e; test -x %s; test -d %s' % (BOX_VENV, BOX_SOURCE), timeout=60)
    src = _train_pit_src()
    if not src.is_file():
        raise BoxIOError('train_pit.py missing: %s' % src)
    put(P, PST + '/train_pit.py', src.read_bytes(), '700')
    out = P.run(
        _train_pit_inner_cmd(expected_master=expected_master),
        timeout=int(os.environ.get('PIT_TRAIN_TIMEOUT_S') or 7200),
        check=False,
    )
    return _parse_train_pit_output(out)


def _ship_train_pit(P: Any) -> None:
    P.run('set -e; test -x %s; test -d %s' % (BOX_VENV, BOX_SOURCE), timeout=60)
    src = _train_pit_src()
    if not src.is_file():
        raise BoxIOError('train_pit.py missing: %s' % src)
    put(P, PST + '/train_pit.py', src.read_bytes(), '700')


def detached_train_command(*, expected_master: str, job_dir: str) -> str:
    """Box-side shell: run train under the job dir and write markers/train.done."""
    # Marker writer reads train.log after the inner command; durable across host HUP.
    inner = _train_pit_inner_cmd(expected_master=expected_master)
    # Escape for embedding inside bash -c '...'.
    return (
        "mkdir -p {JD}/markers; "
        "({INNER}); "
        "python3 - {JD}/markers/train.done {P}/train.log <<'PY'\n"
        "import json,sys\n"
        "marker, logp = sys.argv[1], sys.argv[2]\n"
        "text=open(logp,errors='replace').read() if __import__('os').path.isfile(logp) else ''\n"
        "events={{}}\n"
        "for line in text.splitlines():\n"
        "    line=line.strip()\n"
        "    if not line.startswith('{{'):\n"
        "        continue\n"
        "    try: x=json.loads(line)\n"
        "    except Exception: continue\n"
        "    if isinstance(x,dict) and x.get('event'): events[x['event']]=x\n"
        "rc_ok=('rc=0' in text.splitlines()[-5:]) or ('rc=0' in text)\n"
        "done=events.get('done') or {{}}\n"
        "ck=done.get('checkpoint')\n"
        "doc={{'ok': bool(rc_ok and ck), 'checkpoint': ck, 'events': "
        "{{k: events[k] for k in events}}, 'rc_ok': bool(rc_ok)}}\n"
        "open(marker,'w').write(json.dumps(doc,sort_keys=True)+'\\n')\n"
        "sys.exit(0 if doc['ok'] else 1)\n"
        "PY"
    ).format(JD=job_dir.rstrip('/'), INNER=inner, P=PST)


def place_cand(P: Any, *, candidate_id: str) -> Mapping[str, Any]:
    """drive_item.place_cand — keep the ~18.8 GB candidate ON THE BOX (no scp)."""
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
    return dict(place_tail=out.strip()[-200:], box_candidate_dir=KDIR + '/' + candidate_id)


def free_gpus(P: Any, *, also_vllm: bool = True) -> Mapping[str, Any]:
    """drive_item stage 6 — kill leftover train (and optionally vLLM); require free GPUs.

    When re-running eval after a host restart, ``also_vllm=True`` clears the
    previous host's detached serve processes by recorded pgrep patterns before
    the busy check. Callers that prefer resume must not invoke this while an
    eval job pid is still alive.
    """
    # Keep the historical combined train pattern (tests + live drive_item parity).
    patterns = ["train_pit.py|v83_dp.py"]
    if also_vllm:
        patterns.extend(['vllm serve', 'podctl.py serve', 'pit-box-job-eval'])
    pids: list[str] = []
    for pat in patterns:
        pids_raw = P.run("pgrep -f '%s' || true" % pat.replace("'", ""), check=False, timeout=60)
        pids.extend(p for p in str(pids_raw).split() if p.isdigit())
    # Unique preserve order.
    seen: set[str] = set()
    uniq: list[str] = []
    for p in pids:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    for p in uniq:
        # Kill process group when possible so setsid children die with the leader.
        P.run('kill -TERM -%s 2>/dev/null || kill -TERM %s 2>/dev/null || true' % (p, p), check=False, timeout=30)
    time.sleep(1 if os.environ.get('PIT_FAST_FREE_GPUS') else 3)
    for p in uniq:
        P.run('kill -KILL -%s 2>/dev/null || kill -KILL %s 2>/dev/null || true' % (p, p), check=False, timeout=30)
    time.sleep(2 if os.environ.get('PIT_FAST_FREE_GPUS') else 10)
    used_raw = P.run(
        'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits',
        timeout=60,
    )
    used = [u for u in str(used_raw).split() if u.strip().isdigit() or u.strip().replace('.', '', 1).isdigit()]
    used_mib = [int(float(u)) for u in used]
    if not used_mib:
        raise BoxIOError('free_gpus: nvidia-smi returned no memory.used rows')
    if any(u >= 2000 for u in used_mib):
        # Soft, typed error — controller must NOT release the box on re-adopt.
        raise BoxIOError('GPUs not free after train: %s MiB' % used_mib)
    return dict(killed=uniq, used_mib=used_mib)


def detached_eval_command(
    *,
    job_dir: str,
    candidate_id: str,
    plan_path: str,
    podd: str = '/root/affine-env-probe',
    max_num_seqs: int = 512,
) -> str:
    """Box-side setsid driver: serve → serve_wait → verify_serve_flags → probe.

    Mirrors live ``pe_lane.serve`` / ``serve_wait`` / ``verify_serve_flags`` /
    ``run_probe`` stall rules (finalize-run after 20 min no progress or 90 min
    wall). Always writes ``markers/eval.done`` with status
    ``ok`` / ``finalized`` / ``failed`` so the host never waits out a 2h timeout.
    Plan JSON at ``plan_path`` must be built by ``pe_lane.build_probe_plan``.
    """
    jd = job_dir.rstrip('/')
    # Use replace (not str.format) so the embedded Python braces stay intact.
    # Safe pre-eval cleanup: NEVER pgrep for literal 'vllm serve' etc. inside this
    # setsid/nohup bash -c script — those strings appear in our own cmdline and
    # the wrapper would TERM itself (e1067). Kill only: prior job.pid / its
    # process group, GPU compute-app PIDs, and bracketed-pgrep leftovers, always
    # excluding $$, $PPID, and our process group.
    # After cleanup, poll nvidia-smi until GPUs are free (≤5 min) before serve.
    template = r'''mkdir -p __JD__/markers
SELF=$$; PARENT=$PPID; MYPGID=$(ps -o pgid= -p $$ 2>/dev/null | tr -d " " || true)
_kill_one() {
  p="$1"; [ -z "$p" ] && return 0
  case "$p" in (*[!0-9]*|"") return 0 ;; esac
  [ "$p" = "$SELF" ] && return 0
  [ "$p" = "$PARENT" ] && return 0
  [ -n "$MYPGID" ] && [ "$p" = "$MYPGID" ] && return 0
  pg=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d " " || true)
  [ -n "$MYPGID" ] && [ -n "$pg" ] && [ "$pg" = "$MYPGID" ] && return 0
  kill -TERM "$p" 2>/dev/null || true
}
# Prior stage pidfile / process group (train job leader).
if [ -f __JD__/job.pid ]; then
  opid=$(head -1 __JD__/job.pid 2>/dev/null | tr -d " ")
  if [ -n "$opid" ]; then
    _kill_one "$opid"
    # Negative PGID kill only when opid is a session leader we own — still exclude self.
    if [ "$opid" != "$SELF" ] && [ "$opid" != "$PARENT" ]; then
      kill -TERM -"$opid" 2>/dev/null || true
    fi
  fi
fi
# Processes currently holding GPU compute contexts.
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | while read -r raw; do
  p=$(echo "$raw" | tr -d " ,")
  _kill_one "$p"
done
# Bracketed pgrep: matches real leftovers, not this script's own argv text.
pgrep -f "[t]rain_pit\\.py|[v]83_dp\\.py|[v]llm serve|[p]odctl\\.py serve" 2>/dev/null | while read -r p; do
  _kill_one "$p"
done
python3 - __JD__ __CID__ __PLAN__ __PODD__ __MS__ <<'PY'
import json, os, subprocess, sys, time
jd, cid, plan_path, podd, ms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])

def sh(cmd, check=True, timeout=900):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError('rc=%s cmd=%s stderr=%s' % (p.returncode, cmd[:120], (p.stderr or '')[-300:]))
    return (p.stdout or '')

def write_done(status, **extra):
    doc = dict(ok=(status == 'ok'), status=status, candidate_id=cid, at=time.time(), **extra)
    open(os.path.join(jd, 'markers', 'eval.done'), 'w').write(json.dumps(doc, sort_keys=True) + '\n')
    open(os.path.join(jd, 'RESULT.json'), 'w').write(json.dumps(doc, sort_keys=True) + '\n')
    return doc

def wait_gpus_free(timeout_s=300, thresh_mib=2000, poll_s=5):
    """Poll nvidia-smi until all GPUs report memory.used < thresh (or timeout)."""
    t0 = time.time()
    used = []
    while True:
        raw = sh(
            'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits',
            check=False, timeout=60,
        )
        used = []
        for tok in (raw or '').replace(',', ' ').split():
            try:
                used.append(int(float(tok)))
            except ValueError:
                continue
        if used and all(u < thresh_mib for u in used):
            return used
        if time.time() - t0 >= timeout_s:
            raise RuntimeError(
                'GPUs not free after %.0fs before serve: %s MiB' % (time.time() - t0, used)
            )
        time.sleep(poll_s)

try:
    wait_gpus_free()
    plan = json.load(open(plan_path))
    name = plan['run_name']
    # Live pe_lane.serve
    sh('MAX_SEQS=%d python3 %s/podctl.py serve %s 0.9' % (ms, podd, cid), timeout=900)
    # Live pe_lane.serve_wait: all replicas up + OpenAI models, stable 2 polls.
    t0 = time.time(); ready = 0
    while time.time() - t0 < 900:
        st_raw = sh('python3 %s/podctl.py status' % podd, timeout=120).strip().splitlines()[-1]
        st = json.loads(st_raw)
        up = int(st.get('replicas_up') or st.get('serve_procs') or 0)
        need = int(st.get('replicas') or 8)
        models = sh('curl -s -m 3 http://127.0.0.1:18555/v1/models 2>/dev/null | head -c 200 || true', check=False, timeout=15)
        ok_m = ('data' in (models or '')) or ('model' in (models or '').lower())
        served = st.get('served') or st.get('serve_ckpt')
        if served and served != cid:
            raise RuntimeError('pod serves %s, expected %s' % (str(served)[:12], cid[:12]))
        if up >= need and ok_m:
            ready += 1
            if ready >= 2:
                break
        else:
            ready = 0
        time.sleep(5)
    else:
        raise RuntimeError('serve timeout after %.0fs' % (time.time() - t0))
    # Live pe_lane.verify_serve_flags (fail closed).
    out = sh("ps -eo args | grep -F 'vllm serve' | grep -v grep || true", check=False, timeout=60)
    procs = [l for l in (out or '').splitlines() if ' serve ' in l and '/vllm' in l]
    bad = [l[:200] for l in procs if ('/ckpt/%s' % cid) not in l or '--dtype bfloat16' not in l
           or '--max-model-len 69632' not in l or '--max-num-seqs 512' not in l or 'fp8' in l]
    if len(procs) < need or bad:
        raise RuntimeError('serve parity failed: procs=%d need=%d bad=%s' % (len(procs), need, bad[:1]))
    # Live pe_lane.run_probe start-run + stall rules.
    sh('test ! -f %s/live_math_manifest_subset.json' % podd)
    p = subprocess.run(['python3', '%s/podctl.py' % podd, 'start-run', cid],
                      input=json.dumps(plan).encode(), capture_output=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError('start-run failed: %s' % ((p.stderr or b'')[-300:].decode('utf-8', 'replace')))
    t0 = time.time(); last_p = None; last_t = time.time(); status = 'ok'
    while True:
        st = json.loads(sh('python3 %s/podctl.py status' % podd, timeout=120).strip().splitlines()[-1])
        ru = (st.get('runs') or {}).get(name, {})
        if ru.get('result'):
            break
        pg = ru.get('progress') or {}
        ph = (pg.get('generated'), pg.get('samples'), pg.get('done'))
        if ph != last_p:
            last_p, last_t = ph, time.time()
        if time.time() - last_t > 1200:
            sh('python3 %s/podctl.py finalize-run %s 60' % (podd, name), check=False, timeout=180)
            time.sleep(20); status = 'finalized'; break
        if time.time() - t0 > 5400:
            sh('python3 %s/podctl.py finalize-run %s 60' % (podd, name), check=False, timeout=180)
            status = 'finalized'; break
        time.sleep(10)
    res = json.load(open('%s/runs/%s/result.json' % (podd, name)))
    write_done(status, probe_result=res, run_name=name)
    sys.exit(0)
except Exception as ex:
    write_done('failed', error=str(ex)[:500])
    sys.exit(1)
PY'''
    return (
        template
        .replace('__JD__', jd)
        .replace('__CID__', candidate_id)
        .replace('__PLAN__', plan_path)
        .replace('__PODD__', podd)
        .replace('__MS__', str(int(max_num_seqs)))
    )


def pull_report_only(P: Any, *, candidate_id: str, host_out: Path) -> Mapping[str, Any]:
    """Scp only the small report + descriptor — never the full ~18.8 GB candidate."""
    host_out.mkdir(parents=True, exist_ok=True)
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

    remote_desc = KDIR + '/' + candidate_id + '.descriptor.json'
    alt_desc = PST + '/out/descriptor.json'
    marker = P.run('test -f %s && echo YES || echo NO' % remote_desc, timeout=30, check=False)
    if 'YES' not in str(marker):
        remote_desc = alt_desc
    _scp_from(P, remote_desc, host_out / 'descriptor.json', timeout=120)
    desc = json.loads((host_out / 'descriptor.json').read_text())
    cid = str(desc.get('id') or candidate_id)
    if len(cid) != 64:
        raise BoxIOError('descriptor.id missing')

    fp = ((rep.get('optimizer') or {}).get('sgd') or {}).get('fp32_master') or rep.get('fp32') or {}
    master_dig = str(
        fp.get('output_master_digest')
        or rep.get('candidate_master_digest')
        or rep.get('master_digest')
        or ''
    )
    master_out = host_out / 'fp32-master'
    master_out.mkdir(parents=True, exist_ok=True)
    if len(master_dig) != 64:
        raise BoxIOError('candidate master digest missing from training report')
    (master_out / 'master_digest.txt').write_text(master_dig + '\n')
    # Locate the on-box store entry (digest-generation or digest-only). Host
    # upload_master_from_box pulls every shard from here — never digest-only.
    store_entry = str(
        fp.get('output_store_entry')
        or rep.get('output_store_entry')
        or ''
    )
    # Best-effort locate; GREEN upload_master_from_box fails closed if missing.
    try:
        box_master_dir = _resolve_box_master_dir(
            P, master_digest=master_dig, store_entry=store_entry,
        )
    except BoxIOError:
        box_master_dir = None

    # Box-side path marker — host must NOT treat this as a local tree to scp.
    box_path = 'box:%s/%s' % (KDIR, cid)
    return dict(
        candidate_id=cid,
        candidate_path=box_path,
        candidate_on_box=True,
        box_candidate_dir=KDIR + '/' + cid,
        box_master_dir=box_master_dir,
        candidate_files=dict(desc.get('files') or {}),
        candidate_master_path=str(master_out),
        candidate_master_digest=master_dig,
        report=str(report),
        report_doc=rep,
        descriptor=desc,
        defer_candidate_upload=True,
    )


def _resolve_box_master_dir(
    P: Any,
    *,
    master_digest: str,
    store_entry: str = '',
) -> str:
    """Return the on-box FP32 store directory for this candidate master."""
    candidates: list[str] = []
    if store_entry and '/' not in store_entry:
        candidates.append('%s/%s' % (FP32, store_entry))
    if len(master_digest) == 64:
        # Prefer exact digest-generation entries; fall back to digest prefix match.
        listing = P.run(
            'ls -1 %s 2>/dev/null || true' % FP32,
            timeout=60, check=False,
        )
        for name in str(listing or '').splitlines():
            name = name.strip()
            if not name:
                continue
            if name == master_digest or name.startswith(master_digest + '-'):
                candidates.append('%s/%s' % (FP32, name))
    for path in candidates:
        marker = P.run(
            'test -d %s && ls %s/*.safetensors >/dev/null 2>&1 && echo YES || echo NO'
            % (path, path),
            timeout=60, check=False,
        )
        if 'YES' in str(marker):
            return path
    raise BoxIOError(
        'fp32 master shards missing on box for digest %s (store_entry=%r)'
        % (master_digest[:12], store_entry[:20])
    )


def place_and_pull(P: Any, *, candidate_id: str, host_out: Path) -> Mapping[str, Any]:
    """place_cand + free_gpus + report-only pull (no full candidate scp)."""
    placed = place_cand(P, candidate_id=candidate_id)
    gpus = free_gpus(P)
    pulled = pull_report_only(P, candidate_id=candidate_id, host_out=host_out)
    pulled['place_tail'] = placed.get('place_tail')
    pulled['gpus_after_train'] = gpus
    return pulled


def _r2_local_mirror() -> Optional[Path]:
    return _env_path('PIT_R2_LOCAL_MIRROR')


def _ckpt_r2_prefix(candidate_id: str) -> str:
    prefix = os.environ.get('PIT_R2_CKPT_PREFIX') or (
        'mimo/affine-mimo9b-grpo-20261013/checkpoints'
    )
    return '%s/%s/model' % (prefix.rstrip('/'), candidate_id)


def presign_put_urls(candidate_id: str, files: Mapping[str, str]) -> Mapping[str, str]:
    """Scoped presigned PUT URLs for GREEN candidate upload (box → R2)."""
    inj = _env_path('PIT_PRESIGN_PUT_JSON')
    if inj is not None and inj.is_file():
        doc = json.loads(inj.read_text())
        urls = doc.get('urls') if isinstance(doc, Mapping) else None
        if isinstance(urls, Mapping):
            return {n: str(urls[n]) for n in files if n in urls}
    mirror = _r2_local_mirror()
    if mirror is not None:
        # Test / dry-run: file:// URLs under a local R2 mirror (host verifies sha from disk).
        out: dict[str, str] = {}
        base = mirror / candidate_id
        base.mkdir(parents=True, exist_ok=True)
        for name in files:
            dest = base / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            out[name] = 'file://' + str(dest)
        return out
    # Live: Bucket.presign put_object via host helper / storage.
    # Live: Bucket.presign put_object (requires PIT_LIFE_CFG with bucket credentials).
    try:
        from subnet.storage import Bucket  # type: ignore
        life_cfg = _env_path('PIT_LIFE_CFG', 'AFFINE_LIFE_CFG')
        if life_cfg is None or not life_cfg.is_file():
            raise BoxIOError(
                'PIT_LIFE_CFG required for live R2 put_object presign '
                '(or set PIT_PRESIGN_PUT_JSON / PIT_R2_LOCAL_MIRROR)'
            )
        bucket = Bucket(json.loads(life_cfg.read_text())['bucket'])
        prefix = _ckpt_r2_prefix(candidate_id)
        return {
            n: bucket.presign('%s/%s' % (prefix, n), 'put_object', 7200)
            for n in sorted(files)
        }
    except BoxIOError:
        raise
    except Exception as exc:
        raise BoxIOError('presign put_object failed: %s' % str(exc)[:300]) from exc


def upload_candidate_from_box(
    P: Any,
    *,
    candidate_id: str,
    files: Mapping[str, str],
    box_candidate_dir: Optional[str] = None,
) -> Mapping[str, Any]:
    """GREEN only: box PUTs files to scoped R2 URLs; host verifies sha256 from R2/mirror."""
    if not isinstance(candidate_id, str) or len(candidate_id) != 64:
        raise BoxIOError('candidate_id must be hex64')
    if not files:
        raise BoxIOError('candidate files missing for R2 upload')
    box_dir = box_candidate_dir or (KDIR + '/' + candidate_id)
    urls = presign_put_urls(candidate_id, files)
    missing = [n for n in files if n not in urls]
    if missing:
        raise BoxIOError('presign PUT missing urls for %s' % missing[:5])

    # Push URLs to the box (private, deleted after).
    url_doc = json.dumps(dict(urls=urls, headers={'Content-Type': 'application/octet-stream'}))
    remote_urls = '/tmp/pit-put-%s.urls.private.json' % candidate_id[:12]
    put(P, remote_urls, url_doc.encode(), '600')
    # Box-side PUT (curl for https; cp for file:// mirror).
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
        req = urllib.request.Request(url, data=open(src, 'rb').read(), method='PUT')
        for k, v in headers.items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=3600) as resp:
            if getattr(resp, 'status', 200) not in (200, 201):
                bad.append(('http', name, getattr(resp, 'status', None)))
print('PUT_OK', len(urls), 'BAD', len(bad))
if bad:
    sys.exit(1)
PY
rm -f %s
""".strip() % (remote_urls, box_dir, remote_urls)
    out = P.run(script, timeout=7200)
    if 'PUT_OK' not in out:
        raise BoxIOError('box R2 PUT failed: %s' % out[-400:])

    # Host verifies sha256 from R2 / local mirror (never trust the box alone).
    verified: dict[str, str] = {}
    mirror = _r2_local_mirror()
    for name, expected in files.items():
        if mirror is not None:
            path = mirror / candidate_id / name
            if not path.is_file():
                raise BoxIOError('R2 mirror missing after PUT: %s' % name)
            got = _sha256_file(path)
        else:
            # Live: GET the object via a short-lived GET presign and hash.
            get_urls = None
            try:
                get_urls = ckpt_urls(candidate_id, {name: expected})
            except Exception:
                get_urls = None
            if not get_urls or name not in get_urls:
                raise BoxIOError('cannot GET %s from R2 for sha verify' % name)
            data = urllib.request.urlopen(
                urllib.request.Request(
                    get_urls[name],
                    headers={'User-Agent': 'affine-pit-pe-box/1'},
                ),
                timeout=3600,
            ).read()
            got = hashlib.sha256(data).hexdigest()
        if got != expected:
            raise BoxIOError('R2 sha mismatch for %s: got %s want %s' % (name, got[:12], expected[:12]))
        verified[name] = got

    # Local receipt so pe_verify can confirm without re-fetching 18 GB.
    dst = _upload_root() / candidate_id
    dst.mkdir(parents=True, exist_ok=True)
    receipt = {
        'id': candidate_id,
        'sha256': candidate_id,
        'files': dict(files),
        'verified_files': verified,
        'uploaded_at': time.time(),
        'source': 'pit-box-r2-put',
        'transport': 'presigned-put',
    }
    (dst / 'UPLOAD.json').write_text(json.dumps(receipt, indent=1, sort_keys=True) + '\n')
    (dst / 'CHECKPOINT.json').write_text(json.dumps({
        'id': candidate_id, 'files': dict(files),
    }, indent=1, sort_keys=True) + '\n')
    os.chmod(dst / 'UPLOAD.json', 0o600)
    return dict(ok=True, path=str(dst), verified=len(verified), transport='presigned-put')


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

        # Restart-safe box job: stage markers + RESULT on the box. Host threads
        # die on restart; detached train does not. Re-adopt resumes from markers.
        from ops.parallel_eval.pit.box_job import (
            BoxJobError,
            controller_for_pod,
            marker_path,
        )

        ctrl = controller_for_pod(P, str(item_id))
        ctrl.ensure_dir()
        prior = ctrl.resume_or_fetch()
        if prior and prior.get('candidate_id') and prior.get('candidate_path'):
            # Full RESULT already on box (train+probe finished earlier).
            out = dict(prior)
            out.setdefault('pod', pod)
            out.setdefault('pod_id', getattr(P, 'id', None))
            return out

        poll_s = float(os.environ.get('PIT_BOX_JOB_POLL_S') or 2.0)
        train_timeout = float(os.environ.get('PIT_TRAIN_TIMEOUT_S') or 7200)

        def _do_download() -> dict:
            # drive_item order: boot (+ GRADER_FIX) → push_bundle → bundle_trainer →
            # head → master → inputs (no wipe).
            boot_helpers(P)
            push_probe_bundle(P)
            info.update(bundle_trainer(P))
            info['fetch-head'] = fetch_head_on_box(P, head=head, files=files)
            info['fetch-master'] = fetch_master_on_box(
                P, head=head, expected_master=str(expected),
                head_bf16=str(head_bf16), published=published,
            )
            if info['fetch-master'].get('skipped'):
                raise BoxIOError('no published master -> fail closed')
            with tempfile.TemporaryDirectory(prefix='pit-jd-') as td:
                info['signed_inputs'] = ship_inputs(
                    P, epoch_dir=epoch_dir, epoch=epoch, work_jd=Path(td),
                )
            return dict(ok=True, head=head, epoch=epoch)

        try:
            ctrl.run_stage_once('download', _do_download, poll_s=poll_s, timeout_s=train_timeout)
        except BoxJobError as exc:
            raise BoxIOError('box download stage: %s' % exc) from exc

        # Train: resume polling if already running; else kill stale + detach.
        st = ctrl.status()
        train_marker: dict = {}
        if 'train' in st.markers:
            raw = ctrl.store.read_text(marker_path(ctrl.job_dir, 'train'))
            try:
                train_marker = json.loads(raw) if raw else {}
            except Exception:
                train_marker = {}
        else:
            _ship_train_pit(P)
            cmd = detached_train_command(expected_master=str(expected), job_dir=ctrl.job_dir)

            def _train_host_forbidden() -> dict:
                raise BoxIOError('host must not run sync train when detached path is armed')

            try:
                train_marker = ctrl.run_stage_once(
                    'train', _train_host_forbidden,
                    detached_command=cmd, wait_for_marker=True,
                    poll_s=poll_s, timeout_s=train_timeout,
                )
            except BoxJobError as exc:
                raise BoxIOError('box train stage: %s' % exc) from exc

        if not train_marker.get('ok') and not train_marker.get('checkpoint'):
            # Marker without checkpoint — fail closed (box wrote a failed marker).
            raise BoxIOError('train marker missing checkpoint: %s' % str(train_marker)[:300])
        candidate_id = train_marker.get('checkpoint')
        if len(str(candidate_id)) != 64:
            raise BoxIOError('train done.checkpoint missing')
        info['train-events'] = train_marker.get('events') or {}

        # If a prior host pass already finalized place/pull into the marker, reuse it.
        if train_marker.get('candidate_path') and train_marker.get('report'):
            pulled = dict(train_marker)
            rep = pulled.get('report_doc') or {}
            fp = ((rep.get('optimizer') or {}).get('sgd') or {}).get('fp32_master') or (rep.get('fp32') or {})
            train_gpu = kwargs.get('train_gpu') or pulled.get('train_gpu') or _detect_box_gpu(P)
            result = dict(
                candidate_id=pulled['candidate_id'],
                candidate_path=pulled['candidate_path'],
                candidate_on_box=True,
                box_candidate_dir=pulled.get('box_candidate_dir'),
                box_master_dir=pulled.get('box_master_dir'),
                candidate_files=dict(pulled.get('candidate_files') or {}),
                candidate_master_path=pulled['candidate_master_path'],
                candidate_master_digest=pulled['candidate_master_digest'],
                candidate_sha256=pulled['candidate_id'],
                defer_candidate_upload=True,
                used_fp32_bootstrap_override=False,
                fp32_master_source=fp.get('source') or 'disk',
                report=pulled['report'],
                report_doc=rep,
                training_report=rep,
                train_gpu=train_gpu,
                pod_id=getattr(P, 'id', None),
                pod=pod,
                box_work=PST,
                fetch='r2-box-side',
                artifact_transport='on-box-until-green',
                info=info,
                venv=BOX_VENV,
            )
            return result

        # drive_item: place_cand → free_gpus → keep ~18.8 GB on box until GREEN.
        host_out = _upload_root() / ('train-%s-%d' % (str(item_id)[:16], int(time.time())))
        pulled = place_and_pull(P, candidate_id=str(candidate_id), host_out=host_out)
        info['gpus-after-train'] = pulled.get('gpus_after_train')
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

        result = dict(
            candidate_id=pulled['candidate_id'],
            candidate_path=pulled['candidate_path'],
            candidate_on_box=True,
            box_candidate_dir=pulled.get('box_candidate_dir'),
            box_master_dir=pulled.get('box_master_dir'),
            candidate_files=dict(pulled.get('candidate_files') or {}),
            candidate_master_path=pulled['candidate_master_path'],
            candidate_master_digest=pulled['candidate_master_digest'],
            candidate_sha256=pulled['candidate_id'],
            defer_candidate_upload=True,
            used_fp32_bootstrap_override=False,
            fp32_master_source=fp.get('source') or 'disk',
            report=pulled['report'],
            report_doc=rep,
            training_report=rep,
            train_gpu=train_gpu,
            pod_id=getattr(P, 'id', None),
            pod=pod,
            box_work=PST,
            fetch='r2-box-side',
            artifact_transport='on-box-until-green',
            info=info,
            venv=BOX_VENV,
            # Keep box-side fields used by marker / RESULT resume.
            ok=True,
            checkpoint=str(candidate_id),
            events=info.get('train-events') or {},
        )
        # Enrich train marker so a host restart after place/pull does not redo train.
        try:
            ctrl.mark_stage('train', result)
        except Exception:
            pass
        return result


    def upload_candidate(
        self,
        *,
        candidate_id: str,
        candidate_path: str = '',
        path: str = '',
        pod: Any = None,
        files: Optional[Mapping[str, str]] = None,
        box_candidate_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """GREEN path: box → R2 presigned PUT. RED must not call this.

        ``candidate_path`` / ``path`` may be a ``box:…`` marker (deferred upload).
        Local host trees are still accepted for Scripted / fixture paths.
        """
        src = str(candidate_path or path or '')
        file_map = dict(files or kwargs.get('candidate_files') or {})
        on_box = (
            src.startswith('box:')
            or bool(kwargs.get('candidate_on_box'))
            or bool(kwargs.get('defer_candidate_upload'))
        )
        if on_box:
            P = _as_pod(pod or kwargs.get('pod'))
            box_dir = box_candidate_dir or kwargs.get('box_candidate_dir')
            if not box_dir and src.startswith('box:'):
                box_dir = src[4:]
            if not file_map:
                raise BoxIOError('GREEN R2 upload requires candidate file digests')
            return upload_candidate_from_box(
                P,
                candidate_id=candidate_id,
                files=file_map,
                box_candidate_dir=box_dir,
            )
        # Local host path (tests / already-fetched).
        local = Path(src)
        if not local.exists():
            raise BoxIOError('candidate_path missing for upload: %s' % src)
        dst = _upload_root() / candidate_id
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        if local.is_dir():
            shutil.copytree(local, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, dst / local.name)
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


def destroy_box_state(*, box_id: str, reason: str = 'done', **_ignored) -> Mapping[str, Any]:
    # Accept/ignore extra kwargs (e.g. candidate_id from apply_live) so PIT
    # destroy calls never TypeError on the live pe_box signature.
    return dict(ok=True, box_id=box_id, reason=reason)


def _publish_fp32_master_objects(
    candidate_id: str,
    local_dir: Path,
    files_meta: Mapping[str, Mapping[str, str]],
) -> None:
    """Upload fp32 master shards to R2 (or local mirror) under r2_key paths."""
    mirror = _r2_local_mirror()
    if mirror is not None:
        for name, meta in files_meta.items():
            src = local_dir / name
            if not src.is_file():
                raise BoxIOError('master shard missing for publish: %s' % name)
            dest = mirror / str(meta['r2_key'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        return
    try:
        from subnet.storage import Bucket  # type: ignore
        life_cfg = _env_path('PIT_LIFE_CFG', 'AFFINE_LIFE_CFG')
        if life_cfg is None or not life_cfg.is_file():
            raise BoxIOError(
                'PIT_LIFE_CFG required to publish fp32 master to R2 '
                '(or set PIT_R2_LOCAL_MIRROR for hermetic tests)'
            )
        bucket = Bucket(json.loads(life_cfg.read_text())['bucket'])
        for name, meta in files_meta.items():
            src = local_dir / name
            if not src.is_file():
                raise BoxIOError('master shard missing for publish: %s' % name)
            url = bucket.presign(str(meta['r2_key']), 'put_object', 7200)
            data = src.read_bytes()
            req = urllib.request.Request(url, data=data, method='PUT')
            req.add_header('Content-Type', 'application/octet-stream')
            with urllib.request.urlopen(req, timeout=3600) as resp:
                if getattr(resp, 'status', 200) not in (200, 201):
                    raise BoxIOError('master R2 PUT failed for %s' % name)
    except BoxIOError:
        raise
    except Exception as exc:
        raise BoxIOError('fp32 master R2 publish failed: %s' % str(exc)[:300]) from exc


def _master_files_meta_from_dir(local_dir: Path, *, candidate_id: str) -> dict[str, dict[str, Any]]:
    files_meta: dict[str, dict[str, Any]] = {}
    for f in sorted(p for p in local_dir.rglob('*') if p.is_file()):
        rel = f.relative_to(local_dir).as_posix()
        files_meta[rel] = {
            'sha256': _sha256_file(f),
            'bytes': int(f.stat().st_size),
            'r2_key': 'fp32-master/%s/%s' % (candidate_id, rel),
        }
    return files_meta


def _assert_master_tree_complete(files_meta: Mapping[str, Any]) -> None:
    """Fail closed: require real weight shards + meta.json, not digest-only."""
    if not files_meta:
        raise BoxIOError('fp32 master publish refused: empty master tree')
    shards = [
        n for n in files_meta
        if str(n).endswith('.safetensors') or str(n).endswith('.bin')
    ]
    if not shards:
        raise BoxIOError(
            'fp32 master publish refused: no weight shards '
            '(digest-only master_digest.txt is not sufficient)'
        )
    if 'meta.json' not in files_meta:
        raise BoxIOError(
            'fp32 master publish refused: meta.json missing '
            '(trainer _fp32_load needs tensor_file)'
        )


def _meta_json_r2_ptr(
    files_meta: Mapping[str, Mapping[str, Any]], *, candidate_id: str,
) -> dict:
    """Index meta_json must be {r2_key: ...} — never a synthetic body.

    fetch_master_on_box treats a version key as the whole meta and overwrites
    the real uploaded meta.json, dropping tensor_file so _fp32_load fails.
    """
    meta = files_meta.get('meta.json')
    if not isinstance(meta, Mapping):
        raise BoxIOError('files_meta missing meta.json for index pointer')
    key = str(meta.get('r2_key') or ('fp32-master/%s/meta.json' % candidate_id))
    if not key.endswith('meta.json'):
        raise BoxIOError('meta_json r2_key must end with meta.json: %s' % key)
    return {'r2_key': key}


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _trainer_master_meta(
    local_dir: Path,
    files_meta: Mapping[str, Any],
    *,
    master_digest: str = '',
) -> Mapping[str, Any]:
    """The trainer's own meta.json of a staged master, checked for the next load.

    ``bf16_sha256`` is the FP32 store entry the trainer looks the master up by
    (``FP32/<bf16_sha256>/meta.json``); it is not the checkpoint id. The
    host-recorded digest ties this entry to the candidate that was trained. Fail
    closed here, while the candidate is still on the box, on anything
    ``_fp32_load`` would reject.
    """
    try:
        meta = json.loads((Path(local_dir) / 'meta.json').read_text())
    except Exception as exc:
        raise BoxIOError('fp32 master meta.json unreadable: %s' % str(exc)[:200]) from exc
    if not isinstance(meta, Mapping):
        raise BoxIOError('fp32 master meta.json is not an object')
    if not _is_hex64(meta.get('bf16_sha256')) or not _is_hex64(meta.get('master_digest')):
        raise BoxIOError('fp32 master meta.json lacks bf16_sha256 / master_digest')
    tensor_file = meta.get('tensor_file')
    if not isinstance(tensor_file, Mapping) or not tensor_file:
        raise BoxIOError('fp32 master meta.json has no tensor_file (the trainer cannot load it)')
    absent = sorted({str(v) for v in tensor_file.values()} - set(files_meta))
    if absent:
        raise BoxIOError('fp32 master meta.json names shards that are not in the upload: %s' % absent[:3])
    if master_digest and meta['master_digest'] != master_digest:
        raise BoxIOError(
            'fp32 master meta.json digest %s != host-recorded %s'
            % (str(meta['master_digest'])[:12], master_digest[:12])
        )
    return meta


def _write_master_index(
    *,
    candidate_id: str,
    bf16: str,
    digest: str,
    files_meta: Mapping[str, Mapping[str, Any]],
) -> Path:
    _assert_master_tree_complete(files_meta)
    idx_dir = _fp32_master_index()
    idx_dir.mkdir(parents=True, exist_ok=True)
    idx_path = idx_dir / ('%s.json' % candidate_id)
    idx_doc = {
        'bf16_sha256': bf16,
        'master_digest': digest,
        'files': dict(files_meta),
        # Same shape as live HEAD / k1 fp32-master publish.
        'meta_json': _meta_json_r2_ptr(files_meta, candidate_id=candidate_id),
        'source': 'pit',
        'at': time.time(),
    }
    tmp = idx_path.with_name('.%s.%d.tmp' % (idx_path.name, os.getpid()))
    tmp.write_text(json.dumps(idx_doc, indent=1, sort_keys=True) + '\n')
    os.chmod(tmp, 0o600)
    os.replace(tmp, idx_path)
    return idx_path


def upload_master_from_box(
    P: Any,
    *,
    candidate_id: str,
    master_digest: str,
    bf16_sha256: str = '',
    box_master_dir: Optional[str] = None,
    store_entry: str = '',
) -> Mapping[str, Any]:
    """GREEN: box → R2 for every fp32 master shard; write complete index entry."""
    if not isinstance(candidate_id, str) or len(candidate_id) != 64:
        raise BoxIOError('candidate_id must be hex64')
    digest = str(master_digest or '').strip()
    if len(digest) != 64:
        raise BoxIOError('master_digest must be hex64')
    box_dir = box_master_dir or _resolve_box_master_dir(
        P, master_digest=digest, store_entry=store_entry,
    )
    # Inventory shards on the box (name + sha256 + bytes).
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
        "        out[rel.replace(os.sep, '/')] = dict(sha256=h.hexdigest(), bytes=os.path.getsize(path))\n"
        "print(json.dumps(out, sort_keys=True))\n"
        "PY" % box_dir,
        timeout=1800, check=True,
    )
    try:
        inv = json.loads(inv_raw.strip().splitlines()[-1])
    except Exception as exc:
        raise BoxIOError('box master inventory parse failed: %s' % str(exc)[:200]) from exc
    if not isinstance(inv, dict) or not inv:
        raise BoxIOError('box master inventory empty at %s' % box_dir)
    _assert_master_tree_complete(inv)

    # Stage on host via scp (binary-safe) then publish; keeps verify local.
    dst = _upload_root() / candidate_id / 'fp32-master'
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for rel in sorted(inv):
        local = dst / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        _scp_from(P, '%s/%s' % (box_dir.rstrip('/'), rel), local, timeout=3600)
        got = _sha256_file(local)
        if got != inv[rel]['sha256']:
            raise BoxIOError('master shard sha mismatch after scp: %s' % rel)

    files_meta = _master_files_meta_from_dir(dst, candidate_id=candidate_id)
    _assert_master_tree_complete(files_meta)
    trainer_meta = _trainer_master_meta(dst, files_meta, master_digest=digest)
    # The index names the trainer's store entry: the next box lands the master
    # under FP32/<bf16_sha256>/ and the trainer looks it up by that digest. The
    # caller's bf16_sha256 is the checkpoint id and must not be used here.
    bf16 = str(trainer_meta['bf16_sha256'])
    _publish_fp32_master_objects(candidate_id, dst, files_meta)
    idx_path = _write_master_index(
        candidate_id=candidate_id, bf16=bf16, digest=digest, files_meta=files_meta,
    )
    return dict(
        ok=True,
        path=str(dst),
        candidate_id=candidate_id,
        master_digest=digest,
        index=str(idx_path),
        published=True,
        box_master_dir=box_dir,
        shard_count=sum(
            1 for n in files_meta
            if str(n).endswith('.safetensors') or str(n).endswith('.bin')
        ),
        complete=True,
    )


def upload_master(
    *,
    candidate_id: str,
    path: str = '',
    master_digest: str = '',
    bf16_sha256: str = '',
    **kwargs: Any,
) -> Mapping[str, Any]:
    """Stage master, upload to R2, write PIT_FP32_MASTER_INDEX/<cand>.json.

    Under ``defer_candidate_upload`` / ``candidate_on_box``, shards must come
    from the box (``pod`` + ``box_master_dir``). Digest-only trees are refused.
    """
    if not isinstance(candidate_id, str) or len(candidate_id) != 64:
        raise BoxIOError('candidate_id must be hex64')
    pod = kwargs.get('pod')
    on_box = bool(kwargs.get('candidate_on_box') or kwargs.get('defer_candidate_upload'))
    box_master_dir = kwargs.get('box_master_dir')
    if pod is not None and (on_box or box_master_dir):
        return upload_master_from_box(
            _as_pod(pod),
            candidate_id=candidate_id,
            master_digest=str(master_digest or kwargs.get('master_digest') or ''),
            bf16_sha256=str(bf16_sha256 or kwargs.get('bf16_sha256') or candidate_id),
            box_master_dir=str(box_master_dir) if box_master_dir else None,
            store_entry=str(kwargs.get('store_entry') or ''),
        )

    src = Path(path) if path else None
    if src is None or not src.exists():
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

    digest = str(master_digest or kwargs.get('master_digest') or '').strip()
    if not digest or len(digest) != 64:
        meta_p = dst / 'meta.json'
        if meta_p.is_file():
            try:
                digest = str((json.loads(meta_p.read_text()) or {}).get('master_digest') or '')
            except Exception:
                digest = ''
        if not digest or len(digest) != 64:
            h = hashlib.sha256()
            for f in sorted(p for p in dst.rglob('*') if p.is_file()):
                h.update(_sha256_file(f).encode())
            digest = h.hexdigest()

    files_meta = _master_files_meta_from_dir(dst, candidate_id=candidate_id)
    _assert_master_tree_complete(files_meta)
    trainer_meta = _trainer_master_meta(dst, files_meta, master_digest=digest)
    bf16 = str(trainer_meta['bf16_sha256'])
    _publish_fp32_master_objects(candidate_id, dst, files_meta)
    idx_path = _write_master_index(
        candidate_id=candidate_id, bf16=bf16, digest=digest, files_meta=files_meta,
    )
    return dict(
        ok=True,
        path=str(dst),
        candidate_id=candidate_id,
        master_digest=digest,
        index=str(idx_path),
        published=True,
        shard_count=sum(
            1 for n in files_meta
            if str(n).endswith('.safetensors') or str(n).endswith('.bin')
        ),
        complete=True,
    )


def upload_candidate(*, candidate_id: str, path: str = '', **kwargs: Any) -> Mapping[str, Any]:
    """Module-level GREEN upload seam used by ``LiveApplicator``.

    ``apply_live`` may pass both ``path`` and ``candidate_path``; prefer an
    explicit ``candidate_path`` kwarg and never forward it twice.
    """
    cand_path = kwargs.pop('candidate_path', None)
    if cand_path is None or cand_path == '':
        cand_path = path
    return HostBoxIO().upload_candidate(
        candidate_id=candidate_id,
        candidate_path=str(cand_path or ''),
        path=str(path or ''),
        **kwargs,
    )


BoxIO = HostBoxIO


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv or argv[0] in ('-h', '--help'):
        print('usage: pe_box.py train --pod-json POD.json --head H --master-digest M --epoch N')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
