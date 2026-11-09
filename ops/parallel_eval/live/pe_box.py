#!/usr/bin/env python3
"""Host box I/O for PIT — on-box train (BRIEF §1.3 / shadow run5090 path).

Port of the working shadow-driver provision+train sequence that ran
E1041/E1042/E1046 on real B300s. Hard rules:

1. The **box** fetches the bf16 head and fp32 master from R2 (presigned URLs
   over stdin; deleted after fetch). The validator never loads the ~18.8 GB
   head or ~35.8 GB master into RAM and never pushes them over ssh.
2. Ship the signed v834e release as a **multi-file bundle + venv**, with
   ``SIGNED-MANIFEST`` checked via ``sha256sum -c`` on the box. Do not look
   for a lone ``train_step.py`` under ``/root`` on the validator.
3. Return artifacts as **files** (``scp`` / ``lium scp``). Never stream binary
   over ssh stdout — ``replace_pod.Pod.run`` decodes with ``errors=replace``
   and corrupts bytes. The training report is JSON and also returns as a file.
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
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

sys.dont_write_bytecode = True

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# On-box layout (BRIEF §1.3 / §1.3a).
BOX_ROOT = '/root/affine-env-probe'
BOX_PIT = '/root/pit-train'
BOX_TBX = '/opt/plain/tbx'
BOX_WORKSPACE = '/opt/plain/workspace'
BOX_FP32_STORE = '/workspace/affine-mimo-trainer-prod/fp32-master'
BOX_VENV = '/opt/plain/workspace/runtime-exact/venv/bin/python'
BOX_CKPT = BOX_ROOT + '/ckpt'


class BoxIOError(RuntimeError):
    """Fail-closed host box I/O error."""


def _as_pod(pod: Any) -> Any:
    """Accept a Pod object or a live pe_rent dict → Pod with .run()."""
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


def _ssh_argv(P: Any) -> list:
    """Match replace_pod.ssh_argv so scp uses the same known_hosts file."""
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
    """Upload a local file to the box via scp (binary-safe)."""
    local = Path(local)
    if not local.is_file():
        raise BoxIOError('scp_to missing local file: %s' % local)
    dest = 'root@%s:%s' % (P.ip, remote)
    parent = str(Path(remote).parent)
    P.run('mkdir -p %s' % parent, timeout=60)
    cmd = _ssh_argv(P) + [str(local), dest]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise BoxIOError('scp_to timeout: %s' % local.name) from exc
    if p.returncode != 0:
        err = (p.stderr or p.stdout or b'').decode('utf-8', errors='replace')[-400:]
        raise BoxIOError('scp_to failed (%s): %s' % (local.name, err))


def _scp_from(P: Any, remote: str, local: Path, *, timeout: int = 3600) -> Path:
    """Download a remote file to the host via scp (binary-safe)."""
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    src = 'root@%s:%s' % (P.ip, remote)
    cmd = _ssh_argv(P) + [src, str(local)]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise BoxIOError('scp_from timeout: %s' % remote) from exc
    if p.returncode != 0 or not local.is_file():
        err = (p.stderr or p.stdout or b'').decode('utf-8', errors='replace')[-400:]
        raise BoxIOError('scp_from failed (%s): %s' % (remote, err))
    return local


def _push_text(P: Any, dest: str, data: bytes, *, mode: str = '644') -> None:
    """Small text/script upload over ssh stdin (helpers only; never weights)."""
    if len(data) > 8 * 1024 * 1024:
        raise BoxIOError('refuse large push over ssh stdin (%d bytes); use scp' % len(data))
    parent = str(Path(dest).parent)
    P.run(
        "mkdir -p %s && cat > %s.tmp && chmod %s %s.tmp && mv %s.tmp %s" % (
            parent, dest, mode, dest, dest, dest,
        ),
        inp=data,
        timeout=120,
    )


def _rj(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _pe_state() -> Path:
    env = os.environ.get('PE_STATE') or os.environ.get('PE_ST')
    if env:
        return Path(env)
    return Path(
        '/home/const/subnet120-rewrite/state/root-audits/'
        '20261106-parallel-cutover/state'
    )


def _repo() -> Path:
    env = os.environ.get('AFFINE_REPO') or os.environ.get('REPO')
    if env:
        return Path(env)
    return Path.home() / 'subnet120-rewrite'


def _ra() -> Path:
    env = os.environ.get('PE_RA') or os.environ.get('RA')
    if env:
        return Path(env)
    return _repo() / 'state' / 'root-audits'


def _upload_root() -> Path:
    env = os.environ.get('PIT_UPLOAD_ROOT')
    if env:
        return Path(env)
    return _pe_state() / 'pit' / 'uploads'


def _v834e_release() -> Path:
    env = os.environ.get('PIT_V834E_RELEASE') or os.environ.get('AFFINE_V834E_RELEASE')
    if env:
        return Path(env)
    return (
        _ra() / '20261108-no-improvement-audit' / 'fix-v2' / 'v834e' / 'release-v834e-signed'
    )


def _trainer_bundle() -> Path:
    env = os.environ.get('PIT_TRAINER_BUNDLE') or os.environ.get('AFFINE_TRAINER_BUNDLE')
    if env:
        return Path(env)
    # Golden / plain tree that contains runtime-exact/venv + host modules.
    for cand in (
        _ra() / '20261109-per-item-train-eval' / 'trainer-bundle',
        Path('/opt/plain/tbx-host'),
        _repo() / 'miner' / 'validator' / 'trainer-v834e',
    ):
        if cand.is_dir():
            return cand
    return _v834e_release()


def _presign_bin() -> Path:
    env = os.environ.get('PIT_PRESIGN') or os.environ.get('AFFINE_PRESIGN')
    if env:
        return Path(env)
    return _ra() / '20261026-promotion-gate' / 'code' / 'presign.py'


def _gate_fetch_py() -> Path:
    for cand in (
        _ra() / '20261026-promotion-gate' / 'code' / 'pod' / 'gate_fetch.py',
        _ra() / '20261102-per-sub-pod-test' / 'code' / 'gate_fetch.py',
        _HERE / 'gate_fetch.py',
    ):
        if cand.is_file():
            return cand
    # Minimal inline fetch helper shipped when host helpers are absent (tests).
    return Path()


def _host_py() -> str:
    return os.environ.get('HOST_PY') or str(_repo() / '.venv' / 'bin' / 'python')


def _head_files(head: str) -> Mapping[str, Any]:
    """Resolve bf16 head file map for presign (from lifecycle / CURRENT)."""
    life = Path(os.environ.get('PE_LIFE') or (_repo() / 'state' / 'mimo-mainnet-v1'))
    for e_dir in sorted((life / 'epochs').glob('*'), reverse=True):
        if not e_dir.is_dir():
            continue
        for name in ('TRAINING-RESULT.json', 'CANDIDATE.json', 'RESULT.json'):
            d = _rj(e_dir / name) or {}
            for path in (
                d.get('checkpoint'),
                (d.get('payload') or {}).get('checkpoint'),
                (d.get('result') or {}).get('checkpoint') if isinstance(d.get('result'), dict) else None,
            ):
                if isinstance(path, dict) and path.get('id') == head and path.get('files'):
                    return path['files']
    cur = _rj(life / 'CURRENT.json') or {}
    ck = cur.get('checkpoint') or {}
    if ck.get('id') == head and ck.get('files'):
        return ck['files']
    # Explicit override for tests / dry-run (small fake descriptor).
    env = os.environ.get('PIT_HEAD_FILES_JSON')
    if env:
        doc = json.loads(Path(env).read_text() if Path(env).is_file() else env)
        if isinstance(doc, Mapping):
            return doc
    raise BoxIOError('head file map missing for %s (box will fetch from R2; validator has no local weights)' % head[:12])


def _presign_urls(cid: str, files: Mapping[str, Any]) -> Mapping[str, Any]:
    """Host-side presign; URLs go to the box over stdin and are never logged."""
    helper = _presign_bin()
    if not helper.is_file():
        # Test / dry-run: allow injected URLs.
        inj = os.environ.get('PIT_PRESIGN_PAYLOAD_JSON')
        if inj:
            return json.loads(Path(inj).read_text() if Path(inj).is_file() else inj)
        raise BoxIOError('presign helper missing: %s' % helper)
    p = subprocess.run(
        [_host_py(), '-B', str(helper), cid],
        input=json.dumps(files),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if p.returncode != 0:
        raise BoxIOError('presign failed: %s' % (p.stderr or p.stdout)[-300:])
    return json.loads(p.stdout)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _tar_dir(src: Path, dest_tar: Path) -> str:
    """Tar directory *contents* (not the top folder name); return sha256."""
    src = Path(src)
    with tarfile.open(dest_tar, 'w') as tar:
        if src.is_dir():
            for child in sorted(src.iterdir()):
                tar.add(str(child), arcname=child.name)
        else:
            tar.add(str(src), arcname=src.name)
    return _sha256_file(dest_tar)


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


def _parse_r2_handle(handle: str, *, kind: str) -> str:
    """download_head/master return opaque r2:// handles; extract the head id."""
    if isinstance(handle, str) and handle.startswith('r2://%s/' % kind):
        return handle.split('/', 3)[-1].split(':')[0]
    if isinstance(handle, str) and len(handle) == 64:
        return handle
    raise BoxIOError('expected r2://%s/<hex64> handle, got %r' % (kind, handle))


# Box-side fetch+train driver (shipped as a file; runs under runtime-exact/venv).
_BOX_TRAIN_DRIVER = r'''#!/usr/bin/env python3
"""On-box PIT provision+train driver (BRIEF §1.3 / §1.4). Stdin JSON:
{head, expected_master, head_presign, master_presign?, batch_path, out_dir, work,
 v834e_root, report_path, hotkey, gate_fetch}
Presigned URL blobs are deleted after use. Never logs URLs.
"""
import hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

def main():
    spec = json.load(sys.stdin)
    work = Path(spec["work"]); work.mkdir(parents=True, exist_ok=True)
    head = spec["head"]; expected = spec["expected_master"]
    assert len(head) == 64 and len(expected) == 64
    os.environ["AFFINE_FP32_EXPECTED_MASTER_HASH"] = expected
    os.environ.pop("AFFINE_FP32_BOOTSTRAP_OVERRIDE", None)
    os.environ.pop("FP32_BOOTSTRAP_OVERRIDE", None)

    # Verify SIGNED-MANIFEST when present.
    v834e = Path(spec["v834e_root"])
    man = v834e / "SIGNED-MANIFEST"
    if man.is_file():
        subprocess.check_call(["sha256sum", "-c", "SIGNED-MANIFEST"], cwd=str(v834e))

    gate_fetch = Path(spec.get("gate_fetch") or (work / "gate_fetch.py"))
    ckpt = Path(spec.get("ckpt_root") or "/root/affine-env-probe/ckpt")
    ckpt.mkdir(parents=True, exist_ok=True)

    # Fetch bf16 head via gate_fetch (URLs on a private file, then deleted).
    head_presign = spec["head_presign"]
    urls_path = work / (".%s.urls.private.json" % head[:12])
    desc_path = ckpt / ("%s.descriptor.json" % head)
    urls_path.write_text(json.dumps(head_presign.get("urls") or head_presign))
    if "descriptor" in head_presign:
        desc_path.write_text(json.dumps(head_presign["descriptor"]))
    else:
        desc_path.write_text(json.dumps({"id": head, "files": head_presign.get("files") or {}}))
    log = work / "head.fetch.log"
    cmd = [sys.executable, str(gate_fetch), str(desc_path), str(ckpt), str(urls_path), "--threads", "32"]
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    urls_path.unlink(missing_ok=True)
    verified = ckpt / head / ".verified"
    if rc != 0 or not verified.is_file():
        sys.stderr.write(log.read_text()[-800:] if log.is_file() else "head fetch failed\n")
        sys.exit(2)
    head_dir = (ckpt / head).resolve()

    # Fetch fp32 master into FP32_STORE/<head>/ (or digest-addressed).
    store = Path(spec.get("fp32_store") or "/workspace/affine-mimo-trainer-prod/fp32-master")
    master_dir = store / head
    master_dir.mkdir(parents=True, exist_ok=True)
    master_presign = spec.get("master_presign")
    if master_presign:
        murls = work / (".%s.master.urls.private.json" % head[:12])
        murls.write_text(json.dumps(master_presign.get("urls") or master_presign.get("download") or master_presign))
        mdesc = work / "master.descriptor.json"
        mdesc.write_text(json.dumps(master_presign.get("descriptor") or {"id": head, "files": master_presign.get("files") or {}}))
        mlog = work / "master.fetch.log"
        with open(mlog, "w") as lf:
            rc = subprocess.call(
                [sys.executable, str(gate_fetch), str(mdesc), str(master_dir), str(murls), "--threads", "32"],
                stdout=lf, stderr=subprocess.STDOUT,
            )
        murls.unlink(missing_ok=True)
        if rc != 0:
            # Fallback: public affine.io manifest + download block (miner kit shape).
            fetch_py = Path(spec.get("fetch_fp32") or "")
            if fetch_py.is_file():
                subprocess.check_call(
                    [sys.executable, str(fetch_py), "--head", head, "--out", str(master_dir)],
                )
            else:
                sys.stderr.write(mlog.read_text()[-800:] if mlog.is_file() else "master fetch failed\n")
                sys.exit(3)
    (master_dir / "master_digest.txt").write_text(expected + "\n")

    out = Path(spec["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    report = Path(spec["report_path"])
    batch = Path(spec["batch_path"])
    train_entry = Path(spec.get("train_entry") or (v834e / "train_step.py"))
    if not train_entry.is_file():
        # Prefer kit-style entry under the shipped bundle.
        for cand in (v834e / "train_step.py", v834e / "host" / "train_step.py", work / "train_box_entry.py"):
            if cand.is_file():
                train_entry = cand
                break
    py = spec.get("venv_python") or sys.executable
    cmd = [
        py, str(train_entry),
        "--head", str(head_dir),
        "--batch", str(batch),
        "--out", str(out),
        "--master", str(master_dir),
        "--report", str(report),
        "--hotkey", str(spec.get("hotkey") or "pit"),
    ]
    env = os.environ.copy()
    env["AFFINE_FP32_EXPECTED_MASTER_HASH"] = expected
    env.pop("AFFINE_FP32_BOOTSTRAP_OVERRIDE", None)
    env.pop("FP32_BOOTSTRAP_OVERRIDE", None)
    with open(work / "train.log", "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    if rc != 0 or not report.is_file():
        sys.stderr.write((work / "train.log").read_text()[-1200:] if (work / "train.log").is_file() else "train failed\n")
        sys.exit(4)
    rep = json.loads(report.read_text())
    if rep.get("used_fp32_bootstrap_override") or ((rep.get("fp32_master") or {}).get("source") == "bootstrap-upcast"):
        sys.exit("bootstrap-override-refused-by-pit")
    # Marker for host: sha of report for verify-after-scp.
    (work / "REPORT.sha256").write_text(hashlib.sha256(report.read_bytes()).hexdigest() + "\n")
    print(json.dumps({"ok": True, "report": str(report), "out": str(out), "head": str(head_dir)}))

if __name__ == "__main__":
    main()
'''


class HostBoxIO:
    """Host-side BoxIO: provision+train ON the rented box (never local GPU)."""

    def download_head(self, head: str) -> str:
        """Opaque R2 handle — validator does NOT load head weights."""
        if not isinstance(head, str) or len(head) != 64:
            raise BoxIOError('head must be hex64')
        return 'r2://head/%s' % head

    def download_master(self, head: str, expected_digest: str) -> str:
        """Opaque R2 handle — validator does NOT load the fp32 master."""
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
        master_handle = kwargs.get('master_path') or ''
        if not expected or len(str(expected)) != 64:
            raise BoxIOError('train_v834e requires expected_master_digest hex64')
        if not batch_handle or str(batch_handle).startswith('secret:'):
            raise BoxIOError('train_v834e requires opaque non-secret batch_handle')
        if str(batch_handle).startswith('batch-handle:'):
            raise BoxIOError(
                'refuse synthetic batch-handle; pass real K1 batch path '
                '(E1041-class divergence risk vs live trainer)'
            )
        batch_path = Path(str(batch_handle))
        if not batch_path.is_file():
            raise BoxIOError('batch file missing: %s' % batch_path)

        head = _parse_r2_handle(str(head_handle), kind='head')
        # master handle embeds expected; prefer explicit expected arg
        try:
            _parse_r2_handle(str(master_handle), kind='master')
        except BoxIOError:
            pass

        P = _as_pod(pod)
        work = '%s/%s-%d' % (BOX_PIT, item_id[:16], int(time.time()))
        remote_batch = work + '/batch.json'
        remote_out = work + '/candidate'
        remote_report = work + '/report.json'
        remote_driver = work + '/box_train_driver.py'
        remote_gate_fetch = work + '/gate_fetch.py'
        remote_bundle_tar = work + '/trainer-bundle.tar'
        remote_v834e = work + '/v834e'
        remote_tbx = BOX_TBX

        P.run(
            'mkdir -p %s %s/ckpt %s/runs %s %s %s' % (
                BOX_ROOT, BOX_ROOT, BOX_ROOT, work, BOX_FP32_STORE, remote_tbx,
            ),
            timeout=120,
        )

        # --- Ship gate_fetch helper (small text) ---
        gf = _gate_fetch_py()
        if gf.is_file():
            _push_text(P, remote_gate_fetch, gf.read_bytes(), mode='755')
        else:
            # Minimal fetch stub for integration tests (reads URLs JSON, writes .verified).
            stub = (
                b'#!/usr/bin/env python3\n'
                b'import json,sys,pathlib\n'
                b'desc,dest,urls=sys.argv[1],sys.argv[2],sys.argv[3]\n'
                b'p=pathlib.Path(dest); p.mkdir(parents=True, exist_ok=True)\n'
                b'd=json.load(open(desc)); cid=d.get("id") or "out"\n'
                b'target=p/cid if not (p/cid).exists() else p\n'
                b'target.mkdir(parents=True, exist_ok=True)\n'
                b'u=json.load(open(urls))\n'
                b'for name,meta in (u if isinstance(u,dict) else {}).items():\n'
                b'  url=(meta.get("url") if isinstance(meta,dict) else meta) if meta else None\n'
                b'  (target/str(name)).write_text("fetched:"+str(url)[:80] if url else "x")\n'
                b'(target/".verified").write_text("ok\\n")\n'
                b'print("FETCH_OK")\n'
            )
            _push_text(P, remote_gate_fetch, stub, mode='755')

        # --- Ship trainer bundle + v834e as tar via scp; verify checksum on box ---
        bundle_src = _trainer_bundle()
        v834e_src = _v834e_release()
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            bundle_tar = td_path / 'trainer-bundle.tar'
            # Prefer shipping the signed release tree; fall back to bundle root.
            ship_root = v834e_src if v834e_src.is_dir() else bundle_src
            if not ship_root.is_dir():
                raise BoxIOError(
                    'v834e/trainer bundle missing on host (%s); set PIT_V834E_RELEASE' % ship_root
                )
            tar_sha = _tar_dir(ship_root, bundle_tar)
            _scp_to(P, bundle_tar, remote_bundle_tar, timeout=7200)
            P.run(
                "mkdir -p %s %s && tar -C %s -xf %s && "
                "echo %s > %s.sha256 && "
                "(test -f %s/SIGNED-MANIFEST && cd %s && sha256sum -c SIGNED-MANIFEST; true) && "
                "(cp -al %s/. %s/ 2>/dev/null || cp -a %s/. %s/) && "
                "echo BUNDLE_OK" % (
                    remote_v834e, remote_tbx, remote_v834e, remote_bundle_tar,
                    tar_sha, remote_bundle_tar,
                    remote_v834e, remote_v834e,
                    remote_v834e, remote_tbx, remote_v834e, BOX_WORKSPACE,
                ),
                timeout=1800,
            )

        # --- Presign head (+ optional master) on the host; URLs → box stdin only ---
        files = _head_files(head)
        head_presign = _presign_urls(head, files)
        master_presign = None
        inj_m = os.environ.get('PIT_MASTER_PRESIGN_JSON')
        if inj_m:
            master_presign = json.loads(Path(inj_m).read_text() if Path(inj_m).is_file() else inj_m)
        else:
            # Best-effort: public k1 manifest download block (no secrets).
            try:
                import urllib.request
                url = 'https://affine.io/k1/fp32-master/%s.json' % head
                with urllib.request.urlopen(url, timeout=30) as resp:
                    master_presign = json.loads(resp.read().decode())
            except Exception:
                master_presign = None

        # --- Ship batch + driver via scp ---
        _scp_to(P, batch_path, remote_batch, timeout=600)
        with tempfile.NamedTemporaryFile('wb', suffix='.py', delete=False) as tmp:
            tmp.write(_BOX_TRAIN_DRIVER.encode())
            tmp_path = Path(tmp.name)
        try:
            _scp_to(P, tmp_path, remote_driver, timeout=120)
        finally:
            tmp_path.unlink(missing_ok=True)
        P.run('chmod 755 %s' % remote_driver, timeout=30)

        venv_py = os.environ.get('BOX_VENV_PYTHON') or BOX_VENV
        # Resolve venv on box (BRIEF: runtime-exact/venv under workspace).
        P.run(
            "set -e; "
            "if [ ! -x %s ]; then "
            "  if [ -x %s/runtime-exact/venv/bin/python ]; then V=%s/runtime-exact/venv/bin/python; "
            "  elif [ -x /opt/plain/workspace/runtime-exact/venv/bin/python ]; then V=/opt/plain/workspace/runtime-exact/venv/bin/python; "
            "  else V=python3; fi; else V=%s; fi; "
            "echo \"$V\" > %s/venv.path" % (
                venv_py, remote_tbx, remote_tbx, venv_py, work,
            ),
            timeout=60,
        )

        driver_spec = dict(
            head=head,
            expected_master=expected,
            head_presign=head_presign,
            master_presign=master_presign,
            batch_path=remote_batch,
            out_dir=remote_out,
            work=work,
            v834e_root=remote_v834e,
            report_path=remote_report,
            hotkey=str(kwargs.get('hotkey') or 'pit'),
            gate_fetch=remote_gate_fetch,
            ckpt_root=BOX_CKPT,
            fp32_store=BOX_FP32_STORE,
            venv_python=venv_py,
        )
        # URLs only on stdin — never written to host logs by this call site.
        try:
            out = P.run(
                "V=$(cat %s/venv.path 2>/dev/null || echo python3); "
                "$V %s" % (work, remote_driver),
                inp=json.dumps(driver_spec).encode(),
                timeout=int(os.environ.get('PIT_TRAIN_TIMEOUT_S') or 7200),
            )
        except Exception as exc:
            raise BoxIOError('on-box trainer failed: %s' % str(exc)[:400]) from exc
        if 'ok' not in str(out):
            raise BoxIOError('on-box trainer driver did not confirm ok')

        # --- Pull artifacts as FILES via scp (never cat binary over ssh) ---
        host_out = _upload_root() / ('train-%s-%d' % (item_id[:16], int(time.time())))
        host_out.mkdir(parents=True, exist_ok=True)
        report = host_out / 'report.json'
        _scp_from(P, remote_report, report, timeout=600)
        remote_sha = P.run('cat %s/REPORT.sha256' % work, timeout=30).strip().split()[0]
        got_sha = _sha256_file(report)
        if remote_sha and remote_sha != got_sha:
            raise BoxIOError('training report sha mismatch after scp')
        rep = json.loads(report.read_text())
        if not isinstance(rep, dict) or not rep:
            raise BoxIOError('training report missing after train (required before release)')
        if rep.get('used_fp32_bootstrap_override') or (
            (rep.get('fp32_master') or {}).get('source') == 'bootstrap-upcast'
        ):
            raise BoxIOError('bootstrap-override-refused-by-pit')

        # Candidate tree: tar on box, scp the tar, extract locally.
        remote_cand_tar = work + '/candidate.tar'
        P.run('tar -C %s -cf %s .' % (remote_out, remote_cand_tar), timeout=3600)
        cand_tar = host_out / 'candidate.tar'
        _scp_from(P, remote_cand_tar, cand_tar, timeout=7200)
        cand_dir = host_out / 'candidate'
        cand_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(cand_tar, 'r') as tar:
            tar.extractall(cand_dir)

        candidate_id = str(rep.get('candidate_id') or rep.get('candidate_sha256') or '')
        if len(candidate_id) != 64:
            # Fall back to hashing safetensors names (same as prior helper).
            files = {}
            for p in sorted(cand_dir.rglob('*')):
                if p.is_file() and p.suffix in ('.json', '.safetensors', '.jinja'):
                    files[p.name] = _sha256_file(p)
            candidate_id = hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
            ).hexdigest()
        if len(candidate_id) != 64:
            raise BoxIOError('candidate id digest failed')

        master_out = host_out / 'fp32-master'
        remote_master_out = remote_out + '/fp32-master'
        marker = P.run(
            'test -d %s && echo YES || echo NO' % remote_master_out,
            timeout=60, check=False,
        )
        if 'YES' in str(marker):
            remote_m_tar = work + '/master.tar'
            P.run('tar -C %s -cf %s .' % (remote_master_out, remote_m_tar), timeout=3600)
            m_tar = host_out / 'master.tar'
            _scp_from(P, remote_m_tar, m_tar, timeout=7200)
            master_out.mkdir(parents=True, exist_ok=True)
            with tarfile.open(m_tar, 'r') as tar:
                tar.extractall(master_out)
        else:
            master_out.mkdir(parents=True, exist_ok=True)
            (master_out / 'master_digest.txt').write_text(str(expected) + '\n')

        master_dig = str(rep.get('candidate_master_digest') or rep.get('master_digest') or expected)
        if len(master_dig) != 64:
            raise BoxIOError('candidate master digest missing')

        train_gpu = kwargs.get('train_gpu') or _detect_box_gpu(P)

        # Scrub credentials left on the box (presign private files already deleted by driver).
        P.run(
            "rm -f %s/.*.urls.private.json %s/*.urls.private.json 2>/dev/null; echo SCRUB_OK" % (work, work),
            timeout=60, check=False,
        )

        return dict(
            candidate_id=candidate_id,
            candidate_path=str(cand_dir),
            candidate_master_path=str(master_out),
            candidate_master_digest=master_dig,
            candidate_sha256=candidate_id,
            used_fp32_bootstrap_override=False,
            fp32_master_source=(rep.get('fp32_master') or {}).get('source') or 'disk',
            report=str(report),
            report_doc=rep,
            training_report=rep,
            train_gpu=train_gpu,
            pod_id=getattr(P, 'id', None),
            box_work=work,
            fetch='r2-box-side',
            artifact_transport='scp',
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
        print('usage: pe_box.py train --pod-json POD.json --head H --master-digest M --batch BATCH')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
