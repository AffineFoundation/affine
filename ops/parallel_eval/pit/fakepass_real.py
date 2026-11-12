"""REAL fakepass: forced GREEN through the real train / upload / serve / apply path.

Only the eval verdict is fake. Everything else runs the production code:

* the newest epoch that is already decided and whose batch is public (or the
  ``--epoch`` the operator names, under the same rule), the head ``H`` its
  signed job trains from and the published fp32 master index ``M(H)`` are
  read (never written); an epoch in flight or decided RED is never used;
* a real ``affine-pe-pit-fakepass-*`` box is rented (B300, then B200, then
  H200, each under the price ceiling; never sharded GPUs);
* ``HostBoxIO.train_v834e`` trains on host-v834e (``host.checked``, admitted
  files byte-identical, no overlay) and ``_early_upload_artifacts`` streams
  bf16 + fp32 to real R2 under ``pit-fakepass/<run>/``;
* ``pe_lane.serve`` brings up vLLM and verifies the live flags; a detached box
  job holds "eval" while the host pings the forwarded ssh port;
* ``--restart-host-mid-eval`` re-adopts from sandbox host workers killed by
  exact PID; the box job.pid must survive and nobody may free_gpus/place_cand;
* the FAKE RESULT is written only after eval.done; then the real
  ``LiveApplicator`` / pe_apply / pe_live chain runs against sandbox state to a
  sandbox HEAD-CHAIN, sandbox hash-sync and a minimal bench;
* a second step reloads ``C`` on the same box through ``fetch_master_on_box``
  against the promoted index and the trainer's own ``_fp32_prepare``.

Every write target resolves inside the /tmp sandbox; the live deny-list keys
are only ever rebound to sandbox paths.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Never, Optional, Sequence, Tuple

from subnet.storage import Bucket

from ops.parallel_eval.live import pe_artifact_upload as AU
from ops.parallel_eval.live import pe_box
from ops.parallel_eval.live import pe_verify

from . import PRICE_CEILING_USD_H
from .apply_live import LiveApplicator
from .bench_publish import node_record, write_results
from .box_job import STATUS_RUNNING, controller_for_pod, marker_path
from .config import load_pit_config
from .cutover_apply import confirm_head_advanced, cutover_apply_alert, drive_cutover_apply
from .fakepass_sandbox import (
    BOX_NAME_PREFIX,
    FAKE_Z,
    R2_PREFIX_ROOT,
    R2_PREFIX_TMPL,
    FakepassError,
    FakepassPhase,
    SeveredPod,
    append_timeline,
    assert_env_sandboxed,
    assert_isolated,
    assert_paths_outside_live,
    cleanup_sandbox,
    fake_green_fresh_stamp,
    fake_result_doc,
    has_lium_credentials,
    lium_credential_source,
    liveness_loop,
    phase_diagnostics,
    tcp_ping,
)
from .gpu_policy import effective_gpu_order
from .pe_rent import RentError, RentService, bind_host_rent_service, is_fakepass_name
from .signer import FixtureRootSigner
from .state import ItemRecord, ItemState

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
LIVE_CODE = _HERE.parent / 'live'
ADMITTED_MANIFEST = _HERE / 'host_v834e_admitted.sha256.json'
ROOT_ADMISSION = 'TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json'
HASH_SYNC_SRC = _HERE / 'fakepass_hash_sync.py'
STEP2_SRC = _HERE / 'fakepass_step2_fp32.py'
PUBLIC_MASTER_URL = 'https://affine.io/k1/fp32-master/%s.json'
PUBLIC_TURN_URL = 'https://affine.io/k1/turns/%d/turn.json'
# Most public-record lookups the automatic epoch pick makes before it refuses.
PUBLIC_SCAN_LIMIT = 64

FAKEPASS_GPU_ORDER = tuple(
    g for g in effective_gpu_order(allow_sharded_gpus=False) if g in ('B300', 'B200', 'H200')
)
RENT_TTL = '4h'
VLLM_PORT = 18555
MIN_HOLD_S = 120
MIN_HOLD_RESTART_S = 600
EVAL_DONE_GRACE_S = 300
WORKER_ADOPT_TIMEOUT_S = 900

# Real runs refuse every test-injection seam pe_box / pe_artifact_upload honour.
INJECTION_ENV = (
    'PIT_R2_LOCAL_MIRROR', 'PIT_PRESIGN_PUT_JSON', 'PIT_PRESIGN_MASTER_PUT_JSON',
    'PIT_PRESIGN_PAYLOAD_JSON', 'PIT_PRESIGN_KEYS_JSON', 'PIT_MASTER_PRESIGN_JSON',
    'PIT_MASTER_INDEX_JSON', 'PIT_FP32_MASTER_URL_TMPL', 'PE_FAKE_TRAINER',
)
# Trainer overlays / sharded switches: host-v834e must ship byte-identical.
OVERLAY_ENV = (
    'PIT_TRAINER_HOST', 'AFFINE_TRAINER_HOST', 'PIT_USE_INREPO_TRAINER',
    'PIT_TRAIN_PIT_PY', 'PIT_ZERO3', 'PIT_ALLOW_SHARDED_GPUS',
    'AFFINE_FP32_BOOTSTRAP_OVERRIDE',
)
# Allowed only when the value resolves to the signed default trainer host dir.
TRAINER_HOST_ENV = ('PIT_TRAINER_HOST', 'AFFINE_TRAINER_HOST')
# Modules that bind live roots from os.environ at import time.
LIVE_BOUND_MODULES = (
    'pe_common', 'pe_rent', 'pe_lane', 'pe_apply', 'pe_gate_hook',
    'pe_live', 'pe_lifecycle',
    'ops.parallel_eval.live.pe_common', 'ops.parallel_eval.live.pe_rent',
    'ops.parallel_eval.live.pe_lane', 'ops.parallel_eval.live.pe_apply',
    'ops.parallel_eval.live.pe_gate_hook', 'ops.parallel_eval.live.pe_live',
    'ops.parallel_eval.live.pe_lifecycle',
)

FREE_GPUS_SIG = "pgrep -f 'train_pit.py|v83_dp.py'"
PLACE_CAND_SIGS = ("model-00001-of-*.safetensors", '/.verified', 'descriptor.json')
FORBIDDEN_MID_EVAL = ('free_gpus', 'place_cand', 'place_and_pull')

BENCH_PROMPTS = (
    ('What is 17 + 25? Reply with only the number.', '42'),
    ('What is 9 * 8? Reply with only the number.', '72'),
    ('What is 100 - 37? Reply with only the number.', '63'),
    ('What is 144 / 12? Reply with only the number.', '12'),
)

HOLD_SCRIPT = r'''
import json, sys, time, urllib.request
jd, cid, hold, port = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
t0 = time.time(); polls = ok = 0; last = None
while time.time() - t0 < hold:
    polls += 1
    try:
        with urllib.request.urlopen('http://127.0.0.1:%d/v1/models' % port, timeout=5) as r:
            if '"data"' in r.read().decode(errors='replace'):
                ok += 1
    except Exception as exc:
        last = str(exc)[:200]
    time.sleep(5)
doc = dict(ok=ok > 0, FAKE=True, fakepass_hold=True, candidate_id=cid, hold_s=hold,
           polls=polls, models_ok=ok, last_error=last, at=time.time())
open(jd + '/markers/eval.done', 'w').write(json.dumps(doc, sort_keys=True) + '\n')
'''

BENCH_SCRIPT = r'''
import json, re, sys, time, urllib.request
base = 'http://127.0.0.1:%d/v1' % int(sys.argv[1])
prompts = json.loads(sys.argv[2])
mid = json.load(urllib.request.urlopen(base + '/models', timeout=30))['data'][0]['id']
rows = []
for q, want in prompts:
    body = json.dumps({'model': mid, 'messages': [{'role': 'user', 'content': q}],
                       'temperature': 0, 'presence_penalty': 0, 'max_tokens': 4096}).encode()
    req = urllib.request.Request(base + '/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    out = json.load(urllib.request.urlopen(req, timeout=900))
    ch = out['choices'][0]
    text = (ch.get('message') or {}).get('content') or ''
    nums = re.findall(r'-?\d+', text.split('</think>')[-1])
    rows.append(dict(want=want, got=nums[-1] if nums else None, ok=bool(nums) and nums[-1] == want,
                     chars=len(text), seconds=round(time.time() - t0, 1),
                     finish=ch.get('finish_reason')))
print(json.dumps(dict(model=mid, rows=rows)))
'''


def _die(msg: str) -> Never:
    raise FakepassError(msg)


def _assert_never(phase: FakepassPhase) -> Never:
    raise FakepassError('unhandled fakepass phase: %s' % phase)


def _hex64(v: Any) -> bool:
    s = str(v or '')
    return len(s) == 64 and all(c in '0123456789abcdef' for c in s)


def _rj(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _wj(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _under(path: Any, root: Path) -> bool:
    p = Path(str(path)).resolve()
    r = Path(root).resolve()
    return p == r or str(p).startswith(str(r) + os.sep)


# ---------------------------------------------------------------------------
# Live sources (read-only)
# ---------------------------------------------------------------------------
def _repo_state(env: Mapping[str, str]) -> Path:
    base = env.get('AFFINE_REPO') or env.get('REPO')
    return (Path(base) if base else Path.home() / 'subnet120-rewrite') / 'state'


def live_life_root(env: Mapping[str, str]) -> Path:
    v = env.get('PE_LIFE') or env.get('LIFE_ROOT')
    return Path(v) if v else _repo_state(env) / 'mimo-mainnet-v1'


def live_pe_state(env: Mapping[str, str]) -> Path:
    v = env.get('PE_STATE') or env.get('PE_ST')
    if v:
        return Path(v)
    return _repo_state(env) / 'root-audits' / '20261106-parallel-cutover' / 'state'


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'affine-pit-fakepass/1'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


@dataclass
class LiveSources:
    """Read-only inputs of one run.

    ``head`` is the checkpoint the picked epoch trains from. A decided public
    epoch was promoted, so ``head`` is an earlier head than ``live_head``.
    """
    head: str
    epoch: int
    epoch_dir: Path
    master_digest: str
    head_bf16: str
    master_index: Dict[str, Any]
    master_index_origin: str
    current: Dict[str, Any]
    bundle_json: Path
    provenance: Dict[str, str] = field(default_factory=dict)
    live_head: str = ''
    decision: Dict[str, Any] = field(default_factory=dict)
    public_turn_origin: str = ''

    def as_dict(self) -> Dict[str, Any]:
        return {
            'head': self.head,
            'live_head': self.live_head,
            'epoch': self.epoch,
            'epoch_dir': str(self.epoch_dir),
            'decision': dict(self.decision),
            'public_turn_origin': self.public_turn_origin,
            'master_digest': self.master_digest,
            'head_bf16': self.head_bf16,
            'master_index_origin': self.master_index_origin,
            'bundle_json': str(self.bundle_json),
            'provenance_sha256': dict(self.provenance),
        }


@dataclass(frozen=True)
class EpochPick:
    epoch_dir: Path
    head: str
    decision: Dict[str, Any]
    turn_origin: str
    turn_sha256: str


def _job_checkpoint(job_doc: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = job_doc.get('payload') or {}
    return ((payload.get('manifest') or {}).get('payload') or {}).get('checkpoint') or {}


def epoch_signed_head(epoch_dir: Path) -> Optional[str]:
    """The head a complete, ROOT-signed epoch trains from; None when it is not one."""
    try:
        job = _rj(epoch_dir / 'TRAINING-JOB.json')
        disp = _rj(epoch_dir / 'TRAIN-DISPATCH-FF.json')
    except Exception:
        return None
    ck = _job_checkpoint(job)
    training = (job.get('payload') or {}).get('training') or {}
    signed = bool(
        _hex64(ck.get('id'))
        and isinstance(ck.get('files'), Mapping)
        and disp.get('signature')
        and disp.get('signer') != 'SHADOW-UNSIGNED'
        and isinstance(training.get('config'), Mapping)
        and 'optimizer_step' in training
    )
    return str(ck['id']) if signed else None


def read_public_turn(
    env: Mapping[str, str],
    epoch: int,
    *,
    fetch: Callable[[str], bytes] = _http_get,
) -> Tuple[Optional[bytes], str]:
    """The published transparency record of one epoch, and where it came from.

    ``PIT_PUBLIC_TURNS`` names a local copy of the public ``k1/turns`` tree;
    without it the record is read from the public site. None = not published.
    """
    d = env.get('PIT_PUBLIC_TURNS') or env.get('AFFINE_PUBLIC_TURNS')
    if d:
        p = Path(d) / str(int(epoch)) / 'turn.json'
        return (p.read_bytes() if p.is_file() else None), str(p)
    url = PUBLIC_TURN_URL % int(epoch)
    try:
        return fetch(url), url
    except Exception:
        return None, url


def public_decision(epoch_dir: Path, turn: Optional[bytes]) -> Tuple[Optional[Dict[str, Any]], str]:
    """(decision, '') when the published record shows a decided epoch with a public batch.

    The transparency publisher writes the record only after the gate decision
    is final, and it publishes the batch file only for a GREEN. The record
    must name the sha256 of the local signed dispatch, so a record of another
    dispatch cannot stand in for this epoch.
    """
    if turn is None:
        return None, 'no public turn record (not decided, or not published yet)'
    try:
        doc = json.loads(turn)
    except Exception:
        doc = None
    if not isinstance(doc, Mapping):
        return None, 'public turn record is not a JSON object'
    if doc.get('epoch') != int(epoch_dir.name):
        return None, 'public turn record names epoch %r' % (doc.get('epoch'),)
    dec = doc.get('decision') if isinstance(doc.get('decision'), Mapping) else {}
    verdict = dec.get('decision')
    if verdict not in ('green', 'red'):
        return None, 'public turn record has no final decision'
    batch = doc.get('batch') if isinstance(doc.get('batch'), Mapping) else {}
    if batch.get('published') is not True:
        return None, 'decided %s: the batch is not published' % verdict
    disp = doc.get('dispatch') if isinstance(doc.get('dispatch'), Mapping) else {}
    if disp.get('envelope_file_sha256') != _sha256_file(epoch_dir / 'TRAIN-DISPATCH-FF.json'):
        return None, 'public turn record does not name the local signed dispatch'
    return {
        'decision': verdict,
        'head': dec.get('head'),
        'candidate': dec.get('candidate'),
        'decided_at': dec.get('decided_at'),
        'batch_sha256': batch.get('sha256'),
    }, ''


def select_epoch(
    life: Path,
    env: Mapping[str, str],
    epoch: Optional[int] = None,
    *,
    fetch: Callable[[str], bytes] = _http_get,
    accept: Callable[[EpochPick], str] = lambda _pick: '',
) -> EpochPick:
    """Pick a ROOT-signed epoch that is already decided and whose batch is public.

    Without ``epoch`` this is the newest such epoch. With ``epoch`` it is that
    epoch, under the same rule. An epoch in flight or decided RED is never
    picked: its batch is private. ``accept`` returns a refusal reason for a
    pick the caller cannot use ('' = usable).
    """
    epochs = life / 'epochs'
    lookups: List[int] = []

    def examine(d: Path) -> Tuple[Optional[EpochPick], str]:
        head = epoch_signed_head(d)
        if head is None:
            return None, 'no complete ROOT-signed TRAINING-JOB + TRAIN-DISPATCH-FF'
        lookups.append(int(d.name))
        turn, origin = read_public_turn(env, int(d.name), fetch=fetch)
        decision, why = public_decision(d, turn)
        if decision is None:
            return None, '%s (%s)' % (why, origin)
        pick = EpochPick(d, head, decision, origin, hashlib.sha256(turn or b'').hexdigest())
        why = accept(pick)
        return (None, why) if why else (pick, '')

    if epoch is not None:
        pick, why = examine(epochs / str(int(epoch)))
        if pick is None:
            _die('epoch %d refused: %s' % (int(epoch), why))
        return pick
    if not epochs.is_dir():
        _die('live epochs dir missing: %s' % epochs)
    refused: List[str] = []
    for n in sorted((int(p.name) for p in epochs.iterdir() if p.name.isdigit()), reverse=True):
        if len(lookups) >= PUBLIC_SCAN_LIMIT:
            break
        pick, why = examine(epochs / str(n))
        if pick is not None:
            return pick
        refused.append('e%d: %s' % (n, why))
    _die('no decided public epoch among the newest %d signed epochs (%s); name one with --epoch' % (
        len(lookups), '; '.join(refused[:6]) or 'no epochs'))


def read_live_master_index(
    env: Mapping[str, str],
    head: str,
    *,
    fetch: Callable[[str], bytes] = _http_get,
) -> Tuple[Dict[str, Any], str]:
    d = env.get('PIT_FP32_MASTER_INDEX') or env.get('AFFINE_FP32_MASTER_INDEX')
    if d:
        p = Path(d) / ('%s.json' % head)
        if p.is_file():
            return _rj(p), str(p)
    url = PUBLIC_MASTER_URL % head
    try:
        return json.loads(fetch(url).decode()), url
    except Exception as exc:
        _die('M(H) unpublished for %s (%s): %s' % (head[:12], url, str(exc)[:200]))


def resolve_live_sources(
    env: Mapping[str, str],
    *,
    epoch: Optional[int] = None,
    fetch: Callable[[str], bytes] = _http_get,
) -> LiveSources:
    life = live_life_root(env)
    cur_path = life / 'CURRENT.json'
    if not cur_path.is_file():
        _die('live CURRENT.json missing: %s' % cur_path)
    current = _rj(cur_path)
    live_head = str(((current.get('checkpoint') or {}).get('id')) or '')
    if not _hex64(live_head):
        _die('live CURRENT.checkpoint.id is not hex64')
    masters: Dict[str, Tuple[Dict[str, Any], str]] = {}

    def master_refusal(pick: EpochPick) -> str:
        # fp32 stays fail closed: an epoch without a complete published M(H) is not usable.
        try:
            index, origin = read_live_master_index(env, pick.head, fetch=fetch)
        except FakepassError as exc:
            return str(exc)
        if not _hex64(index.get('master_digest')) or not _hex64(index.get('bf16_sha256')):
            return 'M(H) index for %s lacks master_digest/bf16_sha256' % pick.head[:12]
        if not isinstance(index.get('files'), Mapping) or not index['files']:
            return 'M(H) index for %s has no files' % pick.head[:12]
        masters[pick.head] = (dict(index), origin)
        return ''

    pick = select_epoch(life, env, epoch, fetch=fetch, accept=master_refusal)
    epoch_dir, head = pick.epoch_dir, pick.head
    index, origin = masters[head]
    bundle = live_pe_state(env) / 'BUNDLE.json'
    if not bundle.is_file():
        _die('live PE BUNDLE.json (probe bundle for pe_lane.push_bundle) missing: %s' % bundle)
    prov = {
        str(p): _sha256_file(p)
        for p in (epoch_dir / 'TRAINING-JOB.json', epoch_dir / 'TRAIN-DISPATCH-FF.json', bundle)
    }
    if not origin.startswith('http'):
        prov[origin] = _sha256_file(Path(origin))
    prov[pick.turn_origin] = pick.turn_sha256
    return LiveSources(
        head=head,
        epoch=int(epoch_dir.name),
        epoch_dir=epoch_dir,
        master_digest=str(index['master_digest']),
        head_bf16=str(index['bf16_sha256']),
        master_index=index,
        master_index_origin=origin,
        current=dict(current),
        bundle_json=bundle,
        provenance=prov,
        live_head=live_head,
        decision=dict(pick.decision),
        public_turn_origin=pick.turn_origin,
    )


def pit_flag_doc() -> Dict[str, Any]:
    return {
        'FAKE': True,
        'fakepass_sandbox': True,
        'parallel_eval': {
            'enabled': True,
            'pit': {
                'enabled': True, 'mode': 'live', 'cutover': True,
                'max_boxes': 1, 'allow_sharded_gpus': False,
            },
        },
    }


def stage_live_sources(src: LiveSources, paths: Mapping[str, Path]) -> Dict[str, Path]:
    """Copy the read-only live inputs into the sandbox and seed sandbox state."""
    ep = paths['life'] / 'epochs' / str(src.epoch)
    ep.mkdir(parents=True, exist_ok=True)
    for name in ('TRAINING-JOB.json', 'TRAIN-DISPATCH-FF.json'):
        shutil.copyfile(src.epoch_dir / name, ep / name)
        os.chmod(ep / name, 0o600)
    _wj(paths['fp32_public'] / ('%s.json' % src.head), src.master_index)
    shutil.copyfile(src.bundle_json, paths['pe'] / 'BUNDLE.json')
    job = _rj(ep / 'TRAINING-JOB.json')['payload']
    ck = _job_checkpoint({'payload': job})
    _wj(paths['life'] / 'CURRENT.json', {
        'fakepass_sandbox': True,
        'epoch_index': int(src.epoch),
        'checkpoint': {'id': src.head, 'files': dict(ck.get('files') or {})},
        'optimizer_step': int(job['training']['optimizer_step']),
        # The live authorization names the live head; it is not carried to an earlier head.
        'checkpoint_authorization': (
            src.current.get('checkpoint_authorization') if src.head == src.live_head else None
        ),
    })
    _wj(paths['pe'] / 'HEAD-CHAIN.json', {'genesis': src.head, 'applied': []})
    (paths['pe'] / 'GREENS.json').write_text('[]\n')
    flag = paths['root'] / 'PROMOTION-GATE.json'
    _wj(flag, pit_flag_doc())
    for sub in ('pending', 'apply'):
        (paths['life_state'] / sub).mkdir(parents=True, exist_ok=True)
    (paths['gate'] / 'epochs').mkdir(parents=True, exist_ok=True)
    sync = paths['hash_sync'] / 'sync_expected_hash.py'
    shutil.copyfile(HASH_SYNC_SRC, sync)
    return {'epoch_dir': ep, 'flag': flag, 'hash_sync': sync}


# ---------------------------------------------------------------------------
# Environment + preflight
# ---------------------------------------------------------------------------
def real_env_for_sandbox(
    base_env: Mapping[str, str],
    paths: Mapping[str, Path],
) -> Dict[str, str]:
    """Scrubbed sandbox env plus REAL rebinds; every write key lands in the sandbox."""
    env = dict(base_env)
    for key in INJECTION_ENV:
        env.pop(key, None)
    py_path = [str(REPO)] + [p for p in (env.get('PYTHONPATH') or '').split(os.pathsep) if p]
    env.update({
        'PE_FLAG': str(paths['root'] / 'PROMOTION-GATE.json'),
        'PE_LIFE_STATE': str(paths['life_state']),
        'HG_GATE_STATE': str(paths['gate']),
        'PE_HERE': str(paths['pe_here']),
        'PE_JOBS': str(paths['pe_jobs']),
        'PE_SYNC_EXPECTED_HASH': str(paths['hash_sync'] / 'sync_expected_hash.py'),
        'PE_SYNC_EXPECTED_HASH_PYTHON': sys.executable,
        'PE_SYNC_EXPECTED_HASH_LOG': str(paths['hash_sync'] / 'sync-run.log'),
        'PE_SYNC_EXPECTED_HASH_ALERT': str(paths['hash_sync'] / 'HOOK-ALERT.json'),
        'PYTHONPATH': os.pathsep.join(dict.fromkeys(py_path)),
        'PIT_FAKEPASS_MODE': 'real',
    })
    assert_env_sandboxed(env, paths['root'])
    return env


def check_admitted_trainer_host(host_dir: Path, manifest: Path = ADMITTED_MANIFEST) -> Dict[str, Any]:
    """host-v834e must match the committed admitted inventory exactly (no overlay)."""
    host_dir = Path(host_dir)
    if not host_dir.is_dir():
        _die('trainer-host-v834e dir missing: %s' % host_dir)
    expected = dict(_rj(manifest)['host_files'])
    for rel, sha in sorted(expected.items()):
        p = host_dir / rel
        if not p.is_file():
            _die('admitted host file missing: %s' % rel)
        got = _sha256_file(p)
        if got != sha:
            _die('host file %s drifted from admitted manifest (%s != %s)' % (rel, got[:12], sha[:12]))
    extra = sorted(
        p.relative_to(host_dir).as_posix()
        for p in host_dir.rglob('*.py')
        if p.relative_to(host_dir).as_posix() not in expected
    )
    if extra:
        _die('trainer host carries non-admitted overlay modules: %s' % extra)
    if not (host_dir / ROOT_ADMISSION).is_file():
        _die('%s missing from trainer host (host.checked would fail)' % ROOT_ADMISSION)
    return {'host_dir': str(host_dir), 'admitted_files': len(expected), 'overlay': False}


def names_signed_trainer_host(value: str) -> bool:
    """True when ``value`` resolves to the signed default trainer host dir.

    Naming the signed default explicitly ships the same bytes as leaving the
    variable unset, so it is not an overlay. Any other dir still refuses, and
    ``check_admitted_trainer_host`` runs on whichever dir ships.
    """
    try:
        return Path(value).resolve(strict=True) == pe_box._signed_trainer_host_dir().resolve(strict=True)
    except (OSError, RuntimeError):
        return False


def preflight_real(env: Mapping[str, str]) -> Dict[str, Any]:
    """Refuse REAL unless credentials exist and no injection/overlay seam is set."""
    if not has_lium_credentials(env):
        _die('real mode refused: Lium credentials missing (use --dry-isolation-check or --simulate-local)')
    seams = sorted(k for k in INJECTION_ENV if env.get(k))
    if seams:
        _die('real mode refuses test-injection env: %s' % seams)
    overlays = sorted(
        k for k in OVERLAY_ENV
        if env.get(k) and not (k in TRAINER_HOST_ENV and names_signed_trainer_host(env[k]))
    )
    if overlays:
        _die('real mode refuses trainer overlay / sharded env: %s' % overlays)
    for key in ('PIT_LIFE_CFG', 'PIT_BUNDLE_JSON'):
        v = env.get(key)
        if not v or not Path(v).is_file():
            _die('real mode requires %s (host file)' % key)
    out = check_admitted_trainer_host(pe_box._trainer_host_dir())
    out['trainer_host_env'] = sorted(k for k in TRAINER_HOST_ENV if env.get(k))
    out['lium_credentials'] = lium_credential_source(env)
    return out


@contextlib.contextmanager
def swapped_environ(env: Mapping[str, str]) -> Iterator[None]:
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@dataclass
class LiveModules:
    common: Any
    rent: Any
    lane: Any
    live: Any
    lifecycle: Any
    apply: Any
    hook: Any


def import_live_modules() -> LiveModules:
    """Import the env-binding live modules AFTER the sandbox env is installed.

    These modules bind PE_FLAG / PE_HERE / PE_ST / HG_GATE_STATE / PE_LIFE at
    import time, so they cannot be top-level imports here. Production imports
    them by bare name from the live dir (pe_box / cutover_apply do the same).
    """
    loaded = [n for n in LIVE_BOUND_MODULES if n in sys.modules]
    if loaded:
        _die('live modules already imported with pre-sandbox roots (%s); run REAL fakepass in a fresh process' % loaded)
    live_s = str(LIVE_CODE)
    if live_s not in sys.path:
        sys.path.insert(0, live_s)
    common = importlib.import_module('pe_common')
    rent = importlib.import_module('pe_rent')
    lane = importlib.import_module('pe_lane')
    live = importlib.import_module('pe_live')
    lifecycle = importlib.import_module('pe_lifecycle')
    return LiveModules(
        common=common, rent=rent, lane=lane, live=live, lifecycle=lifecycle,
        apply=live.PA, hook=live.HOOK,
    )


def assert_live_modules_sandboxed(mods: LiveModules, root: Path) -> Dict[str, str]:
    bound = {
        'pe_common.HERE': mods.common.HERE,
        'pe_common.ST': mods.common.ST,
        'pe_common.JOBS': mods.common.JOBS,
        'pe_common.LOGS': mods.common.LOGS,
        'pe_common.GATE_ST': mods.common.GATE_ST,
        'pe_common.EP': mods.common.EP,
        'pe_common.LIFE': mods.common.LIFE,
        'pe_common.FLAG': mods.common.FLAG,
        'pe_live.LST': mods.live.LST,
        'pe_live.life_state': mods.live._life_state_root(),
        'pe_gate_hook.FLAG': mods.hook.FLAG,
        'pe_gate_hook.HERE': mods.hook._HERE,
        'pe_gate_hook.PE_ST': mods.hook.PE_ST,
        'pe_gate_hook.PE_JOBS': mods.hook.PE_JOBS,
        'pe_gate_hook.PE_LIFE': mods.hook.PE_LIFE,
        'pe_lifecycle.flag': mods.lifecycle.flag_path(),
        'pe_lifecycle.state': mods.lifecycle.pe_state_path(),
    }
    bad = {k: str(v) for k, v in bound.items() if not _under(v, root)}
    if bad:
        _die('live module roots escaped the sandbox: %s' % bad)
    return {k: str(v) for k, v in bound.items()}


# ---------------------------------------------------------------------------
# Rent / pod
# ---------------------------------------------------------------------------
def make_fakepass_host_rent(
    rent_mod: Any,
    *,
    order: Sequence[str] = FAKEPASS_GPU_ORDER,
    ceiling: float = float(PRICE_CEILING_USD_H),
    ttl: str = RENT_TTL,
    note: Callable[..., None] = lambda _event, **_extra: None,
) -> Callable[..., Mapping[str, Any]]:
    """B300, then B200, then H200 — each only if an 8-GPU offer is under ceiling.

    ``note(event, **fields)``: the event name is positional, so a timeline
    writer whose first parameter is ``event`` cannot receive it twice.
    """
    order = tuple(order)
    if any(g not in FAKEPASS_GPU_ORDER for g in order):
        _die('fakepass rents only %s (got %s)' % (FAKEPASS_GPU_ORDER, order))

    def _rent(name: str, _ttl: Any = None, *, gpu: Any = None, **_ignored: Any) -> Mapping[str, Any]:
        if not is_fakepass_name(name):
            raise RentError('fakepass rent name must start with %s: %s' % (BOX_NAME_PREFIX, name))
        tried: List[Dict[str, Any]] = []
        for g in order:
            try:
                got_g, row = rent_mod.pick_node([g], fallback=False, price_ceiling=ceiling)
            except Exception as exc:
                tried.append({'gpu': g, 'skip': 'pick_node:%s' % str(exc)[:120]})
                continue
            if not row or got_g != g:
                tried.append({'gpu': g, 'skip': 'no 8x%s offer <= $%.2f/h' % (g, ceiling)})
                continue
            note('rent_attempt', gpu=g, offer_price_h=row.get('price_per_hour'))
            out = rent_mod.rent(name, ttl, gpu=g)
            if not out:
                tried.append({'gpu': g, 'skip': 'lium up failed'})
                continue
            d = dict(out)
            got = str(d.get('gpu') or '')
            price = float(d.get('price_h') or d.get('price_per_hour') or 0.0)
            if got not in order or price > ceiling or not d.get('port'):
                rent_mod.release(d.get('id') or name, reason='fakepass-rent-mismatch')
                raise RentError('fakepass rental %s/%s/$%.2f outside policy (released)' % (
                    d.get('name'), got, price))
            d.update(auto_backup_disabled=True, fakepass=True, rent_tried=tried)
            return d
        raise RentError('no %s offer under $%.2f/h: %s' % ('/'.join(order), ceiling, tried))

    return _rent


def build_rent_service(rent_mod: Any, *, ledger_path: Path, note: Callable[..., None]) -> RentService:
    svc = RentService(
        max_boxes=1,
        max_eval_nodes=99,
        gpu_order=FAKEPASS_GPU_ORDER,
        price_ceiling_usd_h=float(PRICE_CEILING_USD_H),
        disable_auto_backup=True,
        require_host=True,
        shadow=True,
        name_prefix=BOX_NAME_PREFIX,
        share_pe_cap=False,
        ledger_path=str(ledger_path),
    )
    bind_host_rent_service(svc, module_path=rent_mod.__name__)
    if svc.host_release is None or svc.list_pods is None:
        _die('host pe_rent lacks release/list_pods')
    svc.host_rent = make_fakepass_host_rent(rent_mod, note=note)
    svc.offer_precheck_fn = None
    return svc


def classify_box_command(cmd: str) -> Optional[str]:
    if FREE_GPUS_SIG in cmd:
        return 'free_gpus'
    if all(sig in cmd for sig in PLACE_CAND_SIGS):
        return 'place_cand'
    return None


class RecordingPod:
    """Pod wrapper: logs every box command and tags free_gpus / place_cand."""

    def __init__(self, inner: Any, log_path: Path):
        self._inner = inner
        self.log_path = Path(log_path)
        self.records: List[Dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def run(self, cmd: str, *args: Any, **kwargs: Any) -> str:
        row = {'tag': classify_box_command(str(cmd)), 'cmd': str(cmd)[:160], 'pid': os.getpid()}
        self.records.append(row)
        append_timeline(self.log_path, dict(row, event='box_cmd'))
        return self._inner.run(cmd, *args, **kwargs)

    def tags_since(self, mark: int) -> List[str]:
        return [r['tag'] for r in self.records[mark:] if r['tag']]


def read_command_tags(log_path: Path) -> List[str]:
    if not Path(log_path).is_file():
        return []
    tags: List[str] = []
    for line in Path(log_path).read_text().splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get('event') == 'forbidden-call':
            tags.append(str(row.get('tag')))
        elif row.get('tag'):
            tags.append(str(row['tag']))
    return tags


def hold_command(*, job_dir: str, candidate_id: str, hold_s: float, port: int = VLLM_PORT) -> str:
    return "python3 - %s %s %d %d <<'PY'\n%s\nPY" % (
        shlex.quote(job_dir.rstrip('/')), candidate_id, int(hold_s), int(port), HOLD_SCRIPT.strip(),
    )


# ---------------------------------------------------------------------------
# R2
# ---------------------------------------------------------------------------
def life_bucket() -> Any:
    cfg = os.environ.get('PIT_LIFE_CFG') or os.environ.get('AFFINE_LIFE_CFG')
    if not cfg or not Path(cfg).is_file():
        _die('PIT_LIFE_CFG required for R2')
    return Bucket(_rj(Path(cfg))['bucket'])


def run_r2_prefix(run_id: str) -> str:
    return R2_PREFIX_TMPL % run_id


def sweep_r2_prefix(prefix: str, *, bucket: Any = None) -> Dict[str, Any]:
    """Delete every object under a fakepass run prefix (refuses anything else)."""
    prefix = prefix.rstrip('/')
    root = R2_PREFIX_ROOT + '/'
    if not prefix.startswith(root) or len(prefix) <= len(root) or '..' in prefix:
        _die('refuse R2 sweep outside %s<run>: %r' % (root, prefix))
    b = bucket if bucket is not None else life_bucket()
    deleted = 0
    token = None
    while True:
        kw: Dict[str, Any] = dict(Bucket=b.name, Prefix=prefix + '/', MaxKeys=1000)
        if token:
            kw['ContinuationToken'] = token
        page = b.client.list_objects_v2(**kw)
        keys = [{'Key': o['Key']} for o in page.get('Contents') or []]
        if keys:
            b.client.delete_objects(Bucket=b.name, Delete={'Objects': keys, 'Quiet': True})
            deleted += len(keys)
        if not page.get('IsTruncated'):
            break
        token = page.get('NextContinuationToken')
    return {'prefix': prefix, 'deleted': deleted}


def validate_pending_index(doc: Mapping[str, Any], *, candidate_id: str, run_prefix: str, store_entry: str) -> None:
    files = dict(doc.get('files') or {})
    if 'meta.json' not in files:
        _die('pending master index lacks meta.json')
    if not any(str(n).endswith('.safetensors') for n in files):
        _die('pending master index has no weight shards')
    if doc.get('meta_json') != {'r2_key': files['meta.json'].get('r2_key')}:
        _die('pending index meta_json must be exactly {r2_key} of the uploaded meta.json: %r' % doc.get('meta_json'))
    bad = [n for n, m in files.items() if not str((m or {}).get('r2_key') or '').startswith(run_prefix + '/')]
    if bad:
        _die('pending master keys escaped %s: %s' % (run_prefix, bad[:3]))
    if doc.get('bf16_sha256') != store_entry:
        _die('pending index bf16_sha256 %s != trainer store entry %s (checkpoint %s)' % (
            str(doc.get('bf16_sha256'))[:12], store_entry[:12], candidate_id[:12]))


def validate_promoted_index(doc: Mapping[str, Any], *, candidate_id: str, live_prefix: str, master_digest: str) -> None:
    files = dict(doc.get('files') or {})
    if doc.get('pending') is not False:
        _die('promoted index still marked pending')
    if doc.get('master_digest') != master_digest:
        _die('promoted index master_digest != host-recorded digest')
    if 'meta.json' not in files:
        _die('promoted index lacks meta.json')
    mj = doc.get('meta_json')
    if not isinstance(mj, Mapping) or set(mj) != {'r2_key'}:
        _die('promoted index meta_json must be {r2_key} (synthetic inline meta refused): %r' % mj)
    if mj['r2_key'] != files['meta.json'].get('r2_key'):
        _die('promoted meta_json pointer != files[meta.json].r2_key')
    bad = [n for n, m in files.items() if not str((m or {}).get('r2_key') or '').startswith(live_prefix + '/')]
    if bad:
        _die('promoted keys outside %s: %s' % (live_prefix, bad[:3]))


# ---------------------------------------------------------------------------
# Sandbox host worker (restart-host-mid-eval)
# ---------------------------------------------------------------------------
def _write_heartbeat(path: Path, **row: Any) -> None:
    _wj(path, dict(row, at=time.time()))


def host_worker_main(state_path: Path) -> int:
    """Sandbox run_host stand-in: re-adopt via the real train_v834e, then watch."""
    st = _rj(state_path)
    hb = Path(st['heartbeat'])
    cmd_log = Path(st['command_log'])
    if os.environ.get('PIT_FAKEPASS_MODE') != 'real':
        _die('host worker must inherit the REAL fakepass sandbox env')
    assert_env_sandboxed(dict(os.environ), Path(st['sandbox_root']))

    def _forbidden(tag: str) -> Callable[..., Never]:
        def _trip(*_a: Any, **_k: Any) -> Never:
            append_timeline(cmd_log, {'event': 'forbidden-call', 'tag': tag, 'pid': os.getpid()})
            raise pe_box.BoxIOError('fakepass host worker: %s forbidden while eval is live' % tag)
        return _trip

    for tag in FORBIDDEN_MID_EVAL:
        setattr(pe_box, tag, _forbidden(tag))
    assert_live_modules_sandboxed(import_live_modules(), Path(st['sandbox_root']))
    P = RecordingPod(pe_box._as_pod(st['pod']), cmd_log)
    _write_heartbeat(hb, phase='starting', pid=os.getpid())
    receipt = pe_box.HostBoxIO().train_v834e(pod=P, **st['train_kwargs'])
    ctrl = controller_for_pod(P, st['train_kwargs']['item_id'])
    s = ctrl.status()
    base = dict(
        pid=os.getpid(), candidate_id=receipt.get('candidate_id'),
        mid_eval_readopt=bool(receipt.get('mid_eval_readopt')),
        artifacts_uploaded=bool(receipt.get('artifacts_uploaded')),
    )
    _write_heartbeat(hb, phase='adopted', job_pid=s.pid, state=s.state,
                     next_stage=s.next_stage, markers=sorted(s.markers), **base)
    deadline = time.time() + float(st.get('watch_s') or 3600)
    while time.time() < deadline:
        s = ctrl.status()
        if 'eval' in s.markers or s.result is not None:
            _write_heartbeat(hb, phase='eval-done', job_pid=s.pid, **base)
            return 0
        _write_heartbeat(hb, phase='watching', job_pid=s.pid, state=s.state, **base)
        time.sleep(float(st.get('poll_s') or 10))
    _write_heartbeat(hb, phase='watch-timeout', **base)
    return 3


# ---------------------------------------------------------------------------
# REAL run
# ---------------------------------------------------------------------------
@dataclass
class RealFakepass:
    run_id: str
    paths: Dict[str, Path]
    env: Dict[str, str]
    live_roots: List[Path]
    sources: LiveSources
    hold_vllm_s: int = MIN_HOLD_S
    sever_after_green: bool = False
    restart_host_mid_eval: bool = False
    liveness_interval_s: float = 15.0
    evidence_out: Optional[Path] = None
    phase: FakepassPhase = FakepassPhase.ISOLATE
    mods: Optional[LiveModules] = None
    staged: Dict[str, Path] = field(default_factory=dict)
    host_check: Dict[str, Any] = field(default_factory=dict)
    rent: Optional[RentService] = None
    rent_out: Dict[str, Any] = field(default_factory=dict)
    pod: Any = None
    box_id: Optional[str] = None
    box_name: Optional[str] = None
    ctrl: Any = None
    train_kwargs: Dict[str, Any] = field(default_factory=dict)
    receipt: Dict[str, Any] = field(default_factory=dict)
    candidate_id: Optional[str] = None
    store_entry: Optional[str] = None
    early_upload: Dict[str, Any] = field(default_factory=dict)
    serve: Dict[str, Any] = field(default_factory=dict)
    eval_job_pid: Optional[int] = None
    eval_done: Dict[str, Any] = field(default_factory=dict)
    restart_result: Dict[str, Any] = field(default_factory=dict)
    fake_result: Dict[str, Any] = field(default_factory=dict)
    green_stamp: Dict[str, Any] = field(default_factory=dict)
    apply_result: Dict[str, Any] = field(default_factory=dict)
    bench: Dict[str, Any] = field(default_factory=dict)
    second_step: Dict[str, Any] = field(default_factory=dict)
    r2_gc: Dict[str, Any] = field(default_factory=dict)
    released: bool = False
    cleaned: bool = False
    _workers: List[subprocess.Popen] = field(default_factory=list, repr=False)
    _liveness_stop: Optional[threading.Event] = field(default=None, repr=False)
    _liveness_thread: Optional[threading.Thread] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        floor = MIN_HOLD_RESTART_S if self.restart_host_mid_eval else MIN_HOLD_S
        self.hold_vllm_s = max(int(self.hold_vllm_s), floor)

    @property
    def log_path(self) -> Path:
        return self.paths['logs'] / 'timeline.jsonl'

    @property
    def run_prefix(self) -> str:
        return run_r2_prefix(self.run_id)

    @property
    def item_id(self) -> str:
        return 'fakepass-%s' % self.run_id

    def _note(self, event: str, **extra: Any) -> None:
        append_timeline(self.log_path, {'event': event, 'phase': self.phase.value, **extra})

    def _advance(self, nxt: FakepassPhase) -> None:
        self.phase = nxt
        phase_diagnostics(self.log_path, nxt.value)

    def _box_cmd_log(self, label: str) -> Path:
        return self.paths['logs'] / ('pod-%s.jsonl' % label)

    def _require_mods(self) -> LiveModules:
        if self.mods is None:
            _die('live modules not imported (ISOLATE must run first)')
        return self.mods

    # -- phases ---------------------------------------------------------
    def phase_isolate(self) -> None:
        self._advance(FakepassPhase.ISOLATE)
        root = self.paths['root']
        assert_isolated(state_root=root, live_roots=self.live_roots)
        assert_paths_outside_live(self.paths, self.live_roots)
        assert_env_sandboxed(self.env, root)
        if os.environ.get('PIT_FAKEPASS_MODE') != 'real':
            _die('REAL phases must run inside the swapped sandbox environ')
        self.mods = import_live_modules()
        bound = assert_live_modules_sandboxed(self.mods, root)
        self._note('sandbox_ready', run_id=self.run_id, root=str(root), live_sources=self.sources.as_dict(),
                   module_roots=bound, trainer_host=self.host_check)

    def build_rent(self, rent_mod: Any) -> RentService:
        return build_rent_service(
            rent_mod, ledger_path=self.paths['pe'] / 'fakepass-rent-ledger.json',
            note=self._note,
        )

    def phase_rent(self) -> None:
        self._advance(FakepassPhase.RENT)
        mods = self._require_mods()
        self.rent = self.build_rent(mods.rent)
        try:
            out = self.rent.rent(item_id=self.item_id, epoch=self.sources.epoch, attempt=0)
        except RentError as exc:
            raise FakepassError('rent failed: %s' % exc) from exc
        self.rent_out = dict(out)
        self.box_id = str(out.get('box_id') or out.get('id'))
        self.box_name = str(out.get('name') or '')
        if not is_fakepass_name(self.box_name):
            _die('rented name must be fakepass prefix: %s' % self.box_name)
        pod_dict = {k: out.get(k) for k in ('id', 'ip', 'port', 'huid', 'name')}
        self.rent_out['pod'] = pod_dict
        self.pod = RecordingPod(pe_box._as_pod(pod_dict), self._box_cmd_log('main'))
        ping = tcp_ping(str(pod_dict['ip']), int(pod_dict['port'] or 0), timeout_s=5.0)
        if not ping.get('ok'):
            _die('forwarded ssh port %s:%s not reachable after rent' % (pod_dict['ip'], pod_dict['port']))
        self._note('rented', box_id=self.box_id, box_name=self.box_name, gpu=out.get('gpu'),
                   price_h=out.get('price_h'), ssh_port=pod_dict['port'], ping=ping)

    def phase_download_train(self) -> None:
        self._advance(FakepassPhase.DOWNLOAD)
        self.train_kwargs = dict(
            expected_master_digest=self.sources.master_digest,
            batch_handle=str(self.staged['epoch_dir']),
            head_path=pe_box.HostBoxIO().download_head(self.sources.head),
            item_id=self.item_id,
            epoch=int(self.sources.epoch),
            epoch_dir=str(self.staged['epoch_dir']),
            train_gpu=self.rent_out.get('gpu'),
        )
        self._advance(FakepassPhase.TRAIN)
        receipt = dict(pe_box.HostBoxIO().train_v834e(pod=self.pod, **self.train_kwargs))
        cid = str(receipt.get('candidate_id') or '')
        if not _hex64(cid) or cid == self.sources.head:
            _die('train produced no new candidate id')
        if receipt.get('used_fp32_bootstrap_override') or receipt.get('fp32_master_source') != 'disk':
            _die('train must load M(H) from disk (got %r)' % receipt.get('fp32_master_source'))
        done = (receipt.get('events') or {}).get('done') or {}
        if done.get('host') != 'host-v834e':
            _die('train ran on %r, not host-v834e' % done.get('host'))
        if 'root-admission-checked' not in (receipt.get('events') or {}):
            _die('train log lacks root-admission-checked (host.checked)')
        fp = (done.get('fp32') or {})
        if fp.get('input_master_digest') != self.sources.master_digest:
            _die('train input master digest != M(H)')
        self.candidate_id = cid
        self.store_entry = str(fp.get('output_bf16') or '')
        self.receipt = receipt
        self.ctrl = controller_for_pod(self.pod, self.item_id)
        # Evidence only: live trained the same head and batch. A different id is not
        # a failure (the all-reduce order differs between GPU types).
        live_candidate = self.sources.decision.get('candidate')
        self._note('train_done', candidate_id=cid, store_entry=self.store_entry,
                   master_out=receipt.get('candidate_master_digest'), train_gpu=receipt.get('train_gpu'),
                   live_candidate=live_candidate, reproduces_live_candidate=bool(cid == live_candidate))

    def phase_early_upload(self) -> None:
        self._advance(FakepassPhase.EARLY_UPLOAD)
        out = dict(pe_box._early_upload_artifacts(self.pod, self.receipt))
        seed = out.get('live_apply_seed') or {}
        if not (out.get('artifacts_uploaded') and seed.get('candidate_verified') and seed.get('master_verified')):
            _die('early upload did not verify bf16 + fp32 against R2')
        cid = str(self.candidate_id)
        idx_path = AU.pending_master_index_dir() / ('%s.json' % cid)
        if not idx_path.is_file():
            _die('pending master index missing after early upload')
        doc = _rj(idx_path)
        validate_pending_index(doc, candidate_id=cid, run_prefix=self.run_prefix,
                               store_entry=str(self.store_entry))
        self.receipt = out
        self.early_upload = {
            'index': str(idx_path),
            'pending_prefix': AU.pending_r2_prefix(cid),
            'files': sorted(doc['files']),
            'meta_json': doc['meta_json'],
            'bf16_sha256': doc['bf16_sha256'],
            'seed': seed,
        }
        self._note('early_upload', **self.early_upload)

    def start_liveness(self) -> None:
        pod = self.rent_out.get('pod') or {}
        self._liveness_stop = threading.Event()
        rent = self.rent

        def _lium_ps() -> Any:
            if rent is None or rent.list_pods is None:
                return None
            return [
                {k: r.get(k) for k in ('name', 'id', 'status')}
                for r in (rent.list_pods() or []) if is_fakepass_name(str(r.get('name') or ''))
            ]

        self._liveness_thread = threading.Thread(
            target=liveness_loop,
            kwargs=dict(
                log_path=self.log_path, host=str(pod.get('ip')), port=int(pod.get('port') or 0),
                box_name=str(self.box_name), stop_at=time.time() + float(self.hold_vllm_s) + 3600.0,
                interval_s=self.liveness_interval_s, lium_ps_fn=_lium_ps, stop_event=self._liveness_stop,
            ),
            name='fakepass-liveness', daemon=True,
        )
        self._liveness_thread.start()

    def stop_liveness(self) -> None:
        if self._liveness_stop is not None:
            self._liveness_stop.set()
        if self._liveness_thread is not None:
            self._liveness_thread.join(timeout=5.0)

    def phase_hold_vllm(self) -> None:
        self._advance(FakepassPhase.HOLD_VLLM)
        mods = self._require_mods()
        cid = str(self.candidate_id)
        served = dict(mods.lane.serve(self.pod, cid))
        self.serve = {'serve_s': served.get('serve_s'), 'flags': served.get('flags')}
        self._note('vllm_served', **self.serve)
        self.start_liveness()
        mark = len(self.pod.records)
        self.eval_job_pid = int(self.ctrl.store.launch_detached(
            self.ctrl.job_dir, stage='eval',
            command=hold_command(job_dir=self.ctrl.job_dir, candidate_id=cid, hold_s=self.hold_vllm_s),
        ))
        st = self.ctrl.status()
        if st.state != STATUS_RUNNING or st.next_stage != 'eval' or st.pid != self.eval_job_pid:
            _die('eval hold job not running: %s' % st.as_dict())
        self._note('eval_hold_start', job_pid=self.eval_job_pid, seconds=self.hold_vllm_s)
        if self.restart_host_mid_eval:
            self.restart_host_workers()
        deadline = time.time() + float(self.hold_vllm_s) + EVAL_DONE_GRACE_S
        while True:
            st = self.ctrl.status()
            if 'eval' in st.markers:
                break
            if time.time() > deadline:
                _die('eval hold did not finish: %s' % st.as_dict())
            time.sleep(10)
        self.eval_done = json.loads(self.ctrl.store.read_text(marker_path(self.ctrl.job_dir, 'eval')) or '{}')
        if not self.eval_done.get('ok'):
            _die('vLLM went away during the eval hold: %s' % self.eval_done)
        forbidden = [t for t in self.pod.tags_since(mark) if t in FORBIDDEN_MID_EVAL]
        if forbidden:
            _die('host issued %s while eval was live' % forbidden)
        self._wait_workers()
        self._note('eval_hold_end', **self.eval_done)

    def _spawn_worker(self, n: int) -> Tuple[subprocess.Popen, Path, Path]:
        wdir = self.paths['workers']
        hb = wdir / ('w%d.heartbeat.json' % n)
        log = self._box_cmd_log('worker%d' % n)
        state = wdir / ('w%d.state.json' % n)
        _wj(state, {
            'heartbeat': str(hb), 'command_log': str(log), 'sandbox_root': str(self.paths['root']),
            'pod': self.rent_out['pod'], 'train_kwargs': self.train_kwargs,
            'watch_s': float(self.hold_vllm_s) + EVAL_DONE_GRACE_S, 'poll_s': 10,
        })
        out = (wdir / ('w%d.out' % n)).open('ab')
        proc = subprocess.Popen(
            [sys.executable, '-B', '-m', 'ops.parallel_eval.pit.fakepass_real', '--host-worker', str(state)],
            cwd=str(REPO), env=dict(os.environ), stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        out.close()
        self._workers.append(proc)
        return proc, hb, log

    def _await_adopted(self, proc: subprocess.Popen, hb: Path) -> Dict[str, Any]:
        deadline = time.time() + WORKER_ADOPT_TIMEOUT_S
        while time.time() < deadline:
            if hb.is_file():
                doc = _rj(hb)
                if doc.get('phase') in ('adopted', 'watching', 'eval-done'):
                    return doc
            if proc.poll() is not None:
                _die('host worker %d exited rc=%s before adopting' % (proc.pid, proc.returncode))
            time.sleep(2)
        _die('host worker %d did not adopt within %ds' % (proc.pid, WORKER_ADOPT_TIMEOUT_S))

    def _check_adoption(self, doc: Mapping[str, Any], label: str) -> None:
        if doc.get('job_pid') != self.eval_job_pid:
            _die('%s saw job.pid %s, eval pid is %s' % (label, doc.get('job_pid'), self.eval_job_pid))
        if not doc.get('mid_eval_readopt') or doc.get('candidate_id') != self.candidate_id:
            _die('%s did not re-adopt mid-eval: %s' % (label, doc))
        if doc.get('phase') != 'adopted' and doc.get('phase') != 'watching':
            _die('%s adopted after eval finished (raise --hold-vllm-s): %s' % (label, doc))

    def restart_host_workers(self) -> None:
        w1, hb1, log1 = self._spawn_worker(1)
        doc1 = self._await_adopted(w1, hb1)
        self._check_adoption(doc1, 'host worker 1')
        killed = int(w1.pid)
        if killed in (os.getpid(), self.eval_job_pid):
            _die('refuse to kill pid %d' % killed)
        os.kill(killed, signal.SIGKILL)
        w1.wait(timeout=60)
        st = self.ctrl.status()
        if st.state != STATUS_RUNNING or st.pid != self.eval_job_pid:
            _die('eval job.pid changed or died after host kill: %s' % st.as_dict())
        w2, hb2, log2 = self._spawn_worker(2)
        doc2 = self._await_adopted(w2, hb2)
        self._check_adoption(doc2, 'host worker 2')
        st = self.ctrl.status()
        if st.state != STATUS_RUNNING or st.pid != self.eval_job_pid:
            _die('eval job.pid changed after host restart: %s' % st.as_dict())
        tags = read_command_tags(log1) + read_command_tags(log2)
        forbidden = sorted({t for t in tags if t in FORBIDDEN_MID_EVAL})
        if forbidden:
            _die('restarted host issued %s during eval' % forbidden)
        self.restart_result = {
            'killed_host_pid': killed,
            'killed_rc': w1.returncode,
            'new_host_pid': int(w2.pid),
            'eval_pid_unchanged': int(self.eval_job_pid),
            'eval_alive': True,
            'no_free_gpus': 'free_gpus' not in tags,
            'no_place_cand': 'place_cand' not in tags and 'place_and_pull' not in tags,
            'worker_box_cmds': {'w1': len(read_command_lines(log1)), 'w2': len(read_command_lines(log2))},
        }
        self._note('restart_host_mid_eval', **self.restart_result)

    def _wait_workers(self) -> None:
        for proc in self._workers:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.kill(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=30)

    def phase_fake_result(self) -> None:
        self._advance(FakepassPhase.FAKE_RESULT)
        if 'eval' not in self.ctrl.status().markers:
            _die('FAKE RESULT refused before eval.done')
        cid = str(self.candidate_id)
        self.fake_result = fake_result_doc(candidate_id=cid, z=FAKE_Z)
        self.green_stamp = fake_green_fresh_stamp(candidate_id=cid, z=FAKE_Z)
        if self.green_stamp.get('promote_live') is not False:
            _die('fake green_fresh must set promote_live=False')
        self.ctrl.publish_result(dict(self.fake_result, candidate_id=cid))
        _wj(self.paths['pit'] / 'RESULT.json', self.fake_result)
        _wj(self.paths['greens'] / ('%s.json' % cid), self.green_stamp)
        _wj(self.paths['gate'] / 'epochs' / str(self.sources.epoch) / 'DECISION.json', {
            'FAKE': True,
            'decision': 'green',
            'candidate': cid,
            'source': 'pit',
            'evaluated': {'z': FAKE_Z, 'n': 0, 'FAKE': True},
            'decided_at': time.time(),
        })
        self._note('fake_result', result=self.fake_result, stamp=self.green_stamp)
        if self.sever_after_green:
            self.pod = SeveredPod(self.pod)
            self.pod.arm()
            self._note('sever_after_green', armed=True)

    def build_item(self) -> ItemRecord:
        r = self.receipt
        cid = str(self.candidate_id)
        job = _rj(self.staged['epoch_dir'] / 'TRAINING-JOB.json')['payload']
        meta: Dict[str, Any] = {
            'candidate_path': r.get('candidate_path'),
            'candidate_on_box': bool(r.get('candidate_on_box') or r.get('defer_candidate_upload')),
            'box_candidate_dir': r.get('box_candidate_dir'),
            'box_master_dir': r.get('box_master_dir'),
            'candidate_files': dict(r.get('candidate_files') or {}),
            'defer_candidate_upload': bool(r.get('defer_candidate_upload')),
            'candidate_master_path': r.get('candidate_master_path'),
            'candidate_master_digest': r.get('candidate_master_digest'),
            'candidate_sha256': r.get('candidate_sha256') or cid,
            'artifacts_uploaded': bool(r.get('artifacts_uploaded')),
            'optimizer_step': int(job['training']['optimizer_step']) + 1,
            'output_store_entry': self.store_entry,
            'pod': self.pod,
            'train_gpu': r.get('train_gpu'),
            'decided_at_mono': time.time(),
        }
        seed = r.get('live_apply_seed')
        if isinstance(seed, Mapping) and seed.get('candidate_verified') and seed.get('master_verified'):
            meta['live_apply'] = dict(seed)
            meta['candidate_on_box'] = False
            meta['defer_candidate_upload'] = False
        return ItemRecord(
            item_id=self.item_id, registration_index=0, epoch=int(self.sources.epoch),
            state=ItemState.DECIDED, bound_head=self.sources.head,
            bound_master_digest=self.sources.master_digest, candidate_id=cid,
            box_id=self.box_id, box_name=self.box_name, decision='green_fresh', z=FAKE_Z,
            fresh=True, paid=True, shadow=False, meta=meta,
        )

    def phase_apply_tail(self) -> None:
        self._advance(FakepassPhase.APPLY_TAIL)
        self.apply_result = run_apply_chain(
            mods=self._require_mods(), paths=self.paths, item=self.build_item(),
            head=self.sources.head, note=self._note,
        )
        if isinstance(self.pod, SeveredPod):
            if self.pod.ssh_attempts:
                _die('APPLY touched the severed box %d times' % self.pod.ssh_attempts)
            self.apply_result['apply_from_upload'] = True
            self.apply_result['ssh_attempts'] = 0
            self.pod.disarm()
        self._note('apply_tail', **self.apply_result)

    def phase_bench(self) -> None:
        self._advance(FakepassPhase.BENCH)
        cid = str(self.candidate_id)
        out = self.pod.run(
            "python3 - %d %s <<'PY'\n%s\nPY" % (
                VLLM_PORT, shlex.quote(json.dumps(BENCH_PROMPTS)), BENCH_SCRIPT.strip()),
            timeout=3600,
        )
        doc = json.loads(str(out).strip().splitlines()[-1])
        rows = list(doc.get('rows') or [])
        if not rows or any(not r.get('chars') for r in rows):
            _die('minimal bench: empty completions from served %s' % cid[:12])
        score = sum(1 for r in rows if r.get('ok')) / float(len(rows))
        landed = write_results(
            head=cid,
            result={
                'model_id': 'fakepass-%s' % cid[:16],
                'presence_penalty': 0.0,
                'results': [{
                    'benchmark': 'fakepass-minimal', 'score': score, 'n': len(rows),
                    'presence_penalty': 0.0, 'FAKE': True, 'served_model': doc.get('model'),
                    'rows': rows,
                }],
            },
            results_dir_path=str(self.paths['bench']),
            node=node_record(box_id=self.box_id, box_name=self.box_name,
                             gpu=self.rent_out.get('gpu'), gpu_count=8, rent_meta=self.rent_out),
        )
        self.bench = {'score': score, 'n': len(rows), 'landed': landed}
        self._note('bench', **self.bench)

    def phase_second_step(self) -> None:
        self._advance(FakepassPhase.SECOND_STEP)
        self.second_step = run_second_step(
            self.pod, candidate_id=str(self.candidate_id), receipt=self.receipt,
            paths=self.paths, epoch_dir=self.staged['epoch_dir'], note=self._note,
        )

    def phase_release(self) -> None:
        self._advance(FakepassPhase.RELEASE)
        self.release_box()
        self.gc_r2()

    def release_box(self) -> None:
        if self.released or self.rent is None or not self.box_id:
            return
        try:
            out = self.rent.release(self.box_id, reason='fakepass-done')
            self.released = True
            self._note('released', **out)
        except RentError as exc:
            self._note('release_error', error=str(exc)[:200])
            raise

    def gc_r2(self) -> None:
        if self.r2_gc.get('swept'):
            return
        gc: Dict[str, Any] = {}
        if self.candidate_id:
            gc['pending'] = AU.gc_pending_artifacts(str(self.candidate_id))
        gc['sweep'] = sweep_r2_prefix(self.run_prefix)
        gc['swept'] = True
        self.r2_gc = gc
        self._note('r2_gc', **{k: v for k, v in gc.items() if k != 'pending'})

    def phase_cleanup(self) -> None:
        self._advance(FakepassPhase.CLEANUP)
        self.stop_liveness()
        for proc in self._workers:
            if proc.poll() is None:
                os.kill(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
        if self.evidence_out is not None:
            dest = Path(self.evidence_out)
            assert_paths_outside_live({'evidence_out': dest}, self.live_roots)
            shutil.copytree(self.paths['logs'], dest / ('fakepass-%s' % self.run_id), dirs_exist_ok=True)
        cleanup_sandbox(self.paths)
        self.cleaned = True
        self.phase = FakepassPhase.DONE

    def _dispatch_phase(self, phase: FakepassPhase) -> None:
        if phase is FakepassPhase.ISOLATE:
            self.phase_isolate()
        elif phase is FakepassPhase.RENT:
            self.phase_rent()
        elif phase is FakepassPhase.DOWNLOAD:
            self.phase_download_train()
        elif phase is FakepassPhase.TRAIN:
            return
        elif phase is FakepassPhase.EARLY_UPLOAD:
            self.phase_early_upload()
        elif phase is FakepassPhase.HOLD_VLLM:
            self.phase_hold_vllm()
        elif phase is FakepassPhase.FAKE_RESULT:
            self.phase_fake_result()
        elif phase is FakepassPhase.APPLY_TAIL:
            self.phase_apply_tail()
        elif phase is FakepassPhase.BENCH:
            self.phase_bench()
        elif phase is FakepassPhase.SECOND_STEP:
            self.phase_second_step()
        elif phase is FakepassPhase.RELEASE:
            self.phase_release()
        elif phase is FakepassPhase.CLEANUP:
            self.phase_cleanup()
        elif phase is FakepassPhase.DONE:
            return
        else:
            _assert_never(phase)

    def report(self) -> Dict[str, Any]:
        return {
            'ok': True,
            'mode': 'real',
            'run_id': self.run_id,
            'live_sources': self.sources.as_dict(),
            'trainer_host': self.host_check,
            'box_id': self.box_id,
            'box_name': self.box_name,
            'gpu': self.rent_out.get('gpu'),
            'price_h': self.rent_out.get('price_h'),
            'candidate_id': self.candidate_id,
            'store_entry': self.store_entry,
            'early_upload': self.early_upload,
            'serve': self.serve,
            'eval_done': self.eval_done,
            'restart_result': self.restart_result or None,
            'fake_result': self.fake_result,
            'green_stamp': self.green_stamp,
            'apply_result': self.apply_result,
            'bench': self.bench,
            'second_step': self.second_step,
            'r2_gc': self.r2_gc,
            'sever_after_green': bool(self.sever_after_green),
            'promote_live': False,
            'root': str(self.paths['root']),
        }

    def teardown(self) -> List[str]:
        """Release the box and sweep R2 even on failure; always remove the sandbox."""
        errors: List[str] = []
        for step in (self.release_box, self.gc_r2):
            try:
                step()
            except Exception as exc:
                errors.append('%s: %s' % (step.__name__, str(exc)[:200]))
                print(json.dumps({
                    'component': 'pit-fakepass', 'level': 'error', 'event': 'teardown_error',
                    'step': step.__name__, 'box_id': self.box_id, 'box_name': self.box_name,
                    'r2_prefix': self.run_prefix, 'error': str(exc)[:200],
                }), file=sys.stderr, flush=True)
        if not self.cleaned:
            self._dispatch_phase(FakepassPhase.CLEANUP)
        return errors

    def run(self) -> Dict[str, Any]:
        with swapped_environ(self.env):
            try:
                for phase in (
                    FakepassPhase.ISOLATE, FakepassPhase.RENT, FakepassPhase.DOWNLOAD,
                    FakepassPhase.EARLY_UPLOAD, FakepassPhase.HOLD_VLLM, FakepassPhase.FAKE_RESULT,
                    FakepassPhase.APPLY_TAIL, FakepassPhase.BENCH, FakepassPhase.SECOND_STEP,
                    FakepassPhase.RELEASE,
                ):
                    self._dispatch_phase(phase)
                out = self.report()
                _wj(self.paths['root'] / 'PLAN.json', out)
            except BaseException as exc:
                self._note('failed', error=str(exc)[:400], failed_phase=self.phase.value)
                raise
            finally:
                errors = self.teardown()
        if errors:
            _die('fakepass teardown failed: %s' % errors)
        return out


def read_command_lines(log_path: Path) -> List[str]:
    if not Path(log_path).is_file():
        return []
    return [ln for ln in Path(log_path).read_text().splitlines() if '"box_cmd"' in ln]


# ---------------------------------------------------------------------------
# Apply chain (LiveApplicator → pe_apply/pe_live → lifecycle → record_applied)
# ---------------------------------------------------------------------------
def emulate_lifecycle_epoch(
    mods: LiveModules,
    *,
    life_root: Path,
    epoch: int,
) -> Dict[str, Any]:
    """Drive the real install_pe wrapper for one sandbox lifecycle epoch and finalize it."""

    class SandboxLifecycle:
        def _training(self, directory, opening, dispatch, current, now):
            return None

    prior = SandboxLifecycle._training

    def _training(self, *a):
        return prior(self, *a)

    SandboxLifecycle._training = _training
    mods.lifecycle.install_pe(SandboxLifecycle)
    ep = life_root / 'epochs' / str(int(epoch))
    ep.mkdir(parents=True, exist_ok=True)
    cur = _rj(life_root / 'CURRENT.json')
    r = SandboxLifecycle()._training(ep, None, None, cur, time.time())
    if not isinstance(r, Mapping) or r.get('status') != 'updated':
        _die('lifecycle did not offer the GREEN apply: %r' % (r,))
    _wj(life_root / 'CURRENT.json', {
        'fakepass_sandbox': True,
        'epoch_index': int(epoch) + 1,
        'checkpoint': r['checkpoint'],
        'optimizer_step': r['optimizer_step'],
        'checkpoint_authorization': r['checkpoint_authorization'],
    })
    _wj(ep / 'FINALIZED.json', {'payload': {
        'input_checkpoint': cur['checkpoint']['id'],
        'result': {
            'status': 'updated',
            'checkpoint': r['checkpoint'],
            'optimizer_step': r['optimizer_step'],
            'checkpoint_authorization': r['checkpoint_authorization'],
        },
    }})
    confirmed = mods.lifecycle.confirm_applied(life_root, _rj(life_root / 'CURRENT.json'))
    if not confirmed:
        _die('lifecycle did not confirm the applied GREEN')
    return {'lifecycle_epoch': int(epoch), 'confirmed': confirmed, 'checkpoint': r['checkpoint']['id']}


def run_apply_chain(
    *,
    mods: LiveModules,
    paths: Mapping[str, Path],
    item: ItemRecord,
    head: str,
    note: Callable[..., None],
    signer: Any = None,
) -> Dict[str, Any]:
    cid = str(item.candidate_id)
    applicator = LiveApplicator(
        config=load_pit_config(pit_flag_doc()['parallel_eval']),
        accept_pit_decision=mods.apply.accept_pit_decision,
        verify_uploads=pe_verify.verify_uploads,
        upload_master=pe_box.upload_master,
        upload_candidate=pe_box.upload_candidate,
        checkpoint_signer=signer if signer is not None else FixtureRootSigner(),
    )
    res = applicator.apply(item, scored={'decision': 'green', 'evaluated': {'z': FAKE_Z, 'FAKE': True}})
    if not (res.applied and res.master_verified and res.candidate_verified and res.pe_apply_accepted):
        _die('LiveApplicator refused the GREEN: %s' % res.as_dict())
    ready = paths['life_state'] / 'apply' / ('e%d.json' % item.epoch)
    if not ready.is_file() or _rj(ready).get('state') != 'ready':
        _die('pe_live apply_tick did not stage %s: %s' % (ready.name, res.detail))
    note('apply_ready', apply=str(ready), detail=res.detail)
    lifecycle = emulate_lifecycle_epoch(mods, life_root=paths['life'], epoch=int(item.epoch) + 1)
    rec = drive_cutover_apply(pe_live=mods.live, alert_fn=cutover_apply_alert)
    if not {int(item.epoch), int(item.epoch) + 1} & set(rec.get('recorded_epochs') or []):
        _die('record_applied did not record the GREEN (epoch %d): %s' % (item.epoch, rec))
    if not confirm_head_advanced(cid, pe_apply=mods.apply):
        _die('sandbox HEAD did not advance to %s' % cid[:12])
    chain = _rj(paths['pe'] / 'HEAD-CHAIN.json')
    if not any(a.get('new_head') == cid for a in chain.get('applied') or []):
        _die('sandbox HEAD-CHAIN lacks %s' % cid[:12])
    _wj(paths['head_chain'] / 'HEAD-CHAIN.json', chain)
    hash_file = paths['hash_sync'] / 'fp32-expected-master.hash'
    last = paths['hash_sync'] / 'LAST-SYNC.json'
    if not hash_file.is_file() or not last.is_file():
        alert = paths['hash_sync'] / 'ALERT.json'
        _die('sandbox hash-sync did not run: %s' % (alert.read_text()[:400] if alert.is_file() else 'no ALERT'))
    synced = hash_file.read_text().strip()
    if synced != item.meta.get('candidate_master_digest'):
        _die('hash-sync wrote %s, expected candidate master %s' % (synced[:12], str(item.meta.get('candidate_master_digest'))[:12]))
    promoted = AU.public_master_index_dir() / ('%s.json' % cid)
    return {
        'applied': True,
        'head_advanced': True,
        'prev_head': head,
        'new_head': cid,
        'promote_live': False,
        'sandbox_only': True,
        'verify': res.detail.get('verify'),
        'lifecycle': lifecycle,
        'recorded_epochs': rec.get('recorded_epochs'),
        'hash_sync': synced,
        'promoted_index': str(promoted),
    }


# ---------------------------------------------------------------------------
# Second step from the sandbox head on the same box
# ---------------------------------------------------------------------------
def run_second_step(
    P: Any,
    *,
    candidate_id: str,
    receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
    epoch_dir: Path,
    note: Callable[..., None],
) -> Dict[str, Any]:
    cid = candidate_id
    promoted_path = AU.public_master_index_dir() / ('%s.json' % cid)
    published = _rj(promoted_path)
    expected = (paths['hash_sync'] / 'fp32-expected-master.hash').read_text().strip()
    validate_promoted_index(published, candidate_id=cid, live_prefix=AU.live_fp32_r2_prefix(cid),
                            master_digest=expected)
    # Eval and bench are finished; the step needs the GPUs and a cold FP32 store.
    pe_box.free_gpus(P, also_vllm=True)
    P.run('rm -rf %s/* %s/%s %s/%s.descriptor.json %s/in2 && mkdir -p %s/in2/actor && echo OK' % (
        pe_box.FP32, pe_box.KDIR, cid, pe_box.KDIR, cid, pe_box.PST, pe_box.PST), timeout=600)
    files = {n: (v['sha256'] if isinstance(v, Mapping) else v) for n, v in dict(receipt['candidate_files']).items()}
    pending = AU.pending_r2_prefix(cid)
    urls = pe_box.presign_keys({n: '%s/model/%s' % (pending, n) for n in files})
    bf16_s, bf16_b = pe_box.fetch(P, cid, files, urls, pe_box.KDIR)
    P.run(
        'mkdir -p {A} && for f in $(ls {K}/{c}); do ln -f {K}/{c}/$f {A}/$f 2>/dev/null || '
        'cp {K}/{c}/$f {A}/$f; done; ls {A} | wc -l'.format(
            A=pe_box.PST + '/in2/actor/' + cid, K=pe_box.KDIR, c=cid),
        timeout=600,
    )
    master = dict(pe_box.fetch_master_on_box(
        P, head=cid, expected_master=expected, head_bf16=str(published['bf16_sha256']), published=published,
    ))
    job = _rj(epoch_dir / 'TRAINING-JOB.json')['payload']
    step = int(job['training']['optimizer_step']) + 1
    for name, value in (
        ('head.descriptor', {'id': cid, 'files': files}),
        ('manifest', job['manifest']),
        ('train_config', job['training']['config']),
        ('meta', {'optimizer_step': step, 'fakepass_second_step': True}),
    ):
        pe_box.put(P, '%s/in2/%s.json' % (pe_box.PST, name), json.dumps(value).encode(), '600')
    pe_box.put(P, pe_box.PST + '/fakepass_step2_fp32.py', STEP2_SRC.read_bytes(), '700')
    out = P.run(
        'cd {S} && env -u AFFINE_FP32_BOOTSTRAP_OVERRIDE AFFINE_FP32_EXPECTED_MASTER_HASH={X} '
        'CUDA_VISIBLE_DEVICES=0 PYTHONPATH={S} TOKENIZERS_PARALLELISM=false '
        'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {V} -B {P}/fakepass_step2_fp32.py {P}/in2 '
        '> {P}/step2.log 2>&1; echo rc=$?; grep -E "^\\{{" {P}/step2.log | tail -40; tail -20 {P}/step2.log'.format(
            S=pe_box.BOX_SOURCE, X=expected, V=pe_box.BOX_VENV, P=pe_box.PST),
        timeout=7200, check=False,
    )
    events: Dict[str, Any] = {}
    for line in str(out).splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict) and row.get('event'):
            events[row['event']] = row
    prepared = events.get('fp32-prepared') or {}
    if 'rc=0' not in str(out) or not prepared:
        _die('second step failed: %s' % str(out)[-1500:])
    if prepared.get('source') != 'disk':
        _die('second step fp32 source %r (must load the promoted master from disk)' % prepared.get('source'))
    if prepared.get('input_master_digest') != expected or prepared.get('input_bf16_sha256') != published['bf16_sha256']:
        _die('second step loaded %s/%s, promoted index says %s/%s' % (
            str(prepared.get('input_bf16_sha256'))[:12], str(prepared.get('input_master_digest'))[:12],
            str(published['bf16_sha256'])[:12], expected[:12]))
    if 'root-admission-checked' not in events or prepared.get('host') != 'host-v834e':
        _die('second step did not run on admitted host-v834e')
    result = {
        'head': cid,
        'bf16_fetch': {'seconds': bf16_s, 'bytes': bf16_b, 'source': 'r2-pending'},
        'fetch_master_on_box': master,
        'meta_json': published['meta_json'],
        'fp32_prepared': prepared,
        'optimizer_step': step,
        'stops_before_grpo': 'no signed dispatch exists for the sandbox head',
    }
    note('second_step', **result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) == 2 and args[0] == '--host-worker':
        return host_worker_main(Path(args[1]))
    print('usage: python -m ops.parallel_eval.pit.fakepass_real --host-worker STATE.json', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
