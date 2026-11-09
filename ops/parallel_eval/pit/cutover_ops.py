"""PIT live cutover enable/revert helpers (bash scripts + unit tests)."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

from . import PREFIX
from .pe_rent import is_pit_name

BANNED_BASELINE_KEYS = frozenset(('indices', 'raw_seed', 'secret_draw', 'presign'))
CLAIM_GLOB = 'PIT-CLAIM-e*.json'


class TrainerBusyError(RuntimeError):
    """TRAINING-JOB exists without TRAINING-RESULT at latest epoch."""


class MidProbeError(RuntimeError):
    """A live pe_controller lane is mid-probe / mid-serve — refuse cutover stop."""


# STATUS.json phases that mean a lane still holds a box / is mid-eval.
_MID_PROBE_STATUSES = frozenset({
    'renting', 'booting', 'fetching', 'serving', 'probing', 'starting', 'warm-wait',
})


def cutover_lock_path(pe_flag: Path, pe_state: Optional[Path] = None) -> Path:
    if pe_state is not None:
        return pe_state / 'cutover.lock'
    return pe_flag.with_suffix(pe_flag.suffix + '.lock')


def find_latest_pre_pit_backup(pe_flag: Path) -> Optional[Path]:
    parent = pe_flag.parent
    stem = pe_flag.name
    pattern = re.compile(re.escape(stem) + r'\.bak-pre-pit-(\d{8}T\d{6}Z)$')
    candidates: List[Tuple[str, Path]] = []
    for p in parent.iterdir():
        m = pattern.match(p.name)
        if m:
            candidates.append((m.group(1), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def backup_pre_pit(pe_flag: Path, *, now: Optional[datetime] = None) -> Path:
    ts = (now or datetime.now(timezone.utc)).strftime('%Y%m%dT%H%M%SZ')
    dest = pe_flag.parent / ('%s.bak-pre-pit-%s' % (pe_flag.name, ts))
    shutil.copy2(pe_flag, dest)
    return dest


def _detect_indent(text: str) -> Tuple[int, bool]:
    for line in text.splitlines():
        m = re.match(r'^(\s+)"', line)
        if m:
            sp = m.group(1)
            if '\t' in sp:
                return 1, True
            n = len(sp)
            if n >= 2 and n % 2 == 0:
                return 2, False
            return n, False
    return 2, False


def _dumps_preserve_style(doc: Mapping[str, Any], original_text: str) -> str:
    indent, use_tabs = _detect_indent(original_text)
    if use_tabs:
        body = json.dumps(doc, indent=1, sort_keys=True)
        body = body.replace('  ', '\t')
    else:
        body = json.dumps(doc, indent=indent, sort_keys=True)
    if original_text.endswith('\n'):
        body += '\n'
    return body


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    os.chmod(tmp, path.stat().st_mode if path.exists() else 0o644)
    tmp.replace(path)


def read_flag(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def patch_pit_block(
    doc: MutableMapping[str, Any],
    pit_updates: Mapping[str, Any],
    *,
    pe_enabled: Optional[bool] = None,
) -> None:
    pe = doc.setdefault('parallel_eval', {})
    if pe_enabled is not None:
        pe['enabled'] = pe_enabled
    pit = pe.setdefault('pit', {})
    pit.update(dict(pit_updates))


def write_flag_patched(pe_flag: Path, pit_updates: Mapping[str, Any], *, pe_enabled: Optional[bool] = None) -> None:
    raw = pe_flag.read_text()
    doc = json.loads(raw)
    patch_pit_block(doc, pit_updates, pe_enabled=pe_enabled)
    _atomic_write_text(pe_flag, _dumps_preserve_style(doc, raw))


def restore_flag_from_backup(pe_flag: Path, backup: Path) -> None:
    shutil.copy2(backup, pe_flag)


def trainer_idle_check(life_root: Path) -> Optional[int]:
    """Return latest epoch number when idle; None when no epochs. Raises TrainerBusyError if mid-turn."""
    epochs = life_root / 'epochs'
    if not epochs.is_dir():
        return None
    latest: Optional[Path] = None
    for p in sorted(epochs.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else -1):
        if p.name.isdigit():
            latest = p
    if latest is None:
        return None
    job = latest / 'TRAINING-JOB.json'
    res = latest / 'TRAINING-RESULT.json'
    if job.exists() and not res.exists():
        raise TrainerBusyError('trainer mid-turn at e%s (JOB without RESULT)' % latest.name)
    return int(latest.name)


def lane_mid_probe_check(pe_state: Path) -> List[str]:
    """Return list of mid-probe job descriptions; empty when safe to stop pe_controller.

    Inspects ``PE_STATE/jobs/*/STATUS.json`` (and CTL.json state=running). Raises
    nothing — callers decide wait vs abort from the returned list.
    """
    busy: List[str] = []
    jobs = pe_state / 'jobs'
    if not jobs.is_dir():
        return busy
    for jd in sorted(jobs.iterdir()):
        if not jd.is_dir():
            continue
        st = {}
        try:
            st = json.loads((jd / 'STATUS.json').read_text())
        except Exception:
            st = {}
        status = str(st.get('status') or st.get('state') or '').lower()
        if status in _MID_PROBE_STATUSES:
            busy.append('%s status=%s' % (jd.name, status))
            continue
        try:
            ctl = json.loads((jd / 'CTL.json').read_text())
        except Exception:
            ctl = {}
        if str(ctl.get('state') or '').lower() == 'running':
            busy.append('%s ctl=running' % jd.name)
    return busy


def assert_no_mid_probe(pe_state: Path) -> None:
    busy = lane_mid_probe_check(pe_state)
    if busy:
        raise MidProbeError('live lane mid-probe; refuse stopping pe_controller: %s' % '; '.join(busy[:8]))


def warm_off_path(pe_state: Path) -> Path:
    return pe_state / 'WARM-OFF'


def write_warm_off(pe_state: Path) -> Path:
    pe_state.mkdir(parents=True, exist_ok=True)
    p = warm_off_path(pe_state)
    if not p.exists():
        p.write_text('{}\n')
    return p


def remove_warm_off(pe_state: Path) -> None:
    p = warm_off_path(pe_state)
    if p.exists():
        p.unlink()


def pit_claim_roots(
    pe_state: Optional[Path],
    pe_life_state: Optional[Path],
    pit_state: Optional[Path],
    claims_dir: Optional[Path],
) -> List[Path]:
    roots: List[Path] = []
    for p in (pe_state, pe_life_state, pit_state, claims_dir):
        if p is not None:
            roots.append(p)
    out: List[Path] = []
    seen: set = set()
    for r in roots:
        key = str(r.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def clear_pit_claims(roots: List[Path]) -> List[str]:
    removed: List[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.glob(CLAIM_GLOB):
            try:
                p.unlink()
                removed.append(str(p))
            except OSError:
                pass
    return removed


def release_affine_pit_boxes() -> Dict[str, Any]:
    """Best-effort release of affine-pe-pit-* rentals via host pe_rent when importable."""
    out: Dict[str, Any] = dict(released=[], warnings=[])
    listings: List[Mapping[str, Any]] = []
    release_fn = None
    try:
        from .pe_rent import RentService, bind_host_rent_service, is_pit_name

        rent = bind_host_rent_service(RentService())
        list_fn = rent.list_pods
        release_host = rent.host_release
        if list_fn is not None:
            raw = list_fn()
            if isinstance(raw, list):
                listings = [x for x in raw if is_pit_name((x.get('name') or x.get('pod_name') or ''))]
        if release_host is None:
            out['warnings'].append('host pe_rent.release unavailable; skipped box release')
            return out
        for row in listings:
            name = row.get('name') or row.get('pod_name') or ''
            bid = str(row.get('id') or name)
            try:
                if rent.host_release is not None:
                    rent.release(bid, reason='cutover-revert')
                else:
                    release_host(bid, reason='cutover-revert')
                out['released'].append(bid)
            except Exception as exc:
                out['warnings'].append('release %s: %s' % (bid, str(exc)[:120]))
    except Exception as exc:
        out['warnings'].append('pe_rent module unavailable: %s' % str(exc)[:160])
    # Fallback: lium JSON listing when host module missing
    if not listings and not out['released']:
        lium = os.environ.get('LIUM') or os.path.expanduser('~/.lium/bin/lium')
        if Path(lium).is_file():
            try:
                proc = subprocess.run(
                    [lium, 'ps', '--format', 'json'],
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                if proc.returncode == 0:
                    pods = json.loads(proc.stdout)
                    for row in pods:
                        name = row.get('name') or ''
                        if not is_pit_name(name):
                            continue
                        bid = str(row.get('id') or name)
                        try:
                            subprocess.run(
                                [lium, 'release', bid],
                                capture_output=True,
                                text=True,
                                timeout=120,
                                check=False,
                            )
                            out['released'].append(bid)
                        except Exception as exc:
                            out['warnings'].append('lium release %s: %s' % (bid, str(exc)[:120]))
                else:
                    out['warnings'].append('lium ps failed (rc=%s)' % proc.returncode)
            except Exception as exc:
                out['warnings'].append('lium fallback: %s' % str(exc)[:160])
        else:
            out['warnings'].append('no lium binary; skipped pit box release')
    return out


def enable_cutover_on_flag(pe_flag: Path) -> Dict[str, Any]:
    pe_flag = Path(pe_flag)
    doc = read_flag(pe_flag)
    pit = (doc.get('parallel_eval') or {}).get('pit') or {}
    write_flag_patched(
        pe_flag,
        {
            'enabled': True,
            'mode': 'live',
            'cutover': True,
            'revert': False,
            'max_boxes': int(pit.get('max_boxes') or 5),
            'master_upload': pit.get('master_upload') or 'after_green',
            'root_mode': 'delegated_host',
        },
        pe_enabled=True,
    )
    return read_flag(pe_flag)


def revert_cutover_on_flag(pe_flag: Path, *, restore_from: Optional[Path] = None) -> Dict[str, Any]:
    pe_flag = Path(pe_flag)
    if restore_from is not None and Path(restore_from).is_file():
        restore_flag_from_backup(pe_flag, Path(restore_from))
        doc = read_flag(pe_flag)
        pe = doc.setdefault('parallel_eval', {})
        pit = pe.setdefault('pit', {})
        pit.update(enabled=False, cutover=False, revert=True)
        raw = pe_flag.read_text()
        _atomic_write_text(pe_flag, _dumps_preserve_style(doc, raw))
        return doc
    doc = read_flag(pe_flag)
    pit = (doc.get('parallel_eval') or {}).get('pit') or {}
    write_flag_patched(
        pe_flag,
        {
            'enabled': False,
            'mode': pit.get('mode') or 'live',
            'cutover': False,
            'revert': True,
        },
    )
    return read_flag(pe_flag)


def _cli() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    pe_flag = Path(os.environ['PE_FLAG']) if os.environ.get('PE_FLAG') else None
    if cmd == 'trainer-idle':
        life = Path(os.environ.get('LIFE_ROOT', os.environ.get('PE_LIFE', '/tmp/no-life')))
        try:
            ep = trainer_idle_check(life)
        except TrainerBusyError as ex:
            print(ex, flush=True)
            return 3
        if ep is None:
            print('no lifecycle epochs found; continuing', flush=True)
        else:
            print('trainer idle at e%s' % ep, flush=True)
        return 0
    if cmd == 'mid-probe-check':
        pe_state = Path(os.environ.get('PE_STATE') or '/tmp/no-pe-state')
        try:
            assert_no_mid_probe(pe_state)
        except MidProbeError as ex:
            print(ex, flush=True)
            return 3
        print('no mid-probe lanes under %s/jobs' % pe_state, flush=True)
        return 0
    if cmd == 'backup-flag' and pe_flag:
        bak = backup_pre_pit(pe_flag)
        print(str(bak))
        return 0
    if cmd == 'enable-patch' and pe_flag:
        enable_cutover_on_flag(pe_flag)
        print('wrote pit live cutover into', pe_flag)
        return 0
    if cmd == 'revert-patch' and pe_flag:
        restore = os.environ.get('RESTORE_FROM')
        revert_cutover_on_flag(pe_flag, restore_from=Path(restore) if restore else None)
        print('pit.revert=true, enabled=false in', pe_flag)
        return 0
    if cmd == 'clear-claims':
        roots = pit_claim_roots(
            Path(os.environ['PE_STATE']) if os.environ.get('PE_STATE') else None,
            Path(os.environ['PE_LIFE_STATE']) if os.environ.get('PE_LIFE_STATE') else None,
            Path(os.environ['PIT_STATE']) if os.environ.get('PIT_STATE') else None,
            Path(os.environ['CLAIMS_DIR']) if os.environ.get('CLAIMS_DIR') else None,
        )
        removed = clear_pit_claims(roots)
        print('removed %d claim file(s)' % len(removed))
        for p in removed:
            print('  ', p)
        return 0
    if cmd == 'release-pit-boxes':
        out = release_affine_pit_boxes()
        print(json.dumps(out, indent=2))
        for w in out.get('warnings') or []:
            print('warning:', w, file=sys.stderr)
        return 0
    print(
        'usage: cutover_ops.py trainer-idle|mid-probe-check|backup-flag|enable-patch|'
        'revert-patch|clear-claims|release-pit-boxes',
        file=sys.stderr,
    )
    return 2


if __name__ == '__main__':
    raise SystemExit(_cli())
