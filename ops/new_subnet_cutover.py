"""Explicit operator cutover AFTER successful mock; preserves legacy payout guard."""
import argparse
import json
import subprocess
from pathlib import Path


def systemctl(*args):
    return subprocess.run(['systemctl', '--user', *args], capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', default='state/live-chain')
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--mock-evidence', type=Path)
    args = parser.parse_args()
    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker = Path.home()/'.local/state/affine-transition/active'
    names = ['affine-transition-weights', 'affine-hourly-burn']
    inventory = {n: {suffix: systemctl('is-active', f'{n}.{suffix}').stdout.strip()
                     for suffix in ('timer', 'service')} for n in names}
    if not args.activate:
        print(json.dumps({'status': 'inspection_only', 'legacy_guard': marker.exists(), 'writers': inventory}))
        return
    if not args.mock_evidence or not args.mock_evidence.is_file():
        raise RuntimeError('successful mock evidence file required')
    evidence = json.loads(args.mock_evidence.read_text())
    if evidence.get('success') is not True:
        raise RuntimeError('mock evidence must explicitly indicate success=true')
    if not marker.exists():
        raise RuntimeError('original validator guard must already exist before cutover')
    (state/'cutover-before.json').write_text(json.dumps(inventory, indent=2)+'\n')
    for name in names:
        for suffix in ('timer', 'service'):
            result = systemctl('stop', f'{name}.{suffix}')
            if result.returncode and 'not loaded' not in result.stderr and 'not found' not in result.stderr:
                raise RuntimeError(f'could not stop {name}.{suffix}')
        systemctl('disable', f'{name}.timer')
        if systemctl('is-active', f'{name}.service').returncode == 0:
            raise RuntimeError('prior writer remains active')
    (state/'writer.enabled').write_text('Live epoch writer owns SN120 payouts.\n')
    print(json.dumps({'status': 'single_writer_enabled', 'guard_preserved': True}))


if __name__ == '__main__':
    main()
