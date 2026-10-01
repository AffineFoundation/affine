"""Hourly live payouts from controller-authenticated reports; dry-run by default."""
import argparse
import json
import signal
import time
from pathlib import Path

from subnet.chain import ChainAdapter, hourly_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', default='state/live-chain')
    parser.add_argument('--reports', default='state/finalized-reports.json')
    parser.add_argument('--registrations', default='state/epoch-registrations.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--inventory', action='store_true')
    args = parser.parse_args()
    adapter = ChainAdapter(args.state_dir)
    if args.inventory:
        print(json.dumps({'registrations': adapter.registrations()}, indent=2))
        return
    reports_path = Path(args.reports)
    registrations_path = Path(args.registrations)
    if not reports_path.exists() or not registrations_path.exists():
        print(json.dumps({'status': 'waiting_for_verified_live_epochs'}))
        return
    # These files are private local controller outputs, not miner-supplied JSON.
    reports = json.loads(reports_path.read_text())
    registrations = json.loads(registrations_path.read_text())
    end = int(time.time()) // 3600 * 3600
    points = hourly_points(reports, end)
    print(json.dumps(adapter.submit_hour(points, registrations, end, args.execute)))


if __name__ == '__main__':
    def timeout(*_):
        raise TimeoutError('chain worker exceeded 12 minutes')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(720)
    main()
