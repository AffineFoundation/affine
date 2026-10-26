#!/usr/bin/env python3
"""Stage a MiMo source release locally; never activate public discovery."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subnet.mimo_release import build, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--settings', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--include', action='append', default=[])
    args = parser.parse_args()
    record = build(args.source, args.output,
                   settings=json.loads(args.settings.read_bytes()), extra=args.include)
    assert verify(args.output) == record
    print(json.dumps(dict(status=record['status'], source_sha256=record['source_sha256'],
                          files=len(record['source']), output=str(args.output))))


if __name__ == '__main__':
    main()
