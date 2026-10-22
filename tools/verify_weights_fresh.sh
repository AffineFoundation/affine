#!/usr/bin/env bash
# READ-ONLY SN120 weight freshness check. Exit 0 OK, 1 stale, 2 error. See verify_weights_fresh.py.
exec timeout 240 /home/const/subnet120-rewrite/.venv/bin/python -I -B /home/const/subnet120-rewrite/tools/verify_weights_fresh.py "$@"
