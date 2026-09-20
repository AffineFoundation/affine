#!/bin/bash
# Point this clone's git hooks at ops/hooks (pre-commit + pre-push secret
# scan). Idempotent; called by setup.sh. Run by hand after cloning elsewhere:
#   bash ops/hooks/install.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
chmod +x ops/hooks/pre-commit ops/hooks/pre-push ops/hooks/secret_scan.py
git config core.hooksPath ops/hooks
echo "git hooks: core.hooksPath=ops/hooks (pre-commit + pre-push secret scan)"
