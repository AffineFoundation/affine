"""Interpreter start-up hook for the verifiers eval subprocess.

The runner prepends this directory to the eval subprocess's PYTHONPATH, so
`site` imports this module before the eval CLI starts.

* ROLLOUTS_LOOP_GUARD_REPEATS: install the `loop_guard` @stop
  (rollouts.loopguard). Unset or 0 leaves the guard off.
* ROLLOUTS_BLOCK_UPSTREAM=1: block GitHub on the agent phase and strip git
  metadata after setup (rollouts.upstream_guard).

A missing verifiers install, or any other failure, is logged and ignored.
The eval process still starts.
"""

import os
import sys

if os.environ.get("ROLLOUTS_LOOP_GUARD_REPEATS", "0") not in ("", "0"):
    try:
        from rollouts.loopguard import install_verifiers_stop

        install_verifiers_stop()
    except Exception as exc:  # never break the interpreter that hosts us
        print(f"[loop_guard] sitecustomize: not installed ({exc!r})",
              file=sys.stderr)

if os.environ.get("ROLLOUTS_BLOCK_UPSTREAM") == "1":
    try:
        from rollouts.upstream_guard import install

        install()
    except Exception as exc:
        print(f"[upstream_guard] sitecustomize: not installed ({exc!r})",
              file=sys.stderr)
