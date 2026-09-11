"""Interpreter start-up hook for the verifiers eval subprocess.

The runner prepends this directory to the eval subprocess's PYTHONPATH when
the policy has a loop guard, so `site` imports this module before the eval
CLI starts. It installs the `loop_guard` @stop on verifiers' Task base class
(rollouts.loopguard). Any process without ROLLOUTS_LOOP_GUARD_REPEATS in its
environment, or without verifiers importable, is left untouched.
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
