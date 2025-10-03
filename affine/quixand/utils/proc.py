from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RunOutput:
	stdout: bytes
	stderr: bytes
	exit_code: int
	duration_s: float


def run_capture(cmd: List[str], timeout: Optional[int] = None) -> RunOutput:
	start = time.monotonic()
	proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
	dur = time.monotonic() - start
	return RunOutput(stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode, duration_s=dur)


