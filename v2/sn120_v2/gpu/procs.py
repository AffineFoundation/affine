"""Child processes that hold GPUs, and the proof that they let go.

All GPU work runs in child processes (one training job, one vLLM server).
A GPU is free again only when its process is gone, so ``Child.stop`` ends
the whole process group and ``wait_gpus_free`` checks the driver's report.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from ..errors import InfraError


class Process(Protocol):
    def alive(self) -> bool: ...

    def wait(self, timeout_s: float | None = None) -> int | None:
        """Exit code, or None when the process still runs after ``timeout_s``."""
        ...

    def stop(self) -> None: ...


Spawn = Callable[[Sequence[str], Mapping[str, str], Path], Process]


class Child:
    """A real child process in a process group of its own. Its output goes to ``log``."""

    def __init__(self, argv: Sequence[str], env: Mapping[str, str], log: Path, grace_s: float = 30.0):
        self.grace_s = grace_s
        with open(log, "ab") as out:
            self.proc = subprocess.Popen(list(argv), env=dict(env), stdout=out, stderr=subprocess.STDOUT, start_new_session=True)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def wait(self, timeout_s: float | None = None) -> int | None:
        try:
            return self.proc.wait(timeout_s)
        except subprocess.TimeoutExpired:
            return None

    def stop(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                break
            if self.wait(self.grace_s) is not None:
                break
        self.proc.wait()


def gpu_env(gpu_ids: Iterable[int], environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The parent's environment, with only the given GPUs visible to the child."""
    env = dict(os.environ if environ is None else environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpu_ids)
    return env


def stop_all(procs: list[Process]) -> None:
    """Stop every process, then forget it. A failure to stop one does not skip the others."""
    errors = []
    for proc in procs:
        try:
            proc.stop()
        except Exception as exc:  # noqa: BLE001 - every child must get its stop call
            errors.append(exc)
    procs.clear()
    if errors:
        raise InfraError(f"could not stop {len(errors)} GPU process(es): {errors[0]}")


def busy_gpus(report: str, limit_mib: int) -> list[int]:
    """GPU indexes that still hold more than ``limit_mib`` of memory.

    ``report`` is the output of
    ``nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits``.
    A line that cannot be read counts as busy: the check fails closed.
    """
    busy = []
    for position, line in enumerate(line for line in report.splitlines() if line.strip()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            busy.append(position)
        elif int(parts[1]) > limit_mib:
            busy.append(int(parts[0]))
    return busy


def nvidia_smi() -> str:
    argv = ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise InfraError(f"cannot read the GPU state: {exc}") from exc


def wait_gpus_free(
    limit_mib: int,
    timeout_s: float,
    report: Callable[[], str] = nvidia_smi,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Return when no GPU holds more than ``limit_mib``. After ``timeout_s`` it is an ``InfraError``."""
    waited = 0.0
    while busy := busy_gpus(report(), limit_mib):
        if waited >= timeout_s:
            raise InfraError(f"GPUs {busy} are still in use; evaluation must not start")
        sleep(5.0)
        waited += 5.0
