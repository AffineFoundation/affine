"""Secret-free, resumable dispatch and collection of a Generator scorer.

Only fixed SSH/scp/tar commands are available.  The private job is sent to the
pod; no credentials or local environment are forwarded.  A local timeout is
never evidence that the detached remote scorer failed or did not start.
"""

from __future__ import annotations

import fcntl
import io
import json
import math
import os
import shlex
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..canonical import canonical_json, sha256_hex
from ..errors import ConfigError, ContractError, InfraError
from .epoch import epoch_id

REMOTE_ROOT = "/root/gen"
REMOTE_PYTHONPATH = "/root/gen/src"
SCORER_COMMAND = (
    "/root/venv/bin/python -m sn120_v2.generator.scorer run "
    "/root/gen/jobs/{epoch}/job.json"
)
DEFAULT_TIMEOUT_S = 30.0
HEAD_PREPARE_TIMEOUT_S = 900  # ~19 GB public checkpoint plus eight-server restart
MAX_OUTPUT_FILES = 100_000
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024 * 1024
MAX_ARCHIVE_STREAM_BYTES = MAX_ARCHIVE_BYTES + 1024 * 1024


@dataclass(frozen=True, slots=True)
class SSHConfig:
    """Explicit endpoint and local work directory, with no implicit identity."""

    host: str
    port: int
    identity_file: Path = field(repr=False)
    work_dir: Path
    user: str = "root"
    timeout_s: float = DEFAULT_TIMEOUT_S

    def __post_init__(self) -> None:
        # Restrict the host rather than trying to make every possible ssh/scp
        # destination grammar safe.  DNS names and IPv4 are enough for a pod.
        if not isinstance(self.host, str) or not self.host:
            raise ConfigError("Generator SSH host must be nonempty")
        allowed = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
        if not allowed.issuperset(self.host) or self.host.startswith(("-", ".")):
            raise ConfigError("Generator SSH host must be a DNS name or IPv4 address")
        if not isinstance(self.user, str) or not self.user or self.user != self.user.strip():
            raise ConfigError("Generator SSH user must be a nonempty trimmed string")
        if any(c.isspace() or ord(c) < 32 for c in self.user) or self.user.startswith("-") or "\x00" in self.user:
            raise ConfigError("Generator SSH user contains unsafe characters")
        if "@" in self.user or ":" in self.user:
            raise ConfigError("Generator SSH user must not contain an address")
        if type(self.port) is not int or not 1 <= self.port <= 65_535:
            raise ConfigError("Generator SSH port must be an integer in 1..65535")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ConfigError("Generator SSH timeout must be a positive finite number of seconds")
        try:
            identity = Path(self.identity_file).expanduser().resolve(strict=True)
            work = Path(self.work_dir).expanduser().resolve()
        except (OSError, TypeError, ValueError) as exc:
            raise ConfigError(f"invalid Generator SSH paths: {exc}") from exc
        if not identity.is_file():
            raise ConfigError("Generator SSH identity_file must be an existing file")
        object.__setattr__(self, "identity_file", identity)
        object.__setattr__(self, "work_dir", work)
        object.__setattr__(self, "timeout_s", float(self.timeout_s))

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, base_dir: str | Path | None = None
    ) -> "SSHConfig":
        """Load host, port, identity_file/key_path and work_dir from JSON data."""
        if not isinstance(value, Mapping):
            raise ConfigError("Generator SSH config must be a JSON object")
        allowed = {"host", "port", "identity_file", "key_path", "work_dir", "user", "timeout_s", "helpers"}
        unknown = set(value) - allowed
        if unknown:
            raise ConfigError(f"unknown Generator SSH config fields: {', '.join(sorted(map(str, unknown)))}")
        if "identity_file" in value and "key_path" in value:
            raise ConfigError("use exactly one of identity_file and key_path")
        key = value.get("identity_file", value.get("key_path"))
        if not isinstance(key, (str, os.PathLike)) or not os.fspath(key):
            raise ConfigError("Generator SSH config needs identity_file (or key_path)")
        if not isinstance(value.get("work_dir"), (str, os.PathLike)) or not os.fspath(value["work_dir"]):
            raise ConfigError("Generator SSH config needs work_dir")
        if "host" not in value or "port" not in value:
            raise ConfigError("Generator SSH config needs host and port")
        base = Path.cwd() if base_dir is None else Path(base_dir)
        key_path, work_dir = Path(key), Path(value["work_dir"])
        if not key_path.is_absolute():
            key_path = base / key_path
        if not work_dir.is_absolute():
            work_dir = base / work_dir
        return cls(
            host=value["host"], port=value["port"], identity_file=key_path,
            work_dir=work_dir, user=value.get("user", "root"),
            timeout_s=value.get("timeout_s", DEFAULT_TIMEOUT_S),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "SSHConfig":
        path = Path(path).expanduser().resolve()
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise ConfigError(f"cannot read Generator SSH config {path}: {exc}") from exc
        return cls.from_dict(value, base_dir=path.parent)

    load = from_file


@dataclass(frozen=True, slots=True)
class CommandResult:
    """A subprocess-compatible result that test runners can construct."""

    returncode: int
    stdout: str | bytes = ""
    stderr: str | bytes = ""


@dataclass(frozen=True, slots=True)
class RemoteStatus:
    epoch: str
    state: str
    detail: str = ""
    job_sha256: str = ""
    attempts: int = 0
    out_dir: Path | None = None
    results_path: Path | None = None

    @property
    def running(self) -> bool:
        return self.state in ("dispatched", "running")

    @property
    def complete(self) -> bool:
        return self.state == "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch, "state": self.state, "detail": self.detail,
            "job_sha256": self.job_sha256, "attempts": self.attempts,
            "out_dir": str(self.out_dir) if self.out_dir is not None else None,
            "results_path": str(self.results_path) if self.results_path is not None else None,
        }


class Runner(Protocol):
    def __call__(self, argv: Sequence[str], **kwargs: Any) -> Any: ...


class SSHRunner:
    """Argv-only subprocess wrapper with strict, bounded SSH connections."""

    def __init__(self, config: SSHConfig, run: Runner = subprocess.run) -> None:
        self.config = config
        self.run = run

    def _base(self, program: str) -> list[str]:
        if program not in ("ssh", "scp"):
            raise ValueError("only ssh and scp are supported")
        cfg = self.config
        argv = [
            program, "-F", "/dev/null",  # Never inherit a relaxed local ssh_config.
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"ConnectTimeout={max(1, int(cfg.timeout_s))}",
            "-o", "IdentitiesOnly=yes",
            "-i", os.fspath(cfg.identity_file),
        ]
        if program == "scp":
            argv.append("-s")  # SFTP: no legacy scp remote-shell evaluation.
        argv += ["-P" if program == "scp" else "-p", str(cfg.port)]
        return argv

    @property
    def destination(self) -> str:
        return f"{self.config.user}@{self.config.host}"

    def ssh(self, command: str, *, timeout: float | None = None) -> Any:
        return self.execute([*self._base("ssh"), "--", self.destination, command], timeout=timeout)

    def ssh_bytes(self, command: str, *, timeout: float | None = None) -> Any:
        return self.execute(
            [*self._base("ssh"), "--", self.destination, command], timeout=timeout, binary=True
        )

    def scp(self, source: Path, destination: str, *, timeout: float | None = None) -> Any:
        if not destination.startswith(self.destination + ":"):
            raise ContractError("scp destination must use this runner's fixed SSH endpoint")
        remote = destination.split(":", 1)[1]
        if not remote.startswith("/"):
            raise ContractError("scp destination must be an absolute remote path")
        return self.execute([*self._base("scp"), "--", os.fspath(source), destination], timeout=timeout)

    def upload(self, source: Path, remote: str, *, timeout: float | None = None) -> Any:
        return self.scp(source, f"{self.destination}:{remote_path(remote)}", timeout=timeout)

    def execute(
        self, argv: Sequence[str], *, timeout: float | None = None, binary: bool = False
    ) -> Any:
        try:
            # subprocess.run kills and waits for a local child on timeout.
            # The remote nohup process may continue; only its markers decide.
            result = self.run(
                list(argv), timeout=self.config.timeout_s if timeout is None else timeout,
                capture_output=True, text=not binary, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise InfraError("Generator SSH command TimeoutExpired; poll remote markers before retry") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise InfraError(f"Generator SSH command could not run: {type(exc).__name__}") from exc
        returncode = getattr(result, "returncode", None)
        if not isinstance(returncode, int):
            raise InfraError("Generator SSH runner returned no integer exit code")
        if returncode:
            stderr = getattr(result, "stderr", "") or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            raise InfraError(f"Generator SSH command exited {returncode}: {stderr.strip()[-4096:]}")
        return result


def _stdout(result: Any) -> str:
    value = getattr(result, "stdout", "")
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InfraError("Generator remote response is not UTF-8") from exc
    if not isinstance(value, str):
        raise InfraError("Generator remote response has no text stdout")
    return value


def _write_bytes(path: Path, data: bytes) -> None:
    """Complete and fsync a temporary file, then atomically replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes(path, canonical_json(value) + b"\n")


def remote_path(path: str) -> str:
    """One quoted POSIX word for an absolute path without parent components."""
    if (
        not path.startswith("/") or "\x00" in path
        or any(part in ("", ".", "..") for part in path.split("/")[1:])
    ):
        raise ContractError(f"unsafe remote path {path!r}")
    return shlex.quote(path)


def _epoch_serialized(method: Callable[..., RemoteStatus]) -> Callable[..., RemoteStatus]:
    """Serialize each state-check/network/state transaction for one epoch."""
    @wraps(method)
    def locked(self: "ScorerRemote", epoch: str) -> RemoteStatus:
        name = epoch_id(epoch)
        path = self.work_dir / "jobs" / name / ".remote.lock"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                return method(self, name)
        except OSError as exc:
            raise InfraError(f"cannot lock Generator remote epoch state: {exc}") from exc
    return locked


class ScorerRemote:
    """Dispatch once, then collect only a code-zero done marker's public out."""

    def __init__(self, config: SSHConfig, runner: Any = None, helpers: Sequence[Any] = ()) -> None:
        self.config = config
        self.runner = SSHRunner(config) if runner is None else runner
        self.helpers = tuple(helpers)
        self.work_dir = config.work_dir

    @staticmethod
    def _helper_runners(value: Mapping[str, Any], work_dir: str | Path) -> list[SSHRunner]:
        helpers = value.get("helpers", [])
        if not isinstance(helpers, list) or any(not isinstance(helper, Mapping) for helper in helpers):
            raise ConfigError("Generator helpers must be a list of SSH config objects")
        return [SSHRunner(SSHConfig.from_dict({**helper, "work_dir": os.fspath(work_dir)})) for helper in helpers]

    @classmethod
    def from_config(
        cls, value: str | Path | Mapping[str, Any],
        work_dir: str | Path | None = None, runner: Any = None,
    ) -> "ScorerRemote":
        if isinstance(value, Mapping):
            if work_dir is None:
                raise ConfigError("Generator control-plane work_dir is required with inline SSH config")
            config = SSHConfig.from_dict(
                {**value, "work_dir": os.fspath(work_dir)}
            )
        else:
            config = SSHConfig.from_file(value)
            if work_dir is not None and config.work_dir != Path(work_dir).expanduser().resolve():
                raise ConfigError("Generator SSH config work_dir differs from control-plane work_dir")
        return cls(config, runner=runner, helpers=cls._helper_runners(value, work_dir) if isinstance(value, Mapping) else [])

    def _epoch_helper_set(self, epoch: str, state: Mapping[str, Any]) -> tuple[Any, ...]:
        """The epoch's persisted helper set is authoritative across process restarts."""
        count = state.get("shard_count", 1)
        if count == 1:
            return ()
        identities = state.get("helper_identities", [])
        if not isinstance(identities, list) or len(identities) != count - 1:
            raise ConfigError("Generator epoch helper identities are incomplete")
        configured = {}
        for helper in self.helpers:
            config = getattr(helper, "config", helper)
            host = getattr(config, "host", None)
            port = getattr(config, "port", None)
            if host is not None and port is not None:
                configured[(host, port)] = helper
        result = []
        for identity in identities:
            if not isinstance(identity, Mapping):
                raise ConfigError("Generator epoch helper identity is invalid")
            key = (identity.get("host"), identity.get("port"))
            if key not in configured:
                raise ConfigError("a recorded Generator helper is no longer configured")
            result.append(configured[key])
        return tuple(result)

    def _paths(self, epoch: str) -> tuple[str, Path, Path, str, str, str, str]:
        name = epoch_id(epoch)
        root = f"{REMOTE_ROOT}/jobs/{name}"
        return (
            name, self.work_dir / "jobs" / name / "job.json",
            self.work_dir / "jobs" / name / "remote-state.json", root,
            f"{root}/job.json", f"{root}/out", f"{root}/done.json",
        )

    def _read_state(self, path: Path, epoch: str) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise InfraError(f"cannot read Generator remote state: {exc}") from exc
        if not isinstance(value, dict) or value.get("epoch") != epoch or not isinstance(value.get("state"), str):
            raise InfraError("Generator remote state has the wrong epoch or schema")
        if value["state"] not in ("preparing", "dispatched", "running", "collecting", "failed", "complete"):
            raise InfraError("Generator remote state has an unknown state")
        if type(value.get("attempts", 0)) is not int or value.get("attempts", 0) < 0:
            raise InfraError("Generator remote state has an invalid attempt count")
        return value

    def _job(self, path: Path) -> tuple[bytes, str]:
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, ValueError, UnicodeError) as exc:
            raise ConfigError(f"cannot read Generator private job: {exc}") from exc
        if not raw.strip() or not isinstance(value, dict) or not value:
            raise ConfigError("Generator private job must be a nonempty JSON object")
        head = value.get("head_version")
        if not isinstance(head, str) or not head or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in head):
            raise ConfigError("Generator private job has no safe exact head_version")
        return raw, sha256_hex(raw)

    def _status(self, epoch: str, record: Mapping[str, Any], detail: str = "") -> RemoteStatus:
        state = str(record.get("state", "absent"))
        if state == "preparing":
            state = "dispatched"

        out = self.work_dir / "epochs" / epoch / "out"
        complete = state == "complete" and (out / "results.json").is_file()
        return RemoteStatus(
            epoch=epoch, state=state,
            detail=detail or str(record.get("detail", "")),
            job_sha256=str(record.get("job_sha256", "")),
            attempts=int(record.get("attempts", 0)),
            out_dir=out if complete else None,
            results_path=out / "results.json" if complete else None,
        )

    def _failed(self, path: Path, record: dict[str, Any], message: str) -> None:
        record.update(state="failed", detail=message)
        _write_json(path, record)
        raise InfraError(message)

    @_epoch_serialized
    def dispatch(self, epoch: str) -> RemoteStatus:
        name, job_path, state_path, root, remote_job, remote_out, done = self._paths(epoch)
        _, job_digest = self._job(job_path)
        previous = self._read_state(state_path, name)
        epoch_helpers = self._epoch_helper_set(name, previous)
        local_results = self.work_dir / "epochs" / name / "out" / "results.json"
        if not previous and local_results.is_file():
            return self._status(name, {"state": "complete", "job_sha256": job_digest})
        if previous:
            if previous.get("job_sha256") not in (None, "", job_digest):
                raise ConfigError("Generator job changed after dispatch; explicitly reset state or use a new epoch")
            if previous.get("state") == "failed":
                return self._status(name, previous)
            if previous.get("state") == "complete" and local_results.is_file():
                return self._status(name, previous)
            if previous.get("state") == "complete":
                return self._status(name, previous, "completed scorer state is never relaunched; collect to recover output")
            if previous.get("launch_attempted"):
                record = dict(previous)
                if epoch_helpers:
                    # Every primary launch attempt is sticky. Collection, not
                    # dispatch, polls helpers once their launch was attempted.
                    pending = [row for row in previous.get("helpers", []) if not row.get("launch_attempted")]
                    if pending:
                        record["helpers"] = self._dispatch_helpers(name, job_path, state_path, job_digest, previous)
                else:
                    record["helpers"] = []
                if epoch_helpers:
                    _write_json(state_path, record)
                return self._status(name, record, "scorer launch attempted; collect to resume")
            if previous.get("state") == "dispatched":
                if not epoch_helpers:
                    return self._status(name, previous, "copy outcome unknown; collect to resume")
                record = dict(previous)
                record["helpers"] = self._prepare_helpers(name, job_path, state_path, job_digest, previous)
                if any(row.get("state") == "failed" for row in record["helpers"]):
                    record.update(shard_count=1, helper_identities=[], helpers=[], state="preparing", detail="helper unavailable before launch; using primary-only K=1 scoring")
                    _write_json(state_path, record)
                    previous = record
                else:
                    return self._status(name, record, "copy outcome unknown; collect to resume")
            if previous.get("state") != "preparing":
                previous = {}

        primary_launched = bool(previous.get("launch_attempted") or previous.get("state") == "complete")
        record = dict(previous) if previous else {
            "epoch": name, "state": "dispatched",
            "detail": "preparing exact frozen head; scorer has not been launched", "launch_attempted": False,
            "job_sha256": job_digest, "attempts": 1, "dispatched_at": time.time(),
        }
        if not primary_launched:
            record["shard_count"] = 1 + len(self.helpers)
            record["helper_identities"] = [
                {"host": helper.config.host, "port": helper.config.port} for helper in self.helpers
            ]
            epoch_helpers = tuple(self.helpers)
        if epoch_helpers:
            # A later control-plane release can change its grace; the epoch's
            # existing budget contract is immutable once dispatch has started.
            record.setdefault("results_grace_s", 45 * 60)
        _write_json(state_path, record)  # Durable before the first network call.
        head_version = json.loads(job_path.read_text())["head_version"]
        # Prepare every configured helper before any pod launches. If a helper
        # cannot be prepared, this epoch degrades to K=1 with no scorer running.
        if epoch_helpers:
            record["helpers"] = self._prepare_helpers(name, job_path, state_path, job_digest, record)
            if any(row.get("state") == "failed" for row in record["helpers"]):
                record.update(shard_count=1, helper_identities=[], helpers=[], state="preparing", detail="helper unavailable before launch; using primary-only K=1 scoring")
                epoch_helpers = ()
            _write_json(state_path, record)
        qroot, qjob = remote_path(root), remote_path(remote_job)
        qout, qdone = remote_path(remote_out), remote_path(done)
        self.runner.ssh(f"umask 077 && mkdir -p -- {qroot}")
        self.runner.scp(job_path, f"{self.runner.destination}:{qjob}")
        record.update(state="preparing", detail="private job copied; preparing exact frozen head")
        _write_json(state_path, record)
        ensure = (
            f"PYTHONPATH={REMOTE_PYTHONPATH} /root/venv/bin/python -m sn120_v2.generator.ensure_head "
            f"{shlex.quote(head_version)}"
        )
        self.runner.ssh(ensure, timeout=HEAD_PREPARE_TIMEOUT_S)
        if epoch_helpers:
            self._launch_helpers(name, state_path, record)
        # Only an uncertain scorer launch is sticky. Before this point, head
        # preparation can be retried without ever running two scorers.
        record.update(state="dispatched", detail="head prepared; scorer launch attempted", launch_attempted=True)
        _write_json(state_path, record)

        scorer = SCORER_COMMAND.format(epoch=name)
        if record["shard_count"] > 1:
            scorer += f" --shard 0/{record['shard_count']}"
        failed = f"{root}/failed.json"
        # No eval, environment forwarding or user command.  The marker file is
        # complete before mv, and is named done only after scorer exit code 0.
        inner = (
            f"PYTHONPATH={REMOTE_PYTHONPATH} {scorer}; code=$?; "
            f"if [ \"$code\" -eq 0 ]; then kind=done; else kind=failed; fi; "
            f"tmp={root}/.$kind.$$.json; "
            f"printf '%s\\n' \"{{\\\"epoch\\\":\\\"{name}\\\",\\\"code\\\":$code}}\" > \"$tmp\" && "
            f"if [ \"$code\" -eq 0 ]; then mv -- \"$tmp\" {qdone}; "
            f"else mv -- \"$tmp\" {remote_path(failed)}; fi; exit \"$code\""
        )
        # POSIX sh backgrounds a subshell for this whole list.  Redirect that
        # subshell, not merely nohup, so ssh does not wait for scorer stdout.
        launch = (
            f"{{ mkdir -p -- {qout} && nohup /bin/sh -c {shlex.quote(inner)} ; }} "
            f"< /dev/null >> {remote_path(root + '/scorer.log')} 2>&1 &"
        )
        self.runner.ssh(launch)
        record.update(state="dispatched", detail="scorer started asynchronously")
        _write_json(state_path, record)
        return self._status(name, record)

    def _helper_records(self, state_path: Path, previous: Mapping[str, Any], count: int) -> list[dict[str, Any]]:
        records = [dict(row) for row in previous.get("helpers", [])]
        for index in range(1, count - 1):
            path = state_path.with_name(f"remote-state-shard-{index}.json")
            if path.is_file():
                record = json.loads(path.read_text())
                if len(records) < index:
                    records.append(record)
                else:
                    records[index - 1] = record
        while len(records) < count - 1:
            records.append({})
        return records

    def _prepare_helpers(self, name: str, job_path: Path, state_path: Path, job_digest: str, previous: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Prepare all helpers without launching any, including on resume."""
        helpers = self._epoch_helper_set(name, previous)
        records = self._helper_records(state_path, previous, len(helpers) + 1)
        head_version = json.loads(job_path.read_text())["head_version"]
        root = f"{REMOTE_ROOT}/jobs/{name}"
        for index, runner in enumerate(helpers, 1):
            record = records[index - 1]
            if record.get("job_sha256") not in (None, "", job_digest):
                raise ConfigError("Generator helper job changed after dispatch")
            if record.get("launch_attempted") or record.get("state") == "prepared":
                continue
            record.update(epoch=name, state="dispatched", job_sha256=job_digest, launch_attempted=False, shard=index)
            self._write_helper_state(state_path, index, record)
            try:
                runner.ssh(f"umask 077 && mkdir -p -- {remote_path(root)}")
                runner.scp(job_path, f"{runner.destination}:{remote_path(root + '/job.json')}")
                runner.ssh(f"PYTHONPATH={REMOTE_PYTHONPATH} /root/venv/bin/python -m sn120_v2.generator.ensure_head {shlex.quote(head_version)}", timeout=HEAD_PREPARE_TIMEOUT_S)
            except InfraError as exc:
                record.update(state="failed", detail=f"helper preparation failed; launch not attempted: {type(exc).__name__}")
                self._write_helper_state(state_path, index, record)
                continue
            record["state"] = "prepared"
            self._write_helper_state(state_path, index, record)
        return records

    def _launch_helpers(self, name: str, state_path: Path, record: dict[str, Any]) -> None:
        helpers = self._epoch_helper_set(name, record)
        root = f"{REMOTE_ROOT}/jobs/{name}"
        for index, runner in enumerate(helpers, 1):
            helper = record["helpers"][index - 1]
            if helper.get("launch_attempted"):
                continue
            helper.update(state="dispatched", launch_attempted=True)
            self._write_helper_state(state_path, index, helper)
            scorer = SCORER_COMMAND.format(epoch=name) + f" --shard {index}/{len(helpers) + 1}"
            failed = f"{root}/failed.json"
            inner = (
                f"PYTHONPATH={REMOTE_PYTHONPATH} {scorer}; code=$?; "
                f"if [ \"$code\" -eq 0 ]; then kind=done; else kind=failed; fi; "
                f"tmp={root}/.$kind.$$.json; "
                f"printf '%s\\n' \"{{\\\"epoch\\\":\\\"{name}\\\",\\\"code\\\":$code}}\" > \"$tmp\" && "
                f"if [ \"$code\" -eq 0 ]; then mv -- \"$tmp\" {remote_path(root + '/done.json')}; "
                f"else mv -- \"$tmp\" {remote_path(failed)}; fi; exit \"$code\""
            )
            runner.ssh(f"{{ mkdir -p -- {remote_path(root + '/out')} && nohup /bin/sh -c {shlex.quote(inner)} ; }} < /dev/null >> {remote_path(root + '/scorer.log')} 2>&1 &")
            self._write_helper_state(state_path, index, helper)

    def _dispatch_helpers(self, name: str, job_path: Path, state_path: Path, job_digest: str, previous: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Resume unlaunched helpers without touching a launched primary."""
        if not previous:
            return []
        records = self._prepare_helpers(name, job_path, state_path, job_digest, previous)
        if any(row.get("state") == "failed" for row in records):
            # Do not begin a new sharded epoch after its primary is already
            # running K=1, and never claim an unreachable helper is running.
            if not previous.get("launch_attempted"):
                raise InfraError("Generator helper preparation failed before primary launch")
            for row in records:
                if row.get("launch_attempted"):
                    continue
                row["state"] = "failed"
                self._write_helper_state(state_path, row["shard"], row)
            return records
        record = dict(previous)
        record["helpers"] = records
        self._launch_helpers(name, state_path, record)
        return records

    @staticmethod
    def _write_helper_state(state_path: Path, index: int, record: Mapping[str, Any]) -> None:
        _write_json(state_path.with_name(f"remote-state-shard-{index}.json"), record)

    def _marker(self, path: str, *, runner: Any = None) -> dict[str, Any] | None:
        """Read a small bounded marker from the selected pod."""
        # Fixed bounded shell; there is intentionally no public arbitrary SSH.
        script = (
            "if [ -f {path} ]; then bytes=$(wc -c < {path}) || exit 2; "
            "[ \"$bytes\" -le 4096 ] || exit 3; cat -- {path}; fi"
        ).format(path=remote_path(path))
        endpoint = self.runner if runner is None else runner
        text = _stdout(endpoint.ssh(script))
        if not text.strip():
            return None
        if len(text.encode("utf-8")) > 4096:
            raise InfraError("Generator remote marker is too large")
        try:
            marker = json.loads(text)
        except ValueError as exc:
            raise InfraError("Generator remote marker is invalid JSON") from exc
        if not isinstance(marker, dict) or type(marker.get("code")) is not int:
            raise InfraError("Generator remote marker has no integer code")
        return marker

    @_epoch_serialized
    def collect(self, epoch: str) -> RemoteStatus:
        name, _, state_path, _, _, remote_out, done = self._paths(epoch)
        previous = self._read_state(state_path, name)
        epoch_helpers = self._epoch_helper_set(name, previous)
        local_results = self.work_dir / "epochs" / name / "out" / "results.json"
        if not previous:
            if local_results.is_file():
                return self._status(name, {"state": "complete"}, "collected output already preserved")
            return self._status(name, {"state": "absent"}, "dispatch is required before collection")
        if previous.get("state") == "complete" and local_results.is_file():
            return self._status(name, previous)
        if previous.get("state") == "failed":
            return self._status(name, previous)
        record = dict(previous)
        failed = self._marker(f"{REMOTE_ROOT}/jobs/{name}/failed.json")
        if failed is not None:
            self._failed(state_path, record, f"Generator scorer failed remotely with code {failed['code']}")
        marker = self._marker(done)
        if marker is None:
            record.update(state="running", detail="remote scorer has not written a done marker")
            _write_json(state_path, record)
            return self._status(name, record)
        if marker.get("code") != 0:
            self._failed(state_path, record, f"Generator scorer done marker has nonzero code {marker['code']}")
        if marker.get("epoch") not in (None, name):
            self._failed(state_path, record, "Generator scorer done marker names the wrong epoch")
        record.update(state="collecting", detail="successful done marker observed; downloading public out")
        _write_json(state_path, record)
        helper_outs: list[tuple[int, Path]] = []
        helper_records = [dict(row) for row in previous.get("helpers", [])]
        if epoch_helpers:
            pending = []
            helpers_by_index = {
                int(row["shard"]): row for row in previous.get("helpers", [])
                if isinstance(row, Mapping) and type(row.get("shard")) is int
            }
            for index, runner in enumerate(epoch_helpers, 1):
                if helpers_by_index.get(index, {}).get("state") == "failed":
                    continue
                try:
                    marker = self._marker(f"{REMOTE_ROOT}/jobs/{name}/done.json", runner=runner)
                except InfraError:
                    # A timeout/refused connection or temporarily malformed
                    # marker never relaunches a scorer. Poll until results grace;
                    # only this epoch's owned shard can then become unscored.
                    pending.append(index)
                    continue
                if marker is None:
                    pending.append(index)
                    continue
                if marker.get("code") != 0:
                    raise InfraError(f"Generator helper shard {index} done marker has nonzero code {marker['code']}")
                if marker.get("epoch") not in (None, name):
                    raise InfraError(f"Generator helper shard {index} done marker names the wrong epoch")
            if pending:
                job = json.loads((self.work_dir / "jobs" / name / "job.json").read_text())
                deadline = float(job["budget_deadline"])
                grace = float(previous.get("results_grace_s", 45 * 60))
                if time.time() <= deadline + grace:
                    record.update(state="running", detail=f"helpers not complete: {','.join(map(str, pending))}")
                    _write_json(state_path, record)
                    return self._status(name, record)
                # Once the scorer and its collection grace expire, a helper's
                # whole owned shard is unscored. Do not fetch a partial shard
                # or keep the epoch blocked on an unreachable/dead pod.
                for index in pending:
                    helper_records[index - 1].update(state="failed", detail="helper did not complete before the results deadline")
                    self._write_helper_state(state_path, index, helper_records[index - 1])
            record["helpers"] = helper_records
            _write_json(state_path, record)
        try:
            out = self._collect_out(name, remote_out, shard=0 if epoch_helpers else None)
            if epoch_helpers:
                helper_outs = [
                    (index, self._collect_out(name, remote_out, shard=index, runner=runner))
                    for index, runner in enumerate(epoch_helpers, 1)
                    if index not in pending
                ]
                out = self._merge_shards(name, out, helper_outs)
            try:
                results = json.loads((out / "results.json").read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeError) as exc:
                raise InfraError(f"Generator remote results.json is missing or invalid: {exc}") from exc
            if not isinstance(results, dict) or not results:
                raise InfraError("Generator remote results.json must be a nonempty JSON object")
        except Exception as exc:
            record.update(state="collecting", detail=f"collection incomplete; safe to retry: {type(exc).__name__}")
            _write_json(state_path, record)
            if isinstance(exc, InfraError):
                raise
            raise InfraError(f"Generator collection failed: {type(exc).__name__}") from exc
        record.update(state="complete", detail="public scorer out collected", collected_at=time.time())
        _write_json(state_path, record)
        return self._status(name, record)

    def _merge_shards(self, epoch: str, primary: Path, helpers: list[tuple[int, Path]]) -> Path:
        job = json.loads((self.work_dir / "jobs" / epoch / "job.json").read_text())
        documents = [(0, json.loads((primary / "results.json").read_text()))]
        documents += [(index, json.loads((path / "results.json").read_text())) for index, path in helpers]
        helper_paths = [path for _, path in helpers]
        base = documents[0][1]
        for _, document in documents[1:]:
            for key in ("epoch", "head_version", "seed_commitment", "pool_pin"):
                if document.get(key) != base.get(key):
                    raise InfraError("Generator shard result identity mismatch")
        count = len(helpers) + 1
        state_path = self.work_dir / "jobs" / epoch / "remote-state.json"
        if state_path.is_file():
            count = json.loads(state_path.read_text()).get("shard_count", count)
        by_id = {}
        for shard, document in documents:
            if document.get("shard", {}).get("index", shard) != shard or document.get("shard", {}).get("count", count) != count:
                raise InfraError("Generator shard result shard metadata mismatch")
            if sorted(row.get("id", row.get("task_id")) for row in document.get("tasks", [])) != sorted(task["id"] for position, task in enumerate(job["tasks"]) if position % count == shard):
                raise InfraError(f"Generator shard {shard} result does not cover exactly its owned tasks")
            for task in document.get("tasks", []):
                task_id = task.get("id", task.get("task_id"))
                if task_id in by_id:
                    raise InfraError("Generator shard results contain overlapping task IDs")
                by_id[task_id] = task
        ordered = [row["id"] for row in job["tasks"]]
        missing = [task_id for task_id in ordered if task_id not in by_id]
        if missing:
            grace = 45 * 60
            if state_path.is_file():
                grace = json.loads(state_path.read_text()).get("results_grace_s", grace)
            if time.time() <= float(job["budget_deadline"]) + grace:
                raise InfraError("Generator shard results do not cover exactly the job tasks")
            for task_id in missing:
                owner = next(row["hotkey"] for row in job["tasks"] if row["id"] == task_id)
                by_id[task_id] = {
                    "id": task_id, "hotkey": owner, "s": 0, "n": 0, "q": None,
                    "decision": "unscored", "unscored_reason": "time_budget",
                    "reason": "helper did not complete before the scoring budget grace",
                }
        if set(by_id) != set(ordered) or len(by_id) != len(ordered):
            raise InfraError("Generator shard results do not cover exactly the job tasks")
        merged = dict(base)
        merged.pop("shard", None)
        merged["tasks"] = [by_id[task_id] for task_id in ordered]
        merged["timings"] = dict(base.get("timings", {}))
        for key in ("started_at", "ready_at"):
            if all(isinstance(document.get("timings", {}).get(key), str) for _, document in documents):
                merged["timings"][key] = min(document["timings"][key] for _, document in documents)
        if all(isinstance(document.get("timings", {}).get("finished_at"), str) for _, document in documents):
            merged["timings"]["finished_at"] = max(document["timings"]["finished_at"] for _, document in documents)
        merged["timings"]["budget_exhausted"] = any(document.get("timings", {}).get("budget_exhausted") for _, document in documents)
        for key in ("wall_s", "startup_s", "health_s", "generate_s", "grade_s"):
            merged["timings"][key] = max(document.get("timings", {}).get(key, 0) for _, document in documents)
        throughput = dict(base.get("throughput", {}))
        for key in ("rollouts", "new_rollouts", "requests", "prompt_tokens", "completion_tokens", "generate_seconds", "grade_seconds"):
            throughput[key] = sum(document.get("throughput", {}).get(key, 0) for _, document in documents)
        wall = merged["timings"].get("wall_s", 0)
        throughput["rollouts_per_s"] = throughput["new_rollouts"] / wall if wall else 0.0
        throughput["completion_tokens_per_s"] = throughput["completion_tokens"] / wall if wall else 0.0
        throughput["clients_per_server"] = base.get("throughput", {}).get("clients_per_server", 0)
        throughput["servers"] = sum(document.get("throughput", {}).get("servers", 0) for _, document in documents)
        merged["throughput"] = throughput
        merged["servers"] = [server for _, document in documents for server in document.get("servers", [])]
        merged["hotkeys"] = {
            hotkey: dict.fromkeys(("submitted", "eligible", "scored", "accepted", "rejected", "unscored", "rollouts", "passes"), 0)
            for hotkey in dict.fromkeys(task["hotkey"] for task in job["tasks"])
        }
        for row in merged["tasks"]:
            total = merged["hotkeys"][row["hotkey"]]
            total["submitted"] += 1
            total["eligible"] += 1
            total["scored"] += row["n"] >= 8
            total[{"accept": "accepted", "reject": "rejected"}.get(row["decision"], "unscored")] += 1
            total["rollouts"] += row["n"]
            total["passes"] += row["s"]
        merged["publication"] = base.get("publication", {})
        final = self.work_dir / "epochs" / epoch / "out"
        out = self.work_dir / "epochs" / epoch / ".merged-staging"
        if out.exists():
            shutil.rmtree(out)
        if final.exists():
            raise InfraError("merged Generator output already exists")
        existing = set()
        for path in [primary, *helper_paths]:
            for rollout in (path / "rollouts").glob("*.jsonl"):
                if rollout.name in existing:
                    raise InfraError("Generator shard rollouts overlap")
                existing.add(rollout.name)
        if final.exists():
            raise InfraError("merged Generator output already exists")
        shutil.copytree(primary, out)
        for path in helper_paths:
            for rollout in (path / "rollouts").glob("*.jsonl"):
                target = out / "rollouts" / rollout.name
                if target.exists():
                    raise InfraError("Generator shard rollouts overlap")
                shutil.copyfile(rollout, target)
        _write_json(out / "results.json", merged)
        os.replace(out, final)
        return final

    def _collect_out(self, epoch: str, remote_out: str, *, shard: int | None = None, runner: Any = None) -> Path:
        runner = self.runner if runner is None else runner
        suffix = "" if shard is None else f"-shard-{shard}"
        """Download one fixed tar and manually extract only safe public files."""
        staging = self.work_dir / "epochs" / epoch / f".out-download{suffix}"
        final = self.work_dir / "epochs" / epoch / f"out{suffix}"
        if suffix:
            final = self.work_dir / "epochs" / epoch / f"out-shard-{shard}"
        self._clear_staging(staging)
        staging.mkdir(parents=True, mode=0o700)
        try:
            result = runner.ssh_bytes(f"tar -C {remote_path(remote_out)} -czf - --exclude=./rollouts/responses .")
            raw = result if isinstance(result, bytes) else getattr(result, "stdout", b"")
            if not isinstance(raw, bytes) or not raw:
                raise InfraError("Generator remote out archive is empty")
            if len(raw) > MAX_ARCHIVE_STREAM_BYTES:
                raise InfraError("Generator remote out archive exceeds collection bounds")
            try:
                archive = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
            except (tarfile.TarError, OSError, EOFError) as exc:
                raise InfraError("Generator remote out is not a valid tar archive") from exc
            with archive:
                members = archive.getmembers()
                if len(members) > MAX_OUTPUT_FILES:
                    raise InfraError("Generator remote out archive has too many entries")
                total = 0
                for member in members:
                    clean = self._safe_archive_name(member.name)
                    if not clean:
                        continue
                    target = staging.joinpath(*clean.split("/"))
                    # Normal tar includes directories (notably rollouts/).
                    # Links, devices and metadata are never applied locally.
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        raise InfraError("Generator remote out must contain only directories and regular files")
                    total += member.size
                    if total > MAX_ARCHIVE_BYTES:
                        raise InfraError("Generator remote out archive is too large")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        raise InfraError("Generator remote out has an unreadable file")
                    with source, target.open("xb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
            if not (staging / "results.json").is_file():
                raise InfraError("Generator remote out/results.json is required")
            # Publish a new complete out directory in one rename.  Never
            # overwrite an already preserved output with a later download.
            if final.exists() or final.is_symlink():
                if final.is_symlink() or not final.is_dir() or not (final / "results.json").is_file():
                    raise InfraError("local Generator out exists without a safe results.json")
                self._clear_staging(staging)
            else:
                os.replace(staging, final)
            return final
        except Exception:
            self._clear_staging(staging)
            raise

    @staticmethod
    def _safe_archive_name(name: str) -> str:
        if not name or name.startswith("/") or "\x00" in name:
            raise InfraError("Generator out archive contains an unsafe path")
        path = PurePosixPath(name)
        if any(part in ("", "..") for part in path.parts):
            raise InfraError("Generator out archive contains a parent traversal")
        clean = name[2:] if name.startswith("./") else name
        return "" if clean in ("", ".") else clean

    @staticmethod
    def _clear_staging(path: Path) -> None:
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    def status(self, epoch: str, *, refresh: bool = True) -> RemoteStatus:
        """Read durable state, or perform the same non-destructive poll as collect."""
        name, _, state_path, _, _, _, _ = self._paths(epoch)
        record = self._read_state(state_path, name)
        if not refresh or record.get("state") in ("complete", "failed"):
            return self._status(name, record)
        return self.collect(name)


__all__ = [
    "CommandResult", "REMOTE_PYTHONPATH", "REMOTE_ROOT", "RemoteStatus",
    "SCORER_COMMAND", "SSHConfig", "SSHRunner", "ScorerRemote",
]
