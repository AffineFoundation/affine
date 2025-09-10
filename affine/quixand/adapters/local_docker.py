from __future__ import annotations

import json
import os
import shlex
import sys
import time
import uuid
import subprocess
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from ..config import Config, Resources
from ..errors import QSAdapterError, QSSandboxNotFound, QSTimeout
from ..types import CommandResult, Execution, FileInfo, PTYHandle, SandboxHandle, SandboxStatus
from ..utils.fs import atomic_write_text
from .base import SandboxConfig
from ..container import (
    ContainerRuntime,
    DockerRuntime,
    PodmanRuntime,
    ContainerConfig,
    ContainerState,
    ExecConfig,
    VolumeMount,
    PTYSession
)


@dataclass
class LocalHandle(SandboxHandle):
    container_id: str
    id: str
    runtime_name: str  # "docker" or "podman"
    workdir: str
    created_at: datetime
    last_active_at: datetime
    timeout_seconds: int
    metadata: dict


def get_runtime(runtime_name: str) -> ContainerRuntime:
    """Get container runtime instance based on name."""
    if runtime_name == "docker":
        return DockerRuntime()
    elif runtime_name == "podman":
        return PodmanRuntime()
    else:
        # Default to Docker if available, else Podman
        try:
            return DockerRuntime()
        except:
            try:
                return PodmanRuntime()
            except:
                raise RuntimeError("Neither Docker nor Podman is available")


class LocalDockerAdapter:
    name = "local-docker"

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        # Use runtime from config directly
        self.runtime_name = self.cfg.runtime or "docker"
        self.runtime = get_runtime(self.runtime_name)

    # lifecycle
    def create(self, cfg: "SandboxConfig") -> LocalHandle:
        sbx_id = str(uuid.uuid4())
        container_name = f"qs_{sbx_id[:8]}"

        # Prepare volume mounts
        volumes = [
            VolumeMount(
                source=self._ensure_volume_dir(sbx_id),
                target=cfg.workdir,
                read_only=False,
                type="bind"
            )
        ]
        
        # Add user-specified volumes
        if cfg.volumes:
            for vol in cfg.volumes:
                # vol can be either a string "src:dest" or a dict with volume configuration
                if isinstance(vol, str):
                    parts = vol.split(":")
                    if len(parts) >= 2:
                        volumes.append(
                            VolumeMount(
                                source=parts[0],
                                target=parts[1],
                                read_only=len(parts) > 2 and parts[2] == "ro",
                                type="bind"
                            )
                        )
                elif isinstance(vol, dict):
                    volumes.append(
                        VolumeMount(
                            source=vol.get("source"),
                            target=vol.get("target"),
                            read_only=vol.get("read_only", False),
                            type=vol.get("type", "bind")
                        )
                    )

        # Prepare resource limits
        resources = None
        if cfg.resources:
            # Use Resources directly from config
            resources = cfg.resources

        # Create container configuration
        container_config = ContainerConfig(
            name=container_name,
            image=cfg.image,
            workdir=cfg.workdir,
            env={
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                **(cfg.env or {})
            },
            volumes=volumes,
            resources=resources,
            entrypoint=cfg.entrypoint,
            command=cfg.command,
            labels={
                "quixand.id": sbx_id,
                "quixand.created": datetime.utcnow().isoformat(),
            }
        )

        # Create and start container
        container_id = self.runtime.create_container(container_config)
        self.runtime.start_container(container_id)

        h = LocalHandle(
            container_id=container_id,
            id=sbx_id,
            runtime_name=self.runtime_name,
            workdir=cfg.workdir,
            created_at=datetime.utcnow(),
            last_active_at=datetime.utcnow(),
            timeout_seconds=cfg.timeout,
            metadata=cfg.metadata or {},
        )
        self._persist_handle(h)

        # Spawn watchdog in background, fully detached from this process/TTY
        if os.getenv("QS_DISABLE_WATCHDOG") not in {"1", "true", "TRUE", "True"}:
            try:
                interpreter = sys.executable or "python"
                subprocess.Popen(
                    [interpreter, "-m", "quixand.core.watchdog", h.id],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception:
                pass
        return h

    def connect(self, sandbox_id: str) -> LocalHandle:
        state = self._load_state()
        entry = state.get(sandbox_id)
        if not entry:
            raise QSSandboxNotFound(f"Sandbox {sandbox_id} not found")
        return LocalHandle(
            container_id=entry["container_id"],
            id=sandbox_id,
            runtime_name=entry.get("runtime", "docker"),
            workdir=entry["workdir"],
            created_at=datetime.fromisoformat(entry["created_at"]),
            last_active_at=datetime.fromisoformat(entry["last_active_at"]),
            timeout_seconds=entry["timeout_seconds"],
            metadata=entry.get("metadata", {}),
        )

    def shutdown(self, h: LocalHandle) -> None:
        runtime = get_runtime(h.runtime_name)
        runtime.stop_container(h.container_id, timeout=15)
        runtime.remove_container(h.container_id, force=True)
        self._remove_state(h.id)
        # Best-effort cleanup of host-mounted dirs for this sandbox
        self._cleanup_host_dirs(h.id)

    def status(self, h: LocalHandle) -> SandboxStatus:
        runtime = get_runtime(h.runtime_name)
        
        # Get container info to determine actual state
        container_state = "error"
        try:
            info = runtime.get_container_info(h.container_id)
            
            # Map ContainerState to sandbox state string
            state_map = {
                ContainerState.CREATED: "creating",
                ContainerState.RUNNING: "running",
                ContainerState.PAUSED: "running",
                ContainerState.STOPPED: "stopped",
                ContainerState.EXITED: "stopped",
                ContainerState.DEAD: "stopped",
                ContainerState.REMOVING: "stopping",
                ContainerState.ERROR: "error",
            }
            container_state = state_map.get(info.state, "error")
        except Exception:
            container_state = "error"

        deadline = h.last_active_at + timedelta(seconds=h.timeout_seconds)
        return SandboxStatus(
            state=container_state,
            created_at=h.created_at,
            last_active_at=h.last_active_at,
            timeout_at=deadline,
            metadata=h.metadata
        )

    def refresh_timeout(self, h: LocalHandle, seconds: int) -> None:
        h.timeout_seconds = seconds
        h.last_active_at = datetime.utcnow()
        self._persist_handle(h)

    # filesystem
    def fs_write(self, h: LocalHandle, path: str, data: bytes, text: bool) -> None:
        runtime = get_runtime(h.runtime_name)
        local = self._scratch(h, "put.tmp")
        local.write_bytes(data if not text else data.decode("utf-8").encode("utf-8"))
        runtime.copy_to_container(h.container_id, str(local), self._abs(h, path))
        h.last_active_at = datetime.utcnow()
        self._persist_handle(h)

    def fs_read(self, h: LocalHandle, path: str, text: bool) -> bytes | str:
        runtime = get_runtime(h.runtime_name)
        local = self._scratch(h, "get.tmp")
        runtime.copy_from_container(h.container_id, self._abs(h, path), str(local))
        data = local.read_bytes()
        h.last_active_at = datetime.utcnow()
        self._persist_handle(h)
        return data.decode("utf-8", "ignore") if text else data

    def fs_ls(self, h: LocalHandle, path: str) -> list[FileInfo]:
        runtime = get_runtime(h.runtime_name)
        
        # Execute ls command to get file info
        exec_config = ExecConfig(
            command=["sh", "-lc", f"ls -la --time-style=+%s {shlex.quote(self._abs(h, path))}"]
        )
        res = runtime.exec_in_container(h.container_id, exec_config)
        
        out = res.stdout.decode("utf-8", "ignore")
        files: list[FileInfo] = []
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 7:
                continue
            size = int(parts[4]) if parts[4].isdigit() else 0
            mtime = None
            try:
                mtime = datetime.utcfromtimestamp(int(parts[5]))
            except Exception:
                mtime = None
            name = parts[-1]
            is_dir = line.startswith("d")
            files.append(FileInfo(path=name, size=size, is_dir=is_dir, modified_at=mtime))
        return files

    def fs_mkdir(self, h: LocalHandle, path: str, parents: bool) -> None:
        runtime = get_runtime(h.runtime_name)
        flag = "-p" if parents else ""
        exec_config = ExecConfig(
            command=["sh", "-lc", f"mkdir {flag} {shlex.quote(self._abs(h, path))}"]
        )
        runtime.exec_in_container(h.container_id, exec_config)
        h.last_active_at = datetime.utcnow()
        self._persist_handle(h)

    def fs_rm(self, h: LocalHandle, path: str, recursive: bool) -> None:
        runtime = get_runtime(h.runtime_name)
        flag = "-r" if recursive else ""
        exec_config = ExecConfig(
            command=["sh", "-lc", f"rm {flag} -f {shlex.quote(self._abs(h, path))}"]
        )
        runtime.exec_in_container(h.container_id, exec_config)
        h.last_active_at = datetime.utcnow()
        self._persist_handle(h)

    def fs_mv(self, h: LocalHandle, src: str, dst: str) -> None:
        runtime = get_runtime(h.runtime_name)
        exec_config = ExecConfig(
            command=["sh", "-lc", f"mv {shlex.quote(self._abs(h, src))} {shlex.quote(self._abs(h, dst))}"]
        )
        runtime.exec_in_container(h.container_id, exec_config)
        h.last_active_at = datetime.utcnow()
        self._persist_handle(h)

    def fs_put(self, h: LocalHandle, local: str, remote: str) -> None:
        runtime = get_runtime(h.runtime_name)
        runtime.copy_to_container(h.container_id, local, self._abs(h, remote))
        h.last_active_at = datetime.utcnow()
        self._persist_handle(h)

    def fs_get(self, h: LocalHandle, remote: str, local: str) -> None:
        runtime = get_runtime(h.runtime_name)
        runtime.copy_from_container(h.container_id, self._abs(h, remote), local)
        h.last_active_at = datetime.utcnow()
        self._persist_handle(h)

    # processes
    def run(self, h: LocalHandle, cmd: list[str] | str, env: Optional[dict], timeout: Optional[int]) -> CommandResult:
        runtime = get_runtime(h.runtime_name)
        
        # Prepare command
        cmd_str = shlex.join(cmd) if isinstance(cmd, list) else cmd
        if env:
            prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
            cmd_str = f"{prefix} {cmd_str}"
        
        # Create exec configuration
        exec_config = ExecConfig(
            command=["sh", "-lc", cmd_str]
        )
        
        start = time.monotonic()
        try:
            res = runtime.exec_in_container(h.container_id, exec_config, timeout=timeout)
        except TimeoutError:
            raise QSTimeout(f"Command timed out after {timeout}s")
        except Exception as e:
            raise
        
        duration = time.monotonic() - start
        text = res.stdout.decode("utf-8", "ignore")
        
        h.last_active_at = datetime.utcnow()
        self._persist_handle(h)
        
        return CommandResult(
            text=text,
            stdout=res.stdout,
            stderr=res.stderr,
            exit_code=res.exit_code,
            duration_s=duration
        )

    def pty_start(self, h: LocalHandle, cmd: str, env: Optional[dict]) -> PTYHandle:
        """Start a PTY session with interactive streaming support."""
        runtime = get_runtime(h.runtime_name)
        
        # Create PTY session using the container runtime
        session = runtime.create_pty_session(h.container_id, cmd, env)
        
        # Store runtime name in session for later use
        session._runtime_name = h.runtime_name
        
        # Update last active time
        h.last_active_at = datetime.utcnow()
        self._persist_handle(h)
        
        return session

    def pty_send(self, pty: PTYHandle, data: bytes) -> None:
        """Send data to the PTY session."""
        if isinstance(pty, PTYSession):
            runtime = get_runtime(pty._runtime_name)
            # Auto-append newline if not present
            if isinstance(data, str):
                data = data.encode('utf-8')
            if not data.endswith(b'\n'):
                data = data + b'\n'
            runtime.send_pty_input(pty, data)

    def pty_stream(self, pty: PTYHandle) -> Iterable[bytes]:
        """Stream output from the PTY session."""
        if isinstance(pty, PTYSession):
            runtime = get_runtime(pty._runtime_name)
            yield from runtime.stream_pty_output(pty)

    def pty_close(self, pty: PTYHandle) -> None:
        """Close the PTY session."""
        if isinstance(pty, PTYSession):
            runtime = get_runtime(pty._runtime_name)
            runtime.close_pty_session(pty)

    # code conveniences
    def run_code(self, h: LocalHandle, code: str) -> Execution:
        code_path = self._abs(h, ".qs_exec.py")
        self.fs_write(h, code_path, code.encode("utf-8"), text=False)
        res = self.run(h, ["python", code_path], env=None, timeout=None)
        ok = res.exit_code == 0
        return Execution(text=res.text, images=None, files=None, stderr=res.stderr.decode("utf-8", "ignore"), ok=ok)

    def install_pkg(self, h: LocalHandle, spec: str) -> CommandResult:
        return self.run(h, ["python", "-m", "pip", "install", "--no-input", spec], env=None, timeout=None)

    def expose(self, h: LocalHandle, port: int, host_port: Optional[int], proto: str):  # TODO
        return {"container_port": port, "host_port": host_port or port, "proto": proto}

    # helpers
    def _abs(self, h: LocalHandle, path: str) -> str:
        p = Path(path)
        return str(p if p.is_absolute() else Path(h.workdir) / p)

    def _scratch(self, h: LocalHandle, name: str) -> Path:
        p = Config.state_file().parent / "scratch" / h.id
        p.mkdir(parents=True, exist_ok=True)
        return p / name

    def _ensure_volume_dir(self, sbx_id: str) -> str:
        p = Config.state_file().parent / "volumes" / sbx_id
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    def _cleanup_host_dirs(self, sbx_id: str) -> None:
        base = Config.state_file().parent
        for rel in ("scratch", "volumes"):
            try:
                shutil.rmtree(str(base / rel / sbx_id), ignore_errors=True)
            except Exception:
                pass

    def _load_state(self) -> dict:
        if not Config.state_file().exists():
            return {}
        try:
            return json.loads(Config.state_file().read_text())
        except Exception:
            return {}

    def _persist_handle(self, h: LocalHandle) -> None:
        state = self._load_state()
        state[h.id] = {
            "container_id": h.container_id,
            "runtime": h.runtime_name,  # Store runtime name, not "runtime" object
            "workdir": h.workdir,
            "created_at": h.created_at.isoformat(),
            "last_active_at": h.last_active_at.isoformat(),
            "timeout_seconds": h.timeout_seconds,
            "metadata": h.metadata,
            "adapter": self.name,
        }
        atomic_write_text(Config.state_file(), json.dumps(state, indent=2))

    def _remove_state(self, sandbox_id: str) -> None:
        state = self._load_state()
        state.pop(sandbox_id, None)
        atomic_write_text(Config.state_file(), json.dumps(state, indent=2))
