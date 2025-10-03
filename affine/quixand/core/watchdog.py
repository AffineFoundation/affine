from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
import shutil
from pathlib import Path

from ..config import Config
from ..container import DockerRuntime, PodmanRuntime, ContainerRuntime


def _load_state() -> dict:
    try:
        return json.loads(Config.state_file().read_text())
    except Exception:
        return {}


def get_runtime(runtime_name: str) -> ContainerRuntime:
    """Get container runtime instance based on name."""
    if runtime_name == "docker":
        try:
            return DockerRuntime()
        except Exception:
            pass
    elif runtime_name == "podman":
        try:
            return PodmanRuntime()
        except Exception:
            pass
    
    # Fallback: try both
    try:
        return DockerRuntime()
    except Exception:
        try:
            return PodmanRuntime()
        except Exception:
            return None


def main() -> int:
    if len(sys.argv) < 2:
        return 2
    sbx_id = sys.argv[1]
    
    # Poll state until sandbox is removed or timed out
    while True:
        state = _load_state()
        entry = state.get(sbx_id)
        if not entry:
            return 0
        
        try:
            created_at = datetime.fromisoformat(entry["created_at"])  # naive -> treat as UTC
        except Exception:
            created_at = datetime.utcnow()
        
        try:
            last_active_at = datetime.fromisoformat(entry["last_active_at"])  # naive -> treat as UTC
        except Exception:
            # If state is corrupted or missing last_active_at, fall back to created_at
            # rather than "now" to avoid extending lifetime indefinitely.
            last_active_at = created_at
        
        timeout_seconds = int(entry.get("timeout_seconds", 300))
        now = datetime.utcnow()
        deadline = last_active_at + timedelta(seconds=timeout_seconds)
        
        # Apply a hard lifetime cap to prevent a watchdog from running forever
        # in case last_active_at parsing keeps failing or state drifts.
        lifetime_deadline = created_at + timedelta(seconds=max(timeout_seconds * 2, timeout_seconds + 60))
        
        # If idle or lifetime exceeded
        if now >= deadline or now >= lifetime_deadline:
            runtime_name = entry.get("runtime", "docker")
            container_id = entry.get("container_id")
            
            if container_id:
                runtime = get_runtime(runtime_name)
                if runtime:
                    try:
                        # Stop container with timeout
                        runtime.stop_container(container_id, timeout=15)
                        runtime.remove_container(container_id, force=True)
                    except Exception:
                        pass
            
            # Best-effort cleanup of host-mounted dirs for this sandbox
            try:
                base = Config.state_file().parent
                for rel in ("scratch", "volumes"):
                    shutil.rmtree(str(base / rel / sbx_id), ignore_errors=True)
            except Exception:
                pass
            
            # Remove from state
            try:
                state.pop(sbx_id, None)
                Config.state_file().write_text(json.dumps(state, indent=2))
            except Exception:
                pass
            return 0
        
        # If container does not exist anymore, exit and clean state quickly
        runtime_name = entry.get("runtime", "docker")
        container_id = entry.get("container_id")
        
        if container_id:
            runtime = get_runtime(runtime_name)
            if runtime and not runtime.container_exists(container_id):
                try:
                    state.pop(sbx_id, None)
                    Config.state_file().write_text(json.dumps(state, indent=2))
                except Exception:
                    pass
                
                # Also cleanup host-mounted dirs
                try:
                    base = Config.state_file().parent
                    for rel in ("scratch", "volumes"):
                        shutil.rmtree(str(base / rel / sbx_id), ignore_errors=True)
                except Exception:
                    pass
                return 0
        
        time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
