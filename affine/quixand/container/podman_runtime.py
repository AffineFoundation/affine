"""Podman SDK implementation of container runtime."""

from __future__ import annotations

import io
import os
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import podman
    from podman import PodmanClient
    from podman.errors import NotFound, APIError
    HAS_PODMAN = True
except ImportError:
    HAS_PODMAN = False
    PodmanClient = Any

from .base import (
    ContainerRuntime,
    ContainerConfig,
    ContainerInfo,
    ContainerState,
    ExecConfig,
    ExecResult,
    PTYSession,
)
import threading
import queue
import json
import requests


class PodmanRuntime(ContainerRuntime):
    """Podman runtime implementation using Podman SDK."""

    def __init__(self, uri: Optional[str] = None):
        if not HAS_PODMAN:
            raise ImportError("podman package is not installed. Install with: pip install podman")
        
        # Use environment variable or default Podman socket
        uri = uri or os.environ.get("PODMAN_URI", "unix:///run/podman/podman.sock")
        self.client = PodmanClient(base_url=uri)
        
        # Test connection
        try:
            self.client.version()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Podman service: {e}")

    def _ensure_image_exists(self, image: str) -> None:
        """Ensure image exists locally, pull if needed."""
        try:
            self.client.images.get(image)
            return
        except (NotFound, Exception) as e:
            if not ("ImageNotFound" in str(type(e).__name__) or "image not known" in str(e) or "404" in str(e)):
                raise RuntimeError(f"Failed to check image {image}: {e}")

        print(f"Pulling {image} from registry...")
        try:
            pulled_image = self.client.images.pull(image)
            print(f"Image {image} pulled successfully (ID: {pulled_image.id[:12]})")
        except Exception as pull_e:
            raise RuntimeError(
                f"Failed to acquire image {image}. "
                f"The image could not be found locally, imported from Docker, or pulled from a registry. "
                f"Error: {pull_e}"
            )
    
    def create_container(self, config: ContainerConfig) -> str:
        """Create a new container and return its ID."""
        # Ensure image exists
        self._ensure_image_exists(config.image)
        
        # Prepare mounts for Podman
        mounts = []
        for vol in config.volumes:
            mount = {
                'type': vol.type,
                'source': vol.source,
                'target': vol.target,
                'read_only': vol.read_only,
            }
            mounts.append(mount)

        # Build container configuration
        container_kwargs = {
            'image': config.image,
            'name': config.name,
            'working_dir': config.workdir,
            'environment': config.env,
            'labels': config.labels,
            'detach': True,
            'tty': True,
            'stdin_open': True,
        }
        
        # Only add mounts if they exist
        if mounts:
            container_kwargs['mounts'] = mounts

        # Add resource limits
        if config.resources:
            if config.resources.cpu_limit:
                # Podman uses CPU shares or period/quota
                container_kwargs['cpu_shares'] = int(config.resources.cpu_limit * 1024)
            if config.resources.mem_limit:
                container_kwargs['mem_limit'] = config.resources.mem_limit
            if config.resources.pids_limit:
                container_kwargs['pids_limit'] = config.resources.pids_limit
            if config.resources.network:
                container_kwargs['network_mode'] = config.resources.network

        # Handle entrypoint and command
        if config.entrypoint is not None:
            container_kwargs['entrypoint'] = config.entrypoint
        if config.command is not None:
            container_kwargs['command'] = config.command

        # Add port mappings
        if config.ports:
            ports = {}
            for container_port, host_port in config.ports.items():
                ports[f"{container_port}/tcp"] = host_port
            container_kwargs['ports'] = ports

        try:
            container = self.client.containers.create(**container_kwargs)
            return container.id
        except Exception as e:
            raise RuntimeError(f"Failed to create container: {e}")

    def start_container(self, container_id: str) -> None:
        """Start a created container."""
        try:
            container = self.client.containers.get(container_id)
            container.start()
        except NotFound:
            raise ValueError(f"Container {container_id} not found")
        except Exception as e:
            raise RuntimeError(f"Failed to start container: {e}")

    def stop_container(self, container_id: str, timeout: int = 10) -> None:
        """Stop a running container."""
        try:
            container = self.client.containers.get(container_id)
            container.stop(timeout=timeout)
        except NotFound:
            # Container doesn't exist, consider it stopped
            pass
        except Exception as e:
            # Log error but don't raise - container might already be stopped
            pass

    def remove_container(self, container_id: str, force: bool = False) -> None:
        """Remove a container."""
        try:
            container = self.client.containers.get(container_id)
            container.remove(force=force)
        except NotFound:
            # Container already removed
            pass
        except Exception as e:
            if force:
                # Force removal requested, ignore errors
                pass
            else:
                raise RuntimeError(f"Failed to remove container: {e}")

    def get_container_info(self, container_id: str) -> ContainerInfo:
        """Get information about a container."""
        try:
            container = self.client.containers.get(container_id)
            attrs = container.attrs
            
            # Map Podman state to our ContainerState
            state_str = attrs.get('State', {}).get('Status', '').lower()
            state_map = {
                'created': ContainerState.CREATED,
                'running': ContainerState.RUNNING,
                'paused': ContainerState.PAUSED,
                'stopped': ContainerState.STOPPED,
                'exited': ContainerState.EXITED,
                'dead': ContainerState.DEAD,
                'removing': ContainerState.REMOVING,
            }
            state = state_map.get(state_str, ContainerState.ERROR)

            # Parse timestamps - handle nanosecond precision
            def parse_podman_timestamp(timestamp_str):
                """Parse Podman timestamp with nanosecond precision."""
                if not timestamp_str or timestamp_str == '0001-01-01T00:00:00Z':
                    return None
                # Remove nanoseconds if present (keep only up to microseconds)
                # Format: 2025-09-03T14:12:12.334389548+00:00
                if '.' in timestamp_str:
                    parts = timestamp_str.split('.')
                    # Take first 6 digits of fractional seconds
                    fractional = parts[1][:6].ljust(6, '0')
                    # Find timezone part
                    tz_idx = max(fractional.find('+'), fractional.find('-'), fractional.find('Z'))
                    if tz_idx > 0:
                        tz_part = fractional[tz_idx:]
                        fractional = fractional[:tz_idx]
                    else:
                        # Look in the rest of the original fractional part
                        rest = parts[1][6:]
                        tz_idx = max(rest.find('+'), rest.find('-'), rest.find('Z'))
                        if tz_idx >= 0:
                            tz_part = rest[tz_idx:]
                        else:
                            tz_part = ''
                    timestamp_str = f"{parts[0]}.{fractional}{tz_part}"
                
                # Replace Z with +00:00 for ISO format
                timestamp_str = timestamp_str.replace('Z', '+00:00')
                return datetime.fromisoformat(timestamp_str)
            
            created_at = parse_podman_timestamp(attrs.get('Created', ''))
            started_at = parse_podman_timestamp(attrs.get('State', {}).get('StartedAt'))
            finished_at = parse_podman_timestamp(attrs.get('State', {}).get('FinishedAt'))

            return ContainerInfo(
                id=container.id,
                name=container.name,
                state=state,
                created_at=created_at,
                started_at=started_at,
                finished_at=finished_at,
                exit_code=attrs.get('State', {}).get('ExitCode'),
                labels=container.labels or {},
            )
        except NotFound:
            raise ValueError(f"Container {container_id} not found")
        except Exception as e:
            raise RuntimeError(f"Failed to inspect container: {e}")

    def container_exists(self, container_id: str) -> bool:
        """Check if container exists."""
        try:
            self.client.containers.get(container_id)
            return True
        except NotFound:
            return False
        except Exception:
            return False

    def exec_in_container(
        self,
        container_id: str,
        config: ExecConfig,
        timeout: Optional[int] = None
    ) -> ExecResult:
        """Execute a command in a running container."""
        start_time = time.time()
        
        try:
            container = self.client.containers.get(container_id)
            
            # Prepare exec configuration
            exec_kwargs = {
                'cmd': config.command,
                'tty': config.tty,
                'stdin': config.stdin,
                'stdout': True,
                'stderr': True,
                'privileged': config.privileged,
            }
            
            if config.env:
                exec_kwargs['environment'] = config.env
            if config.workdir:
                exec_kwargs['workdir'] = config.workdir
            if config.user:
                exec_kwargs['user'] = config.user

            # Execute command with timeout handling
            if timeout:
                import threading
                result = {'exit_code': -1, 'stdout': b'', 'stderr': b''}
                error = [None]
                
                def run_exec():
                    try:
                        exec_result = container.exec_run(**exec_kwargs)
                        result['exit_code'] = exec_result[0]
                        result['stdout'] = exec_result[1] if isinstance(exec_result[1], bytes) else exec_result[1].encode()
                    except Exception as e:
                        error[0] = e
                
                thread = threading.Thread(target=run_exec)
                thread.daemon = True
                thread.start()
                thread.join(timeout)
                
                if thread.is_alive():
                    raise TimeoutError(f"Command timed out after {timeout}s")
                
                if error[0]:
                    raise error[0]
                
                exit_code = result['exit_code']
                stdout = result['stdout']
                stderr = result['stderr']
            else:
                # Execute without timeout
                exec_result = container.exec_run(**exec_kwargs)
                exit_code = exec_result[0]
                stdout = exec_result[1] if isinstance(exec_result[1], bytes) else exec_result[1].encode()
                stderr = b''
            
            duration = time.time() - start_time
            
            return ExecResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration
            )
            
        except NotFound:
            raise ValueError(f"Container {container_id} not found")
        except TimeoutError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to execute command: {e}")

    def copy_to_container(
        self,
        container_id: str,
        source: str,
        dest: str
    ) -> None:
        """Copy file or directory from host to container."""
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path {source} does not exist")

        # Determine destination directory and filename
        dest_path = Path(dest)
        if source_path.is_file():
            # For files, we need to separate directory and filename
            dest_dir = str(dest_path.parent) if dest_path.parent != Path('.') else '/'
            dest_name = dest_path.name
        else:
            # For directories, use dest as is
            dest_dir = dest
            dest_name = '.'

        # Create tar archive in memory
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            if source_path.is_file():
                # Add single file with the destination name
                tar.add(source_path, arcname=dest_name)
            else:
                # Add directory
                tar.add(source_path, arcname=dest_name)
        
        tar_stream.seek(0)
        
        try:
            container = self.client.containers.get(container_id)
            # Put archive to container (dest_dir must be a directory)
            container.put_archive(dest_dir, tar_stream.read())
        except NotFound:
            raise ValueError(f"Container {container_id} not found")
        except Exception as e:
            raise RuntimeError(f"Failed to copy to container: {e}")

    def copy_from_container(
        self,
        container_id: str,
        source: str,
        dest: str
    ) -> None:
        """Copy file or directory from container to host."""
        try:
            container = self.client.containers.get(container_id)
            
            # Get archive from container
            bits, stat = container.get_archive(source)
            
            # Write to tar stream
            tar_stream = io.BytesIO()
            for chunk in bits:
                tar_stream.write(chunk)
            tar_stream.seek(0)
            
            dest_path = Path(dest)
            
            # Extract archive
            with tarfile.open(fileobj=tar_stream, mode='r') as tar:
                members = tar.getmembers()
                
                if len(members) == 1 and members[0].isfile():
                    # Single file - extract directly to destination
                    member = members[0]
                    # Extract to parent directory with the desired filename
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Extract file content
                    file_obj = tar.extractfile(member)
                    if file_obj:
                        dest_path.write_bytes(file_obj.read())
                else:
                    # Multiple files or directory - extract to directory
                    if not dest_path.exists():
                        dest_path.mkdir(parents=True, exist_ok=True)
                    tar.extractall(path=dest_path)
                
        except NotFound:
            raise ValueError(f"Container {container_id} not found")
        except Exception as e:
            raise RuntimeError(f"Failed to copy from container: {e}")

    def list_containers(self, all: bool = False) -> List[ContainerInfo]:
        """List containers."""
        containers = self.client.containers.list(all=all)
        result = []
        
        for container in containers:
            attrs = container.attrs
            
            # Map Podman state to our ContainerState  
            state_str = attrs.get('State', '').lower()
            state_map = {
                'created': ContainerState.CREATED,
                'running': ContainerState.RUNNING,
                'paused': ContainerState.PAUSED,
                'stopped': ContainerState.STOPPED,
                'exited': ContainerState.EXITED,
                'dead': ContainerState.DEAD,
                'removing': ContainerState.REMOVING,
            }
            state = state_map.get(state_str, ContainerState.ERROR)
            
            # Parse created timestamp - handle nanosecond precision
            def parse_podman_timestamp(timestamp_str):
                """Parse Podman timestamp with nanosecond precision."""
                if not timestamp_str or timestamp_str == '0001-01-01T00:00:00Z':
                    return None
                # Remove nanoseconds if present (keep only up to microseconds)
                if '.' in timestamp_str:
                    parts = timestamp_str.split('.')
                    # Take first 6 digits of fractional seconds
                    fractional = parts[1][:6].ljust(6, '0')
                    # Find timezone part
                    tz_idx = max(fractional.find('+'), fractional.find('-'), fractional.find('Z'))
                    if tz_idx > 0:
                        tz_part = fractional[tz_idx:]
                        fractional = fractional[:tz_idx]
                    else:
                        # Look in the rest of the original fractional part
                        rest = parts[1][6:]
                        tz_idx = max(rest.find('+'), rest.find('-'), rest.find('Z'))
                        if tz_idx >= 0:
                            tz_part = rest[tz_idx:]
                        else:
                            tz_part = ''
                    timestamp_str = f"{parts[0]}.{fractional}{tz_part}"
                
                # Replace Z with +00:00 for ISO format
                timestamp_str = timestamp_str.replace('Z', '+00:00')
                return datetime.fromisoformat(timestamp_str)
            
            created_at = parse_podman_timestamp(attrs.get('Created', ''))
            
            result.append(ContainerInfo(
                id=container.id,
                name=container.name,
                state=state,
                created_at=created_at,
                labels=container.labels or {},
            ))
        
        return result

    def get_container_logs(
        self,
        container_id: str,
        stdout: bool = True,
        stderr: bool = True,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        tail: Optional[int] = None
    ) -> str:
        """Get container logs."""
        try:
            container = self.client.containers.get(container_id)
            
            kwargs = {
                'stdout': stdout,
                'stderr': stderr,
                'stream': False,
            }
            
            if since:
                kwargs['since'] = since
            if until:
                kwargs['until'] = until  
            if tail:
                kwargs['tail'] = str(tail)
            
            logs = container.logs(**kwargs)
            if isinstance(logs, bytes):
                return logs.decode('utf-8', errors='ignore')
            return logs
            
        except NotFound:
            raise ValueError(f"Container {container_id} not found")
        except Exception as e:
            raise RuntimeError(f"Failed to get logs: {e}")

    def wait_container(self, container_id: str, timeout: Optional[int] = None) -> int:
        """Wait for container to stop and return exit code."""
        try:
            container = self.client.containers.get(container_id)
            
            # Podman doesn't have a direct wait with timeout
            # We'll implement it with polling
            start_time = time.time()
            while True:
                container.reload()
                if container.status != 'running':
                    return container.attrs.get('State', {}).get('ExitCode', 0)
                
                if timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Timeout waiting for container after {timeout}s")
                
                time.sleep(0.5)
                
        except NotFound:
            raise ValueError(f"Container {container_id} not found")
        except TimeoutError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to wait for container: {e}")
    
    def create_pty_session(
        self,
        container_id: str,
        command: str,
        env: Optional[Dict[str, str]] = None
    ) -> PTYSession:
        """Create an interactive PTY session with the container."""
        try:
            # Create PTY session (Podman uses exec but through HTTP API for streaming)
            session = PTYSession(container_id)
            
            # Start streaming thread for Podman
            self._start_pty_stream_podman(session, container_id, command, env)
            
            return session
        except NotFound:
            raise ValueError(f"Container {container_id} not found")
        except Exception as e:
            raise RuntimeError(f"Failed to create PTY session: {e}")
    
    def _start_pty_stream_podman(self, session: PTYSession, container_id: str, command: str, env: Optional[Dict[str, str]]) -> None:
        """Start the streaming thread for Podman PTY session using subprocess for reliability."""
        def stream_handler():
            import subprocess
            import select
            import fcntl
            import os
            
            try:
                # Build podman exec command
                podman_cmd = ["podman", "exec", "-it"]
                if env:
                    for key, val in env.items():
                        podman_cmd.extend(["-e", f"{key}={val}"])
                podman_cmd.extend([container_id, "/bin/sh", "-c", command])
                
                # Start podman process with PTY
                proc = subprocess.Popen(
                    podman_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0
                )
                session._process = proc
                
                # Make stdout non-blocking
                if proc.stdout:
                    flags = fcntl.fcntl(proc.stdout.fileno(), fcntl.F_GETFL)
                    fcntl.fcntl(proc.stdout.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
                
                # Stream output and handle input
                while not session._closed and proc.poll() is None:
                    # Check for output
                    if proc.stdout:
                        ready, _, _ = select.select([proc.stdout], [], [], 0.1)
                        if ready:
                            try:
                                chunk = proc.stdout.read(4096)
                                if chunk:
                                    session.output_queue.put(chunk)
                                else:
                                    # EOF received
                                    break
                            except Exception:
                                pass
                    
                    # Check for input
                    try:
                        data = session.input_queue.get_nowait()
                        if data and proc.stdin:
                            proc.stdin.write(data)
                            proc.stdin.flush()
                    except queue.Empty:
                        pass
                    except Exception:
                        break
                
                # Process ended
                session._closed = True
                if proc.poll() is None:
                    proc.terminate()
                    
            except Exception as e:
                if not session._closed:
                    print(f"PTY stream error: {e}")
            finally:
                session._closed = True
        
        session._stream_thread = threading.Thread(target=stream_handler, daemon=True)
        session._stream_thread.start()
    
    def send_pty_input(self, session: PTYSession, data: bytes) -> None:
        """Send input data to the PTY session."""
        if not session._closed:
            session.input_queue.put(data)
    
    def stream_pty_output(self, session: PTYSession):
        """Stream output from the PTY session."""
        while not session._closed:
            try:
                chunk = session.output_queue.get(timeout=0.1)
                yield chunk
            except queue.Empty:
                # Check if stream thread is still alive
                if session._stream_thread and not session._stream_thread.is_alive():
                    # Drain any remaining output
                    while not session.output_queue.empty():
                        try:
                            yield session.output_queue.get_nowait()
                        except queue.Empty:
                            break
                    break
                continue
            except Exception:
                break
    
    def close_pty_session(self, session: PTYSession) -> None:
        """Close the PTY session."""
        session._closed = True
        
        # Terminate subprocess if exists
        if hasattr(session, '_process') and session._process:
            try:
                session._process.terminate()
                session._process.wait(timeout=1)
            except:
                pass
        
        # Wait for thread to finish
        if session._stream_thread:
            session._stream_thread.join(timeout=1)