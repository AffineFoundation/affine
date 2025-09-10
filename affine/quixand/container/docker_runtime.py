"""Docker SDK implementation of container runtime."""

from __future__ import annotations

import io
import logging
import os
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Suppress Docker SDK debug logs
logging.getLogger('docker.utils.config').setLevel(logging.WARNING)
logging.getLogger('docker.auth').setLevel(logging.WARNING)

try:
    import docker
    from docker import DockerClient
    from docker.models.containers import Container
    from docker.errors import NotFound, APIError
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False
    DockerClient = Any
    Container = Any

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


class DockerRuntime(ContainerRuntime):
    """Docker runtime implementation using Docker SDK."""

    def __init__(self, base_url: Optional[str] = None):
        if not HAS_DOCKER:
            raise ImportError("docker package is not installed. Install with: pip install docker")
        
        # Use environment variable or default Docker socket
        base_url = base_url or os.environ.get("DOCKER_HOST", "unix://var/run/docker.sock")
        self.client: DockerClient = docker.DockerClient(base_url=base_url)
        
        # Test connection
        try:
            self.client.ping()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Docker daemon: {e}")

    def _ensure_image_exists(self, image: str) -> None:
        """Ensure image exists locally, pull if needed."""
        try:
            # Check if image exists
            self.client.api.inspect_image(image)
        except NotFound:
            # Image doesn't exist, pull it
            print(f"Image {image} not found locally, pulling...")
            try:
                for line in self.client.api.pull(image, stream=True, decode=True):
                    if 'status' in line:
                        status = line.get('status', '')
                        progress = line.get('progress', '')
                        if progress:
                            print(f"  {status}: {progress}")
                        elif status and status not in ['Pulling fs layer', 'Waiting', 'Download complete']:
                            print(f"  {status}")
                print(f"Image {image} pulled successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to pull image {image}: {e}")
    
    def create_container(self, config: ContainerConfig) -> str:
        """Create a new container and return its ID."""
        # Ensure image exists
        self._ensure_image_exists(config.image)
        
        # Prepare volume mounts
        volumes = {}
        binds = {}
        for vol in config.volumes:
            volumes[vol.target] = {}
            bind_config = {
                'bind': vol.target,
                'mode': 'ro' if vol.read_only else 'rw'
            }
            binds[vol.source] = bind_config

        # Prepare resource limits
        host_config_kwargs = {
            'binds': binds,
            'network_mode': config.resources.network if config.resources else 'bridge',
        }
        
        if config.resources:
            if config.resources.cpu_limit:
                # Convert CPU cores to nano CPUs (1 core = 1e9 nano CPUs)
                host_config_kwargs['nano_cpus'] = int(config.resources.cpu_limit * 1e9)
            if config.resources.mem_limit:
                host_config_kwargs['mem_limit'] = config.resources.mem_limit
            if config.resources.pids_limit:
                host_config_kwargs['pids_limit'] = config.resources.pids_limit

        # Prepare port bindings
        if config.ports:
            port_bindings = {}
            exposed_ports = {}
            for container_port, host_port in config.ports.items():
                port_key = f"{container_port}/tcp"
                exposed_ports[port_key] = {}
                port_bindings[port_key] = host_port
            host_config_kwargs['port_bindings'] = port_bindings
        else:
            exposed_ports = None

        # Create host config
        host_config = self.client.api.create_host_config(**host_config_kwargs)

        # Create container
        container_kwargs = {
            'image': config.image,
            'name': config.name,
            'working_dir': config.workdir,
            'environment': config.env,
            'volumes': list(volumes.keys()) if volumes else None,
            'host_config': host_config,
            'labels': config.labels,
            'detach': True,
            'tty': True,
            'stdin_open': True,
        }

        if config.entrypoint is not None:
            container_kwargs['entrypoint'] = config.entrypoint
        if config.command is not None:
            container_kwargs['command'] = config.command            
        if exposed_ports:
            container_kwargs['ports'] = exposed_ports

        try:
            response = self.client.api.create_container(**container_kwargs)
            return response['Id']
        except APIError as e:
            raise RuntimeError(f"Failed to create container: {e}")

    def start_container(self, container_id: str) -> None:
        """Start a created container."""
        try:
            self.client.api.start(container_id)
        except NotFound:
            raise ValueError(f"Container {container_id} not found")
        except APIError as e:
            raise RuntimeError(f"Failed to start container: {e}")

    def stop_container(self, container_id: str, timeout: int = 10) -> None:
        """Stop a running container."""
        try:
            self.client.api.stop(container_id, timeout=timeout)
        except NotFound:
            # Container doesn't exist, consider it stopped
            pass
        except APIError as e:
            # Log error but don't raise - container might already be stopped
            pass

    def remove_container(self, container_id: str, force: bool = False) -> None:
        """Remove a container."""
        try:
            self.client.api.remove_container(container_id, force=force)
        except NotFound:
            # Container already removed
            pass
        except APIError as e:
            if force:
                # Force removal requested, ignore errors
                pass
            else:
                raise RuntimeError(f"Failed to remove container: {e}")

    def get_container_info(self, container_id: str) -> ContainerInfo:
        """Get information about a container."""
        try:
            data = self.client.api.inspect_container(container_id)
            
            # Map Docker state to our ContainerState
            state_str = data['State']['Status'].lower()
            state_map = {
                'created': ContainerState.CREATED,
                'running': ContainerState.RUNNING,
                'paused': ContainerState.PAUSED,
                'exited': ContainerState.EXITED,
                'dead': ContainerState.DEAD,
                'removing': ContainerState.REMOVING,
                'restarting': ContainerState.RUNNING,
            }
            state = state_map.get(state_str, ContainerState.ERROR)

            # Parse timestamps - handle nanosecond precision
            def parse_docker_timestamp(timestamp_str):
                """Parse Docker timestamp with nanosecond precision."""
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
            
            created_at = parse_docker_timestamp(data['Created'])
            started_at = parse_docker_timestamp(data['State'].get('StartedAt'))
            finished_at = parse_docker_timestamp(data['State'].get('FinishedAt'))

            return ContainerInfo(
                id=data['Id'],
                name=data['Name'].lstrip('/'),
                state=state,
                created_at=created_at,
                started_at=started_at,
                finished_at=finished_at,
                exit_code=data['State'].get('ExitCode'),
                labels=data.get('Config', {}).get('Labels', {}),
            )
        except NotFound:
            raise ValueError(f"Container {container_id} not found")
        except Exception as e:
            raise RuntimeError(f"Failed to inspect container: {e}")

    def container_exists(self, container_id: str) -> bool:
        """Check if container exists."""
        try:
            self.client.api.inspect_container(container_id)
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
            # Create exec instance
            exec_kwargs = {
                'container': container_id,
                'cmd': config.command,
                'stdout': True,
                'stderr': True,
                'stdin': config.stdin,
                'tty': config.tty,
                'privileged': config.privileged,
            }
            
            if config.env:
                exec_kwargs['environment'] = config.env
            if config.workdir:
                exec_kwargs['workingdir'] = config.workdir
            if config.user:
                exec_kwargs['user'] = config.user

            exec_id = self.client.api.exec_create(**exec_kwargs)['Id']
            
            # Start exec with timeout handling
            if timeout:
                # For timeout, we need to handle it manually
                import threading
                result = {'stdout': b'', 'stderr': b'', 'exit_code': -1}
                error = [None]
                
                def run_exec():
                    try:
                        output = self.client.api.exec_start(
                            exec_id,
                            detach=config.detach,
                            stream=False
                        )
                        # Output is combined stdout/stderr when stream=False
                        result['stdout'] = output
                        exec_info = self.client.api.exec_inspect(exec_id)
                        result['exit_code'] = exec_info.get('ExitCode', 0)
                    except Exception as e:
                        error[0] = e
                
                thread = threading.Thread(target=run_exec)
                thread.daemon = True
                thread.start()
                thread.join(timeout)
                
                if thread.is_alive():
                    # Timeout occurred
                    raise TimeoutError(f"Command timed out after {timeout}s")
                
                if error[0]:
                    raise error[0]
                    
                stdout = result['stdout']
                stderr = result['stderr']
                exit_code = result['exit_code']
            else:
                # No timeout, execute normally
                output = self.client.api.exec_start(
                    exec_id,
                    detach=config.detach,
                    stream=False
                )
                stdout = output
                stderr = b''
                exec_info = self.client.api.exec_inspect(exec_id)
                exit_code = exec_info.get('ExitCode', 0)
            
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
            # Put archive to container (dest must be a directory)
            self.client.api.put_archive(
                container_id,
                dest_dir,
                tar_stream.read()
            )
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
            # Get archive from container
            bits, stat = self.client.api.get_archive(container_id, source)
            
            # Write to temporary tar file and extract
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
        containers = self.client.api.containers(all=all)
        result = []
        
        for container in containers:
            # Map Docker state to our ContainerState
            state_str = container['State'].lower()
            state_map = {
                'created': ContainerState.CREATED,
                'running': ContainerState.RUNNING,
                'paused': ContainerState.PAUSED,
                'exited': ContainerState.EXITED,
                'dead': ContainerState.DEAD,
                'removing': ContainerState.REMOVING,
                'restarting': ContainerState.RUNNING,
            }
            state = state_map.get(state_str, ContainerState.ERROR)
            
            # Convert timestamp
            created_at = datetime.fromtimestamp(container['Created'])
            
            result.append(ContainerInfo(
                id=container['Id'],
                name=container['Names'][0].lstrip('/') if container['Names'] else '',
                state=state,
                created_at=created_at,
                labels=container.get('Labels', {}),
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
        kwargs = {
            'container': container_id,
            'stdout': stdout,
            'stderr': stderr,
            'stream': False,
        }
        
        if since:
            kwargs['since'] = since
        if until:
            kwargs['until'] = until
        if tail:
            kwargs['tail'] = tail
        
        try:
            logs = self.client.api.logs(**kwargs)
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
            result = self.client.api.wait(container_id, timeout=timeout)
            return result.get('StatusCode', 0)
        except NotFound:
            raise ValueError(f"Container {container_id} not found")
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
            # Create exec instance for PTY
            exec_kwargs = {
                'container': container_id,
                'cmd': ['/bin/sh', '-c', command],
                'stdout': True,
                'stderr': True,
                'stdin': True,
                'tty': True,
            }
            if env:
                exec_kwargs['environment'] = env
            
            exec_response = self.client.api.exec_create(**exec_kwargs)
            exec_id = exec_response['Id']
            
            # Create PTY session
            session = PTYSession(container_id, exec_id)
            
            # Start streaming thread
            self._start_pty_stream(session)
            
            return session
        except NotFound:
            raise ValueError(f"Container {container_id} not found")
        except Exception as e:
            raise RuntimeError(f"Failed to create PTY session: {e}")
    
    def _start_pty_stream(self, session: PTYSession) -> None:
        """Start the streaming thread for Docker PTY session."""
        def stream_handler():
            try:
                # Start exec with socket for bidirectional communication
                socket = self.client.api.exec_start(
                    session.exec_id,
                    detach=False,
                    tty=True,
                    stream=True,
                    socket=True
                )
                session._socket = socket
                
                # Start reader thread for output
                def read_output():
                    while not session._closed:
                        try:
                            socket._sock.settimeout(0.1)
                            chunk = socket._sock.recv(4096)
                            if chunk:
                                session.output_queue.put(chunk)
                            else:
                                # Socket closed - EOF received
                                session._closed = True
                                break
                        except Exception:
                            if not session._closed:
                                continue
                            break
                
                reader_thread = threading.Thread(target=read_output, daemon=True)
                reader_thread.start()
                
                # Handle input in main streaming thread
                while not session._closed:
                    try:
                        data = session.input_queue.get(timeout=0.1)
                        if data:
                            socket._sock.send(data)
                    except queue.Empty:
                        continue
                    except Exception:
                        break
                
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
        
        # Close socket if exists
        if session._socket:
            try:
                session._socket.close()
            except:
                pass
        
        # Wait for thread to finish
        if session._stream_thread:
            session._stream_thread.join(timeout=1)