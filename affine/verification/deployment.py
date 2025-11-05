"""Model deployment management using SSH and SGLang."""

import os
import asyncio
import random
import subprocess
from typing import Optional, Dict
from dataclasses import dataclass

from affine.setup import logger

# Global SSH tunnel manager
_ssh_tunnels: Dict[int, subprocess.Popen] = {}


@dataclass
class DeploymentInfo:
    """Information about a deployed model."""

    model: str
    revision: str
    endpoint: str  # http://host:port/v1
    container_id: str
    host: str
    port: int


class ModelDeployment:
    """Manage model deployment on GPU machines using SSH and SGLang."""

    def __init__(
        self,
        mode: Optional[str] = None,
        ssh_host: Optional[str] = None,
        ssh_key: Optional[str] = None,
        work_dir: Optional[str] = None,
    ):
        """Initialize deployment manager.

        Args:
            mode: Deployment mode ('remote' or 'local')
            ssh_host: SSH host (e.g., user@gpu-server.com)
            ssh_key: Path to SSH private key
            work_dir: Working directory on host
        """
        self.mode = mode or os.getenv("GPU_MODE", "remote")
        self.ssh_host = ssh_host or os.getenv("GPU_HOST", "")
        self.ssh_key = ssh_key or os.getenv("GPU_SSH_KEY", "")
        self.ssh_port = os.getenv("GPU_SSH_PORT", "22")
        self.work_dir = work_dir or os.getenv("GPU_WORK_DIR", "/opt/affine/models")

        if self.mode not in ["remote", "local"]:
            raise ValueError(f"Invalid GPU_MODE: {self.mode}. Must be 'remote' or 'local'")

        if self.mode == "remote" and not self.ssh_host:
            raise ValueError("GPU_HOST must be set when GPU_MODE=remote")

        # SGLang Docker image
        self.sglang_image = os.getenv("SGLANG_DOCKER_IMAGE", "lmsysorg/sglang:latest")

        # Port range for model serving
        self.port_range_start = int(os.getenv("GPU_PORT_RANGE_START", "30000"))
        self.port_range_end = int(os.getenv("GPU_PORT_RANGE_END", "31000"))

        if self.mode == "local":
            logger.info("Initialized deployment manager for local GPU")
        else:
            logger.info(f"Initialized deployment manager for remote GPU: {self.ssh_host}")

    async def _run_command(self, command: str, timeout: int = 300) -> tuple[int, str, str]:
        """Run command on GPU host (local or remote via SSH).

        Args:
            command: Command to run
            timeout: Command timeout in seconds

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        if self.mode == "local":
            # Run command locally
            logger.debug(f"Running local command: {command}")

            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                return proc.returncode, stdout.decode("utf-8", errors="ignore"), stderr.decode("utf-8", errors="ignore")
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise TimeoutError(f"Command timed out after {timeout}s")

        else:
            # Run command via SSH
            ssh_cmd = ["ssh"]

            if self.ssh_key:
                ssh_cmd.extend(["-i", self.ssh_key])

            # Add port if not default
            if self.ssh_port != "22":
                ssh_cmd.extend(["-p", self.ssh_port])

            # SSH options
            ssh_cmd.extend([
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=10",
                self.ssh_host,
                command,
            ])

            logger.debug(f"Running SSH command: {' '.join(ssh_cmd)}")

            # Run command
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                return proc.returncode, stdout.decode("utf-8", errors="ignore"), stderr.decode("utf-8", errors="ignore")
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise TimeoutError(f"SSH command timed out after {timeout}s")

    async def _run_ssh_command(self, command: str, timeout: int = 300) -> tuple[int, str, str]:
        """Legacy wrapper for _run_command (for backwards compatibility)."""
        return await self._run_command(command, timeout)

    async def _find_available_port(self) -> int:
        """Find an available port on remote host.

        Returns:
            Available port number
        """
        # Try random ports in range
        for _ in range(20):
            port = random.randint(self.port_range_start, self.port_range_end)

            # Check if port is in use
            cmd = f"docker ps --filter publish={port} --format '{{{{.ID}}}}'"
            ret, stdout, _ = await self._run_command(cmd)

            if ret == 0 and not stdout.strip():
                return port

        raise RuntimeError("No available ports in range")

    async def deploy_model(
        self,
        model: str,
        revision: str,
        hf_token: Optional[str] = None,
    ) -> DeploymentInfo:
        """Deploy a model on remote GPU using SGLang in Docker.

        Args:
            model: HuggingFace model name
            revision: Model revision
            hf_token: HuggingFace token for downloading

        Returns:
            DeploymentInfo object
        """
        logger.info(f"Deploying model {model}@{revision} on {self.ssh_host}")

        # Get HF token
        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN", "")

        if not hf_token:
            logger.warning("HF_TOKEN not set - model download may fail for private models or hit rate limits")

        # Find available port
        port = await self._find_available_port()

        # Generate container name
        container_name = f"sglang_{model.replace('/', '_')}_{revision[:8]}_{port}"

        # Build Docker run command
        # Using Docker-in-Docker with SGLang
        hf_env = f"-e HF_TOKEN={hf_token} -e HUGGING_FACE_HUB_TOKEN={hf_token}" if hf_token else ""
        docker_cmd = f"""
docker run -d \\
  --name {container_name} \\
  --gpus all \\
  --ipc=host \\
  -p {port}:30000 \\
  {hf_env} \\
  --shm-size 16g \\
  {self.sglang_image} \\
  python3 -m sglang.launch_server \\
    --model-path {model} \\
    --revision {revision} \\
    --host 0.0.0.0 \\
    --port 30000 \\
    --mem-fraction-static 0.7 \\
    --trust-remote-code
        """.strip()

        # Run Docker command
        logger.info(f"Starting container {container_name} on port {port}")
        ret, stdout, stderr = await self._run_command(docker_cmd, timeout=600)

        if ret != 0:
            raise RuntimeError(f"Failed to start container: {stderr}")

        container_id = stdout.strip()
        logger.info(f"Container started: {container_id[:12]}")

        # Wait for model to be ready (check health)
        logger.info("Waiting for model to be ready...")
        await self._wait_for_model_ready(port, timeout=300)

        # For remote mode, setup SSH tunnel
        if self.mode == "remote":
            await self._setup_ssh_tunnel(port)

        # Build endpoint URL
        # For remote mode, use localhost since we access via SSH tunnel
        endpoint = f"http://localhost:{port}/v1"

        deployment = DeploymentInfo(
            model=model,
            revision=revision,
            endpoint=endpoint,
            container_id=container_id,
            host="localhost",
            port=port,
        )

        logger.info(f"Model deployed successfully: {endpoint}")
        return deployment

    async def _setup_ssh_tunnel(self, port: int) -> None:
        """Setup SSH tunnel for accessing remote port.

        Args:
            port: Remote port to forward
        """
        global _ssh_tunnels

        # Kill existing tunnel for this port
        if port in _ssh_tunnels:
            try:
                _ssh_tunnels[port].terminate()
                _ssh_tunnels[port].wait(timeout=5)
            except:
                pass
            del _ssh_tunnels[port]

        # Build SSH command for port forwarding
        ssh_cmd = [
            "ssh", "-f", "-N",
            "-i", self.ssh_key,
            "-p", self.ssh_port,
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=60",
            "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=no",
            "-L", f"{port}:localhost:{port}",
            self.ssh_host,
        ]

        logger.info(f"Setting up SSH tunnel: localhost:{port} -> {self.ssh_host}:{port}")

        # Start SSH tunnel
        try:
            proc = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Give it a moment to establish
            await asyncio.sleep(2)

            # Check if tunnel is still running
            if proc.poll() is not None:
                # SSH process exited - might mean tunnel already exists
                # Check if port is accessible
                logger.warning(f"SSH tunnel process exited (code: {proc.poll()}), checking if port is accessible")
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                try:
                    result = sock.connect_ex(('localhost', port))
                    sock.close()
                    if result == 0:
                        logger.info(f"Port {port} is already accessible (tunnel may exist on host)")
                        return
                    else:
                        raise RuntimeError(f"SSH tunnel failed and port {port} is not accessible")
                except Exception as check_e:
                    raise RuntimeError(f"SSH tunnel failed to start (exit code: {proc.poll()}) and port check failed: {check_e}")

            _ssh_tunnels[port] = proc
            logger.info(f"SSH tunnel established for port {port}")

        except Exception as e:
            logger.error(f"Failed to setup SSH tunnel for port {port}: {e}")
            raise

    async def _wait_for_model_ready(self, port: int, timeout: int = 300) -> None:
        """Wait for model to be ready by checking v1/models endpoint.

        Args:
            port: Port to check
            timeout: Max wait time in seconds
        """
        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if container is running and API is accessible
            # Try /v1/models endpoint which is standard for OpenAI-compatible APIs
            cmd = f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{port}/v1/models || echo '000'"
            ret, stdout, _ = await self._run_command(cmd, timeout=10)

            if ret == 0:
                status_code = stdout.strip()
                if status_code == "200":
                    logger.info("Model is ready")
                    return
                elif status_code != "000":
                    logger.debug(f"Waiting for model... (status: {status_code})")

            # Wait before retry
            await asyncio.sleep(5)

        raise TimeoutError(f"Model failed to become ready within {timeout}s")

    async def cleanup_deployment(self, container_id: str) -> None:
        """Clean up a deployed model.

        Args:
            container_id: Container ID to clean up
        """
        logger.info(f"Cleaning up container {container_id[:12]}")

        # Stop and remove container
        cmd = f"docker stop {container_id} && docker rm {container_id}"
        ret, stdout, stderr = await self._run_command(cmd, timeout=60)

        if ret != 0:
            logger.warning(f"Failed to cleanup container: {stderr}")
        else:
            logger.info(f"Container {container_id[:12]} cleaned up")

    async def cleanup_all_sglang_containers(self) -> None:
        """Clean up all SGLang containers on the GPU host.

        This is useful to clear old containers before starting verification.
        """
        logger.info("Cleaning up all SGLang containers on GPU host...")

        # List all containers with name starting with "sglang_"
        list_cmd = "docker ps -a --filter name=^sglang_ --format '{{.ID}}'"
        ret, stdout, stderr = await self._run_command(list_cmd, timeout=30)

        if ret != 0:
            logger.warning(f"Failed to list containers: {stderr}")
            return

        container_ids = [cid.strip() for cid in stdout.strip().split('\n') if cid.strip()]

        if not container_ids:
            logger.info("No SGLang containers found to clean up")
            return

        logger.info(f"Found {len(container_ids)} SGLang containers to clean up")

        # Stop and remove all containers
        for container_id in container_ids:
            try:
                cmd = f"docker stop {container_id} && docker rm {container_id}"
                ret, stdout, stderr = await self._run_command(cmd, timeout=60)
                if ret == 0:
                    logger.info(f"Cleaned up container {container_id[:12]}")
                else:
                    logger.warning(f"Failed to clean up container {container_id[:12]}: {stderr}")
            except Exception as e:
                logger.warning(f"Error cleaning up container {container_id[:12]}: {e}")

        logger.info("SGLang container cleanup completed")

    async def test_inference(
        self,
        endpoint: str,
        prompt: str,
        model: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> Dict:
        """Test inference on deployed model.

        Args:
            endpoint: Model endpoint (e.g., http://host:port/v1)
            prompt: Prompt to test
            model: Model name
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Response dictionary
        """
        import aiohttp

        url = f"{endpoint}/chat/completions"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()

    async def list_deployments(self) -> list[Dict]:
        """List all active SGLang deployments.

        Returns:
            List of deployment info
        """
        # List all SGLang containers
        cmd = "docker ps --filter name=sglang_ --format '{{.ID}}\\t{{.Names}}\\t{{.Ports}}'"
        ret, stdout, stderr = await self._run_command(cmd)

        if ret != 0:
            raise RuntimeError(f"Failed to list containers: {stderr}")

        deployments = []
        for line in stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 3:
                container_id = parts[0]
                name = parts[1]
                ports = parts[2]

                deployments.append({
                    "container_id": container_id,
                    "name": name,
                    "ports": ports,
                })

        return deployments
