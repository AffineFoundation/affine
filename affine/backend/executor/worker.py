"""
Executor Worker - Individual Task Executor

Each worker fetches tasks for a specific environment and executes them.
Uses authenticated API endpoints with wallet signature verification.
"""

import asyncio
import time
import traceback
import aiohttp
from typing import Optional, Dict, Any
from dataclasses import dataclass

from affine.core.setup import logger, wallet
from affine.core.models import SampleSubmission


@dataclass
class WorkerMetrics:
    """Metrics for a worker."""
    
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    last_task_at: Optional[float] = None


class ExecutorWorker:
    """Worker that executes sampling tasks for a specific environment.
    
    Uses authenticated API endpoints with wallet signature verification.
    """
    
    def __init__(
        self,
        worker_id: int,
        env: str,
        api_base_url: str,
        poll_interval: int = 5,
    ):
        """Initialize executor worker.
        
        Args:
            worker_id: Unique worker ID
            env: Environment to execute tasks for (e.g., "affine:sat")
            api_base_url: API server URL
            poll_interval: How often to poll for tasks (seconds)
        """
        self.worker_id = worker_id
        self.env = env
        self.api_base_url = api_base_url
        self.poll_interval = poll_interval
        
        self.running = False
        self.metrics = WorkerMetrics()
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Environment executor (will be initialized lazily)
        self.env_executor = None
        
        # Executor identity (from wallet)
        if wallet:
            self.hotkey = wallet.hotkey.ss58_address
        else:
            self.hotkey = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)  # 5 min timeout
            )
        return self._session
    
    async def _init_env_executor(self):
        """Initialize environment executor lazily."""
        if self.env_executor is not None:
            return

        try:
            # Use affine's environment wrapper (not affinetes directly)
            from affine.core.environments import create_environment

            # create_environment returns the SDK environment instance
            self.env_executor = await create_environment(self.env)

            logger.info(f"[Worker-{self.worker_id}] Initialized {self.env} executor")

        except Exception as e:
            logger.error(f"[Worker-{self.worker_id}] Failed to initialize executor: {e}")
            raise
    
    async def start(self):
        """Start the worker."""
        logger.info(f"[Worker-{self.worker_id}] Starting worker for {self.env}...")

        # Validate wallet
        if not wallet or not self.hotkey:
            raise RuntimeError("Wallet not configured for worker")
        
        self.running = True
        
        # Initialize environment executor
        await self._init_env_executor()
        
        # Start execution loop
        await self._execution_loop()
    
    async def stop(self):
        """Stop the worker."""
        logger.info(f"[Worker-{self.worker_id}] Stopping worker...")
        self.running = False
    
    def _sign_message(self, message: str) -> str:
        """Sign a message using the wallet.
        
        Args:
            message: Message to sign
            
        Returns:
            Hex-encoded signature
        """
        if not wallet:
            raise RuntimeError("Wallet not configured")
        
        # Sign the message
        signature = wallet.hotkey.sign(message.encode())
        return signature.hex()
    
    def _get_auth_headers(self, message: Optional[str] = None) -> Dict[str, str]:
        """Get authentication headers for API requests.
        
        Args:
            message: Message to sign (defaults to timestamp-based message)
            
        Returns:
            Headers dict with hotkey, signature, and message
        """
        if message is None:
            message = f"executor:{self.hotkey}:{int(time.time())}"
        
        signature = self._sign_message(message)
        
        return {
            "X-Executor-Hotkey": self.hotkey,
            "X-Executor-Signature": signature,
            "X-Executor-Message": message,
        }
    
    async def _fetch_task(self) -> Optional[Dict]:
        """Fetch a pending task from API with authentication.
        
        Uses weighted random selection via TaskPoolManager.
        
        Returns:
            Task dictionary or None if no task available
        """
        try:
            url = f"{self.api_base_url}/api/v1/tasks/fetch"
            params = {"env": self.env}
            
            # Get authentication headers
            headers = self._get_auth_headers()
            
            session = await self._get_session()
            async with session.post(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    response = await resp.json()
                    
                    if response and "task" in response and response["task"] is not None:
                        task = response["task"]
                        logger.info(
                            f"[Worker-{self.worker_id}] Fetched task: "
                            f"uuid={task.get('task_uuid', 'N/A')[:8]}... "
                            f"task_id={task.get('task_id')} "
                            f"miner={task.get('miner_hotkey', '')[:12]}..."
                        )
                        return task
            
            return None
        
        except Exception as e:
            logger.error(f"[Worker-{self.worker_id}] Error fetching task: {e}")
            return None
    
    async def _check_consecutive_errors(self, miner_hotkey: str) -> int:
        """Check consecutive errors for a miner.
        
        Args:
            miner_hotkey: Miner hotkey
        
        Returns:
            Number of consecutive errors
        """
        try:
            url = f"{self.api_base_url}/api/v1/miners/{miner_hotkey}/consecutive-errors"
            session = await self._get_session()
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    response = await resp.json()
                    return response.get("consecutive_errors", 0)
            
            return 0
        
        except Exception as e:
            logger.error(f"[Worker-{self.worker_id}] Error checking consecutive errors: {e}")
            return 0
    
    async def _pause_miner(self, miner_hotkey: str, consecutive_errors: int) -> bool:
        """Pause a miner due to consecutive errors.
        
        Args:
            miner_hotkey: Miner hotkey
            consecutive_errors: Number of consecutive errors
        
        Returns:
            True if successful
        """
        try:
            url = f"{self.api_base_url}/api/v1/miners/{miner_hotkey}/pause"
            
            # Exponential backoff: 10 minutes * 2^(errors-1), max 2 hours
            base_duration = 600  # 10 minutes
            duration = min(base_duration * (2 ** (consecutive_errors - 1)), 7200)
            
            data = {
                "duration_seconds": int(duration),
                "reason": f"Consecutive errors: {consecutive_errors}"
            }
            
            session = await self._get_session()
            async with session.put(url, json=data) as resp:
                if resp.status == 200:
                    logger.warning(
                        f"[Worker-{self.worker_id}] Paused miner {miner_hotkey[:8]}... "
                        f"for {duration}s due to {consecutive_errors} consecutive errors"
                    )
                    return True
            
            return False
        
        except Exception as e:
            logger.error(f"[Worker-{self.worker_id}] Error pausing miner: {e}")
            return False
    
    async def _execute_task(self, task: Dict) -> tuple[SampleSubmission, bool]:
        """Execute a sampling task.
        
        Args:
            task: Task dictionary from API response
                - task_id: Dataset index (string)
                - task_uuid: Queue UUID (string)
                - miner_hotkey: Miner's hotkey
                - model_revision: Model revision
                - model: Model repo/name
                - env: Environment
        
        Returns:
            Tuple of (SampleSubmission, is_http_error)
            - SampleSubmission: Signed submission with score, latency, extra
            - is_http_error: True if error is HTTP-related (report to API), False if executor error (retry)
        """
        start_time = time.time()
        
        try:
            # Extract task parameters
            model = task["model"]
            task_id = int(task["task_id"])  # Dataset index as integer
            task_uuid = task.get("task_uuid", "")
            miner_hotkey = task["miner_hotkey"]
            
            logger.info(
                f"[Worker-{self.worker_id}] Executing task: "
                f"uuid={task_uuid[:8]}... miner={miner_hotkey[:12]}... model={model} task_id={task_id}"
            )
            
            # Execute sampling using affine's environment wrapper
            # env_executor.evaluate() accepts miner=None and dynamic kwargs
            result = await self.env_executor.evaluate(
                miner=None,
                model=model,
                base_url="https://llm.chutes.ai/v1",
                task_id=task_id,
            )
            
            execution_time = time.time() - start_time
            self.metrics.total_execution_time += execution_time
            
            # Build SampleSubmission (only contains task_uuid, score, latency, extra)
            submission = SampleSubmission(
                task_uuid=task_uuid,
                score=float(result.score),
                latency_ms=int(result.latency_seconds * 1000),
                extra=result.extra or {},
                signature="",  # Will be signed below
            )
            
            # Sign submission using wallet
            if wallet:
                sign_data = submission.get_sign_data()
                signature_bytes = wallet.hotkey.sign(sign_data.encode())
                submission.signature = signature_bytes.hex()
            
            logger.info(
                f"[Worker-{self.worker_id}] Task completed: "
                f"uuid={task_uuid[:8]}... score={submission.score:.4f} "
                f"latency={submission.latency_ms}ms time={execution_time:.2f}s"
            )
            
            return submission, False  # Success, not HTTP error
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.total_execution_time += execution_time
            
            error_str = str(e)
            logger.error(f"[Worker-{self.worker_id}] Task execution failed: {error_str}")
            traceback.print_exc()
            
            # Classify error type
            is_http_error = self._is_http_error(error_str)
            
            if is_http_error:
                # HTTP error: report to API with zero score
                submission = SampleSubmission(
                    task_uuid=task.get("task_uuid", ""),
                    score=0.0,
                    latency_ms=int(execution_time * 1000),
                    extra={"error": error_str, "error_type": self._classify_error(error_str)},
                    signature="",
                )
                
                # Sign the failed submission
                if wallet:
                    sign_data = submission.get_sign_data()
                    signature_bytes = wallet.hotkey.sign(sign_data.encode())
                    submission.signature = signature_bytes.hex()
                
                return submission, True  # HTTP error, report to API
            else:
                # Executor error: don't report, let task retry
                logger.warning(f"[Worker-{self.worker_id}] Executor error (will retry): {error_str}")
                raise  # Re-raise to trigger retry in execution loop
    
    async def _submit_result(self, task: Dict, submission: SampleSubmission) -> bool:
        """Submit task result to API with authentication.
        
        Args:
            task: Original task from API (includes miner metadata)
            submission: Signed sample submission (only contains task_uuid, score, latency, extra, signature)
        
        Returns:
            True if successful
        """
        try:
            # Get authentication headers
            headers = self._get_auth_headers()
            
            session = await self._get_session()
            
            # Submit to new /api/v1/samples/submit endpoint
            # API will merge miner metadata from task queue
            url = f"{self.api_base_url}/api/v1/samples/submit"
            submit_data = {
                "task_uuid": submission.task_uuid,
                "score": submission.score,
                "latency_ms": submission.latency_ms,
                "extra": submission.extra,
                "signature": submission.signature,
            }
            
            async with session.post(url, json=submit_data, headers=headers) as resp:
                if resp.status == 200:
                    response = await resp.json()
                    logger.info(
                        f"[Worker-{self.worker_id}] Submitted result: "
                        f"task_uuid={submission.task_uuid[:8]}... "
                        f"score={submission.score:.4f}"
                    )
                    
                    self.metrics.tasks_completed += 1
                    return True
                elif resp.status == 400:
                    # Bad request (e.g., invalid signature, task not found)
                    error_detail = await resp.text()
                    logger.error(
                        f"[Worker-{self.worker_id}] Failed to submit (400): {error_detail}"
                    )
                    self.metrics.tasks_failed += 1
                    return False
                else:
                    logger.warning(
                        f"[Worker-{self.worker_id}] Failed to submit (status={resp.status})"
                    )
                    self.metrics.tasks_failed += 1
                    return False
        
        except Exception as e:
            logger.error(f"[Worker-{self.worker_id}] Error submitting result: {e}")
            self.metrics.tasks_failed += 1
            return False
    
    def _is_http_error(self, error_message: str) -> bool:
        """Check if error is HTTP-related (should be reported to API).
        
        Args:
            error_message: Error message
        
        Returns:
            True if HTTP error, False if executor error (should retry)
        """
        error_lower = error_message.lower()
        
        # HTTP errors that should be reported to API
        http_keywords = [
            "chute", "cold", "timeout", "connect", "network",
            "hash", "mismatch", "model", "404", "503", "502",
            "unavailable", "connection", "refused"
        ]
        
        return any(keyword in error_lower for keyword in http_keywords)
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error type from error message.
        
        Args:
            error_message: Error message
        
        Returns:
            Error type string
        """
        error_lower = error_message.lower()
        
        if "chute" in error_lower or "cold" in error_lower:
            return "chutes_error"
        elif "timeout" in error_lower:
            return "timeout_error"
        elif "connect" in error_lower or "network" in error_lower:
            return "network_error"
        elif "hash" in error_lower or "mismatch" in error_lower:
            return "hash_mismatch"
        elif "model" in error_lower:
            return "model_error"
        else:
            return "unknown_error"
    
    async def _execution_loop(self):
        """Main execution loop."""
        while self.running:
            try:
                # Fetch task (already assigned to this executor via authenticated endpoint)
                task = await self._fetch_task()
                
                if task is None:
                    # No task available, wait before polling again
                    await asyncio.sleep(self.poll_interval)
                    continue
                
                # Execute task
                try:
                    submission, is_http_error = await self._execute_task(task)
                    
                    # Submit result (both success and HTTP errors)
                    await self._submit_result(task, submission)
                    
                except Exception as e:
                    # Executor error (not HTTP): don't submit, task will be retried
                    logger.warning(
                        f"[Worker-{self.worker_id}] Executor error, task will be retried: {e}"
                    )
                    # Task lock will timeout and be released automatically
                    continue
                
                # Update metrics
                self.metrics.last_task_at = time.time()
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                logger.error(f"[Worker-{self.worker_id}] Error in execution loop: {e}")
                traceback.print_exc()
                await asyncio.sleep(self.poll_interval)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker metrics.
        
        Returns:
            Dictionary of metrics
        """
        avg_time = (
            self.metrics.total_execution_time / self.metrics.tasks_completed
            if self.metrics.tasks_completed > 0
            else 0
        )
        
        return {
            "worker_id": self.worker_id,
            "env": self.env,
            "running": self.running,
            "tasks_completed": self.metrics.tasks_completed,
            "tasks_failed": self.metrics.tasks_failed,
            "avg_execution_time": avg_time,
            "last_task_at": self.metrics.last_task_at,
        }