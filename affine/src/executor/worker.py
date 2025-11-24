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
from affine.utils.api_client import create_api_client, APIClient


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
        poll_interval: int = 5,
    ):
        """Initialize executor worker.
        
        Args:
            worker_id: Unique worker ID
            env: Environment to execute tasks for (e.g., "affine:sat")
            poll_interval: How often to poll for tasks (seconds)
        """
        self.worker_id = worker_id
        self.env = env
        self.poll_interval = poll_interval
        
        self.running = False
        self.metrics = WorkerMetrics()
        self._chutes_session: Optional[aiohttp.ClientSession] = None  # Separate session for Chutes API
        
        # API client for affine backend
        self.api_client: Optional[APIClient] = None
        
        # Environment executor (will be initialized lazily)
        self.env_executor = None
        
        # Executor identity (from wallet)
        if wallet:
            self.hotkey = wallet.hotkey.ss58_address
        else:
            self.hotkey = None
    
    async def _get_chutes_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session for Chutes API calls."""
        if self._chutes_session is None or self._chutes_session.closed:
            self._chutes_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)  # 5 min timeout
            )
        return self._chutes_session
    
    async def _init_env_executor(self):
        """Initialize environment executor lazily."""
        if self.env_executor is not None:
            return

        try:
            # Use affine's environment wrapper (not affinetes directly)
            from affine.core.environments import create_environment

            # create_environment returns the SDK environment instance
            self.env_executor = await create_environment(self.env)

            logger.info(f"[{self.env}] Initialized {self.env} executor")

        except Exception as e:
            logger.error(f"[{self.env}] Failed to initialize executor: {e}")
            raise
    
    async def initialize(self):
        """Initialize the worker (must be called serially)."""
        logger.info(f"[{self.env}] Initializing worker for {self.env}...")

        # Validate wallet
        if not wallet or not self.hotkey:
            raise RuntimeError("Wallet not configured for worker")
        
        # Initialize API client
        self.api_client = create_api_client()
        
        # Initialize environment executor
        await self._init_env_executor()
        
        logger.info(f"[{self.env}] Worker initialized for {self.env}")
    
    def start(self):
        """Start the worker execution loop (returns immediately, loop runs in background)."""
        logger.info(f"[{self.env}] Starting execution loop for {self.env}...")
        self.running = True
        # Create background task for execution loop
        asyncio.create_task(self._execution_loop())
    
    async def stop(self):
        """Stop the worker."""
        logger.info(f"[{self.env}] Stopping worker...")
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
            message = str(int(time.time()))
        
        signature = self._sign_message(message)
        
        return {
            "X-Hotkey": self.hotkey,
            "X-Signature": signature,
            "X-Message": message,
        }
    
    async def _fetch_task(self) -> Optional[Dict]:
        """Fetch next task from API."""
        try:
            headers = self._get_auth_headers()
            response = await self.api_client.post(
                "/tasks/fetch",
                params={"env": self.env},
                headers=headers
            )

            task = response.get("task") if isinstance(response, dict) else None
            if not task:
                logger.info(f"[{self.env}] No task available")
                return None

            logger.info(
                f"[{self.env}] Fetched task: "
                f"uuid={str(task.get('task_uuid','N/A'))[:8]}... "
                f"task_id={task.get('task_id')} "
                f"miner={(task.get('miner_hotkey',''))[:12]}..."
            )
            return task

        except Exception:
            logger.exception(f"[{self.env}] Error fetching task")
            return None


    
    async def _pause_miner(self, miner_hotkey: str, consecutive_errors: int) -> bool:
        """Pause a miner due to consecutive errors.
        
        Args:
            miner_hotkey: Miner hotkey
            consecutive_errors: Number of consecutive errors
        
        Returns:
            True if successful
        """
        try:
            # Exponential backoff: 10 minutes * 2^(errors-1), max 2 hours
            base_duration = 600  # 10 minutes
            duration = min(base_duration * (2 ** (consecutive_errors - 1)), 7200)
            
            data = {
                "duration_seconds": int(duration),
                "reason": f"Consecutive errors: {consecutive_errors}"
            }
            
            await self.api_client.put(f"/miners/{miner_hotkey}/pause", json=data)
            
            logger.warning(
                f"[{self.env}] Paused miner {miner_hotkey[:8]}... "
                f"for {duration}s due to {consecutive_errors} consecutive errors"
            )
            return True
        
        except Exception as e:
            logger.error(f"[{self.env}] Error pausing miner: {e}")
            return False
    
    async def _get_chute_slug(self, chute_id: str) -> Optional[str]:
        """Get chute slug from chute ID.
        
        Args:
            chute_id: Chute deployment ID (required)
        
        Returns:
            Chute slug or None if not found
        
        Raises:
            ValueError: If chute_id is empty or invalid
        """
        if not chute_id or not chute_id.strip():
            raise ValueError("chute_id cannot be empty")
        
        try:
            import os
            url = f"https://api.chutes.ai/chutes/{chute_id}"
            token = os.getenv("CHUTES_API_KEY", "")
            
            if not token:
                raise ValueError("CHUTES_API_KEY not configured")
            
            headers = {"Authorization": token}
            
            # Use separate session for Chutes API
            session = await self._get_chutes_session()
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    logger.error(
                        f"[{self.env}] Failed to get chute info for {chute_id}: "
                        f"status={resp.status}"
                    )
                    return None
                
                chute_info = await resp.json()
                slug = chute_info.get("slug")
                
                if not slug:
                    logger.error(
                        f"[{self.env}] No slug found in chute info for {chute_id}"
                    )
                    return None
                
                logger.debug(
                    f"[{self.env}] Resolved chute_id={chute_id} to slug={slug}"
                )
                return slug
        
        except Exception as e:
            logger.error(
                f"[{self.env}] Error fetching chute slug for {chute_id}: {e}"
            )
            return None
    
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
                - chute_id: Chute deployment ID
        
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
            chute_id = task["chute_id"]
            
            logger.info(
                f"[{self.env}] Executing task: "
                f"uuid={task_uuid[:8]}... miner={miner_hotkey[:12]}... model={model} task_id={task_id}"
            )
            
            # Validate chute_id (required)
            if not chute_id:
                raise ValueError(
                    f"chute_id is required but missing for task {task_uuid[:8]}... "
                    f"miner={miner_hotkey[:12]}..."
                )
            
            # Resolve slug from chute_id
            slug = await self._get_chute_slug(chute_id)
            if not slug:
                raise ValueError(
                    f"Failed to resolve slug for chute_id={chute_id} "
                    f"(task_uuid={task_uuid[:8]}..., miner={miner_hotkey[:12]}...)"
                )
            
            # Build dynamic URL
            base_url = f"https://{slug}.chutes.ai/v1"
            logger.info(
                f"[{self.env}] Using dynamic URL: {base_url} "
                f"(chute_id={chute_id}, slug={slug})"
            )
            
            # Execute sampling using affine's environment wrapper
            result = await self.env_executor.evaluate(
                model=model,
                base_url=base_url,
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
                f"[{self.env}] Task completed: "
                f"uuid={task_uuid[:8]}... env={self.env} score={submission.score:.4f} "
                f"latency={submission.latency_ms/1000}s time={execution_time:.2f}s"
            )
            
            return submission
        
        except Exception as e:
            logger.warning(f"[{self.env}] Executor error (will retry): {e}")
            raise
    
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
            
            # Submit to /samples/submit endpoint
            # API will merge miner metadata from task queue
            submit_data = {
                "task_uuid": submission.task_uuid,
                "score": submission.score,
                "latency_ms": submission.latency_ms,
                "extra": submission.extra,
                "signature": submission.signature,
            }
            
            response = await self.api_client.post(
                "/samples/submit",
                json=submit_data,
                headers=headers
            )
            
            logger.info(
                f"[{self.env}] Submitted result: "
                f"task_uuid={submission.task_uuid[:8]}... "
                f"score={submission.score:.4f} {response}"
            )
            self.metrics.tasks_completed += 1
            return True
        
        except Exception as e:
            logger.error(f"[{self.env}] Error submitting result: {e}")
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
                    submission = await self._execute_task(task)

                    # Submit result (both success and HTTP errors)
                    await self._submit_result(task, submission)
                    
                except Exception as e:
                    # Executor error (not HTTP): don't submit, task will be retried
                    logger.warning(
                        f"[{self.env}] Executor error, task will be retried: {e}"
                    )
                    # Task lock will timeout and be released automatically
                    continue
                
                # Update metrics
                self.metrics.last_task_at = time.time()
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                logger.error(f"[{self.env}] Error in execution loop: {e}")
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