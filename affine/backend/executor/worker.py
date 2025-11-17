"""
Executor Worker - Individual Task Executor

Each worker fetches tasks for a specific environment and executes them.
Uses authenticated API endpoints with wallet signature verification.
"""

import asyncio
import time
import traceback
from typing import Optional, Dict, Any
from dataclasses import dataclass

from affine.core.http_client import AsyncHTTPClient
from affine.core.setup import logger, wallet
from affine.core.models import Result


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
        
        self.http_client = AsyncHTTPClient(timeout=300)  # 5 min timeout for sampling
        self.running = False
        self.metrics = WorkerMetrics()
        
        # Environment executor (will be initialized lazily)
        self.env_executor = None
        
        # Executor identity (from wallet)
        if wallet:
            self.hotkey = wallet.hotkey.ss58_address
        else:
            self.hotkey = None
    
    async def _init_env_executor(self):
        """Initialize environment executor lazily."""
        if self.env_executor is not None:
            return
        
        try:
            # Parse environment name
            parts = self.env.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid environment format: {self.env}")
            
            env_type, env_name = parts
            
            # Import and initialize environment executor
            if env_type == "affine":
                from affinetes.environments.affine.executor import AffineExecutor
                self.env_executor = AffineExecutor(env_name.upper())
            else:
                raise ValueError(f"Unsupported environment type: {env_type}")
            
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
        
        Uses the authenticated endpoint that verifies executor signature.
        
        Returns:
            Task dictionary or None if no task available
        """
        try:
            url = f"{self.api_base_url}/api/v1/tasks/fetch-authenticated"
            params = {"env": self.env}
            
            # Get authentication headers
            headers = self._get_auth_headers()
            
            response = await self.http_client.post(url, params=params, headers=headers)
            
            if response and "tasks" in response and len(response["tasks"]) > 0:
                task = response["tasks"][0]
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
            response = await self.http_client.get(url)
            
            if response:
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
            
            response = await self.http_client.put(url, json=data)
            
            if response:
                logger.warning(
                    f"[Worker-{self.worker_id}] Paused miner {miner_hotkey[:8]}... "
                    f"for {duration}s due to {consecutive_errors} consecutive errors"
                )
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"[Worker-{self.worker_id}] Error pausing miner: {e}")
            return False
    
    async def _execute_task(self, task: Dict) -> Result:
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
            Evaluation result
        """
        start_time = time.time()
        
        try:
            # Extract task parameters (new schema)
            miner_hotkey = task["miner_hotkey"]
            model_revision = task["model_revision"]
            model = task["model"]
            task_id = int(task["task_id"])  # Dataset index as integer
            task_uuid = task.get("task_uuid", "")
            
            logger.info(
                f"[Worker-{self.worker_id}] Executing task: "
                f"miner={miner_hotkey[:12]}... model={model} task_id={task_id}"
            )
            
            # Create a Miner object for the environment executor
            # Note: We don't have slug, so we'll pass model/base_url directly
            from affine.core.models import Miner
            miner_obj = Miner(
                uid=-1,  # UID not available in task queue
                hotkey=miner_hotkey,
                model=model,
                revision=model_revision,
                slug=None,  # No slug available
            )
            
            # Execute sampling using environment executor
            # Since we don't have slug, pass model and base_url directly in eval_kwargs
            # The environment executor will use these to override miner defaults
            result = await self.env_executor.evaluate(
                miner=miner_obj,
                model=model,
                base_url=f"https://llm.chutes.ai/v1",  # Default Chutes API
                task_id=task_id,
            )
            
            # Sign result
            if wallet:
                result.sign(wallet)
            
            execution_time = time.time() - start_time
            self.metrics.total_execution_time += execution_time
            
            logger.info(
                f"[Worker-{self.worker_id}] Task completed: "
                f"miner={miner_hotkey[:12]}... task_id={task_id} "
                f"success={result.success} time={execution_time:.2f}s"
            )
            
            return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.total_execution_time += execution_time
            
            logger.error(f"[Worker-{self.worker_id}] Task execution failed: {e}")
            traceback.print_exc()
            
            # Return failed result
            from affine.core.models import Miner
            return Result(
                miner=Miner(
                    uid=-1,  # UID not available in new schema
                    hotkey=task.get("miner_hotkey", ""),
                    revision=task.get("model_revision", ""),
                    model=task.get("model", ""),
                ),
                env=self.env,
                score=0.0,
                latency_seconds=0.0,
                success=False,
                error=str(e),
                task_id=int(task.get("task_id", 0)),
            )
    
    async def _submit_result(self, task: Dict, result: Result) -> bool:
        """Submit task result to API with authentication.
        
        Args:
            task: Original task from API (includes task_uuid and task_id)
            result: Evaluation result
        
        Returns:
            True if successful
        """
        try:
            # Get task identifiers (new schema)
            task_uuid = task.get("task_uuid", "")
            task_id = int(task.get("task_id", 0))  # Dataset index
            miner_hotkey = task.get("miner_hotkey", "")
            
            # Get authentication headers
            headers = self._get_auth_headers()
            
            if result.success:
                # First submit sample result
                url = f"{self.api_base_url}/api/v1/samples"
                sample_data = result.dict()
                
                response = await self.http_client.post(url, json=sample_data, headers=headers)
                
                if response:
                    sample_id = response.get("sample_id", "")
                    
                    # Mark task completed using authenticated endpoint
                    complete_url = f"{self.api_base_url}/api/v1/tasks/complete-authenticated"
                    complete_params = {
                        "task_uuid": task_uuid,
                        "dataset_task_id": task_id,
                        "success": True,
                    }
                    
                    await self.http_client.post(
                        complete_url,
                        params=complete_params,
                        headers=headers
                    )
                    
                    logger.info(
                        f"[Worker-{self.worker_id}] Submitted result: "
                        f"sample_id={sample_id} task_uuid={task_uuid[:8]}..."
                    )
                    
                    self.metrics.tasks_completed += 1
                    return True
                else:
                    logger.warning(f"[Worker-{self.worker_id}] Failed to submit sample")
                    self.metrics.tasks_failed += 1
                    return False
            
            else:
                # Task failed - report error
                error_type = self._classify_error(result.error or "Unknown error")
                error_message = result.error or "Unknown error"
                
                # Mark task failed using authenticated endpoint
                complete_url = f"{self.api_base_url}/api/v1/tasks/complete-authenticated"
                complete_params = {
                    "task_uuid": task_uuid,
                    "dataset_task_id": task_id,
                    "success": False,
                    "error_message": f"{error_type}: {error_message}",
                }
                
                await self.http_client.post(
                    complete_url,
                    params=complete_params,
                    headers=headers
                )
                
                logger.warning(
                    f"[Worker-{self.worker_id}] Task failed: "
                    f"task_uuid={task_uuid[:8]}... error={error_type}"
                )
                
                # Check if we need to pause miner due to consecutive errors
                consecutive_errors = await self._check_consecutive_errors(miner_hotkey)
                
                # Default threshold is 3 (will be configurable via API)
                error_threshold = 3
                if consecutive_errors >= error_threshold:
                    await self._pause_miner(miner_hotkey, consecutive_errors)
                
                self.metrics.tasks_failed += 1
                return True
        
        except Exception as e:
            logger.error(f"[Worker-{self.worker_id}] Error submitting result: {e}")
            self.metrics.tasks_failed += 1
            return False
    
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
                
                # Task is already assigned when fetched from authenticated endpoint
                # No need to mark started separately
                
                # Execute task
                result = await self._execute_task(task)
                
                # Submit result
                await self._submit_result(task, result)
                
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