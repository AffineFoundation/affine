"""
Executor Worker - Individual Task Executor

Each worker fetches tasks for a specific environment and executes them.
Uses authenticated API endpoints with wallet signature verification.
"""

import os
import asyncio
import time
import traceback
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import bittensor as bt
from affine.core.setup import logger
from affine.core.models import SampleSubmission
from affine.utils.api_client import create_api_client, APIClient


@dataclass
class WorkerMetrics:
    """Metrics for a worker."""
    
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    last_task_at: Optional[float] = None
    fetch_count: int = 0
    total_fetch_time: float = 0.0


class ExecutorWorker:
    """Worker that executes sampling tasks for a specific environment.
    
    Uses authenticated API endpoints with wallet signature verification.
    """
    
    def __init__(
        self,
        worker_id: int,
        env: str,
        wallet: bt.wallet,
        max_concurrent_tasks: int = 5,
        batch_size: int = 20,
    ):
        """Initialize executor worker.
        
        Args:
            worker_id: Unique worker ID
            env: Environment to execute tasks for (e.g., "affine:sat")
            wallet: Bittensor wallet for signing
            max_concurrent_tasks: Maximum number of concurrent task executions (default: 5)
            batch_size: Number of tasks to fetch per request (default: 20)
        """
        self.worker_id = worker_id
        self.env = env
        self.wallet = wallet
        self.hotkey = wallet.hotkey.ss58_address
        self.max_concurrent_tasks = max_concurrent_tasks
        self.batch_size = batch_size
        
        self.running = False
        self.metrics = WorkerMetrics()
        
        # API client for affine backend
        self.api_client: Optional[APIClient] = None
        
        # Environment executor (will be initialized lazily)
        self.env_executor = None
        
        # Task queue and semaphore for concurrent execution
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.execution_semaphore: Optional[asyncio.Semaphore] = None
        self.executor_tasks: List[asyncio.Task] = []

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
        if not self.wallet or not self.hotkey:
            raise RuntimeError("Wallet not configured for worker")
        
        # Initialize API client
        self.api_client = create_api_client()
        
        # Initialize environment executor
        await self._init_env_executor()
        
        logger.info(f"[{self.env}] Worker initialized for {self.env}")
    
    def start(self):
        """Start the worker fetch and execution loops (returns immediately, loops run in background)."""
        self.running = True
        
        # Initialize semaphore for concurrent execution
        self.execution_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # Create background task for fetch loop
        asyncio.create_task(self._fetch_loop())
        
        # Create multiple executor tasks for concurrent execution
        for i in range(self.max_concurrent_tasks):
            task = asyncio.create_task(self._execution_worker(i))
            self.executor_tasks.append(task)
        
        logger.info(
            f"[{self.env}] Started fetch loop (batch_size: {self.batch_size}) "
            f"and {self.max_concurrent_tasks} executor workers"
        )
    
    async def stop(self):
        """Stop the worker and all executor tasks."""
        logger.info(f"[{self.env}] Stopping worker...")
        self.running = False
        
        # Cancel all executor tasks
        for task in self.executor_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete
        if self.executor_tasks:
            await asyncio.gather(*self.executor_tasks, return_exceptions=True)
        
        logger.info(f"[{self.env}] Worker stopped, cancelled {len(self.executor_tasks)} executor tasks")
    
    def _sign_message(self, message: str) -> str:
        """Sign a message using the wallet.
        
        Args:
            message: Message to sign
            
        Returns:
            Hex-encoded signature
        """
        if not self.wallet:
            raise RuntimeError("Wallet not configured")
        
        # Sign the message
        signature = self.wallet.hotkey.sign(message.encode())
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
    
    async def _fetch_tasks_batch(self) -> List[Dict]:
        """Fetch batch of tasks from API with latency tracking.
        
        Returns:
            List of task dicts (may be empty if no tasks available)
        """
        start_time = time.time()
        
        try:
            headers = self._get_auth_headers()
            response = await self.api_client.post(
                "/tasks/fetch",
                params={"env": self.env, "batch_size": self.batch_size},
                headers=headers
            )

            if not isinstance(response, dict):
                logger.debug(f"[{self.env}] Invalid response format")
                return []
            
            # Handle batch response
            tasks = response.get("tasks", [])
            
            if not tasks:
                logger.debug(f"[{self.env}] No tasks available")
                return []

            logger.debug(
                f"[{self.env}] Fetched {len(tasks)} tasks (requested {self.batch_size})"
            )
            return tasks

        except Exception as e:
            # Silently ignore fetch errors and continue polling
            logger.debug(f"[{self.env}] Failed to fetch tasks: {e}")
            return []
        
        finally:
            # Record fetch latency
            fetch_time = (time.time() - start_time) * 1000  # Convert to ms
            self.metrics.fetch_count += 1
            self.metrics.total_fetch_time += fetch_time


    
    async def _execute_task(self, task: Dict) -> SampleSubmission:
        """Execute a sampling task.
        
        Args:
            task: Task dictionary from API response
                - task_id: Dataset index (string)
                - task_uuid: Queue UUID (string)
                - miner_hotkey: Miner's hotkey
                - model_revision: Model revision
                - model: Model repo/name
                - env: Environment
                - chute_slug: Chute URL slug (pre-resolved by API)
        
        Returns:
            SampleSubmission: Signed submission with score, latency, extra, signature
        """
        start_time = time.time()
        
        try:
            # Extract task parameters
            model = task["model"]
            task_id = int(task["task_id"])  # Dataset index as integer
            task_uuid = task.get("task_uuid", "")
            miner_hotkey = task["miner_hotkey"]
            chute_slug = task.get("chute_slug", "")
            
            logger.debug(
                f"[{self.env}] Executing task: "
                f"uuid={task_uuid[:8]}... miner={miner_hotkey[:12]}... model={model} task_id={task_id}"
            )
            
            # Validate chute_slug (required, pre-resolved by API from miners table)
            if not chute_slug:
                raise ValueError(
                    f"chute_slug is required but missing for task {task_uuid[:8]}... "
                    f"miner={miner_hotkey[:12]}..."
                )
            
            base_url = f"https://{chute_slug}.chutes.ai/v1"
            # Execute sampling using affine's environment wrapper
            result = await self.env_executor.evaluate(
                model=model,
                base_url=base_url,
                task_id=task_id,
            )
            
            execution_time = time.time() - start_time
            self.metrics.total_execution_time += execution_time
            
            # Build SampleSubmission (only contains task_uuid, score, latency, extra)
            # Pass through raw score from environment without normalization
            # IMPORTANT: Merge Result.error into extra if present
            extra = result.extra or {}
            if result.error:
                extra["error"] = result.error
            
            submission = SampleSubmission(
                task_uuid=task_uuid,
                score=float(result.score),
                latency_ms=int(result.latency_seconds * 1000),
                extra=extra,
                signature="",  # Will be signed below
            )
            
            # Sign submission using wallet
            sign_data = submission.get_sign_data()
            signature_bytes = self.wallet.hotkey.sign(sign_data.encode())
            submission.signature = signature_bytes.hex()
            
            # Format aligned RESULT log
            logger.info(
                f"[RESULT] U{task.get('miner_uid'):<4} │ {self.env:<20} │ {submission.score:10.3f} │ "
                f"task_id={task_id:<4} │ {execution_time:6.3f}s"
            )
            
            return submission
        
        except Exception as e:
            # Re-raise to be handled by _execution_loop
            # Don't log here - will be logged as [RESULT] FAILED
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
                "/tasks/submit",
                json=submit_data,
                headers=headers
            )
            
            logger.debug(
                f"[{self.env}] Submitted result: "
                f"task_uuid={submission.task_uuid[:8]}... "
                f"score={submission.score:.4f} {response}"
            )
            self.metrics.tasks_completed += 1
            return True

        except Exception as e:
            self.metrics.tasks_failed += 1
            raise
    
    async def _fetch_loop(self):
        """Fetch loop driven by queue size.
        
        Continuously fetches tasks when queue is low, maintaining buffer of
        max_concurrent_tasks * 2 tasks in queue.
        """
        logger.info(f"[{self.env}] Fetch loop started (batch_size={self.batch_size})")
        
        while self.running:
            try:
                # Check queue size before fetching
                current_queue_size = self.task_queue.qsize()
                
                # If queue has enough tasks, wait a bit
                if current_queue_size >= self.max_concurrent_tasks:
                    await asyncio.sleep(1)
                    continue

                # Fetch batch of tasks from API
                tasks = await self._fetch_tasks_batch()
                
                num_tasks = len(tasks)
                if num_tasks > 0:
                    # Put all tasks in queue
                    for task in tasks:
                        await self.task_queue.put(task)
                    
                    logger.debug(f"[{self.env}] Queued {num_tasks} tasks (queue_size={self.task_queue.qsize()})")
                else:
                    # No tasks available, wait before retry
                    await asyncio.sleep(1)
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                logger.error(f"[{self.env}] Error in fetch loop: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)
        
        logger.info(f"[{self.env}] Fetch loop stopped")
    
    async def _execution_worker(self, worker_idx: int):
        """Execution worker that processes tasks from queue concurrently.
        
        Args:
            worker_idx: Index of this execution worker
        """
        logger.debug(f"[{self.env}] Execution worker {worker_idx} started")
        
        while self.running:
            try:
                # Get task from queue (wait if empty)
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Acquire semaphore for concurrent execution control
                async with self.execution_semaphore:
                    logger.debug(
                        f"[{self.env}] Worker {worker_idx} executing task "
                        f"uuid={task.get('task_uuid', 'unknown')[:8]}..."
                    )
                    
                    # Execute task
                    try:
                        submission = await self._execute_task(task)
                        # Submit result (both success and HTTP errors)
                        await self._submit_result(task, submission)
                        
                    except Exception as e:
                        miner_uid = task.get('miner_uid')
                        task_id = task.get('task_id', 'N/A')
                        
                        # Brief error message
                        error_brief = str(e)[:200]
                        
                        logger.info(
                            f"[RESULT] U{miner_uid:<4} │ {self.env:<20} │ FAILED     │ "
                            f"task_id={task_id:<6} │ {error_brief}"
                        )
                        # Task lock will timeout and be released automatically
                    
                    finally:
                        self.task_queue.task_done()
                        # Update metrics
                        self.metrics.last_task_at = time.time()
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                logger.error(f"[{self.env}] Error in execution worker {worker_idx}: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)
        
        logger.debug(f"[{self.env}] Execution worker {worker_idx} stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker metrics.
        
        Returns:
            Dictionary of metrics including:
            - running_tasks: Number of tasks currently being executed (held by semaphore)
            - pending_tasks: Number of tasks waiting in queue
            - avg_fetch_time_ms: Average fetch latency in milliseconds
        """
        avg_time = (
            self.metrics.total_execution_time / self.metrics.tasks_completed
            if self.metrics.tasks_completed > 0
            else 0
        )
        
        avg_fetch_time = (
            self.metrics.total_fetch_time / self.metrics.fetch_count
            if self.metrics.fetch_count > 0
            else 0
        )
        
        # Calculate running tasks: max_concurrent - available semaphore slots
        running_tasks = 0
        if self.execution_semaphore is not None:
            running_tasks = self.max_concurrent_tasks - self.execution_semaphore._value
        
        # Get pending tasks from queue
        pending_tasks = self.task_queue.qsize()
        
        return {
            "worker_id": self.worker_id,
            "env": self.env,
            "running": self.running,
            "tasks_completed": self.metrics.tasks_completed,
            "tasks_failed": self.metrics.tasks_failed,
            "running_tasks": running_tasks,
            "pending_tasks": pending_tasks,
            "avg_execution_time": avg_time,
            "avg_fetch_time_ms": avg_fetch_time,
            "last_task_at": self.metrics.last_task_at,
        }