"""
Executor Worker - Individual Task Executor

Each worker fetches tasks for a specific environment and executes them.
Uses authenticated API endpoints with wallet signature verification.
"""

import os
import asyncio
import time
import traceback
import aiohttp
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


class ExecutorWorker:
    """Worker that executes sampling tasks for a specific environment.
    
    Uses authenticated API endpoints with wallet signature verification.
    """
    
    def __init__(
        self,
        worker_id: int,
        env: str,
        wallet: bt.wallet,
        fetch_rate_per_hour: int = 1800,
        max_concurrent_tasks: int = 5,
    ):
        """Initialize executor worker.
        
        Args:
            worker_id: Unique worker ID
            env: Environment to execute tasks for (e.g., "affine:sat")
            wallet: Bittensor wallet for signing
            fetch_rate_per_hour: Number of tasks to fetch per hour (default: 1800, i.e., 1 task per 2 seconds)
            max_concurrent_tasks: Maximum number of concurrent task executions (default: 5)
        """
        self.worker_id = worker_id
        self.env = env
        self.wallet = wallet
        self.hotkey = wallet.hotkey.ss58_address
        self.fetch_rate_per_hour = fetch_rate_per_hour
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Calculate fetch interval in seconds
        self.fetch_interval = 3600 / fetch_rate_per_hour  # seconds between fetches
        
        self.running = False
        self.metrics = WorkerMetrics()
        self._chutes_session: Optional[aiohttp.ClientSession] = None  # Separate session for Chutes API
        
        # API client for affine backend
        self.api_client: Optional[APIClient] = None
        
        # Environment executor (will be initialized lazily)
        self.env_executor = None
        
        # Task queue and semaphore for concurrent execution
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.execution_semaphore: Optional[asyncio.Semaphore] = None
        self.executor_tasks: List[asyncio.Task] = []

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
        if not self.wallet or not self.hotkey:
            raise RuntimeError("Wallet not configured for worker")
        
        # Initialize API client
        self.api_client = create_api_client()
        
        # Initialize environment executor
        await self._init_env_executor()
        
        logger.info(f"[{self.env}] Worker initialized for {self.env}")
    
    def start(self):
        """Start the worker fetch and execution loops (returns immediately, loops run in background)."""
        logger.info(f"[{self.env}] Starting fetch and execution loops for {self.env}...")
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
            f"[{self.env}] Started fetch loop (rate: {self.fetch_rate_per_hour}/hour, "
            f"interval: {self.fetch_interval:.2f}s) and {self.max_concurrent_tasks} executor workers"
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
                logger.debug(f"[{self.env}] No task available")
                return None

            logger.debug(
                f"[{self.env}] Fetched task: "
                f"uuid={str(task.get('task_uuid'))[:8]}... "
                f"task_id={task.get('task_id')} "
                f"uid={task.get('miner_uid')} "
                f"miner={(task.get('miner_hotkey',''))[:12]}..."
            )
            return task

        except Exception as e:
            # Silently ignore fetch errors and continue polling
            logger.debug(f"[{self.env}] Failed to fetch task: {e}")
            return None


    
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
                    logger.debug(
                        f"[{self.env}] Failed to get chute info for {chute_id}: "
                        f"status={resp.status}"
                    )
                    return None
                
                chute_info = await resp.json()
                slug = chute_info.get("slug")
                
                if not slug:
                    logger.debug(
                        f"[{self.env}] No slug found in chute info for {chute_id}"
                    )
                    return None
                
                logger.debug(
                    f"[{self.env}] Resolved chute_id={chute_id} to slug={slug}"
                )
                return slug
        
        except Exception as e:
            logger.debug(
                f"[{self.env}] Error fetching chute slug for {chute_id}: {e}"
            )
            return None
    
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
                - chute_id: Chute deployment ID
        
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
            chute_id = task["chute_id"]
            
            logger.debug(
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
            
            base_url = f"https://{slug}.chutes.ai/v1"
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
            submission = SampleSubmission(
                task_uuid=task_uuid,
                score=float(result.score),
                latency_ms=int(result.latency_seconds * 1000),
                extra=result.extra or {},
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
            logger.error(f"[{self.env}] Error submitting result: {e}")
            self.metrics.tasks_failed += 1
            return False
    
    async def _fetch_loop(self):
        """Fetch loop that continuously fetches tasks and puts them in queue."""
        logger.info(f"[{self.env}] Fetch loop started")
        
        while self.running:
            try:
                # Check queue size before fetching
                current_queue_size = self.task_queue.qsize()
                if current_queue_size > 10:
                    # Queue has too many pending tasks, sleep to let workers catch up
                    logger.debug(f"[{self.env}] Queue size {current_queue_size} > 10, sleeping to let workers catch up")
                    await asyncio.sleep(5)
                    continue

                # Fetch task from API
                task = await self._fetch_task()
                
                if task is None:
                    # No task available, sleep for 5 seconds
                    await asyncio.sleep(5)
                    continue
                
                # Put task in queue for execution
                await self.task_queue.put(task)
                logger.debug(f"[{self.env}] Task added to queue, queue size: {self.task_queue.qsize()}")
                
                # Wait for next fetch based on rate limit
                await asyncio.sleep(self.fetch_interval)
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                logger.error(f"[{self.env}] Error in fetch loop: {e}")
                traceback.print_exc()
                await asyncio.sleep(5)
        
        logger.info(f"[{self.env}] Fetch loop stopped")
    
    async def _execution_worker(self, worker_idx: int):
        """Execution worker that processes tasks from queue concurrently.
        
        Args:
            worker_idx: Index of this execution worker
        """
        logger.info(f"[{self.env}] Execution worker {worker_idx} started")
        
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
                        # Executor error: log in aligned format with brief error
                        miner_hotkey = task.get('miner_hotkey', 'unknown')
                        miner_uid = task.get('miner_uid')
                        task_id = task.get('task_id', 'N/A')
                        
                        # Brief error message
                        error_brief = str(e)[:200]
                        
                        logger.info(
                            f"[RESULT] U{miner_uid:<4} │ {self.env:<20} │ FAILED      │ "
                            f"task_id={task_id:<6} │ {error_brief}"
                        )
                        # Task lock will timeout and be released automatically
                    
                    finally:
                        # Mark task as done in queue
                        self.task_queue.task_done()
                        # Update metrics
                        self.metrics.last_task_at = time.time()
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                logger.error(f"[{self.env}] Error in execution worker {worker_idx}: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)
        
        logger.info(f"[{self.env}] Execution worker {worker_idx} stopped")
    
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