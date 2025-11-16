#!/usr/bin/env python3
"""
Scheduler monitoring API server

Provides HTTP endpoints for monitoring scheduler status.
"""

import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn


def create_monitor_app(scheduler_monitor) -> FastAPI:
    """Create FastAPI application for monitoring
    
    Args:
        scheduler_monitor: SchedulerMonitor instance
    
    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Affine Scheduler Monitor",
        description="Real-time monitoring API for sampling scheduler",
        version="1.0.0",
    )
    
    @app.get("/")
    async def root():
        """API root - basic information"""
        return {
            "service": "Affine Scheduler Monitor",
            "version": "1.0.0",
            "endpoints": {
                "/status": "Complete scheduler status",
                "/status/summary": "Summary statistics",
                "/status/miners": "Per-miner statistics",
                "/status/queue": "Queue statistics",
                "/status/workers": "Worker statistics",
                "/status/env/{env_name}": "Environment-specific statistics",
                "/health": "Health check",
            }
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {"status": "healthy"}
    
    @app.get("/status")
    async def get_full_status():
        """Get complete scheduler status
        
        Returns comprehensive monitoring data including:
        - Scheduler summary (uptime, miner counts)
        - Queue statistics (size, throughput, per-env breakdown)
        - Worker utilization
        - Detailed per-miner statistics (sampling rates, errors, status)
        - Per-environment aggregated statistics
        """
        try:
            status = scheduler_monitor.get_status_dict()
            return JSONResponse(content=status)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/status/summary")
    async def get_summary():
        """Get summary statistics only
        
        Returns high-level overview without detailed miner data.
        """
        try:
            status = scheduler_monitor.get_status()
            return {
                "timestamp": status.timestamp,
                "uptime_seconds": status.uptime_seconds,
                "miners": {
                    "total": status.total_miners,
                    "active": status.active_miners,
                    "paused": status.paused_miners,
                },
                "evaluation": {
                    "total_samples_1h": status.evaluation_summary.total_samples_1h,
                    "total_samples_24h": status.evaluation_summary.total_samples_24h,
                    "avg_samples_per_hour": status.evaluation_summary.avg_samples_per_hour,
                    "projected_daily_samples": status.evaluation_summary.projected_daily_samples,
                    "active_sampling_miners": status.evaluation_summary.active_sampling_miners,
                    "total_errors_1h": status.evaluation_summary.total_errors_1h,
                    "miners_with_errors": status.evaluation_summary.miners_with_errors,
                },
                "queue": {
                    "current_size": status.queue.current_size,
                    "max_size": status.queue.max_size,
                    "is_paused": status.queue.is_paused,
                    "is_warning": status.queue.is_warning,
                    "enqueue_rate": status.queue.enqueue_rate,
                    "dequeue_rate": status.queue.dequeue_rate,
                },
                "workers": {
                    "total": status.workers.total_workers,
                    "active": status.workers.active_workers,
                    "utilization": status.workers.utilization,
                    "tasks_per_minute": status.workers.tasks_per_minute,
                },
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/status/miners")
    async def get_miners_status(
        status_filter: Optional[str] = None,
        has_errors: Optional[bool] = None,
    ):
        """Get per-miner statistics with optional filters
        
        Args:
            status_filter: Filter by status ("active", "paused")
            has_errors: Filter miners with errors (true) or without (false)
        
        Returns:
            Overall evaluation summary plus list of miner statistics sorted by UID
        """
        try:
            status = scheduler_monitor.get_status()
            miners = status.miners
            
            # Apply filters
            if status_filter:
                miners = [m for m in miners if m.status == status_filter]
            
            if has_errors is not None:
                if has_errors:
                    miners = [m for m in miners if m.error_count_1h > 0]
                else:
                    miners = [m for m in miners if m.error_count_1h == 0]
            
            return {
                "evaluation_summary": {
                    "total_samples_1h": status.evaluation_summary.total_samples_1h,
                    "total_samples_24h": status.evaluation_summary.total_samples_24h,
                    "avg_samples_per_hour": status.evaluation_summary.avg_samples_per_hour,
                    "projected_daily_samples": status.evaluation_summary.projected_daily_samples,
                    "samples_by_env_1h": status.evaluation_summary.samples_by_env_1h,
                    "samples_by_env_24h": status.evaluation_summary.samples_by_env_24h,
                    "total_participating_miners": status.evaluation_summary.total_participating_miners,
                    "active_sampling_miners": status.evaluation_summary.active_sampling_miners,
                    "total_errors_1h": status.evaluation_summary.total_errors_1h,
                    "miners_with_errors": status.evaluation_summary.miners_with_errors,
                    "current_throughput_per_minute": status.evaluation_summary.current_throughput_per_minute,
                },
                "total": len(miners),
                "miners": [
                    {
                        "uid": m.uid,
                        "hotkey": m.hotkey,
                        "model": m.model,
                        "status": m.status,
                        "pause_reason": m.pause_reason,
                        "env_rate_multipliers": m.env_rate_multipliers,
                        "configured_daily_rate": m.configured_daily_rate,
                        "effective_daily_rate": m.effective_daily_rate,
                        "samples_1h": m.samples_1h,
                        "total_samples_1h": m.total_samples_1h,
                        "samples_24h": m.samples_24h,
                        "total_samples_24h": m.total_samples_24h,
                        "consecutive_errors": m.consecutive_errors,
                        "last_error": m.last_error,
                        "error_count_1h": m.error_count_1h,
                    }
                    for m in miners
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/status/miners/{uid}")
    async def get_miner_status(uid: int):
        """Get detailed status for a specific miner
        
        Args:
            uid: Miner UID
        
        Returns:
            Detailed miner statistics
        """
        try:
            status = scheduler_monitor.get_status()
            miner = next((m for m in status.miners if m.uid == uid), None)
            
            if not miner:
                raise HTTPException(status_code=404, detail=f"Miner {uid} not found")
            
            return {
                "uid": miner.uid,
                "hotkey": miner.hotkey,
                "model": miner.model,
                "status": miner.status,
                "pause_until": miner.pause_until,
                "pause_reason": miner.pause_reason,
                "env_rate_multipliers": miner.env_rate_multipliers,
                "configured_daily_rate": miner.configured_daily_rate,
                "effective_daily_rate": miner.effective_daily_rate,
                "samples_1h": miner.samples_1h,
                "total_samples_1h": miner.total_samples_1h,
                "samples_24h": miner.samples_24h,
                "total_samples_24h": miner.total_samples_24h,
                "consecutive_errors": miner.consecutive_errors,
                "last_error": miner.last_error,
                "last_error_time": miner.last_error_time,
                "error_count_1h": miner.error_count_1h,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/status/queue")
    async def get_queue_status():
        """Get detailed queue statistics"""
        try:
            status = scheduler_monitor.get_status()
            queue = status.queue
            
            return {
                "current_size": queue.current_size,
                "max_size": queue.max_size,
                "warning_threshold": queue.warning_threshold,
                "pause_threshold": queue.pause_threshold,
                "resume_threshold": queue.resume_threshold,
                "is_paused": queue.is_paused,
                "is_warning": queue.is_warning,
                "utilization": queue.current_size / queue.max_size if queue.max_size > 0 else 0,
                "throughput": {
                    "enqueue_rate": queue.enqueue_rate,
                    "dequeue_rate": queue.dequeue_rate,
                    "net_rate": queue.enqueue_rate - queue.dequeue_rate,
                },
                "totals": {
                    "enqueued": queue.total_enqueued,
                    "dequeued": queue.total_dequeued,
                },
                "env_breakdown": queue.env_breakdown,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/status/workers")
    async def get_worker_status():
        """Get detailed worker statistics"""
        try:
            status = scheduler_monitor.get_status()
            workers = status.workers
            
            return {
                "total_workers": workers.total_workers,
                "active_workers": workers.active_workers,
                "idle_workers": workers.idle_workers,
                "utilization": workers.utilization,
                "tasks_per_minute": workers.tasks_per_minute,
                "efficiency": {
                    "is_saturated": workers.utilization > 0.9,
                    "has_capacity": workers.utilization < 0.7,
                    "recommendation": (
                        "Consider increasing workers" if workers.utilization > 0.9
                        else "Capacity available" if workers.utilization < 0.7
                        else "Optimal utilization"
                    )
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/status/env/{env_name}")
    async def get_env_status(env_name: str):
        """Get statistics for a specific environment
        
        Args:
            env_name: Environment name (e.g., "affine:sat", "agentgym:webshop")
        
        Returns:
            Environment-specific statistics
        """
        try:
            status = scheduler_monitor.get_status()
            
            if env_name not in status.env_stats:
                raise HTTPException(status_code=404, detail=f"Environment {env_name} not found")
            
            env_stat = status.env_stats[env_name]
            
            # Get miners sampling this environment
            active_miners = [
                {
                    "uid": m.uid,
                    "samples_1h": m.samples_1h.get(env_name, 0),
                    "samples_24h": m.samples_24h.get(env_name, 0),
                }
                for m in status.miners
                if env_name in m.samples_1h and m.status == "active"
            ]
            
            return {
                "env_name": env_name,
                "total_samples_1h": env_stat['total_samples_1h'],
                "total_samples_24h": env_stat['total_samples_24h'],
                "queue_size": env_stat['queue_size'],
                "active_miners": env_stat['active_miners'],
                "sampling_rate_per_hour": env_stat['total_samples_1h'],
                "projected_daily_samples": env_stat['total_samples_1h'] * 24,
                "miners": active_miners,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


async def run_monitor_api(scheduler_monitor, host: str = "0.0.0.0", port: int = 8000):
    """Run monitoring API server
    
    Args:
        scheduler_monitor: SchedulerMonitor instance
        host: Host to bind to
        port: Port to listen on
    """
    app = create_monitor_app(scheduler_monitor)
    
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    
    await server.serve()


async def start_monitoring_server(
    monitor,
    host: str = "0.0.0.0",
    port: int = 8765
):
    """Start the monitoring API server for scheduler
    
    Args:
        monitor: SchedulerMonitor instance
        host: Host to bind to
        port: Port to bind to
    """
    app = create_monitor_app(monitor)
    
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False  # Disable access logs to reduce noise
    )
    server = uvicorn.Server(config)
    
    # Run server without signals to allow proper cancellation
    try:
        await server.serve()
    except asyncio.CancelledError:
        # Gracefully shutdown on cancellation
        await server.shutdown()
        raise