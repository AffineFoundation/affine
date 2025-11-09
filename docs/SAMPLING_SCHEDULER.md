# Sampling Scheduler

## Overview

The new sampling scheduler implements independent coroutines for each miner, achieving higher concurrency utilization and more flexible rate control.

## Quick Start

### Configuration

All configuration parameters can be set via environment variables (see below) or use default values.

### Run

```bash
# Start the sampling scheduler
af runner

# Enable monitoring API (recommended)
af runner --enable-monitoring
```

## Environment Variable Configuration

### Core Parameters

| Variable | Default | Description |
|---------|---------|-------------|
| `DAILY_RATE_PER_ENV` | 100 | **Deprecated**: Global default sampling rate, now configurable per environment |
| `LOW_SAMPLE_THRESHOLD` | 200 | Low sample threshold (total samples) |
| `LOW_SAMPLE_MULTIPLIER` | 3 | Low sample multiplier |
| `NUM_EVALUATION_WORKERS` | 20 | Number of evaluation workers |

**Note**: It's now recommended to set the `DEFAULT_DAILY_RATE` attribute for each environment individually in `affine/affine/tasks.py`.

### Queue Control

| Variable | Default | Description |
|---------|---------|-------------|
| `QUEUE_MAX_SIZE` | 100000 | Maximum queue capacity |
| `QUEUE_WARNING_THRESHOLD` | 5000 | Warning threshold |
| `QUEUE_PAUSE_THRESHOLD` | 8000 | Pause enqueue threshold |
| `QUEUE_RESUME_THRESHOLD` | 3000 | Resume enqueue threshold |
| `BATCH_SIZE` | 50 | Batch buffer size |
| `BATCH_FLUSH_INTERVAL` | 30 | Batch flush interval (seconds) |

### Timing Control

| Variable | Default | Description |
|---------|---------|-------------|
| `MINER_REFRESH_INTERVAL` | 600 | Miner list refresh interval (seconds) |
| `SINK_BATCH_SIZE` | 300 | Upload batch size |
| `SINK_MAX_WAIT` | 300 | Maximum upload wait time (seconds) |
| `CHUTES_ERROR_PAUSE_DURATION` | 900 | Chutes error pause duration (seconds) |
| `CHUTES_ERROR_THRESHOLD` | 3 | Chutes error threshold |

## Independent Sampling Rates per Environment

### Setting Independent Rates

Starting from the new version, each environment can be configured with an independent sampling rate instead of using the global `DAILY_RATE_PER_ENV` parameter. This allows setting optimal sampling rates based on environment complexity and execution time.

### Configuration Method

Modify the corresponding environment class in `affine/affine/tasks.py` by adding the `DEFAULT_DAILY_RATE` attribute:

```python
@register_env(EnvType.AGENTGYM, "agentgym:babyai")
class BABYAI(AgentGymSDKEnv):
    """BABYAI environment for SDK"""
    DEFAULT_DATA_LEN = 4000
    DEFAULT_REPLICAS = 1
    DEFAULT_DAILY_RATE = 120  # Sample each miner 120 times per day
    
    @property
    def env_name(self) -> str:
        return "agentgym:babyai"
```

### Current Environment Configuration

| Environment | Sampling Rate | Notes |
|------------|--------------|-------|
| `affine:sat` | 100/day | Standard rate |
| `affine:abd` | 100/day | Standard rate |
| `affine:ded` | 100/day | Standard rate |
| `agentgym:babyai` | 120/day | Fast (simple tasks) |
| `agentgym:webshop` | 100/day | Standard rate |
| `agentgym:textcraft` | 100/day | Standard rate |
| `agentgym:alfworld` | 80/day | Slow (complex tasks) |
| `agentgym:sciworld` | 80/day | Slow (complex tasks) |

### Rate Setting Guidelines

Choose appropriate sampling rates based on environment characteristics:

**Simple/Fast Environments** (120-150/day):
- Short task execution time (< 30s)
- Low compute resource consumption
- Example: BABYAI

**Medium Complexity Environments** (100/day):
- Medium task execution time (30-120s)
- Moderate compute resource consumption
- Example: WEBSHOP, SAT, ABD, DED

**Complex/Slow Environments** (80/day):
- Long task execution time (> 120s)
- High compute resource consumption
- Example: ALFWORLD, SCIWORLD

### Dynamic Rate Multiplier

Regardless of the environment's base sampling rate, when a miner's 24-hour total sample count is below the threshold, rate multiplication is triggered:

- **Trigger Condition**: 24h total samples < `LOW_SAMPLE_THRESHOLD` (default 200)
- **Multiplier**: `LOW_SAMPLE_MULTIPLIER` (default 3)
- **Formula**: `Actual Rate = DEFAULT_DAILY_RATE × Multiplier`

Examples:
- BABYAI base rate 120/day → 360/day after acceleration
- SCIWORLD base rate 80/day → 240/day after acceleration

## Core Features

### 1. Independent Sampling

Each miner runs an independent sampling coroutine:
- Not blocked by other miners
- Maximize concurrency utilization
- Avoid batch synchronization waits

### 2. Dynamic Rate Adjustment

Automatically adjust based on 24-hour sample count:
- Total samples < 200: sampling rate × 3
- Total samples ≥ 200: normal sampling rate

### 3. Backpressure Control

Automatically control on queue buildup:
- Queue > 5000: emit warning
- Queue > 8000: pause all sampling
- Queue < 3000: resume sampling

### 4. Fair Scheduling

Batch buffering mechanism ensures fairness:
- Each sampler buffers 50 tasks
- Submit batches uniformly when full
- Avoid starvation issues

### 5. Chutes Error Detection

Intelligently detect Chutes service errors:
- Insufficient API balance (Invalid API key)
- Instance unavailable (No instances available)
- Pause for 15 minutes after 3 consecutive errors

## Architecture Components

### Component List

```
affine/scheduler/
├── __init__.py          # Module exports
├── config.py            # Configuration management
├── models.py            # Data models
├── queue.py             # Task queue (backpressure control)
├── sampler.py           # Miner sampler
├── worker.py            # Evaluation worker
├── scheduler.py         # Main scheduler
├── monitor.py           # Status monitoring
├── api.py               # Monitoring API server
└── initializer.py       # Historical data initialization
```

### Data Flow

```
MinerSampler (per UID)
    ↓ Generate tasks
TaskQueue (backpressure control)
    ↓ Batch scheduling
EvaluationWorker (20 concurrent)
    ↓ Execute evaluation
ResultQueue
    ↓ Batch upload
Storage (R2/S3)
```

## Scheduler Initialization

The scheduler supports loading historical sampling data on startup to:
- Avoid unnecessary 3x acceleration for miners that already meet the sample threshold
- Initialize miner state with last sample times
- Resume sampling seamlessly after restarts

### Initialization Process

1. **Load Summary**: Read validator summary to get total sample counts per miner
2. **Load Recent Samples**: Read last 600 blocks (~2 hours) to get last sample timestamps
3. **Initialize Samplers**: Set rate multiplier and last sample times based on historical data

### Data Sources

- **Summary File**: Total sample counts (determines if 3x acceleration is needed)
- **Recent Blocks**: Last sample timestamps per environment (for state display)

See [SCHEDULER_INITIALIZATION_DESIGN.md](SCHEDULER_INITIALIZATION_DESIGN.md) for detailed design.

## Monitoring

### Status Logs

Status output every 30 seconds:

```
[STATUS] samplers=95 (paused=3) task_queue=1234 result_queue=56 enqueued=12345 dequeued=12000
```

- `samplers`: Active sampler count
- `paused`: Paused sampler count
- `task_queue`: Task queue size
- `result_queue`: Result queue size
- `enqueued`: Total enqueued
- `dequeued`: Total dequeued

### Warning Logs

Queue buildup warnings:

```
[TaskQueue] WARNING: Queue size 5234 exceeds warning threshold 5000
[TaskQueue] PAUSED: Queue size 8123 exceeds pause threshold 8000
[TaskQueue] RESUMED: Queue size 2987 below resume threshold 3000
```

Chutes error pauses:

```
[MinerSampler] U123 paused for 900s due to Chutes errors (3 consecutive)
```

### Monitoring API

The scheduler provides a comprehensive HTTP monitoring API. See [MONITORING_API.md](MONITORING_API.md) for details.

Quick start:
```bash
# Start runner with monitoring API
af runner --enable-monitoring

# Access monitoring endpoints
curl http://localhost:8765/status/summary
curl http://localhost:8765/status/miners
curl http://localhost:8765/status/queue
```

## Performance Comparison

### Legacy Scheduler

- **Concurrency utilization**: 60-70%
- **Batch waiting**: Significant idle time at end of each round
- **Response latency**: Affected by slowest task

### New Scheduler

- **Concurrency utilization**: 95%+
- **Batch waiting**: No batch synchronization
- **Response latency**: Independent sampling, no blocking

## Production Usage

The sampling scheduler is now the default and only implementation. All validators use this scheduler architecture for:

- Efficient independent sampling per miner
- Dynamic rate adjustment based on historical data
- Queue management with backpressure control
- Real-time monitoring capabilities

The legacy batch-based scheduler has been deprecated and removed.

## Troubleshooting

### Persistent Queue Buildup

**Symptoms**: Queue size continuously growing, frequent pause/resume

**Cause**: Insufficient evaluation workers

**Solution**: Increase `NUM_EVALUATION_WORKERS`

```bash
export NUM_EVALUATION_WORKERS=40
```

### Slow Sampling Rate

**Symptoms**: Daily sample count far below expected

**Cause**: Rate setting too low

**Solution**: Increase environment's `DEFAULT_DAILY_RATE`

### Frequent Miner Pauses

**Symptoms**: Many miners showing paused status

**Cause**: Chutes service issues or error threshold too low

**Solution**: Adjust error threshold

```bash
export CHUTES_ERROR_THRESHOLD=5
export CHUTES_ERROR_PAUSE_DURATION=600
```

## Best Practices

### 1. Set Appropriate Worker Count

Adjust based on server performance:
- CPU intensive: workers = CPU cores
- IO intensive: workers = 2-4 × CPU cores

### 2. Monitor Queue Size

Regularly check queue size in logs, adjust parameters as needed.

### 3. Set Reasonable Sampling Rates

Consider:
- Total miner count
- Number of environments
- Server capacity

Recommendation: `daily_rate × num_miners × num_envs / 86400 ≤ worker throughput`

### 4. Regular Log Cleanup

The new scheduler generates more debug logs, configure log rotation.

## Future Extensions

### Backend-Validator Architecture

The new scheduler is prepared for future architecture:

1. **Task Provider Abstraction**: Can easily replace with API-based task fetching
2. **Result Upload Decoupling**: Can independently replace upload target
3. **Configuration Design**: All parameters dynamically adjustable

### Migration Example

```python
# Current: Local task generation
class LocalTaskProvider:
    async def get_tasks(self, miner, envs):
        for env in envs:
            yield Task(env=env.env_name, miner=miner)

# Future: API task fetching
class APITaskProvider:
    async def get_tasks(self, miner, envs):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_URL}/tasks/{miner.uid}") as resp:
                tasks = await resp.json()
                for task in tasks:
                    yield Task(**task)
```

## Technical Support

If you encounter issues, check:
1. Environment variable configuration
2. Log output (especially WARNING/ERROR levels)
3. Queue monitoring metrics

---

**Last Updated**: 2025-11-09