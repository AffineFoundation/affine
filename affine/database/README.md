# Affine Database Module

DynamoDB-based data storage layer for the Affine validator project.

## Architecture

```
affine/database/
├── __init__.py              # Module exports
├── client.py                # DynamoDB client management
├── schema.py                # Table schema definitions
├── tables.py                # Table initialization & management
├── base_dao.py              # Base DAO with common operations
├── dao/                     # Data Access Objects
│   ├── sample_results.py    # Sampling results with compression
│   ├── task_queue.py        # Task queue management
│   ├── execution_logs.py    # Execution history tracking
│   ├── scores.py            # Score snapshots by block
│   ├── system_config.py     # Dynamic configuration
│   ├── miner_metadata.py    # Miner metadata & statistics
│   └── data_retention.py    # Data retention policies
├── migrate.py               # R2 to DynamoDB migration
├── cli.py                   # Database management CLI
└── README.md                # This file
```

## Quick Start

### 1. Configure Environment

```bash
export AWS_REGION="us-east-1"
export DYNAMODB_TABLE_PREFIX="affine_dev"
```

### 2. Initialize Tables

```bash
python -m affine.database.cli init
```

### 3. Run Tests

```bash
python -m affine.database.cli test
```

### 4. Migrate Data from R2

```bash
# Migrate last 100k blocks
python -m affine.database.cli migrate --tail 100000

# Migrate specific number of results
python -m affine.database.cli migrate --tail 100000 --max-results 10000
```

## Table Schemas

### 1. sample_results
- **Purpose**: Store sampling results with compressed conversation data
- **PK**: `MINER#{hotkey}#REV#{revision}`
- **SK**: `ENV#{env}#TIME#{timestamp}#ID#{uuid}`
- **Features**: Conversation compression, no TTL (custom cleanup)

### 2. task_queue
- **Purpose**: Manage sampling tasks with status tracking
- **PK**: `MINER#{hotkey}#REV#{revision}`
- **SK**: `TASK#{task_id}`
- **Features**: Pre-execution validation, 7-day TTL

### 3. execution_logs
- **Purpose**: Track execution history with automatic cleanup
- **PK**: `MINER#{hotkey}`
- **SK**: `TIME#{timestamp}#ID#{uuid}`
- **Features**: 7-day TTL, error tracking

### 4. scores
- **Purpose**: Store score snapshots organized by block
- **PK**: `SCORE#{block_number}`
- **SK**: `MINER#{hotkey}`
- **Features**: Block-based organization, 30-day TTL, GSI for latest lookup

### 5. system_config
- **Purpose**: Dynamic configuration parameters
- **PK**: `CONFIG`
- **SK**: `PARAM#{param_name}`
- **Features**: Version tracking, no TTL

### 6. miner_metadata
- **Purpose**: Miner metadata and operational status
- **PK**: `MINER#{hotkey}`
- **SK**: `METADATA`
- **Features**: Chutes status, pause mechanism, statistics

### 7. data_retention_policy
- **Purpose**: Data retention policies for cleanup
- **PK**: `RETENTION#{hotkey}`
- **SK**: `METADATA`
- **Features**: Protect historical top-3 miners

## Usage Examples

### Save Sampling Result

```python
from affine.database import init_client
from affine.database.dao import SampleResultsDAO

await init_client()

sample_dao = SampleResultsDAO()

await sample_dao.save_sample(
    miner_hotkey="5xxx...",
    model_revision="v1.0.0",
    model_hash="abc123",
    uid=42,
    env="L3",
    subset="test_subset",
    dataset_index=0,
    task_id="task_001",
    score=0.95,
    success=True,
    latency_ms=150,
    conversation={"messages": [...]},  # Will be compressed
    validator_hotkey="validator_hotkey",
    block_number=1000000,
    signature="signature_hex"
)
```

### Create and Execute Task

```python
from affine.database.dao import TaskQueueDAO, MinerMetadataDAO

task_dao = TaskQueueDAO()
miner_dao = MinerMetadataDAO()

# Create task
task = await task_dao.create_task(
    miner_hotkey="5xxx...",
    model_revision="v1.0.0",
    model_hash="abc123",
    env="L3",
    validator_hotkey="owner_hotkey"
)

# Before execution: validate miner state
metadata = await miner_dao.get_metadata("5xxx...")
if metadata["model_hash"] != task["model_hash"]:
    # Miner state changed, invalidate task
    await task_dao.complete_task("5xxx...", "v1.0.0", task["task_id"])
else:
    # Execute task
    await task_dao.start_task("5xxx...", "v1.0.0", task["task_id"])
    # ... perform sampling ...
    await task_dao.complete_task("5xxx...", "v1.0.0", task["task_id"])
```

### Save Score Snapshot

```python
from affine.database.dao import ScoresDAO

score_dao = ScoresDAO()

await score_dao.save_score(
    block_number=1000000,
    miner_hotkey="5xxx...",
    uid=42,
    model_revision="v1.0.0",
    overall_score=0.95,
    confidence_interval_lower=0.90,
    confidence_interval_upper=0.98,
    average_score=0.94,
    scores_by_layer={"L3": 0.95, "L4": 0.93},
    scores_by_env={"env1": 0.96, "env2": 0.94},
    total_samples=100,
    is_eligible=True,
    meets_criteria=True
)

# Get latest scores
latest = await score_dao.get_latest_scores()
print(f"Block: {latest['block_number']}")
print(f"Scores: {latest['scores']}")
```

### Dynamic Configuration

```python
from affine.database.dao import SystemConfigDAO

config_dao = SystemConfigDAO()

# Set parameter
await config_dao.set_param(
    param_name="max_concurrent_tasks",
    param_value=10,
    param_type="int",
    description="Maximum concurrent sampling tasks"
)

# Get parameter
max_tasks = await config_dao.get_param_value("max_concurrent_tasks", default=5)
```

### Data Retention

```python
from affine.database.dao import DataRetentionDAO

retention_dao = DataRetentionDAO()

# Protect top-3 miners
await retention_dao.set_protected(
    miner_hotkey="5xxx...",
    reason="Historical rank #1"
)

# Check protection
is_protected = await retention_dao.is_protected("5xxx...")
```

## CLI Commands

```bash
# Initialize tables
python -m affine.database.cli init

# List tables
python -m affine.database.cli list

# Run tests
python -m affine.database.cli test

# Reset tables (WARNING: deletes all data)
python -m affine.database.cli reset

# Migrate from R2
python -m affine.database.cli migrate --tail 100000
```

## Design Principles

1. **Razorblade Simplicity**: Minimal code, maximum clarity
2. **No Backward Compatibility**: Clean slate, no legacy baggage
3. **Engineering Best Practices**: Proper abstraction, DRY, SOLID
4. **Performance**: Compression, batching, efficient queries
5. **Data Protection**: Smart cleanup preserves historical top-3

## Cost Optimization

- **On-Demand Billing**: Pay only for what you use
- **Compression**: Conversation data compressed ~70%
- **TTL**: Automatic cleanup of old data
- **Smart Retention**: Protect important data, clean the rest
- **Batch Operations**: Reduce API calls

## Performance Tips

1. Use `compact=True` when loading datasets (memory efficient)
2. Batch writes when possible (up to 25 items)
3. Use GSI for cross-partition queries
4. Enable point-in-time recovery for critical tables
5. Monitor with CloudWatch metrics

## Migration Strategy

1. **Phase 1**: Initialize DynamoDB tables
2. **Phase 2**: Direct cutover (stop old system, start new)
3. **Phase 3**: Background migration of historical data

See [`migrate.py`](migrate.py) for implementation details.