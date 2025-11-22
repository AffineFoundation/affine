"""
DynamoDB table schema definitions

Defines table structures with partition keys, sort keys, and indexes.
"""

from typing import Dict, Any, List


def get_table_name(base_name: str) -> str:
    """Get full table name with prefix."""
    from affine.database.client import get_table_prefix
    return f"{get_table_prefix()}_{base_name}"


# Sample Results Table
#
# Design Philosophy:
# - PK combines the 3 most frequent query dimensions: hotkey + revision + env
# - SK uses task_id for natural ordering
# - uid removed (mutable, should query via bittensor metadata -> hotkey first)
# - GSI for efficient timestamp range queries (incremental updates)
# - block_number stored but not indexed (no block query requirement)
#
# Query Patterns:
# 1. Get samples by hotkey+revision+env -> Query by PK
# 2. Get samples by hotkey+revision (all envs) -> Query with PK prefix + filter
# 3. Get samples by hotkey (all revisions) -> Scan with hotkey prefix + filter
# 4. Get samples by timestamp range -> Use timestamp-index GSI (gsi_partition='SAMPLE' AND timestamp > :since)
# 5. Get samples by uid -> Query bittensor metadata first to get hotkey+revision, then query here
#
# GSI Design:
# - gsi_partition: Fixed value "SAMPLE" for all records (partition key)
# - timestamp: Milliseconds since epoch (range key, supports > < BETWEEN)
# - This design enables efficient Query operations for incremental updates
SAMPLE_RESULTS_SCHEMA = {
    "TableName": get_table_name("sample_results"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # MINER#{hotkey}#REV#{revision}#ENV#{env}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # TASK#{task_id}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "gsi_partition", "AttributeType": "S"},
        {"AttributeName": "timestamp", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "timestamp-index",
            "KeySchema": [
                {"AttributeName": "gsi_partition", "KeyType": "HASH"},   # Fixed "SAMPLE"
                {"AttributeName": "timestamp", "KeyType": "RANGE"},      # Sortable timestamp
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


# Task Queue Table (Task Pool)
# Schema design:
# - PK: ENV#{env} - partition by environment
# - SK: STATUS#{status}#UUID#{uuid} - status and unique identifier only
# - GSI1: miner-revision-index for querying pending tasks by miner+revision
# - GSI2: task-uuid-index for reverse lookup by uuid (for task completion/removal)
#
# Query patterns:
# 1. Weighted random task selection (by API Server TaskPoolManager):
#    - Get valid miners from cache
#    - Query pending task count per miner in this env
#    - Weighted random select miner (higher task count = higher probability)
#    - Fetch first task from selected miner (FIFO)
#    - Apply lock to prevent concurrent assignment
# 2. Task completion/removal: Query by task_uuid via GSI2, delete by pk+sk
# 3. Check miner pending tasks: Query by miner+revision via GSI1
# 4. Cleanup invalid tasks: Scan and delete tasks for miners not in valid list
#
# Design rationale:
# - Task pool uses weighted random selection for fairness (not true randomness)
# - Fairness = distribute sampling load proportionally across all miners
# - Task locks managed in memory by API Server (single instance)
# - Periodic cleanup removes tasks for miners with changed hotkey/revision
TASK_QUEUE_SCHEMA = {
    "TableName": get_table_name("task_queue"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "gsi1_pk", "AttributeType": "S"},
        {"AttributeName": "gsi1_sk", "AttributeType": "S"},
        {"AttributeName": "task_uuid", "AttributeType": "S"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "miner-revision-index",
            "KeySchema": [
                {"AttributeName": "gsi1_pk", "KeyType": "HASH"},
                {"AttributeName": "gsi1_sk", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
        {
            "IndexName": "task-uuid-index",
            "KeySchema": [
                {"AttributeName": "task_uuid", "KeyType": "HASH"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


# Execution Logs Table
EXECUTION_LOGS_SCHEMA = {
    "TableName": get_table_name("execution_logs"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# TTL settings (applied after table creation)
EXECUTION_LOGS_TTL = {
    "AttributeName": "ttl",
}


# Scores Table
SCORES_SCHEMA = {
    "TableName": get_table_name("scores"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "latest_marker", "AttributeType": "S"},
        {"AttributeName": "block_number", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "latest-block-index",
            "KeySchema": [
                {"AttributeName": "latest_marker", "KeyType": "HASH"},
                {"AttributeName": "block_number", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# TTL settings (applied after table creation)
SCORES_TTL = {
    "AttributeName": "ttl",
}


# System Config Table
SYSTEM_CONFIG_SCHEMA = {
    "TableName": get_table_name("system_config"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


# Data Retention Policy Table
DATA_RETENTION_SCHEMA = {
    "TableName": get_table_name("data_retention_policy"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


# Miners Table
# Schema design:
# - PK: UID#{uid} - unique primary key, each UID has only one record
# - No SK needed - single record per UID
# - GSI1: is-valid-index for querying valid/invalid miners
# - GSI2: hotkey-index for querying miner by hotkey
#
# Query patterns:
# 1. Get miner by UID: Direct get by PK
# 2. Get all valid miners: Query GSI1 with is_valid=true
# 3. Get miner by hotkey: Query GSI2 with hotkey
# 4. Get miners by model hash: Scan with filter (for anti-plagiarism)
MINERS_SCHEMA = {
    "TableName": get_table_name("miners"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "is_valid", "AttributeType": "S"},
        {"AttributeName": "hotkey", "AttributeType": "S"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "is-valid-index",
            "KeySchema": [
                {"AttributeName": "is_valid", "KeyType": "HASH"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
        {
            "IndexName": "hotkey-index",
            "KeySchema": [
                {"AttributeName": "hotkey", "KeyType": "HASH"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


# Miner Scores Table
# Stores detailed scoring records for each miner at each block
MINER_SCORES_SCHEMA = {
    "TableName": get_table_name("miner_scores"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # SNAPSHOT#{block_number}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # MINER#{hotkey}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "block_number", "AttributeType": "N"},
        {"AttributeName": "hotkey", "AttributeType": "S"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "block-number-index",
            "KeySchema": [
                {"AttributeName": "block_number", "KeyType": "HASH"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
        {
            "IndexName": "hotkey-index",
            "KeySchema": [
                {"AttributeName": "hotkey", "KeyType": "HASH"},
                {"AttributeName": "block_number", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# TTL settings for miner_scores
MINER_SCORES_TTL = {
    "AttributeName": "ttl",
}


# Score Snapshots Table
# Stores metadata for each scoring calculation
SCORE_SNAPSHOTS_SCHEMA = {
    "TableName": get_table_name("score_snapshots"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # BLOCK#{block_number}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # TIME#{timestamp}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "latest_marker", "AttributeType": "S"},
        {"AttributeName": "timestamp", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "latest-index",
            "KeySchema": [
                {"AttributeName": "latest_marker", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# TTL settings for score_snapshots
SCORE_SNAPSHOTS_TTL = {
    "AttributeName": "ttl",
}


# All table schemas
ALL_SCHEMAS = [
    SAMPLE_RESULTS_SCHEMA,
    TASK_QUEUE_SCHEMA,
    EXECUTION_LOGS_SCHEMA,
    SCORES_SCHEMA,
    SYSTEM_CONFIG_SCHEMA,
    DATA_RETENTION_SCHEMA,
    MINERS_SCHEMA,
    MINER_SCORES_SCHEMA,
    SCORE_SNAPSHOTS_SCHEMA,
]