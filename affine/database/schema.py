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
# - GSI for timestamp range queries only
# - block_number stored but not indexed (no block query requirement)
#
# Query Patterns:
# 1. Get samples by hotkey+revision+env -> Query by PK
# 2. Get samples by hotkey+revision (all envs) -> Query with PK prefix + filter
# 3. Get samples by hotkey (all revisions) -> Scan with hotkey prefix + filter
# 4. Get samples by timestamp range -> Use timestamp-index GSI
# 5. Get samples by uid -> Query bittensor metadata first to get hotkey+revision, then query here
SAMPLE_RESULTS_SCHEMA = {
    "TableName": get_table_name("sample_results"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # MINER#{hotkey}#REV#{revision}#ENV#{env}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # TASK#{task_id}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "timestamp", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "timestamp-index",
            "KeySchema": [
                {"AttributeName": "timestamp", "KeyType": "HASH"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


# Task Queue Table
# New schema design:
# - PK: ENV#{env} - partition by environment
# - SK: STATUS#{status}#PRIORITY#{priority}#CREATED#{timestamp}#UUID#{uuid}
# - GSI1: miner-revision-index for querying by miner+revision
# - GSI2: task-uuid-index for reverse lookup by uuid
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


# All table schemas
ALL_SCHEMAS = [
    SAMPLE_RESULTS_SCHEMA,
    TASK_QUEUE_SCHEMA,
    EXECUTION_LOGS_SCHEMA,
    SCORES_SCHEMA,
    SYSTEM_CONFIG_SCHEMA,
    DATA_RETENTION_SCHEMA,
]