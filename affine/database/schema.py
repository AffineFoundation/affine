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
SAMPLE_RESULTS_SCHEMA = {
    "TableName": get_table_name("sample_results"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # Partition key
        {"AttributeName": "sk", "KeyType": "RANGE"},  # Sort key
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "env", "AttributeType": "S"},
        {"AttributeName": "timestamp", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "env-timestamp-index",
            "KeySchema": [
                {"AttributeName": "env", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


# Task Queue Table
TASK_QUEUE_SCHEMA = {
    "TableName": get_table_name("task_queue"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "status", "AttributeType": "S"},
        {"AttributeName": "created_at", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "status-created-index",
            "KeySchema": [
                {"AttributeName": "status", "KeyType": "HASH"},
                {"AttributeName": "created_at", "KeyType": "RANGE"},
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


# Miner Metadata Table
MINER_METADATA_SCHEMA = {
    "TableName": get_table_name("miner_metadata"),
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
    MINER_METADATA_SCHEMA,
    DATA_RETENTION_SCHEMA,
]