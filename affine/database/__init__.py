"""
Database module for Affine validator

Provides DynamoDB-based storage with clean DAO abstraction layer.
"""

from affine.database.client import get_client, init_client, close_client
from affine.database.tables import init_tables

__all__ = [
    "get_client",
    "init_client", 
    "close_client",
    "init_tables",
]