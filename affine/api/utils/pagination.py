"""
Pagination Utilities

Helpers for handling pagination in API responses.
"""

from typing import Optional, Tuple
from fastapi import Query
from affine.api.config import config


def get_pagination_params(
    limit: Optional[int] = Query(
        default=None,
        description="Maximum number of results to return",
        ge=1,
        le=None,
    ),
    start_from: Optional[str] = Query(
        default=None,
        description="Pagination cursor to start from",
    ),
) -> Tuple[int, Optional[str]]:
    """
    Extract and validate pagination parameters from query string.
    
    Args:
        limit: Maximum results to return (None = default)
        start_from: Pagination cursor
        
    Returns:
        Tuple of (limit, start_from)
    """
    # Use default limit if not provided
    if limit is None:
        limit = config.DEFAULT_PAGE_SIZE
    
    # Enforce max page size
    if limit > config.MAX_PAGE_SIZE:
        limit = config.MAX_PAGE_SIZE
    
    return limit, start_from


def create_pagination_info(
    total: Optional[int],
    limit: int,
    next_cursor: Optional[str],
) -> dict:
    """
    Create pagination metadata dictionary.
    
    Args:
        total: Total number of items (if known)
        limit: Limit used for this request
        next_cursor: Cursor for next page (if any)
        
    Returns:
        Pagination info dict
    """
    return {
        "total": total,
        "limit": limit,
        "next_cursor": next_cursor,
    }