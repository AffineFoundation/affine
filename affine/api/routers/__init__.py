"""
API Routers

Route definitions for all API endpoints.
"""

from affine.api.routers.samples import router as samples_router
from affine.api.routers.tasks import router as tasks_router
from affine.api.routers.miners import router as miners_router
from affine.api.routers.scores import router as scores_router
from affine.api.routers.config import router as config_router
from affine.api.routers.logs import router as logs_router

__all__ = [
    "samples_router",
    "tasks_router",
    "miners_router",
    "scores_router",
    "config_router",
    "logs_router",
    "chain_router",
]