"""
ChefoodAI API Routes
Main router configuration for all API endpoints
"""

from fastapi import APIRouter
import structlog

from api.endpoints import (
    auth, users, recipes, ai, meal_planning, analytics, health, shopping_lists
)
logger = structlog.get_logger()

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["authentication"]
)

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["users"]
)

api_router.include_router(
    recipes.router,
    prefix="/recipes",
    tags=["recipes"]
)

api_router.include_router(
    ai.router,
    prefix="/ai",
    tags=["artificial-intelligence"]
)

api_router.include_router(
    meal_planning.router,
    prefix="/meal-plans",
    tags=["meal-planning"]
)

api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["analytics"]
)

# AI Shopping Enhancement Router with error handling
try:
    from ai.shopping_enhancement import router as ai_shopping_router
    api_router.include_router(
        ai_shopping_router,
        prefix="/ai-shopping",
        tags=["ai-shopping-enhancement"]
    )
    logger.info("AI Shopping Enhancement router included successfully")
except ImportError as e:
    logger.warning(f"AI Shopping Enhancement not available: {e}")
except Exception as e:
    logger.error(f"Failed to include AI Shopping Enhancement router: {e}")

api_router.include_router(
    shopping_lists.router,
    prefix="/shopping-lists",
    tags=["shopping-lists"]
)

logger.info("API routes configured successfully")
