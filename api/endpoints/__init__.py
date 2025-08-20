"""
ChefoodAI API Endpoints
All API endpoint modules
"""

# Import all endpoint modules
from . import health, auth, users, recipes, ai, meal_planning, analytics, shopping_lists

__all__ = [
    "health",
    "auth", 
    "users",
    "recipes",
    "ai",
    "meal_planning", 
    "analytics",
    "shopping_lists"
]