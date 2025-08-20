"""
ChefoodAI Database Models
Central import module for all database models
"""

from .users import (
    User, 
    UserProfile, 
    UserDietaryRestriction, 
    UserHealthGoal, 
    UserPreference, 
    UserAddress, 
    UserSession,
    UserTier, 
    UserStatus, 
    DietaryRestrictionType, 
    HealthGoalType
)

# Import models as they are created
# from .recipes import Recipe, Ingredient, RecipeIngredient, RecipeInstruction, RecipeNutrition
# from .meal_planning import MealPlan, MealPlanDay, MealPlanRecipe  
# from .grocery import ShoppingList, ShoppingListItem, GroceryStore
# from .analytics import UserAnalytics, RecipeAnalytics, MealPlanAnalytics

__all__ = [
    # User models
    "User",
    "UserProfile", 
    "UserDietaryRestriction",
    "UserHealthGoal",
    "UserPreference",
    "UserAddress",
    "UserSession",
    
    # Enums
    "UserTier",
    "UserStatus", 
    "DietaryRestrictionType",
    "HealthGoalType",
    
    # Recipe models (to be added)
    # "Recipe",
    # "Ingredient", 
    # "RecipeIngredient",
    # "RecipeInstruction",
    # "RecipeNutrition",
    
    # Meal planning models (to be added)
    # "MealPlan",
    # "MealPlanDay",
    # "MealPlanRecipe",
    
    # Grocery models (to be added)  
    # "ShoppingList",
    # "ShoppingListItem",
    # "GroceryStore",
    
    # Analytics models (to be added)
    # "UserAnalytics",
    # "RecipeAnalytics", 
    # "MealPlanAnalytics",
]