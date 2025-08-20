"""
ChefoodAI Meal Planning Schemas
Pydantic models for meal planning API requests and responses
"""

from typing import List, Optional, Dict, Any
from datetime import date, datetime
from pydantic import BaseModel, Field
from enum import Enum


class MealType(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"
    BRUNCH = "brunch"
    DESSERT = "dessert"


class MealPlanStatus(str, Enum):
    DRAFT = "draft"
    GENERATING = "generating"
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"
    DELETED = "deleted"


class DietaryRestriction(str, Enum):
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    GLUTEN_FREE = "gluten-free"
    DAIRY_FREE = "dairy-free"
    KETO = "keto"
    PALEO = "paleo"
    LOW_CARB = "low-carb"
    NUT_FREE = "nut-free"


class MealPlanCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    duration_days: int = Field(7, ge=1, le=30)
    start_date: Optional[date] = None
    family_size: int = Field(4, ge=1, le=20)
    budget_per_week: Optional[float] = Field(None, ge=0)
    cooking_time_available: Optional[int] = Field(
        None, ge=10, le=600
    )  # minutes
    target_calories_per_day: Optional[int] = Field(
        None, ge=800, le=5000
    )
    goals: List[str] = Field(default_factory=list)
    dietary_restrictions: List[DietaryRestriction] = Field(
        default_factory=list
    )
    cuisine_preferences: List[str] = Field(default_factory=list)
    skill_level: Optional[str] = Field(
        "intermediate", 
        pattern="^(beginner|intermediate|advanced)$"
    )
    meal_prep_style: Optional[str] = Field(
        "mixed", 
        pattern="^(daily|batch|mixed)$"
    )


class MealPlanUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    status: Optional[MealPlanStatus] = None
    budget_per_week: Optional[float] = Field(None, ge=0)
    cooking_time_available: Optional[int] = Field(None, ge=10, le=600)
    target_calories_per_day: Optional[int] = Field(None, ge=800, le=5000)


class RecipeInfo(BaseModel):
    id: Optional[str] = None
    title: str
    description: Optional[str] = None
    prep_time_minutes: int
    cook_time_minutes: int
    total_time_minutes: int
    servings: int
    difficulty_level: str
    cuisine_type: str
    calories_per_serving: Optional[int] = None
    protein_per_serving: Optional[float] = None
    carbs_per_serving: Optional[float] = None
    fat_per_serving: Optional[float] = None


class MealPlanMealResponse(BaseModel):
    id: int
    meal_type: MealType
    scheduled_time: Optional[str] = None
    servings: int
    notes: Optional[str] = None
    is_prepared: bool = False
    recipe: Optional[RecipeInfo] = None


class MealPlanDayResponse(BaseModel):
    id: int
    date: date
    day_number: int
    day_of_week: str
    target_calories: Optional[int] = None
    target_protein: Optional[float] = None
    target_carbs: Optional[float] = None
    target_fat: Optional[float] = None
    notes: Optional[str] = None
    is_rest_day: bool = False
    special_occasion: Optional[str] = None
    status: str = "pending"
    completed_at: Optional[datetime] = None
    meals: List[MealPlanMealResponse] = Field(default_factory=list)


class MealPlanAnalytics(BaseModel):
    total_calories: int
    total_protein: float
    total_carbs: float
    total_fat: float
    nutrition_score: float
    variety_score: float
    cost_estimate: Optional[float] = None


class MealPlanResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    start_date: date
    end_date: date
    status: MealPlanStatus
    goals: List[str]
    dietary_restrictions: List[str]
    family_size: int
    budget_per_week: Optional[float] = None
    cooking_time_available: Optional[int] = None
    target_calories_per_day: Optional[int] = None
    cuisine_preferences: List[str]
    skill_level: str
    meal_prep_style: str
    days: List[MealPlanDayResponse] = Field(default_factory=list)
    analytics: Optional[MealPlanAnalytics] = None
    created_at: datetime
    updated_at: datetime
    generated_at: Optional[datetime] = None


class MealPlanListResponse(BaseModel):
    meal_plans: List[MealPlanResponse]
    total: int
    page: int
    limit: int
    has_next: bool
    has_prev: bool


class MealSwapRequest(BaseModel):
    new_recipe_id: Optional[str] = None
    meal_type: Optional[MealType] = None
    dietary_preferences: List[str] = Field(default_factory=list)
    max_prep_time: Optional[int] = None


class ShoppingListItem(BaseModel):
    ingredient: str
    quantity: str
    unit: str
    category: str
    estimated_cost: Optional[float] = None
    is_purchased: bool = False


class ShoppingListResponse(BaseModel):
    id: int
    meal_plan_id: int
    week_number: int
    items: List[ShoppingListItem]
    total_estimated_cost: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class NutritionAnalysisResponse(BaseModel):
    meal_plan_id: int
    daily_averages: Dict[str, float]
    weekly_totals: Dict[str, float]
    nutrition_goals_met: Dict[str, bool]
    recommendations: List[str]
    variety_analysis: Dict[str, Any]


class MealPlanFeedbackCreate(BaseModel):
    meal_plan_id: int
    rating: int = Field(..., ge=1, le=5)
    feedback_text: Optional[str] = Field(None, max_length=1000)
    liked_meals: List[int] = Field(default_factory=list)
    disliked_meals: List[int] = Field(default_factory=list)
    suggested_improvements: Optional[str] = Field(None, max_length=500)


class MealPlanTemplateResponse(BaseModel):
    id: int
    name: str
    description: str
    duration_days: int
    target_audience: str
    dietary_focus: List[str]
    difficulty_level: str
    estimated_cost: str
    popularity_score: float
    created_by: str
    thumbnail_url: Optional[str] = None 