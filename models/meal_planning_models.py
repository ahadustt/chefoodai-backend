"""
ChefoodAI Meal Planning Models
SQLAlchemy models for advanced meal planning features
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Date, Boolean, Float, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime, date
import uuid
import enum

from core.database import Base

class MealPlanStatus(str, enum.Enum):
    """Meal plan status enumeration"""
    DRAFT = "draft"
    GENERATING = "generating"
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"
    DELETED = "deleted"

class MealType(str, enum.Enum):
    """Meal type enumeration"""
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"
    BRUNCH = "brunch"
    DESSERT = "dessert"

class MealPlan(Base):
    """
    Main meal plan model
    Represents a complete meal planning period with AI-generated meals
    """
    __tablename__ = "meal_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Basic information
    name = Column(String(200), nullable=False)
    description = Column(Text)
    start_date = Column(Date, nullable=False, index=True)
    end_date = Column(Date, nullable=False, index=True)
    status = Column(Enum(MealPlanStatus), default=MealPlanStatus.DRAFT, index=True)
    
    # Goals and preferences
    goals = Column(ARRAY(String), default=list)  # ['weight_loss', 'muscle_gain', 'balanced']
    dietary_restrictions = Column(ARRAY(String), default=list)
    preferences = Column(JSON, default=dict)
    
    # Nutritional targets
    target_calories_per_day = Column(Integer)
    target_protein_per_day = Column(Float)
    target_carbs_per_day = Column(Float)
    target_fat_per_day = Column(Float)
    target_fiber_per_day = Column(Float)
    
    # Planning parameters
    family_size = Column(Integer, default=1)
    budget_per_week = Column(Float)
    cooking_time_available = Column(Integer)  # minutes per day
    meal_prep_style = Column(String(50))  # 'daily', 'batch', 'mixed'
    
    # AI generation metadata
    ai_model_used = Column(String(100))
    generation_prompt = Column(Text)
    generation_parameters = Column(JSON, default=dict)
    
    # Status tracking
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    generated_at = Column(DateTime)
    deleted_at = Column(DateTime)
    error_message = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="meal_plans")
    days = relationship("MealPlanDay", back_populates="meal_plan", cascade="all, delete-orphan")
    shopping_lists = relationship("MealPlanShoppingList", back_populates="meal_plan", cascade="all, delete-orphan")
    analytics = relationship("MealPlanAnalytics", back_populates="meal_plan", cascade="all, delete-orphan")
    feedback = relationship("MealPlanFeedback", back_populates="meal_plan", cascade="all, delete-orphan")

class MealPlanDay(Base):
    """
    Individual day within a meal plan
    Contains all meals for a specific date
    """
    __tablename__ = "meal_plan_days"
    
    id = Column(Integer, primary_key=True, index=True)
    meal_plan_id = Column(Integer, ForeignKey("meal_plans.id"), nullable=False, index=True)
    
    # Day information
    date = Column(Date, nullable=False, index=True)
    day_number = Column(Integer, nullable=False)  # 1, 2, 3, etc.
    day_of_week = Column(String(20))  # 'monday', 'tuesday', etc.
    
    # Daily targets
    target_calories = Column(Integer)
    target_protein = Column(Float)
    target_carbs = Column(Float)
    target_fat = Column(Float)
    
    # Meal timing
    meal_times = Column(JSON, default=dict)  # {'breakfast': '08:00', 'lunch': '12:30'}
    
    # Daily notes and customizations
    notes = Column(Text)
    is_rest_day = Column(Boolean, default=False)
    special_occasion = Column(String(100))  # 'birthday', 'date_night', etc.
    
    # Status
    status = Column(String(20), default='planned')  # 'planned', 'in_progress', 'completed'
    completed_at = Column(DateTime)
    
    # Relationships
    meal_plan = relationship("MealPlan", back_populates="days")
    meals = relationship("MealPlanMeal", back_populates="meal_plan_day", cascade="all, delete-orphan")

class MealPlanMeal(Base):
    """
    Individual meal within a meal plan day
    Links recipes to specific meal slots
    """
    __tablename__ = "meal_plan_meals"
    
    id = Column(Integer, primary_key=True, index=True)
    meal_plan_day_id = Column(Integer, ForeignKey("meal_plan_days.id"), nullable=False, index=True)
    recipe_id = Column(Integer, ForeignKey("recipes.id"), nullable=False, index=True)
    
    # Meal details
    meal_type = Column(Enum(MealType), nullable=False, index=True)
    scheduled_time = Column(String(5))  # '08:00', '12:30'
    servings = Column(Integer, default=1)
    order_index = Column(Integer, default=0)  # Order within the meal type
    
    # Nutritional targets for this meal
    target_calories = Column(Integer)
    target_protein = Column(Float)
    target_carbs = Column(Float)
    target_fat = Column(Float)
    
    # Customizations
    notes = Column(Text)
    ingredient_substitutions = Column(JSON, default=dict)
    scaling_factor = Column(Float, default=1.0)  # For adjusting recipe portions
    
    # Preparation tracking
    is_prep_ahead = Column(Boolean, default=False)
    prep_date = Column(Date)
    prep_notes = Column(Text)
    
    # Status
    status = Column(String(20), default='planned')  # 'planned', 'prepped', 'cooking', 'completed'
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    meal_plan_day = relationship("MealPlanDay", back_populates="meals")
    recipe = relationship("Recipe")

class MealPlanShoppingList(Base):
    """
    Generated shopping list for a meal plan
    Aggregates all ingredients with smart organization
    """
    __tablename__ = "meal_plan_shopping_lists"
    
    id = Column(Integer, primary_key=True, index=True)
    meal_plan_id = Column(Integer, ForeignKey("meal_plans.id"), nullable=False, index=True)
    
    # List metadata
    name = Column(String(200))
    week_number = Column(Integer)  # For weekly shopping lists
    
    # Organized ingredients
    ingredients_by_category = Column(JSON, default=dict)  # Organized by store sections
    pantry_items = Column(JSON, default=list)  # Items likely already owned
    fresh_items = Column(JSON, default=list)  # Items that need to be fresh
    
    # Shopping optimization
    store_layout_optimized = Column(Boolean, default=False)
    estimated_shopping_time = Column(Integer)  # minutes
    preferred_stores = Column(ARRAY(String), default=list)
    
    # Cost estimation
    estimated_cost = Column(Float)
    cost_breakdown_by_category = Column(JSON, default=dict)
    budget_alerts = Column(JSON, default=list)
    
    # List status
    total_items = Column(Integer, default=0)
    checked_items = Column(Integer, default=0)
    completion_percentage = Column(Float, default=0.0)
    
    # Smart features
    seasonal_substitutions = Column(JSON, default=dict)
    bulk_buying_opportunities = Column(JSON, default=list)
    coupon_matches = Column(JSON, default=list)
    
    # Timestamps
    generated_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    shopping_completed_at = Column(DateTime)
    
    # Relationships
    meal_plan = relationship("MealPlan", back_populates="shopping_lists")

class MealPlanAnalytics(Base):
    """
    Comprehensive analytics for meal plans
    Tracks nutritional balance, variety, and success metrics
    """
    __tablename__ = "meal_plan_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    meal_plan_id = Column(Integer, ForeignKey("meal_plans.id"), nullable=False, index=True)
    
    # Nutritional analytics
    total_calories = Column(Float)
    avg_calories_per_day = Column(Float)
    calorie_distribution = Column(JSON, default=dict)  # By meal type
    
    total_protein = Column(Float)
    total_carbohydrates = Column(Float)
    total_fat = Column(Float)
    total_fiber = Column(Float)
    total_sodium = Column(Float)
    total_sugar = Column(Float)
    
    # Macro balance
    protein_percentage = Column(Float)
    carb_percentage = Column(Float)
    fat_percentage = Column(Float)
    
    # Variety and diversity
    unique_recipes_count = Column(Integer)
    unique_ingredients_count = Column(Integer)
    cuisine_diversity_score = Column(Float)  # 0-100
    recipe_diversity_score = Column(Float)  # 0-100
    
    # Goal achievement
    nutritional_balance_score = Column(Float)  # 0-100
    goal_achievement_score = Column(Float)  # 0-100
    dietary_compliance_score = Column(Float)  # 0-100
    
    # Cost and time analytics
    estimated_total_cost = Column(Float)
    avg_cost_per_meal = Column(Float)
    total_prep_time = Column(Integer)  # minutes
    total_cook_time = Column(Integer)  # minutes
    avg_meal_complexity = Column(Float)  # 1-5 scale
    
    # Daily breakdown
    daily_breakdown = Column(JSON, default=list)  # Detailed daily stats
    
    # Distribution analysis
    cuisine_distribution = Column(JSON, default=dict)
    difficulty_distribution = Column(JSON, default=dict)
    meal_type_distribution = Column(JSON, default=dict)
    cooking_method_distribution = Column(JSON, default=dict)
    
    # Health insights
    allergen_exposure = Column(JSON, default=dict)
    nutrient_density_score = Column(Float)
    inflammatory_foods_count = Column(Integer)
    processed_foods_percentage = Column(Float)
    
    # Sustainability metrics
    environmental_impact_score = Column(Float)
    local_ingredients_percentage = Column(Float)
    seasonal_ingredients_percentage = Column(Float)
    food_waste_prediction = Column(Float)
    
    # Timestamps
    generated_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    meal_plan = relationship("MealPlan", back_populates="analytics")

class MealPlanTemplate(Base):
    """
    Reusable meal plan templates
    For common meal planning patterns and user favorites
    """
    __tablename__ = "meal_plan_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)  # Null for public templates
    
    # Template information
    name = Column(String(200), nullable=False)
    description = Column(Text)
    category = Column(String(50))  # 'weight_loss', 'muscle_gain', 'family', etc.
    tags = Column(ARRAY(String), default=list)
    
    # Template configuration
    duration_days = Column(Integer, default=7)
    goals = Column(ARRAY(String), default=list)
    dietary_restrictions = Column(ARRAY(String), default=list)
    preferences = Column(JSON, default=dict)
    
    # Nutritional targets
    target_calories_per_day = Column(Integer)
    macro_ratios = Column(JSON, default=dict)  # protein, carb, fat percentages
    
    # Template metadata
    difficulty_level = Column(String(20))  # 'beginner', 'intermediate', 'advanced'
    time_commitment = Column(String(20))  # 'low', 'medium', 'high'
    budget_level = Column(String(20))  # 'budget', 'moderate', 'premium'
    
    # Usage statistics
    usage_count = Column(Integer, default=0)
    average_rating = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0)
    
    # Visibility
    is_public = Column(Boolean, default=False)
    is_featured = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")

class MealPlanFeedback(Base):
    """
    User feedback on meal plans and individual meals
    Used for improving AI recommendations
    """
    __tablename__ = "meal_plan_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    meal_plan_id = Column(Integer, ForeignKey("meal_plans.id"), nullable=False, index=True)
    meal_id = Column(Integer, ForeignKey("meal_plan_meals.id"), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Feedback type
    feedback_type = Column(String(50))  # 'rating', 'made_it', 'substitution', 'issue'
    
    # Ratings (1-5 scale)
    overall_rating = Column(Integer)
    taste_rating = Column(Integer)
    difficulty_rating = Column(Integer)
    time_rating = Column(Integer)  # How accurate was time estimate
    cost_rating = Column(Integer)
    
    # Boolean feedback
    made_recipe = Column(Boolean)
    would_make_again = Column(Boolean)
    followed_exactly = Column(Boolean)
    
    # Detailed feedback
    comments = Column(Text)
    improvements_suggested = Column(Text)
    substitutions_made = Column(JSON, default=dict)
    issues_encountered = Column(JSON, default=list)
    
    # Context
    cooking_experience = Column(String(20))  # User's experience level when making
    household_size = Column(Integer)
    occasion = Column(String(50))  # 'weeknight', 'special', 'meal_prep'
    
    # Feedback metadata
    is_helpful = Column(Boolean, default=True)
    verified_made = Column(Boolean, default=False)  # Verified they actually made it
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    meal_date = Column(Date)  # When they actually made/planned to make the meal
    
    # Relationships
    meal_plan = relationship("MealPlan", back_populates="feedback")
    meal = relationship("MealPlanMeal")
    user = relationship("User")

class NutritionalGoal(Base):
    """
    User's nutritional goals and targets
    Used for personalizing meal plans
    """
    __tablename__ = "nutritional_goals"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Goal information
    name = Column(String(100), nullable=False)
    description = Column(Text)
    goal_type = Column(String(50))  # 'weight_loss', 'muscle_gain', 'maintenance', 'custom'
    
    # Caloric targets
    target_calories_per_day = Column(Integer)
    calorie_adjustment_factor = Column(Float, default=1.0)  # Multiplier for activity level
    
    # Macronutrient targets (grams)
    target_protein_grams = Column(Float)
    target_carbs_grams = Column(Float)
    target_fat_grams = Column(Float)
    target_fiber_grams = Column(Float)
    
    # Macronutrient ratios (percentages, should sum to 100)
    protein_percentage = Column(Float)
    carb_percentage = Column(Float)
    fat_percentage = Column(Float)
    
    # Micronutrient targets
    target_sodium_mg = Column(Float)
    target_sugar_grams = Column(Float)
    target_saturated_fat_grams = Column(Float)
    
    # Special considerations
    meal_frequency = Column(Integer, default=3)  # Meals per day
    include_snacks = Column(Boolean, default=True)
    hydration_target_ml = Column(Integer)
    
    # Time-based goals
    start_date = Column(Date)
    target_date = Column(Date)
    weekly_weight_change_goal = Column(Float)  # kg per week (+ for gain, - for loss)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    
    # Progress tracking
    current_weight_kg = Column(Float)
    target_weight_kg = Column(Float)
    body_fat_percentage = Column(Float)
    muscle_mass_kg = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")

class MealTemplate(Base):
    """
    Reusable meal templates for quick meal plan generation
    Stores common meal combinations and patterns
    """
    __tablename__ = "meal_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)  # Null for public templates
    
    # Template information
    name = Column(String(200), nullable=False)
    description = Column(Text)
    meal_type = Column(Enum(MealType), nullable=False, index=True)
    
    # Template configuration
    recipe_ids = Column(ARRAY(Integer), default=list)  # Primary recipes
    alternative_recipe_ids = Column(ARRAY(Integer), default=list)  # Alternative options
    
    # Nutritional profile
    target_calories = Column(Integer)
    target_protein = Column(Float)
    target_carbs = Column(Float)
    target_fat = Column(Float)
    
    # Meal characteristics
    preparation_time = Column(Integer)  # minutes
    difficulty_level = Column(String(20))
    cuisine_type = Column(String(50))
    dietary_tags = Column(ARRAY(String), default=list)
    
    # Usage context
    best_for_goals = Column(ARRAY(String), default=list)  # Which goals this template supports
    seasonal_preference = Column(String(20))  # 'spring', 'summer', 'fall', 'winter', 'any'
    occasion_tags = Column(ARRAY(String), default=list)  # 'quick', 'comfort', 'elegant', etc.
    
    # Template metadata
    is_public = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)  # Verified by nutritionist/chef
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)  # Based on user feedback
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")