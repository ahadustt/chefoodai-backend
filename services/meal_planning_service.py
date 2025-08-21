"""
ChefoodAI Premium Meal Planning Service
Advanced AI-powered meal planning with nutritional optimization and smart scheduling
"""

import asyncio
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import json
import logging

from core.config import get_settings
from models.meal_planning_models import (
    MealPlan, MealPlanDay, MealPlanMeal, MealPlanShoppingList,
    MealPlanAnalytics, NutritionalGoal, MealTemplate
)
from models.users import User
from models.recipe_models import Recipe, RecipeIngredient
from services.ai_service import ai_service
from services.nutrition_service import nutrition_service
from utils.date_utils import get_week_dates, get_month_dates

settings = get_settings()
logger = logging.getLogger(__name__)

class MealPlanningService:
    def __init__(self):
        self.default_meals_per_day = 3
        self.max_plan_days = 30  # Increased for premium users
        self.concurrent_limit = 10  # Parallel AI calls limit
        
        # Premium features
        self.premium_features = {
            'max_days': 30,
            'concurrent_recipes': 10,
            'image_generation': True,  # Always enabled for all users now
            'advanced_analytics': True,
            'real_time_progress': True
        }
        
        # Free plan limits - UPDATED: Images enabled for all users
        self.free_limits = {
            'max_days': 7,
            'concurrent_recipes': 3,
            'image_generation': True,  # Changed from False - images for everyone!
            'advanced_analytics': False,
            'real_time_progress': False
        }
        
        # Meal timing defaults
        self.default_meal_times = {
            'breakfast': '08:00',
            'lunch': '12:30',
            'dinner': '18:30',
            'snack': '15:00'
        }
        
        # Nutritional priorities by goal
        self.goal_priorities = {
            'weight_loss': {
                'calories': 0.3,
                'protein': 0.25,
                'fiber': 0.2,
                'carbs': 0.15,
                'fat': 0.1
            },
            'muscle_gain': {
                'protein': 0.35,
                'calories': 0.25,
                'carbs': 0.2,
                'fat': 0.15,
                'fiber': 0.05
            },
            'balanced': {
                'calories': 0.2,
                'protein': 0.2,
                'carbs': 0.2,
                'fat': 0.2,
                'fiber': 0.2
            },
            'low_carb': {
                'carbs': 0.4,
                'protein': 0.25,
                'fat': 0.2,
                'calories': 0.15
            }
        }
    
    async def create_meal_plan(
        self,
        user: User,
        plan_data: Dict[str, Any],
        db: Session
    ) -> MealPlan:
        """
        Create a comprehensive AI-powered meal plan
        
        Args:
            user: User requesting the meal plan
            plan_data: Meal plan configuration
            db: Database session
            
        Returns:
            Complete meal plan with recipes and analytics
        """
        try:
            # Validate plan parameters
            validated_data = await self._validate_plan_data(plan_data, user)
            
            # Create base meal plan
            meal_plan = MealPlan(
                user_id=user.id,
                name=validated_data['name'],
                description=validated_data.get('description', ''),
                start_date=validated_data['start_date'],
                end_date=validated_data['end_date'],
                goals=validated_data['goals'],
                dietary_restrictions=validated_data.get('dietary_restrictions', []),
                preferences=validated_data.get('preferences', {}),
                target_calories_per_day=validated_data.get('target_calories'),
                family_size=validated_data.get('family_size', 1),
                budget_per_week=validated_data.get('budget_per_week'),
                cooking_time_available=validated_data.get('cooking_time_available'),
                status='generating',
                created_at=datetime.utcnow()
            )
            
            db.add(meal_plan)
            db.commit()
            db.refresh(meal_plan)
            
            # Generate meal plan content using AI
            await self._generate_meal_plan_content(meal_plan, user, db, validated_data)
            
            # Calculate nutritional analytics
            await self._calculate_meal_plan_analytics(meal_plan, db)
            
            # Generate shopping list
            await self._generate_shopping_list(meal_plan, db)
            
            # Mark as completed
            meal_plan.status = 'active'
            meal_plan.generated_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"Meal plan created successfully for user {user.id}: {meal_plan.id}")
            return meal_plan
            
        except Exception as e:
            logger.error(f"Failed to create meal plan for user {user.id}: {e}")
            if 'meal_plan' in locals():
                meal_plan.status = 'failed'
                meal_plan.error_message = str(e)
                db.commit()
            raise
    
    async def create_meal_plan_blazing_fast(
        self,
        user: User,
        plan_data: Dict[str, Any],
        db: Session,
        websocket_manager=None,
        user_id: str = None
    ) -> MealPlan:
        """
        🚀 BLAZING FAST meal plan generation with parallel processing and AI images
        
        Features:
        - Concurrent recipe generation (10x faster)
        - AI-generated images for every meal
        - Real-time WebSocket progress updates
        - Background stats processing
        - Smart caching and optimization
        """
        import asyncio
        import time
        from concurrent.futures import ThreadPoolExecutor
        from datetime import datetime, timedelta
        
        start_time = time.time()
        
        try:
            # Validate plan parameters
            validated_data = await self._validate_plan_data(plan_data, user)
            
            # Determine user limits - check multiple possible plan attributes
            is_premium = False
            
            # Check various possible plan attributes (different user models have different fields)
            if hasattr(user, 'plan') and user.plan == 'premium':
                is_premium = True
            elif hasattr(user, 'tier') and str(user.tier).lower() in ['premium', 'enterprise']:
                is_premium = True
            elif hasattr(user, 'subscription_type') and user.subscription_type == 'premium':
                is_premium = True
            elif hasattr(user, 'is_premium') and user.is_premium:
                is_premium = True
            else:
                # For development/testing - enable premium features for all users
                # TODO: Remove this in production when user plans are properly implemented
                is_premium = True
                logger.info("🚀 Enabling premium features for development (image generation enabled)")
            
            limits = self.premium_features if is_premium else self.free_limits
            
            # Create base meal plan (FAST)
            meal_plan = MealPlan(
                user_id=user.id,
                name=validated_data['name'],
                description=validated_data.get('description', ''),
                start_date=validated_data['start_date'],
                end_date=validated_data['end_date'],
                goals=validated_data['goals'],
                dietary_restrictions=validated_data.get('dietary_restrictions', []),
                preferences=validated_data.get('preferences', {}),
                target_calories_per_day=validated_data.get('target_calories'),
                family_size=validated_data.get('family_size', 1),
                budget_per_week=validated_data.get('budget_per_week'),
                cooking_time_available=validated_data.get('cooking_time_available'),
                status='generating',
                created_at=datetime.utcnow()
            )
            
            db.add(meal_plan)
            db.commit()
            db.refresh(meal_plan)
            
            # Send initial progress
            if websocket_manager and user_id:
                await websocket_manager.send_message(user_id, {
                    "type": "meal_plan_progress",
                    "meal_plan_id": str(meal_plan.id),
                    "progress": 10,
                    "message": "🚀 Initializing blazing fast generation..."
                })
            
            # PARALLEL GENERATION PHASE
            logger.info(f"🚀 Starting BLAZING FAST generation for {meal_plan.id}")
            
            # Calculate meal requirements
            plan_days = (validated_data['end_date'] - validated_data['start_date']).days + 1
            meals_per_day = validated_data.get('meals_per_day', 3)
            total_meals = plan_days * meals_per_day
            
            # Generate meal structure
            meal_structure = self._create_meal_structure(plan_days, meals_per_day, validated_data)
            
            # Progress update
            if websocket_manager and user_id:
                await websocket_manager.send_message(user_id, {
                    "type": "meal_plan_progress",
                    "meal_plan_id": str(meal_plan.id),
                    "progress": 20,
                    "message": f"📋 Planning {total_meals} meals across {plan_days} days..."
                })
            
            # CONCURRENT RECIPE + IMAGE GENERATION
            recipes_with_images = await self._generate_recipes_with_images_parallel(
                meal_structure=meal_structure,
                validated_data=validated_data,
                user=user,
                db=db,
                limits=limits,
                websocket_manager=websocket_manager,
                user_id=user_id,
                meal_plan_id=str(meal_plan.id)
            )
            
            # Save generated meals to database (BATCH OPERATION)
            await self._save_meals_batch(meal_plan, recipes_with_images, db)
            
            # Progress update
            if websocket_manager and user_id:
                await websocket_manager.send_message(user_id, {
                    "type": "meal_plan_progress",
                    "meal_plan_id": str(meal_plan.id),
                    "progress": 85,
                    "message": "📊 Calculating nutrition and generating shopping list..."
                })
            
            # BACKGROUND PROCESSING (Non-blocking)
            background_tasks = [
                self._calculate_meal_plan_analytics_fast(meal_plan, db),
                self._generate_shopping_list_fast(meal_plan, db)
            ]
            
            # Run background tasks concurrently
            await asyncio.gather(*background_tasks, return_exceptions=True)
            
            # Mark as completed
            meal_plan.status = 'active'
            meal_plan.generated_at = datetime.utcnow()
            generation_time = time.time() - start_time
            meal_plan.generation_time_seconds = round(generation_time, 2)
            db.commit()
            
            # Final progress update
            if websocket_manager and user_id:
                await websocket_manager.send_message(user_id, {
                    "type": "meal_plan_progress",
                    "meal_plan_id": str(meal_plan.id),
                    "progress": 100,
                    "message": f"✅ Meal plan ready! Generated in {generation_time:.1f}s",
                    "completed": True,
                    "meal_plan": {
                        "id": str(meal_plan.id),
                        "name": meal_plan.name,
                        "total_meals": total_meals,
                        "generation_time": generation_time,
                        "has_images": limits['image_generation']
                    }
                })
            
            logger.info(f"🎉 BLAZING FAST generation completed in {generation_time:.2f}s for meal plan {meal_plan.id}")
            return meal_plan
            
        except Exception as e:
            logger.error(f"🚨 BLAZING FAST generation failed for user {user.id}: {e}")
            if 'meal_plan' in locals():
                meal_plan.status = 'failed'
                meal_plan.error_message = str(e)
                db.commit()
                
                # Error progress update
                if websocket_manager and user_id:
                    await websocket_manager.send_message(user_id, {
                        "type": "meal_plan_progress",
                        "meal_plan_id": str(meal_plan.id),
                        "progress": 0,
                        "message": f"❌ Generation failed: {str(e)}",
                        "error": True
                    })
            raise
    
    def _create_meal_structure(self, days: int, meals_per_day: int, validated_data: Dict) -> List[Dict]:
        """Create optimized meal structure for parallel generation"""
        meal_types = ['breakfast', 'lunch', 'dinner', 'snack'][:meals_per_day]
        meal_structure = []
        
        for day in range(days):
            day_date = validated_data['start_date'] + timedelta(days=day)
            for meal_type in meal_types:
                meal_structure.append({
                    'day': day + 1,
                    'date': day_date,
                    'meal_type': meal_type,
                    'dietary_restrictions': validated_data.get('dietary_restrictions', []),
                    'preferences': validated_data.get('preferences', {}),
                    'family_size': validated_data.get('family_size', 1),
                    'target_calories': self._calculate_meal_calories(
                        meal_type, 
                        validated_data.get('target_calories')
                    )
                })
        
        return meal_structure
    
    async def _generate_recipes_with_images_parallel(
        self,
        meal_structure: List[Dict],
        validated_data: Dict,
        user: User,
        db: Session,
        limits: Dict,
        websocket_manager=None,
        user_id: str = None,
        meal_plan_id: str = None
    ) -> List[Dict]:
        """🚀 Generate recipes and images concurrently for maximum speed"""
        import asyncio
        from services.ai_service import ai_service
        
        async def generate_single_meal_with_image(meal_info: Dict, index: int) -> Dict:
            """Generate a single meal with AI image"""
            try:
                # Generate recipe using AI
                recipe_data = await self._generate_single_recipe_fast(meal_info, ai_service)
                
                # Generate image if premium
                image_url = None
                if limits['image_generation'] and recipe_data:
                    image_url = await vertex_service.generate_recipe_image(
                        recipe_data.get('title', ''),
                        recipe_data.get('description', '')
                    )
                
                # Update progress
                progress = 25 + (60 * (index + 1) / len(meal_structure))
                if websocket_manager and user_id:
                    await websocket_manager.send_message(user_id, {
                        "type": "meal_plan_progress",
                        "meal_plan_id": meal_plan_id,
                        "progress": min(progress, 85),
                        "message": f"🍳 Generated: {recipe_data.get('title', 'Recipe')}..."
                    })
                
                return {
                    **meal_info,
                    'recipe': recipe_data,
                    'image_url': image_url,
                    'generated_at': datetime.utcnow()
                }
                
            except Exception as e:
                logger.error(f"Failed to generate meal {index}: {e}")
                return {
                    **meal_info,
                    'recipe': self._get_fallback_recipe(meal_info),
                    'image_url': None,
                    'error': str(e)
                }
        
        # Execute concurrent generation with controlled concurrency
        concurrent_limit = limits['concurrent_recipes']
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def generate_with_semaphore(meal_info: Dict, index: int) -> Dict:
            async with semaphore:
                return await generate_single_meal_with_image(meal_info, index)
        
        # Start all tasks concurrently
        tasks = [
            generate_with_semaphore(meal_info, i) 
            for i, meal_info in enumerate(meal_structure)
        ]
        
        # Wait for all generations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = [
            result for result in results 
            if not isinstance(result, Exception) and result.get('recipe')
        ]
        
        logger.info(f"🎯 Generated {len(successful_results)}/{len(meal_structure)} meals successfully")
        return successful_results
    
    async def _validate_plan_data(self, plan_data: Dict[str, Any], user: User) -> Dict[str, Any]:
        """Validate and normalize meal plan data"""
        
        # Required fields
        if 'name' not in plan_data:
            plan_data['name'] = f"Meal Plan - {datetime.now().strftime('%B %Y')}"
        
        if 'start_date' not in plan_data:
            plan_data['start_date'] = date.today()
        elif isinstance(plan_data['start_date'], str):
            plan_data['start_date'] = datetime.strptime(plan_data['start_date'], '%Y-%m-%d').date()
        
        # Calculate end date if not provided
        if 'end_date' not in plan_data:
            days = plan_data.get('duration_days', 7)
            plan_data['end_date'] = plan_data['start_date'] + timedelta(days=days-1)
        elif isinstance(plan_data['end_date'], str):
            plan_data['end_date'] = datetime.strptime(plan_data['end_date'], '%Y-%m-%d').date()
        
        # Validate date range
        date_diff = (plan_data['end_date'] - plan_data['start_date']).days + 1
        if date_diff > self.max_plan_days:
            raise ValueError(f"Meal plan cannot exceed {self.max_plan_days} days")
        
        if date_diff < 1:
            raise ValueError("End date must be after start date")
        
        # Set default goals
        if 'goals' not in plan_data:
            plan_data['goals'] = ['balanced']
        
        # Calculate target calories based on user profile and goals
        if 'target_calories' not in plan_data:
            plan_data['target_calories'] = await self._calculate_target_calories(user, plan_data['goals'])
        
        # Merge user dietary restrictions and preferences
        user_restrictions = user.dietary_preferences or []
        plan_restrictions = plan_data.get('dietary_restrictions', [])
        plan_data['dietary_restrictions'] = list(set(user_restrictions + plan_restrictions))
        
        # Set user preferences
        plan_data['preferences'] = {
            **{
                'cuisine_preferences': user.cuisine_preferences or [],
                'skill_level': user.skill_level or 'intermediate',
                'cooking_goals': user.cooking_goals or []
            },
            **plan_data.get('preferences', {})
        }
        
        return plan_data
    
    async def _calculate_target_calories(self, user: User, goals: List[str]) -> int:
        """Calculate target daily calories based on user profile and goals"""
        
        # Base metabolic rate estimation (simplified Harris-Benedict)
        # In production, this would use more comprehensive user data
        base_calories = 2000  # Default
        
        # Adjust based on goals
        if 'weight_loss' in goals:
            base_calories = int(base_calories * 0.85)  # 15% deficit
        elif 'muscle_gain' in goals:
            base_calories = int(base_calories * 1.15)  # 15% surplus
        elif 'maintenance' in goals:
            base_calories = base_calories
        
        # Adjust based on user plan (premium users get more personalized calculations)
        if user.plan in ['premium', 'enterprise']:
            # More sophisticated calculation would go here
            pass
        
        return base_calories
    
    async def _generate_meal_plan_content(
        self,
        meal_plan: MealPlan,
        user: User,
        db: Session,
        plan_data: Dict[str, Any]
    ):
        """Generate meal plan content using AI"""
        
        current_date = meal_plan.start_date
        day_number = 1
        
        while current_date <= meal_plan.end_date:
            # Create meal plan day
            plan_day = MealPlanDay(
                meal_plan_id=meal_plan.id,
                date=current_date,
                day_number=day_number,
                target_calories=meal_plan.target_calories_per_day,
                meal_times=self.default_meal_times.copy()
            )
            
            db.add(plan_day)
            db.commit()
            db.refresh(plan_day)
            
            # Generate meals for the day
            await self._generate_day_meals(plan_day, meal_plan, user, db, plan_data)
            
            current_date += timedelta(days=1)
            day_number += 1
    
    async def _generate_day_meals(
        self,
        plan_day: MealPlanDay,
        meal_plan: MealPlan,
        user: User,
        db: Session,
        plan_data: Dict[str, Any]
    ):
        """Generate meals for a specific day using AI"""
        
        # Define meal types and calorie distribution based on family size
        family_size = plan_data.get('family_size', 1)
        
        if family_size > 2:
            # Include snacks for larger families
            meals_config = [
                {'type': 'breakfast', 'calories_percentage': 0.25},
                {'type': 'lunch', 'calories_percentage': 0.35},
                {'type': 'dinner', 'calories_percentage': 0.30},
                {'type': 'snack', 'calories_percentage': 0.10}
            ]
        else:
            # Just main meals for smaller families/individuals
            meals_config = [
                {'type': 'breakfast', 'calories_percentage': 0.30},
                {'type': 'lunch', 'calories_percentage': 0.40},
                {'type': 'dinner', 'calories_percentage': 0.30}
            ]
        
        daily_calories = plan_day.target_calories
        
        for meal_config in meals_config:
            meal_calories = int(daily_calories * meal_config['calories_percentage'])
            
            # Generate AI prompt for meal
            ai_prompt = await self._create_meal_generation_prompt(
                meal_config['type'], meal_calories, meal_plan, user, plan_data
            )
            
            # Get recipe recommendation from AI
            recipe_data = await ai_service.generate_meal_recipe(
                user=user,
                prompt=ai_prompt,
                meal_type=meal_config['type'],
                target_calories=meal_calories,
                dietary_restrictions=meal_plan.dietary_restrictions,
                preferences=meal_plan.preferences
            )
            
            if recipe_data:
                # Create or find recipe
                recipe = await self._create_or_find_recipe(recipe_data, db)
                
                # Create meal plan meal
                plan_meal = MealPlanMeal(
                    meal_plan_day_id=plan_day.id,
                    recipe_id=recipe.id,
                    meal_type=meal_config['type'],
                    scheduled_time=self.default_meal_times.get(meal_config['type']),
                    servings=meal_plan.family_size,
                    target_calories=meal_calories,
                    order_index=len(meals_config) if meal_config['type'] != 'snack' else 3
                )
                
                db.add(plan_meal)
            
            # Small delay to avoid overwhelming the AI service
            await asyncio.sleep(0.5)
        
        db.commit()
    
    async def _create_meal_generation_prompt(
        self,
        meal_type: str,
        target_calories: int,
        meal_plan: MealPlan,
        user: User,
        plan_data: Dict[str, Any]
    ) -> str:
        """Create AI prompt for meal generation"""
        
        prompt_parts = [
            f"Generate a {meal_type} recipe for a meal plan.",
            f"Target calories: {target_calories}",
            f"Servings: {meal_plan.family_size}",
        ]
        
        if meal_plan.dietary_restrictions:
            prompt_parts.append(f"Dietary restrictions: {', '.join(meal_plan.dietary_restrictions)}")
        
        if meal_plan.cooking_time_available:
            prompt_parts.append(f"Maximum cooking time: {meal_plan.cooking_time_available} minutes")
        
        if meal_plan.goals:
            prompt_parts.append(f"Nutritional goals: {', '.join(meal_plan.goals)}")
        
        preferences = meal_plan.preferences or {}
        if preferences.get('cuisine_preferences'):
            prompt_parts.append(f"Preferred cuisines: {', '.join(preferences['cuisine_preferences'])}")
        
        skill_level = preferences.get('skill_level', 'intermediate')
        prompt_parts.append(f"Cooking skill level: {skill_level}")
        
        if meal_plan.budget_per_week:
            daily_budget = meal_plan.budget_per_week / 7
            meal_budget = daily_budget / 4  # 4 meals per day
            prompt_parts.append(f"Target budget per meal: ${meal_budget:.2f}")
        
        return " ".join(prompt_parts)
    
    async def _create_or_find_recipe(self, recipe_data: Dict[str, Any], db: Session) -> Recipe:
        """Create new recipe or find existing one"""
        
        # Check if similar recipe exists
        existing_recipe = db.query(Recipe).filter(
            Recipe.name.ilike(f"%{recipe_data['name']}%")
        ).first()
        
        if existing_recipe:
            return existing_recipe
        
        # Create new recipe
        recipe = Recipe(
            name=recipe_data['name'],
            description=recipe_data.get('description', ''),
            instructions=recipe_data.get('instructions', []),
            prep_time_minutes=recipe_data.get('prep_time_minutes', 15),
            cook_time_minutes=recipe_data.get('cook_time_minutes', 30),
            total_time_minutes=recipe_data.get('total_time_minutes', 45),
            servings=recipe_data.get('servings', 4),
            difficulty=recipe_data.get('difficulty', 'intermediate'),
            cuisine_type=recipe_data.get('cuisine_type'),
            meal_types=recipe_data.get('meal_types', []),
            dietary_tags=recipe_data.get('dietary_tags', []),
            nutrition_per_serving=recipe_data.get('nutrition_per_serving', {}),
            calories_per_serving=recipe_data.get('calories_per_serving'),
            created_at=datetime.utcnow(),
            is_ai_generated=True
        )
        
        db.add(recipe)
        db.commit()
        db.refresh(recipe)
        
        # Add ingredients
        ingredients = recipe_data.get('ingredients', [])
        for ingredient_data in ingredients:
            recipe_ingredient = RecipeIngredient(
                recipe_id=recipe.id,
                name=ingredient_data['name'],
                amount=ingredient_data.get('amount', ''),
                unit=ingredient_data.get('unit', ''),
                category=ingredient_data.get('category', 'other'),
                optional=ingredient_data.get('optional', False)
            )
            db.add(recipe_ingredient)
        
        db.commit()
        return recipe
    
    async def _calculate_meal_plan_analytics(self, meal_plan: MealPlan, db: Session):
        """Calculate comprehensive analytics for meal plan"""
        
        # Get all meals in the plan
        plan_days = db.query(MealPlanDay).filter(
            MealPlanDay.meal_plan_id == meal_plan.id
        ).all()
        
        total_stats = {
            'total_calories': 0,
            'total_protein': 0,
            'total_carbs': 0,
            'total_fat': 0,
            'total_fiber': 0,
            'total_recipes': 0,
            'unique_ingredients': set(),
            'cuisine_distribution': {},
            'difficulty_distribution': {},
            'meal_type_distribution': {}
        }
        
        daily_stats = []
        
        for plan_day in plan_days:
            day_stats = {
                'date': plan_day.date,
                'calories': 0,
                'protein': 0,
                'carbs': 0,
                'fat': 0,
                'fiber': 0,
                'meals': []
            }
            
            # Calculate daily nutrition
            for meal in plan_day.meals:
                if meal.recipe and meal.recipe.nutrition_per_serving:
                    nutrition = meal.recipe.nutrition_per_serving
                    servings = meal.servings or 1
                    
                    meal_calories = (nutrition.get('calories', 0) * servings)
                    meal_protein = (nutrition.get('protein', 0) * servings)
                    meal_carbs = (nutrition.get('carbohydrates', 0) * servings)
                    meal_fat = (nutrition.get('fat', 0) * servings)
                    meal_fiber = (nutrition.get('fiber', 0) * servings)
                    
                    day_stats['calories'] += meal_calories
                    day_stats['protein'] += meal_protein
                    day_stats['carbs'] += meal_carbs
                    day_stats['fat'] += meal_fat
                    day_stats['fiber'] += meal_fiber
                    
                    # Add to totals
                    total_stats['total_calories'] += meal_calories
                    total_stats['total_protein'] += meal_protein
                    total_stats['total_carbs'] += meal_carbs
                    total_stats['total_fat'] += meal_fat
                    total_stats['total_fiber'] += meal_fiber
                    total_stats['total_recipes'] += 1
                    
                    # Track distributions
                    cuisine = meal.recipe.cuisine_type or 'other'
                    total_stats['cuisine_distribution'][cuisine] = \
                        total_stats['cuisine_distribution'].get(cuisine, 0) + 1
                    
                    difficulty = meal.recipe.difficulty or 'intermediate'
                    total_stats['difficulty_distribution'][difficulty] = \
                        total_stats['difficulty_distribution'].get(difficulty, 0) + 1
                    
                    meal_type = meal.meal_type
                    total_stats['meal_type_distribution'][meal_type] = \
                        total_stats['meal_type_distribution'].get(meal_type, 0) + 1
                    
                    # Collect ingredients
                    for ingredient in meal.recipe.ingredients:
                        total_stats['unique_ingredients'].add(ingredient.name.lower())
                    
                    day_stats['meals'].append({
                        'type': meal.meal_type,
                        'recipe_name': meal.recipe.name,
                        'calories': meal_calories,
                        'protein': meal_protein
                    })
            
            daily_stats.append(day_stats)
        
        # Calculate averages
        num_days = len(plan_days)
        if num_days > 0:
            avg_stats = {
                'avg_calories_per_day': total_stats['total_calories'] / num_days,
                'avg_protein_per_day': total_stats['total_protein'] / num_days,
                'avg_carbs_per_day': total_stats['total_carbs'] / num_days,
                'avg_fat_per_day': total_stats['total_fat'] / num_days,
                'avg_fiber_per_day': total_stats['total_fiber'] / num_days
            }
        else:
            avg_stats = {}
        
        # Create analytics record
        analytics = MealPlanAnalytics(
            meal_plan_id=meal_plan.id,
            total_calories=total_stats['total_calories'],
            avg_calories_per_day=avg_stats.get('avg_calories_per_day', 0),
            total_protein=total_stats['total_protein'],
            total_carbohydrates=total_stats['total_carbs'],
            total_fat=total_stats['total_fat'],
            total_fiber=total_stats['total_fiber'],
            unique_ingredients_count=len(total_stats['unique_ingredients']),
            recipe_diversity_score=self._calculate_diversity_score(total_stats),
            nutritional_balance_score=self._calculate_balance_score(avg_stats, meal_plan.goals),
            daily_breakdown=daily_stats,
            cuisine_distribution=total_stats['cuisine_distribution'],
            difficulty_distribution=total_stats['difficulty_distribution'],
            generated_at=datetime.utcnow()
        )
        
        db.add(analytics)
        db.commit()
    
    def _calculate_diversity_score(self, stats: Dict[str, Any]) -> float:
        """Calculate recipe diversity score (0-100)"""
        
        # Factors: cuisine variety, ingredient variety, difficulty variety
        cuisine_variety = len(stats['cuisine_distribution']) / 10  # Max 10 cuisines
        ingredient_variety = len(stats['unique_ingredients']) / 100  # Max 100 ingredients
        difficulty_variety = len(stats['difficulty_distribution']) / 4  # Max 4 difficulties
        
        # Weight the factors
        diversity_score = (
            cuisine_variety * 0.4 +
            ingredient_variety * 0.4 +
            difficulty_variety * 0.2
        ) * 100
        
        return min(100, diversity_score)
    
    def _calculate_balance_score(self, avg_stats: Dict[str, Any], goals: List[str]) -> float:
        """Calculate nutritional balance score based on goals"""
        
        if not avg_stats:
            return 0
        
        # Get goal priorities
        primary_goal = goals[0] if goals else 'balanced'
        priorities = self.goal_priorities.get(primary_goal, self.goal_priorities['balanced'])
        
        # Calculate score based on hitting nutritional targets
        # This is simplified - in production would use more sophisticated calculations
        balance_score = 75  # Base score
        
        # Adjust for calorie target (example logic)
        target_calories = 2000  # Would come from meal plan
        actual_calories = avg_stats.get('avg_calories_per_day', 0)
        
        if actual_calories > 0:
            calorie_ratio = min(actual_calories, target_calories) / max(actual_calories, target_calories)
            balance_score += (calorie_ratio - 0.8) * 25  # Bonus for being within 20%
        
        return max(0, min(100, balance_score))
    
    async def _generate_shopping_list(self, meal_plan: MealPlan, db: Session):
        """Generate optimized shopping list for meal plan"""
        
        # Collect all ingredients
        ingredient_totals = {}
        ingredient_categories = {}
        
        plan_days = db.query(MealPlanDay).filter(
            MealPlanDay.meal_plan_id == meal_plan.id
        ).all()
        
        for plan_day in plan_days:
            for meal in plan_day.meals:
                if meal.recipe:
                    servings_multiplier = meal.servings or 1
                    
                    for ingredient in meal.recipe.ingredients:
                        key = ingredient.name.lower()
                        
                        if key not in ingredient_totals:
                            ingredient_totals[key] = {
                                'name': ingredient.name,
                                'total_amount': 0,
                                'unit': ingredient.unit or '',
                                'category': ingredient.category or 'other',
                                'recipes': []
                            }
                        
                        # Parse amount (simplified - would need more robust parsing)
                        try:
                            amount = float(ingredient.amount or 0) * servings_multiplier
                            ingredient_totals[key]['total_amount'] += amount
                        except (ValueError, TypeError):
                            # Handle non-numeric amounts
                            ingredient_totals[key]['total_amount'] = ingredient.amount or 'as needed'
                        
                        ingredient_totals[key]['recipes'].append(meal.recipe.name)
                        ingredient_categories[ingredient.category or 'other'] = \
                            ingredient_categories.get(ingredient.category or 'other', 0) + 1
        
        # Organize by category
        organized_list = {}
        for ingredient_data in ingredient_totals.values():
            category = ingredient_data['category']
            if category not in organized_list:
                organized_list[category] = []
            
            organized_list[category].append({
                'name': ingredient_data['name'],
                'amount': ingredient_data['total_amount'],
                'unit': ingredient_data['unit'],
                'recipes': list(set(ingredient_data['recipes']))  # Remove duplicates
            })
        
        # Calculate estimated cost (simplified)
        estimated_cost = len(ingredient_totals) * 3.50  # $3.50 average per ingredient
        
        # Create shopping list record
        shopping_list = MealPlanShoppingList(
            meal_plan_id=meal_plan.id,
            ingredients_by_category=organized_list,
            total_items=len(ingredient_totals),
            estimated_cost=estimated_cost,
            generated_at=datetime.utcnow()
        )
        
        db.add(shopping_list)
        db.commit()
    
    async def get_meal_plan(self, meal_plan_id: int, user: User, db: Session) -> Optional[MealPlan]:
        """Get meal plan with all related data including days and meals"""
        from sqlalchemy.orm import joinedload
        
        meal_plan = db.query(MealPlan).options(
            joinedload(MealPlan.days).joinedload(MealPlanDay.meals).joinedload(MealPlanMeal.recipe)
        ).filter(
            and_(
                MealPlan.id == meal_plan_id,
                MealPlan.user_id == user.id
            )
        ).first()
        
        return meal_plan
    
    async def get_user_meal_plans(
        self,
        user: User,
        db: Session,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None
    ) -> Tuple[List[MealPlan], int]:
        """Get user's meal plans with pagination"""
        
        query = db.query(MealPlan).filter(MealPlan.user_id == user.id)
        
        if status:
            query = query.filter(MealPlan.status == status)
        
        total = query.count()
        meal_plans = query.order_by(MealPlan.created_at.desc()).offset(offset).limit(limit).all()
        
        return meal_plans, total
    
    async def update_meal_plan(
        self,
        meal_plan_id: int,
        updates: Dict[str, Any],
        user: User,
        db: Session
    ) -> Optional[MealPlan]:
        """Update meal plan"""
        
        meal_plan = await self.get_meal_plan(meal_plan_id, user, db)
        if not meal_plan:
            return None
        
        # Update allowed fields
        updatable_fields = ['name', 'description', 'status', 'preferences']
        for field, value in updates.items():
            if field in updatable_fields:
                setattr(meal_plan, field, value)
        
        meal_plan.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(meal_plan)
        
        return meal_plan
    
    async def delete_meal_plan(self, meal_plan_id: int, user: User, db: Session) -> Dict[str, Any]:
        """Soft delete meal plan and all related data"""
        
        meal_plan = await self.get_meal_plan(meal_plan_id, user, db)
        if not meal_plan:
            return {"success": False, "error": "Meal plan not found"}
        
        # Count affected items for user feedback
        total_days = len(meal_plan.days)
        total_meals = sum(len(day.meals) for day in meal_plan.days)
        total_shopping_lists = len(meal_plan.shopping_lists) if hasattr(meal_plan, 'shopping_lists') else 0
        
        try:
            # Soft delete the meal plan
            meal_plan.status = 'deleted'
            meal_plan.deleted_at = datetime.utcnow()
            
            # Soft delete all related data
            for day in meal_plan.days:
                day.status = 'deleted'
                for meal in day.meals:
                    meal.status = 'deleted'
            
            # Soft delete shopping lists if they exist
            if hasattr(meal_plan, 'shopping_lists'):
                for shopping_list in meal_plan.shopping_lists:
                    shopping_list.status = 'deleted'
            
            # Soft delete analytics if they exist
            if hasattr(meal_plan, 'analytics'):
                for analytics in meal_plan.analytics:
                    analytics.status = 'deleted'
            
            db.commit()
            
            logger.info(f"✅ Soft deleted meal plan {meal_plan_id}: {total_days} days, {total_meals} meals, {total_shopping_lists} shopping lists")
            
            return {
                "success": True,
                "message": "Meal plan deleted successfully",
                "affected_items": {
                    "days": total_days,
                    "meals": total_meals,
                    "shopping_lists": total_shopping_lists
                }
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Failed to delete meal plan {meal_plan_id}: {e}")
            return {"success": False, "error": f"Failed to delete meal plan: {str(e)}"}
        
    async def permanently_delete_meal_plan(self, meal_plan_id: int, user: User, db: Session) -> bool:
        """Permanently delete meal plan (admin only)"""
        
        meal_plan = db.query(MealPlan).filter(
            and_(
                MealPlan.id == meal_plan_id,
                MealPlan.user_id == user.id
            )
        ).first()
        
        if not meal_plan:
            return False
        
        try:
            # Hard delete with SQLAlchemy cascade
            db.delete(meal_plan)
            db.commit()
            
            logger.info(f"✅ Permanently deleted meal plan {meal_plan_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Failed to permanently delete meal plan {meal_plan_id}: {e}")
            return False
    
    async def regenerate_day(
        self,
        meal_plan_id: int,
        date: date,
        user: User,
        db: Session
    ) -> Optional[MealPlanDay]:
        """Regenerate meals for a specific day"""
        
        meal_plan = await self.get_meal_plan(meal_plan_id, user, db)
        if not meal_plan:
            return None
        
        # Find the day
        plan_day = db.query(MealPlanDay).filter(
            and_(
                MealPlanDay.meal_plan_id == meal_plan_id,
                MealPlanDay.date == date
            )
        ).first()
        
        if not plan_day:
            return None
        
        # Delete existing meals
        db.query(MealPlanMeal).filter(
            MealPlanMeal.meal_plan_day_id == plan_day.id
        ).delete()
        
        # Regenerate meals
        plan_data = {
            'goals': meal_plan.goals,
            'dietary_restrictions': meal_plan.dietary_restrictions,
            'preferences': meal_plan.preferences
        }
        
        await self._generate_day_meals(plan_day, meal_plan, user, db, plan_data)
        
        return plan_day
    
    async def swap_meal(
        self,
        meal_id: int,
        user: User,
        db: Session,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Optional[MealPlanMeal]:
        """Swap a meal with an AI-generated alternative"""
        
        meal = db.query(MealPlanMeal).join(MealPlanDay).join(MealPlan).filter(
            and_(
                MealPlanMeal.id == meal_id,
                MealPlan.user_id == user.id
            )
        ).first()
        
        if not meal:
            return None
        
        # Generate alternative recipe
        meal_plan = meal.meal_plan_day.meal_plan
        ai_prompt = await self._create_meal_generation_prompt(
            meal.meal_type,
            meal.target_calories,
            meal_plan,
            user,
            {'preferences': preferences or meal_plan.preferences or {}}
        )
        
        recipe_data = await ai_service.generate_meal_recipe(
            user=user,
            prompt=ai_prompt,
            meal_type=meal.meal_type,
            target_calories=meal.target_calories,
            dietary_restrictions=meal_plan.dietary_restrictions,
            preferences=preferences or meal_plan.preferences
        )
        
        if recipe_data:
            # Create new recipe
            new_recipe = await self._create_or_find_recipe(recipe_data, db)
            
            # Update meal
            meal.recipe_id = new_recipe.id
            meal.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(meal)
        
        return meal

    async def delete_recipe_with_meal_plan_updates(
        self, 
        recipe_id: str, 
        user: User, 
        db: Session
    ) -> Dict[str, Any]:
        """
        Enhanced recipe deletion with proper meal plan relationship handling
        """
        try:
            # 1. Find all meal plans using this recipe
            affected_meal_plans_query = db.query(MealPlan).join(
                MealPlanDay
            ).join(
                MealPlanMeal
            ).filter(
                MealPlanMeal.recipe_id == recipe_id,
                MealPlan.user_id == user.id,
                MealPlan.status != 'deleted'
            ).distinct()
            
            affected_meal_plans = affected_meal_plans_query.all()
            
            # 2. Count meals that will be affected
            affected_meals = db.query(MealPlanMeal).join(
                MealPlanDay
            ).join(
                MealPlan
            ).filter(
                MealPlanMeal.recipe_id == recipe_id,
                MealPlan.user_id == user.id,
                MealPlan.status != 'deleted'
            ).all()
            
            meal_plan_names = [mp.name for mp in affected_meal_plans]
            total_affected_meals = len(affected_meals)
            
            logger.info(
                f"🔍 Recipe {recipe_id} used in {len(affected_meal_plans)} "
                f"meal plans, {total_affected_meals} meals total"
            )
            
            # 3. Handle meal plan updates
            for meal_plan in affected_meal_plans:
                for day in meal_plan.days:
                    meals_to_update = [
                        m for m in day.meals 
                        if m.recipe_id == recipe_id and m.status != 'deleted'
                    ]
                    
                    for meal in meals_to_update:
                        # Option A: Remove meal entirely (soft delete)
                        meal.status = 'deleted'
                        meal.notes = (meal.notes or '') + \
                            f" [Recipe '{recipe_id}' was deleted]"
                        
                        # Option B: Mark as needing substitution (alternative)
                        # meal.recipe_id = None
                        # meal.status = 'needs_substitution'
                        # meal.notes = f"Original recipe deleted, needs substitution"
                
                # Update meal plan timestamp
                meal_plan.updated_at = datetime.utcnow()
            
            # 4. Delete recipe and its components (let SQLAlchemy handle cascade)
            recipe = db.query(Recipe).filter(
                Recipe.id == recipe_id,
                Recipe.user_id == user.id
            ).first()
            
            if recipe:
                # Soft delete recipe
                recipe.status = 'deleted'
                recipe.deleted_at = datetime.utcnow()
            
            db.commit()
            
            result = {
                "success": True,
                "message": "Recipe deleted successfully",
                "affected_meal_plans": meal_plan_names,
                "removed_meals": total_affected_meals,
                "details": {
                    "recipe_id": recipe_id,
                    "meal_plans_updated": len(affected_meal_plans),
                    "meals_removed": total_affected_meals
                }
            }
            
            logger.info(
                f"✅ Recipe {recipe_id} deleted: "
                f"{len(affected_meal_plans)} meal plans updated, "
                f"{total_affected_meals} meals removed"
            )
            
            return result
            
        except Exception as e:
            db.rollback()
            error_msg = f"Failed to delete recipe {recipe_id}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "affected_meal_plans": [],
                "removed_meals": 0
            }
    
    async def restore_meal_plan(
        self, 
        meal_plan_id: int, 
        user: User, 
        db: Session
    ) -> Dict[str, Any]:
        """Restore a soft-deleted meal plan"""
        
        meal_plan = db.query(MealPlan).filter(
            and_(
                MealPlan.id == meal_plan_id,
                MealPlan.user_id == user.id,
                MealPlan.status == 'deleted'
            )
        ).first()
        
        if not meal_plan:
            return {"success": False, "error": "Deleted meal plan not found"}
        
        try:
            # Restore meal plan
            meal_plan.status = 'active'
            meal_plan.deleted_at = None
            meal_plan.updated_at = datetime.utcnow()
            
            # Restore related data
            restored_days = 0
            restored_meals = 0
            
            for day in meal_plan.days:
                if day.status == 'deleted':
                    day.status = 'planned'
                    restored_days += 1
                    
                for meal in day.meals:
                    if meal.status == 'deleted':
                        meal.status = 'planned'
                        restored_meals += 1
            
            db.commit()
            
            logger.info(
                f"✅ Restored meal plan {meal_plan_id}: "
                f"{restored_days} days, {restored_meals} meals"
            )
            
            return {
                "success": True,
                "message": "Meal plan restored successfully",
                "restored_items": {
                    "days": restored_days,
                    "meals": restored_meals
                }
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Failed to restore meal plan {meal_plan_id}: {e}")
            return {"success": False, "error": f"Failed to restore: {str(e)}"}

    async def _generate_single_recipe_fast(self, meal_info: Dict, ai_service) -> Dict:
        """Generate a single recipe quickly"""
        try:
            # Use the AI service client directly instead of schemas
            
            # Create optimized AI request
            prompt = f"Create a {meal_info['meal_type']} recipe for {meal_info['family_size']} people"
            
            # Generate recipe using the AI service client
            response = await ai_service.generate_recipe(
                ingredients=[],  # Will be generated by AI
                cuisine=meal_info.get('cuisine'),
                difficulty=meal_info.get('difficulty', 'medium'),
                dietary_restrictions=meal_info.get('dietary_restrictions', []),
                cooking_time=meal_info.get('cooking_time'),
                servings=meal_info.get('family_size', 2)
            )
            
            if response and response.get('success'):
                # Return the recipe data from the response
                return response.get('recipe', {})
            
            return self._get_fallback_recipe(meal_info)
            
        except Exception as e:
            logger.error(f"Fast recipe generation failed: {e}")
            return self._get_fallback_recipe(meal_info)
    
    def _get_fallback_recipe(self, meal_info: Dict) -> Dict:
        """Get a fallback recipe when AI generation fails"""
        meal_type = meal_info.get('meal_type', 'meal')
        
        fallback_recipes = {
            'breakfast': {
                'title': 'Quick Breakfast Bowl',
                'description': 'A nutritious and quick breakfast option',
                'ingredients': ['Oats', 'Milk', 'Berries', 'Honey'],
                'instructions': ['Combine oats and milk', 'Add berries and honey', 'Mix and serve'],
                'prep_time_minutes': 5,
                'cook_time_minutes': 0,
                'servings': meal_info.get('family_size', 1),
                'calories_per_serving': 250
            },
            'lunch': {
                'title': 'Simple Salad Bowl',
                'description': 'Fresh and healthy lunch salad',
                'ingredients': ['Mixed greens', 'Protein', 'Vegetables', 'Dressing'],
                'instructions': ['Wash greens', 'Add protein and vegetables', 'Toss with dressing'],
                'prep_time_minutes': 10,
                'cook_time_minutes': 0,
                'servings': meal_info.get('family_size', 1),
                'calories_per_serving': 350
            },
            'dinner': {
                'title': 'Balanced Dinner Plate',
                'description': 'Complete dinner with protein and vegetables',
                'ingredients': ['Protein source', 'Vegetables', 'Whole grains', 'Seasonings'],
                'instructions': ['Prepare protein', 'Cook vegetables', 'Serve with grains'],
                'prep_time_minutes': 15,
                'cook_time_minutes': 25,
                'servings': meal_info.get('family_size', 1),
                'calories_per_serving': 450
            }
        }
        
        return fallback_recipes.get(meal_type, fallback_recipes['lunch'])
    
    def _parse_recipe_response(self, content: str, meal_info: Dict) -> Dict:
        """Parse AI response into structured recipe format"""
        import json
        
        try:
            # Try to parse as JSON first
            if content.strip().startswith('{'):
                return json.loads(content)
            
            # Fallback parsing for text format
            lines = content.split('\n')
            recipe = {
                'title': f"{meal_info['meal_type'].title()} Recipe",
                'description': '',
                'ingredients': [],
                'instructions': [],
                'prep_time_minutes': 15,
                'cook_time_minutes': 20,
                'servings': meal_info.get('family_size', 1),
                'calories_per_serving': meal_info.get('target_calories', 350)
            }
            
            # Simple text parsing
            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if 'ingredients' in line.lower():
                    current_section = 'ingredients'
                elif 'instructions' in line.lower() or 'steps' in line.lower():
                    current_section = 'instructions'
                elif line.startswith(('1.', '2.', '3.', '-', '*')):
                    if current_section == 'ingredients':
                        recipe['ingredients'].append(line.lstrip('123456789.-* '))
                    elif current_section == 'instructions':
                        recipe['instructions'].append(line.lstrip('123456789.-* '))
                elif current_section is None and recipe['title'] == f"{meal_info['meal_type'].title()} Recipe":
                    recipe['title'] = line
            
            return recipe
            
        except Exception as e:
            logger.error(f"Recipe parsing failed: {e}")
            return self._get_fallback_recipe(meal_info)
    
    def _calculate_meal_calories(self, meal_type: str, total_daily_calories: int) -> int:
        """Calculate target calories for a specific meal type"""
        if not total_daily_calories:
            total_daily_calories = 2000  # Default
        
        calorie_distribution = {
            'breakfast': 0.25,
            'lunch': 0.35,
            'dinner': 0.35,
            'snack': 0.05
        }
        
        return int(total_daily_calories * calorie_distribution.get(meal_type, 0.33))
    
    async def _save_meals_batch(self, meal_plan: MealPlan, recipes_with_images: List[Dict], db: Session):
        """Save all generated meals to database using the CORRECT schema (planned_meals table)"""
        try:
            import uuid
            from sqlalchemy import text
            
            saved_meals_count = 0
            
            for meal_data in recipes_with_images:
                try:
                    # Get meal type ID from the database using the actual schema
                    meal_type_query = text("SELECT id FROM meal_planning.meal_types WHERE LOWER(name) = LOWER(:meal_type) LIMIT 1")
                    meal_type_result = db.execute(meal_type_query, {"meal_type": meal_data['meal_type']}).fetchone()
                    
                    if not meal_type_result:
                        # Fallback to a default meal type if not found
                        default_type_result = db.execute(text("SELECT id FROM meal_planning.meal_types LIMIT 1")).fetchone()
                        meal_type_id = default_type_result[0] if default_type_result else None
                    else:
                        meal_type_id = meal_type_result[0]
                    
                    if not meal_type_id:
                        logger.error(f"No meal types found in database - cannot save meal")
                        continue
                    
                    # Create recipe in the database first using the actual schema
                    recipe_info = meal_data.get('recipe', {})
                    recipe_id = str(uuid.uuid4())
                    
                    # Get organization_id from user
                    org_id_query = text("SELECT organization_id FROM core.users WHERE id = :user_id")
                    org_result = db.execute(org_id_query, {"user_id": meal_plan.user_id}).fetchone()
                    organization_id = org_result[0] if org_result else str(uuid.uuid4())
                    
                    # Insert recipe into recipes table using the actual schema
                    recipe_insert = text("""
                        INSERT INTO recipe.recipes (
                            id, organization_id, created_by_user_id, title, slug, description, 
                            difficulty_level, prep_time_minutes, cook_time_minutes, servings,
                            calories_per_serving, status, is_public, created_at, updated_at
                        ) VALUES (
                            :recipe_id, :organization_id, :user_id, :title, :slug, :description,
                            :difficulty, :prep_time, :cook_time, :servings, :calories,
                            :status, :is_public, :created_at, :updated_at
                        )
                    """)
                    
                    title = recipe_info.get('title', f'AI Recipe {saved_meals_count + 1}')
                    slug = title.lower().replace(' ', '-').replace('[^a-z0-9-]', '')[:100]
                    
                    db.execute(recipe_insert, {
                        "recipe_id": recipe_id,
                        "organization_id": organization_id,
                        "user_id": meal_plan.user_id,
                        "title": title,
                        "slug": slug,
                        "description": recipe_info.get('description', 'AI-generated recipe'),
                        "difficulty": recipe_info.get('difficulty_level', 'intermediate'),
                        "prep_time": recipe_info.get('prep_time_minutes', 15),
                        "cook_time": recipe_info.get('cook_time_minutes', 30),
                        "servings": recipe_info.get('servings', meal_plan.family_size or 1),
                        "calories": recipe_info.get('calories_per_serving', None),
                        "status": 'published',
                        "is_public": False,
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    })
                    
                    # Save image to recipe_images table if we have one
                    image_url = meal_data.get('image_url')
                    if image_url:
                        try:
                            image_id = str(uuid.uuid4())
                            image_insert = text("""
                                INSERT INTO recipe.recipe_images (
                                    id, recipe_id, image_url, caption, is_primary, display_order, created_at
                                ) VALUES (
                                    :image_id, :recipe_id, :image_url, :caption, :is_primary, :display_order, :created_at
                                )
                            """)
                            
                            db.execute(image_insert, {
                                "image_id": image_id,
                                "recipe_id": recipe_id,
                                "image_url": image_url,
                                "caption": f"AI-generated image for {title}",
                                "is_primary": True,
                                "display_order": 1,
                                "created_at": datetime.utcnow()
                            })
                            logger.info(f"✅ Saved image for recipe: {title}")
                        except Exception as img_error:
                            logger.error(f"Failed to save image for recipe {recipe_id}: {img_error}")
                            # Continue even if image save fails
                    
                    # CRITICAL FIX: Save ingredients and instructions for meal plan recipes
                    # This was the missing piece causing empty recipe data!
                    
                    # Save ingredients
                    ingredients = recipe_info.get('ingredients', [])
                    if ingredients:
                        logger.info(f"🥗 Saving {len(ingredients)} ingredients for meal plan recipe: {title}")
                        
                        # Get default unit
                        default_unit_result = db.execute(
                            text("SELECT id FROM recipe.units WHERE name = 'piece' LIMIT 1")
                        ).fetchone()
                        
                        if not default_unit_result:
                            # Create default unit if it doesn't exist
                            default_unit_id = str(uuid.uuid4())
                            db.execute(text("""
                                INSERT INTO recipe.units (id, name, abbreviation, type, system)
                                VALUES (:id, 'piece', 'pc', 'count', 'both')
                            """), {"id": default_unit_id})
                        else:
                            default_unit_id = default_unit_result[0]
                        
                        for idx, ingredient in enumerate(ingredients):
                            try:
                                ingredient_id = str(uuid.uuid4())
                                recipe_ingredient_id = str(uuid.uuid4())
                                
                                # Parse ingredient string
                                ingredient_str = ingredient if isinstance(ingredient, str) else str(ingredient)
                                parts = ingredient_str.split(None, 2)
                                quantity = 1.0
                                unit_name = "piece"
                                ingredient_name = ingredient_str
                                
                                # Try to parse quantity and unit
                                if len(parts) >= 2:
                                    try:
                                        quantity = float(parts[0].replace('/', '.'))
                                        if len(parts) >= 3:
                                            unit_name = parts[1]
                                            ingredient_name = parts[2]
                                        else:
                                            ingredient_name = parts[1]
                                    except (ValueError, IndexError):
                                        ingredient_name = ingredient_str
                                
                                # Get or create unit
                                unit_result = db.execute(
                                    text("SELECT id FROM recipe.units WHERE name = :name OR abbreviation = :abbr LIMIT 1"),
                                    {"name": unit_name.lower(), "abbr": unit_name.lower()}
                                ).fetchone()
                                unit_id = unit_result[0] if unit_result else default_unit_id
                                
                                # Create ingredient record
                                db.execute(text("""
                                    INSERT INTO recipe.ingredients (id, name)
                                    VALUES (:id, :name)
                                    ON CONFLICT (name) DO NOTHING
                                """), {"id": ingredient_id, "name": ingredient_name})
                                
                                # Get actual ingredient ID
                                actual_ing_result = db.execute(
                                    text("SELECT id FROM recipe.ingredients WHERE name = :name"),
                                    {"name": ingredient_name}
                                ).fetchone()
                                actual_ingredient_id = actual_ing_result[0] if actual_ing_result else ingredient_id
                                
                                # Link ingredient to recipe
                                db.execute(text("""
                                    INSERT INTO recipe.recipe_ingredients (
                                        id, recipe_id, ingredient_id, quantity, unit_id, 
                                        display_order, preparation_notes
                                    ) VALUES (
                                        :id, :recipe_id, :ingredient_id, :quantity, :unit_id,
                                        :display_order, :preparation_notes
                                    )
                                """), {
                                    "id": recipe_ingredient_id,
                                    "recipe_id": recipe_id,
                                    "ingredient_id": actual_ingredient_id,
                                    "quantity": quantity,
                                    "unit_id": unit_id,
                                    "display_order": idx,
                                    "preparation_notes": ingredient_str
                                })
                                
                            except Exception as ing_error:
                                logger.error(f"Failed to save ingredient {idx}: {ing_error}")
                                continue
                    
                    # Save instructions
                    instructions = recipe_info.get('instructions', [])
                    if instructions:
                        logger.info(f"📝 Saving {len(instructions)} instructions for meal plan recipe: {title}")
                        
                        for step_num, instruction in enumerate(instructions, 1):
                            try:
                                step_id = str(uuid.uuid4())
                                instruction_text = instruction if isinstance(instruction, str) else str(instruction)
                                
                                db.execute(text("""
                                    INSERT INTO recipe.recipe_steps (
                                        id, recipe_id, step_number, instruction
                                    ) VALUES (
                                        :id, :recipe_id, :step_number, :instruction
                                    )
                                """), {
                                    "id": step_id,
                                    "recipe_id": recipe_id,
                                    "step_number": step_num,
                                    "instruction": instruction_text
                                })
                                
                            except Exception as step_error:
                                logger.error(f"Failed to save instruction step {step_num}: {step_error}")
                                continue
                    
                    # Save nutrition info if available
                    nutrition = recipe_info.get('nutrition_info', recipe_info.get('nutrition', {}))
                    if nutrition and isinstance(nutrition, dict):
                        try:
                            nutrition_id = str(uuid.uuid4())
                            db.execute(text("""
                                INSERT INTO recipe.recipe_nutrition (
                                    id, recipe_id, calories, protein_g, carbohydrates_g, 
                                    fat_g, fiber_g, sugar_g, sodium_mg
                                ) VALUES (
                                    :id, :recipe_id, :calories, :protein, :carbs,
                                    :fat, :fiber, :sugar, :sodium
                                )
                            """), {
                                "id": nutrition_id,
                                "recipe_id": recipe_id,
                                "calories": nutrition.get('calories', 0),
                                "protein": nutrition.get('protein_g', 0),
                                "carbs": nutrition.get('carbohydrates_g', 0),
                                "fat": nutrition.get('fat_g', 0),
                                "fiber": nutrition.get('fiber_g', 0),
                                "sugar": nutrition.get('sugar_g', 0),
                                "sodium": nutrition.get('sodium_mg', 0)
                            })
                            logger.info(f"📊 Saved nutrition for meal plan recipe: {title}")
                        except Exception as nutr_error:
                            logger.error(f"Failed to save nutrition: {nutr_error}")
                    
                    # Save chef tips if available
                    chef_tips = recipe_info.get('chef_tips', [])
                    if chef_tips:
                        try:
                            # Convert list to array format for PostgreSQL
                            chef_tips_array = chef_tips if isinstance(chef_tips, list) else [chef_tips]
                            db.execute(text("""
                                UPDATE recipe.recipes 
                                SET chef_tips = :tips 
                                WHERE id = :recipe_id
                            """), {
                                "tips": chef_tips_array,
                                "recipe_id": recipe_id
                            })
                            logger.info(f"💡 Saved chef tips for meal plan recipe: {title}")
                        except Exception as tips_error:
                            logger.error(f"Failed to save chef tips: {tips_error}")
                    
                    # Calculate meal date
                    meal_date = meal_plan.start_date + timedelta(days=meal_data['day'] - 1)
                    
                    # Insert into planned_meals table (the table that actually exists!)
                    planned_meal_id = str(uuid.uuid4())
                    planned_meal_insert = text("""
                        INSERT INTO meal_planning.planned_meals (
                            id, meal_plan_id, recipe_id, meal_date, meal_type_id,
                            servings, notes, is_prepared, created_at
                        ) VALUES (
                            :planned_meal_id, :meal_plan_id, :recipe_id, :meal_date, :meal_type_id,
                            :servings, :notes, :is_prepared, :created_at
                        )
                    """)
                    
                    db.execute(planned_meal_insert, {
                        "planned_meal_id": planned_meal_id,
                        "meal_plan_id": meal_plan.id,
                        "recipe_id": recipe_id,
                        "meal_date": meal_date,
                        "meal_type_id": meal_type_id,
                        "servings": recipe_info.get('servings', meal_plan.family_size or 1),
                        "notes": f"AI-generated {meal_data['meal_type']} for day {meal_data['day']}",
                        "is_prepared": False,
                        "created_at": datetime.utcnow()
                    })
                    
                    saved_meals_count += 1
                    
                except Exception as meal_error:
                    logger.error(f"Failed to save individual meal: {meal_error}")
                    continue
            
            # Commit all changes
            db.commit()
            
            logger.info(f"✅ FIXED: Saved {saved_meals_count}/{len(recipes_with_images)} meals for plan {meal_plan.id}")
            
            if saved_meals_count == 0:
                raise Exception("No meals were saved - check meal types and database connectivity")
            
        except Exception as e:
            logger.error(f"Failed to save meals batch (FIXED VERSION): {e}")
            db.rollback()
            raise
    
    async def _calculate_meal_plan_analytics_fast(self, meal_plan: MealPlan, db: Session):
        """Fast background calculation of meal plan analytics"""
        try:
            # This runs in background - non-blocking
            logger.info(f"🔄 Starting background analytics for plan {meal_plan.id}")
            
            # Get all planned meals through proper relationships
            from models.meal_planning_models import MealPlanDay, MealPlanMeal
            planned_meals = db.query(MealPlanMeal).join(MealPlanDay).filter(
                MealPlanDay.meal_plan_id == meal_plan.id
            ).all()
            
            # Calculate totals
            total_calories = sum(
                meal.calories_per_serving * meal.servings 
                for meal in planned_meals
            )
            total_prep_time = sum(meal.prep_time_minutes for meal in planned_meals)
            total_cook_time = sum(meal.cook_time_minutes for meal in planned_meals)
            
            # Update meal plan with analytics
            meal_plan.total_calories = total_calories
            meal_plan.total_prep_time_minutes = total_prep_time
            meal_plan.total_cook_time_minutes = total_cook_time
            meal_plan.analytics_calculated_at = datetime.utcnow()
            
            db.commit()
            logger.info(f"✅ Analytics completed for plan {meal_plan.id}")
            
        except Exception as e:
            logger.error(f"Analytics calculation failed: {e}")
    
    async def _generate_shopping_list_fast(self, meal_plan: MealPlan, db: Session):
        """Fast background generation of shopping list"""
        try:
            logger.info(f"🔄 Generating shopping list for plan {meal_plan.id}")
            
            # Get all planned meals through proper relationships
            from models.meal_planning_models import MealPlanDay, MealPlanMeal
            planned_meals = db.query(MealPlanMeal).join(MealPlanDay).filter(
                MealPlanDay.meal_plan_id == meal_plan.id
            ).all()
            
            # Aggregate ingredients
            ingredient_totals = {}
            
            for meal in planned_meals:
                # Get recipe ingredients from the related recipe
                if meal.recipe and hasattr(meal.recipe, 'ai_metadata'):
                    recipe_data = meal.recipe.ai_metadata or {}
                    ingredients = recipe_data.get('ingredients', [])
                
                for ingredient in ingredients:
                    # Simple aggregation
                    if ingredient in ingredient_totals:
                        ingredient_totals[ingredient] += 1
                    else:
                        ingredient_totals[ingredient] = 1
            
            # Create shopping list entries
            from models.meal_planning_models import ShoppingListItem
            
            shopping_items = []
            for ingredient, quantity in ingredient_totals.items():
                item = ShoppingListItem(
                    meal_plan_id=meal_plan.id,
                    ingredient_name=ingredient,
                    quantity=f"{quantity} servings",
                    is_purchased=False,
                    created_at=datetime.utcnow()
                )
                shopping_items.append(item)
            
            # Batch insert shopping list
            db.add_all(shopping_items)
            db.commit()
            
            logger.info(f"✅ Shopping list generated with {len(shopping_items)} items")
            
        except Exception as e:
            logger.error(f"Shopping list generation failed: {e}")

# Create singleton instance
meal_planning_service = MealPlanningService()