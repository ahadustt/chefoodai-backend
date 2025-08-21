"""
ChefoodAI Nutrition Service
Advanced nutritional analysis and optimization for meal planning
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, date, timedelta
import logging
import asyncio

from models.users import User
from models.meal_planning_models import NutritionalGoal, MealPlan
from models.recipe_models import Recipe, RecipeIngredient
from core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

@dataclass
class NutritionalProfile:
    """Complete nutritional profile for foods/recipes"""
    calories: float = 0.0
    protein: float = 0.0  # grams
    carbohydrates: float = 0.0  # grams
    fat: float = 0.0  # grams
    fiber: float = 0.0  # grams
    sugar: float = 0.0  # grams
    sodium: float = 0.0  # mg
    saturated_fat: float = 0.0  # grams
    cholesterol: float = 0.0  # mg
    
    # Vitamins (% daily value)
    vitamin_a: float = 0.0
    vitamin_c: float = 0.0
    vitamin_d: float = 0.0
    vitamin_e: float = 0.0
    vitamin_k: float = 0.0
    thiamin: float = 0.0
    riboflavin: float = 0.0
    niacin: float = 0.0
    vitamin_b6: float = 0.0
    folate: float = 0.0
    vitamin_b12: float = 0.0
    
    # Minerals (% daily value)
    calcium: float = 0.0
    iron: float = 0.0
    magnesium: float = 0.0
    phosphorus: float = 0.0
    potassium: float = 0.0
    zinc: float = 0.0
    copper: float = 0.0
    manganese: float = 0.0
    selenium: float = 0.0
    
    # Additional metrics
    glycemic_index: Optional[float] = None
    antioxidant_score: Optional[float] = None
    inflammatory_score: Optional[float] = None

@dataclass
class DietaryRecommendation:
    """Dietary recommendations and insights"""
    recommendation_type: str  # 'increase', 'decrease', 'maintain', 'caution'
    nutrient: str
    current_amount: float
    target_amount: float
    priority: int  # 1-5, higher is more important
    reason: str
    specific_foods: List[str]
    health_impact: str

class NutritionService:
    def __init__(self):
        # USDA nutrition database (simplified - in production would use real database)
        self.nutrition_database = self._load_nutrition_database()
        
        # Daily value references (based on 2000 calorie diet)
        self.daily_values = {
            'protein': 50,  # grams
            'carbohydrates': 300,  # grams  
            'fat': 65,  # grams
            'fiber': 25,  # grams
            'sodium': 2300,  # mg
            'sugar': 50,  # grams
            'saturated_fat': 20,  # grams
            'cholesterol': 300,  # mg
            'vitamin_c': 90,  # mg
            'calcium': 1000,  # mg
            'iron': 18,  # mg
            'potassium': 3500,  # mg
        }
        
        # Nutritional priorities by goal
        self.goal_nutrient_priorities = {
            'weight_loss': {
                'protein': 1.2,  # Higher protein for satiety
                'fiber': 1.3,   # Higher fiber for fullness
                'calories': 0.8, # Lower calories
                'sugar': 0.7,   # Lower sugar
                'fat': 0.9      # Moderate fat reduction
            },
            'muscle_gain': {
                'protein': 1.5,  # Much higher protein
                'calories': 1.2, # Higher calories
                'carbohydrates': 1.1, # Higher carbs for energy
                'fat': 1.0      # Normal fat
            },
            'heart_health': {
                'saturated_fat': 0.6, # Much lower saturated fat
                'sodium': 0.7,        # Lower sodium
                'fiber': 1.4,         # Higher fiber
                'omega_3': 1.5        # Higher omega-3
            },
            'diabetes_friendly': {
                'sugar': 0.5,    # Much lower sugar
                'fiber': 1.4,    # Higher fiber
                'protein': 1.1,  # Slightly higher protein
                'complex_carbs': 1.2 # Prefer complex carbs
            }
        }
    
    def _load_nutrition_database(self) -> Dict[str, NutritionalProfile]:
        """Load nutrition database (simplified version)"""
        # In production, this would load from USDA database or similar
        return {
            # Sample entries - would have thousands in production
            'chicken_breast': NutritionalProfile(
                calories=165, protein=31, fat=3.6, carbohydrates=0,
                sodium=74, iron=0.7, vitamin_b6=0.5
            ),
            'brown_rice': NutritionalProfile(
                calories=111, protein=2.6, fat=0.9, carbohydrates=23,
                fiber=1.8, magnesium=43, manganese=1.1
            ),
            'broccoli': NutritionalProfile(
                calories=25, protein=3, fat=0.3, carbohydrates=5,
                fiber=2.3, vitamin_c=81, vitamin_k=92, folate=57
            ),
            'olive_oil': NutritionalProfile(
                calories=884, protein=0, fat=100, carbohydrates=0,
                vitamin_e=14, vitamin_k=60
            ),
            # ... many more ingredients would be here
        }
    
    async def analyze_recipe_nutrition(
        self, 
        recipe: Recipe, 
        servings: int = None
    ) -> NutritionalProfile:
        """
        Analyze nutritional content of a recipe
        
        Args:
            recipe: Recipe to analyze
            servings: Number of servings (default uses recipe servings)
            
        Returns:
            Complete nutritional profile
        """
        try:
            servings = servings or recipe.servings or 1
            total_nutrition = NutritionalProfile()
            
            # Analyze each ingredient
            for ingredient in recipe.ingredients:
                ingredient_nutrition = await self._get_ingredient_nutrition(
                    ingredient.name, 
                    ingredient.amount,
                    ingredient.unit
                )
                
                # Add to total
                total_nutrition = self._add_nutritional_profiles(
                    total_nutrition, 
                    ingredient_nutrition
                )
            
            # Calculate per serving
            per_serving_nutrition = self._divide_nutritional_profile(
                total_nutrition, 
                servings
            )
            
            # Add calculated metrics
            per_serving_nutrition.glycemic_index = await self._estimate_glycemic_index(recipe)
            per_serving_nutrition.antioxidant_score = await self._calculate_antioxidant_score(recipe)
            per_serving_nutrition.inflammatory_score = await self._calculate_inflammatory_score(recipe)
            
            return per_serving_nutrition
            
        except Exception as e:
            logger.error(f"Failed to analyze recipe nutrition for {recipe.id}: {e}")
            return NutritionalProfile()  # Return empty profile on error
    
    async def _get_ingredient_nutrition(
        self, 
        ingredient_name: str, 
        amount: str, 
        unit: str
    ) -> NutritionalProfile:
        """Get nutritional information for a single ingredient"""
        
        # Normalize ingredient name
        normalized_name = self._normalize_ingredient_name(ingredient_name)
        
        # Look up in database
        base_nutrition = self.nutrition_database.get(
            normalized_name, 
            NutritionalProfile()  # Default empty profile
        )
        
        # Calculate amount multiplier
        amount_multiplier = self._calculate_amount_multiplier(amount, unit)
        
        # Scale nutrition based on amount
        scaled_nutrition = self._multiply_nutritional_profile(
            base_nutrition, 
            amount_multiplier
        )
        
        return scaled_nutrition
    
    def _normalize_ingredient_name(self, name: str) -> str:
        """Normalize ingredient name for database lookup"""
        # Remove common descriptors and standardize
        name = name.lower().strip()
        
        # Remove common prefixes/suffixes
        removals = [
            'fresh ', 'frozen ', 'dried ', 'organic ', 'raw ', 'cooked ',
            'large ', 'medium ', 'small ', 'whole ', 'chopped ', 'sliced ',
            'diced ', 'minced ', 'ground ', ' pieces', ' chunks'
        ]
        
        for removal in removals:
            name = name.replace(removal, '')
        
        # Handle plurals
        if name.endswith('es'):
            name = name[:-2]
        elif name.endswith('s') and not name.endswith('ss'):
            name = name[:-1]
        
        return name.strip()
    
    def _calculate_amount_multiplier(self, amount: str, unit: str) -> float:
        """Calculate multiplier for scaling nutrition based on amount"""
        try:
            # Parse amount (handle fractions, ranges, etc.)
            if not amount or amount.lower() in ['to taste', 'as needed']:
                return 0.1  # Minimal amount for seasonings
            
            # Handle fractions
            if '/' in amount:
                parts = amount.split('/')
                if len(parts) == 2:
                    amount_val = float(parts[0]) / float(parts[1])
                else:
                    amount_val = 1.0
            else:
                # Extract numeric part
                import re
                numbers = re.findall(r'\d+\.?\d*', amount)
                amount_val = float(numbers[0]) if numbers else 1.0
            
            # Unit conversions (simplified)
            unit_multipliers = {
                'cup': 1.0,
                'cups': 1.0,
                'tablespoon': 0.0625,  # 1/16 cup
                'tablespoons': 0.0625,
                'tbsp': 0.0625,
                'teaspoon': 0.0208,    # 1/48 cup
                'teaspoons': 0.0208,
                'tsp': 0.0208,
                'pound': 16.0,         # 16 oz
                'pounds': 16.0,
                'lb': 16.0,
                'ounce': 1.0,
                'ounces': 1.0,
                'oz': 1.0,
                'gram': 0.035,         # ~1/28 oz
                'grams': 0.035,
                'g': 0.035,
                'kilogram': 35.0,      # ~35 oz
                'kg': 35.0,
                'liter': 4.2,          # ~4.2 cups
                'liters': 4.2,
                'ml': 0.004,           # ~1/250 cup
                'milliliter': 0.004,
                'pint': 2.0,           # 2 cups
                'quart': 4.0,          # 4 cups
                'gallon': 16.0,        # 16 cups
                'piece': 0.5,          # Estimate for average piece
                'pieces': 0.5,
                'clove': 0.01,         # Small amount for garlic clove
                'cloves': 0.01
            }
            
            unit_multiplier = unit_multipliers.get(unit.lower(), 1.0)
            
            return amount_val * unit_multiplier
            
        except Exception:
            return 1.0  # Default multiplier
    
    def _add_nutritional_profiles(
        self, 
        profile1: NutritionalProfile, 
        profile2: NutritionalProfile
    ) -> NutritionalProfile:
        """Add two nutritional profiles together"""
        return NutritionalProfile(
            calories=profile1.calories + profile2.calories,
            protein=profile1.protein + profile2.protein,
            carbohydrates=profile1.carbohydrates + profile2.carbohydrates,
            fat=profile1.fat + profile2.fat,
            fiber=profile1.fiber + profile2.fiber,
            sugar=profile1.sugar + profile2.sugar,
            sodium=profile1.sodium + profile2.sodium,
            saturated_fat=profile1.saturated_fat + profile2.saturated_fat,
            cholesterol=profile1.cholesterol + profile2.cholesterol,
            vitamin_a=profile1.vitamin_a + profile2.vitamin_a,
            vitamin_c=profile1.vitamin_c + profile2.vitamin_c,
            calcium=profile1.calcium + profile2.calcium,
            iron=profile1.iron + profile2.iron,
            potassium=profile1.potassium + profile2.potassium,
            # Add other nutrients...
        )
    
    def _multiply_nutritional_profile(
        self, 
        profile: NutritionalProfile, 
        multiplier: float
    ) -> NutritionalProfile:
        """Multiply nutritional profile by a factor"""
        return NutritionalProfile(
            calories=profile.calories * multiplier,
            protein=profile.protein * multiplier,
            carbohydrates=profile.carbohydrates * multiplier,
            fat=profile.fat * multiplier,
            fiber=profile.fiber * multiplier,
            sugar=profile.sugar * multiplier,
            sodium=profile.sodium * multiplier,
            saturated_fat=profile.saturated_fat * multiplier,
            cholesterol=profile.cholesterol * multiplier,
            vitamin_a=profile.vitamin_a * multiplier,
            vitamin_c=profile.vitamin_c * multiplier,
            calcium=profile.calcium * multiplier,
            iron=profile.iron * multiplier,
            potassium=profile.potassium * multiplier,
            # Multiply other nutrients...
        )
    
    def _divide_nutritional_profile(
        self, 
        profile: NutritionalProfile, 
        divisor: float
    ) -> NutritionalProfile:
        """Divide nutritional profile by a factor"""
        if divisor == 0:
            return NutritionalProfile()
        
        return self._multiply_nutritional_profile(profile, 1.0 / divisor)
    
    async def _estimate_glycemic_index(self, recipe: Recipe) -> float:
        """Estimate glycemic index of recipe (simplified)"""
        # This would use more sophisticated calculation in production
        carb_sources = ['rice', 'bread', 'pasta', 'potato', 'sugar', 'fruit']
        
        high_gi_ingredients = 0
        total_carb_ingredients = 0
        
        for ingredient in recipe.ingredients:
            name = self._normalize_ingredient_name(ingredient.name)
            for carb_source in carb_sources:
                if carb_source in name:
                    total_carb_ingredients += 1
                    if carb_source in ['sugar', 'white rice', 'white bread', 'potato']:
                        high_gi_ingredients += 1
                    break
        
        if total_carb_ingredients == 0:
            return 35.0  # Low GI for no-carb recipes
        
        gi_ratio = high_gi_ingredients / total_carb_ingredients
        return 35 + (gi_ratio * 35)  # Scale from 35 (low) to 70 (high)
    
    async def _calculate_antioxidant_score(self, recipe: Recipe) -> float:
        """Calculate antioxidant score based on ingredients"""
        # High antioxidant foods
        high_antioxidant_foods = {
            'blueberries': 10, 'spinach': 8, 'broccoli': 7, 'tomato': 6,
            'bell pepper': 6, 'carrot': 5, 'onion': 4, 'garlic': 5,
            'green tea': 9, 'dark chocolate': 8, 'nuts': 6, 'berries': 9
        }
        
        total_score = 0
        for ingredient in recipe.ingredients:
            name = self._normalize_ingredient_name(ingredient.name)
            for food, score in high_antioxidant_foods.items():
                if food in name:
                    total_score += score
                    break
        
        return min(100, total_score * 2)  # Scale to 0-100
    
    async def _calculate_inflammatory_score(self, recipe: Recipe) -> float:
        """Calculate inflammatory score (lower is better)"""
        # Pro-inflammatory foods (higher scores)
        inflammatory_foods = {
            'sugar': 8, 'refined flour': 6, 'processed meat': 7,
            'fried': 9, 'trans fat': 10, 'high fructose': 9
        }
        
        # Anti-inflammatory foods (negative scores)
        anti_inflammatory_foods = {
            'turmeric': -5, 'ginger': -4, 'salmon': -6, 'olive oil': -3,
            'leafy greens': -4, 'berries': -3, 'nuts': -2, 'avocado': -3
        }
        
        total_score = 0
        
        for ingredient in recipe.ingredients:
            name = self._normalize_ingredient_name(ingredient.name)
            
            # Check inflammatory foods
            for food, score in inflammatory_foods.items():
                if food in name:
                    total_score += score
                    break
            
            # Check anti-inflammatory foods
            for food, score in anti_inflammatory_foods.items():
                if food in name:
                    total_score += score  # Score is negative
                    break
        
        return max(0, total_score)  # 0 is best (no inflammation)
    
    async def generate_nutrition_recommendations(
        self, 
        user: User, 
        current_nutrition: NutritionalProfile,
        goals: List[str] = None
    ) -> List[DietaryRecommendation]:
        """Generate personalized nutrition recommendations"""
        
        recommendations = []
        goals = goals or user.cooking_goals or ['balanced']
        
        # Get user's nutritional goals
        target_nutrition = await self._calculate_target_nutrition(user, goals)
        
        # Analyze gaps and excesses
        recommendations.extend(
            await self._analyze_macronutrient_balance(
                current_nutrition, target_nutrition, goals
            )
        )
        
        recommendations.extend(
            await self._analyze_micronutrient_needs(
                current_nutrition, target_nutrition, user
            )
        )
        
        recommendations.extend(
            await self._analyze_health_markers(
                current_nutrition, goals
            )
        )
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        return recommendations[:10]  # Return top 10 recommendations
    
    async def _calculate_target_nutrition(
        self, 
        user: User, 
        goals: List[str]
    ) -> NutritionalProfile:
        """Calculate target nutrition based on user profile and goals"""
        
        # Base requirements (simplified calculation)
        base_calories = 2000  # Would calculate based on age, gender, weight, activity
        
        # Adjust for goals
        for goal in goals:
            if goal in self.goal_nutrient_priorities:
                multipliers = self.goal_nutrient_priorities[goal]
                base_calories *= multipliers.get('calories', 1.0)
        
        # Calculate macros
        protein_calories = base_calories * 0.2  # 20% protein
        carb_calories = base_calories * 0.5     # 50% carbs
        fat_calories = base_calories * 0.3      # 30% fat
        
        return NutritionalProfile(
            calories=base_calories,
            protein=protein_calories / 4,  # 4 calories per gram
            carbohydrates=carb_calories / 4,
            fat=fat_calories / 9,  # 9 calories per gram
            fiber=25,  # Daily recommendation
            sodium=2300,  # Daily limit in mg
            # ... other targets
        )
    
    async def _analyze_macronutrient_balance(
        self,
        current: NutritionalProfile,
        target: NutritionalProfile,
        goals: List[str]
    ) -> List[DietaryRecommendation]:
        """Analyze macronutrient balance and generate recommendations"""
        
        recommendations = []
        
        # Protein analysis
        protein_ratio = current.protein / target.protein if target.protein > 0 else 0
        if protein_ratio < 0.8:
            recommendations.append(DietaryRecommendation(
                recommendation_type='increase',
                nutrient='protein',
                current_amount=current.protein,
                target_amount=target.protein,
                priority=4,
                reason='Protein intake is below target for your goals',
                specific_foods=['lean meats', 'fish', 'eggs', 'legumes', 'Greek yogurt'],
                health_impact='Supports muscle maintenance and satiety'
            ))
        
        # Fiber analysis
        if current.fiber < 20:
            recommendations.append(DietaryRecommendation(
                recommendation_type='increase',
                nutrient='fiber',
                current_amount=current.fiber,
                target_amount=25,
                priority=3,
                reason='Fiber intake is below recommended levels',
                specific_foods=['vegetables', 'fruits', 'whole grains', 'legumes'],
                health_impact='Improves digestion and heart health'
            ))
        
        # Sodium analysis
        if current.sodium > 2300:
            recommendations.append(DietaryRecommendation(
                recommendation_type='decrease',
                nutrient='sodium',
                current_amount=current.sodium,
                target_amount=2300,
                priority=5,
                reason='Sodium intake exceeds recommended limits',
                specific_foods=['processed foods', 'restaurant meals', 'added salt'],
                health_impact='Reduces risk of high blood pressure'
            ))
        
        return recommendations
    
    async def _analyze_micronutrient_needs(
        self,
        current: NutritionalProfile,
        target: NutritionalProfile,
        user: User
    ) -> List[DietaryRecommendation]:
        """Analyze micronutrient needs"""
        
        recommendations = []
        
        # Vitamin C
        if current.vitamin_c < 70:
            recommendations.append(DietaryRecommendation(
                recommendation_type='increase',
                nutrient='vitamin_c',
                current_amount=current.vitamin_c,
                target_amount=90,
                priority=2,
                reason='Vitamin C intake may be insufficient',
                specific_foods=['citrus fruits', 'bell peppers', 'broccoli', 'strawberries'],
                health_impact='Supports immune system and collagen production'
            ))
        
        # Iron (especially for women)
        if current.iron < 15:
            recommendations.append(DietaryRecommendation(
                recommendation_type='increase',
                nutrient='iron',
                current_amount=current.iron,
                target_amount=18,
                priority=3,
                reason='Iron intake may be insufficient',
                specific_foods=['lean red meat', 'spinach', 'lentils', 'tofu'],
                health_impact='Prevents anemia and supports energy levels'
            ))
        
        return recommendations
    
    async def _analyze_health_markers(
        self,
        current: NutritionalProfile,
        goals: List[str]
    ) -> List[DietaryRecommendation]:
        """Analyze health markers and inflammatory potential"""
        
        recommendations = []
        
        # High inflammatory score
        if hasattr(current, 'inflammatory_score') and current.inflammatory_score > 20:
            recommendations.append(DietaryRecommendation(
                recommendation_type='decrease',
                nutrient='inflammatory_foods',
                current_amount=current.inflammatory_score,
                target_amount=10,
                priority=4,
                reason='Current diet may promote inflammation',
                specific_foods=['processed foods', 'sugar', 'refined carbs'],
                health_impact='Reduces inflammation and chronic disease risk'
            ))
        
        # Low antioxidant score
        if hasattr(current, 'antioxidant_score') and current.antioxidant_score < 30:
            recommendations.append(DietaryRecommendation(
                recommendation_type='increase',
                nutrient='antioxidants',
                current_amount=current.antioxidant_score,
                target_amount=60,
                priority=3,
                reason='Antioxidant intake could be improved',
                specific_foods=['berries', 'leafy greens', 'colorful vegetables'],
                health_impact='Protects cells from oxidative damage'
            ))
        
        return recommendations

    async def optimize_meal_nutrition(
        self,
        recipes: List[Recipe],
        target_nutrition: NutritionalProfile,
        dietary_restrictions: List[str] = None
    ) -> List[Recipe]:
        """Optimize recipe selection for nutritional targets"""
        
        # This would implement optimization algorithm
        # For now, return original recipes
        return recipes
    
    async def calculate_meal_plan_nutrition_score(
        self,
        meal_plan: MealPlan,
        user_goals: List[str]
    ) -> Dict[str, float]:
        """Calculate comprehensive nutrition score for meal plan"""
        
        scores = {
            'overall_score': 0.0,
            'macro_balance': 0.0,
            'micro_completeness': 0.0,
            'variety_score': 0.0,
            'goal_alignment': 0.0,
            'health_score': 0.0
        }
        
        # Implementation would analyze all meals in plan
        # and calculate scores based on nutritional completeness
        
        return scores

# Create singleton instance
nutrition_service = NutritionService()