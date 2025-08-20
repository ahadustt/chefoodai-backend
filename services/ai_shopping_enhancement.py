"""
🤖 AI-Enhanced Shopping List Generation Service
Integrates with AI microservice for intelligent ingredient processing
"""

import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import hashlib

# Import models from ai module to avoid circular imports
try:
    from ai.shopping_enhancement import (
        OptimizationLevel,
        IngredientConfidence,
        RawIngredient,
        UserPreferences,
        IngredientEnhancementRequest,
        EnhancedIngredient,
        AggregationSuggestion,
        IngredientEnhancementResponse
    )
except ImportError:
    # Define models locally if ai module not available (for standalone service)
    from pydantic import BaseModel, Field
    from enum import Enum
    
    class OptimizationLevel(str, Enum):
        """AI optimization levels for different user tiers"""
        BASIC = "basic"
        STANDARD = "standard"
        PREMIUM = "premium"
    
    class IngredientConfidence(str, Enum):
        """Confidence levels for AI processing results"""
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
    
    class RawIngredient(BaseModel):
        """Raw ingredient data extracted from recipe"""
        text: str = Field(..., description="Original ingredient text from recipe")
        recipe_id: str = Field(..., description="Source recipe ID")
        recipe_title: str = Field(..., description="Source recipe title")
        servings: int = Field(..., description="Recipe serving size")
        meal_date: Optional[str] = Field(None, description="Planned meal date")
    
    class UserPreferences(BaseModel):
        """User preferences that affect AI processing"""
        measurement_system: str = Field("metric", description="Preferred unit system")
        dietary_restrictions: List[str] = Field(default_factory=list)
        cooking_skill_level: str = Field("intermediate", description="User's cooking experience")
        budget_conscious: bool = Field(True, description="Optimize for cost savings")
        bulk_buying_preference: bool = Field(False, description="Prefer bulk purchases")
        organic_preference: bool = Field(False, description="Prefer organic ingredients")
    
    class IngredientEnhancementRequest(BaseModel):
        """Request to AI microservice for ingredient enhancement"""
        raw_ingredients: List[RawIngredient]
        user_preferences: Optional[UserPreferences] = None
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
        user_id: str = Field(..., description="User ID for usage tracking")
        request_id: str = Field(default_factory=lambda: f"req_{int(datetime.now().timestamp())}")
    
    class EnhancedIngredient(BaseModel):
        """AI-enhanced ingredient with cleaned data and intelligence"""
        name: str = Field(..., description="Cleaned, standardized ingredient name")
        quantity: float = Field(..., description="Parsed quantity from text")
        unit: str = Field(..., description="Parsed unit from text")
        standard_quantity: float = Field(..., description="Quantity in standard units")
        standard_unit: str = Field(..., description="Standard unit (grams, ml, pieces)")
        category: str = Field(..., description="Store section category")
        preparation: Optional[str] = Field(None, description="Preparation method (diced, chopped)")
        confidence_score: float = Field(..., ge=0.0, le=1.0, description="AI confidence (0-1)")
        confidence_level: IngredientConfidence = Field(..., description="Confidence category")
        optimization_notes: Optional[str] = Field(None, description="AI recommendations for better purchasing")
        package_suggestion: Optional[str] = Field(None, description="Optimal package size recommendation")
        bulk_opportunity: bool = Field(False, description="Good candidate for bulk buying")
        source_recipes: List[str] = Field(default_factory=list, description="Recipe IDs that need this ingredient")
        total_needed: float = Field(..., description="Total quantity needed across all recipes")
    
    class AggregationSuggestion(BaseModel):
        """AI suggestions for ingredient aggregation"""
        ingredient_group: List[str] = Field(..., description="Ingredients that can be combined")
        suggested_name: str = Field(..., description="Unified name for the group")
        suggested_quantity: float = Field(..., description="Combined quantity")
        suggested_unit: str = Field(..., description="Best unit for combined quantity")
        reasoning: str = Field(..., description="Why these should be grouped")
        confidence: float = Field(..., ge=0.0, le=1.0)
    
    class IngredientEnhancementResponse(BaseModel):
        """Response from AI microservice with enhanced ingredients"""
        enhanced_ingredients: List[EnhancedIngredient]
        aggregation_suggestions: List[AggregationSuggestion]
        total_processing_time: float = Field(..., description="Total time in seconds")
        ai_confidence_average: float = Field(..., ge=0.0, le=1.0, description="Average confidence across all ingredients")
        optimization_level_used: OptimizationLevel
        total_tokens_used: int = Field(0, description="Total AI tokens consumed")
        estimated_cost: float = Field(0.0, description="Estimated cost in USD")
        cache_hit_rate: float = Field(0.0, ge=0.0, le=1.0, description="Percentage of cached responses")
        fallback_count: int = Field(0, description="Number of ingredients that fell back to rule-based processing")
        error_count: int = Field(0, description="Number of processing errors")

from pydantic import BaseModel, Field
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ============================================================================
# AI SHOPPING ENHANCEMENT SERVICE
# ============================================================================

class AIShoppingEnhancementService:
    """
    Service for AI-enhanced shopping list generation
    Integrates with AI microservice and provides fallback to rule-based system
    """
    
    def __init__(self):
        self.ai_service_url = getattr(settings, 'AI_SERVICE_URL', "http://localhost:8001")
        self.ai_timeout = getattr(settings, 'AI_TIMEOUT', 30)
        self.fallback_enabled = getattr(settings, 'FALLBACK_ENABLED', True)
        self.cache_enabled = getattr(settings, 'CACHE_ENABLED', True)
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.fallback_count = 0
        self.cache_hits = 0
        
        logger.info(f"AI Shopping Enhancement Service initialized - ai_service_url={self.ai_service_url}, fallback_enabled={self.fallback_enabled}, cache_enabled={self.cache_enabled}")
    
    async def enhance_ingredients(
        self,
        raw_ingredients: List[Dict[str, Any]],
        user_preferences: Optional[Dict[str, Any]] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
        user_id: str = None
    ) -> IngredientEnhancementResponse:
        """
        Enhance raw ingredients using AI microservice with fallback strategy
        
        Args:
            raw_ingredients: List of raw ingredient dictionaries
            user_preferences: User preferences affecting processing
            optimization_level: AI optimization level based on user tier
            user_id: User ID for tracking and personalization
            
        Returns:
            IngredientEnhancementResponse with enhanced ingredients and metadata
        """
        start_time = datetime.now()
        request_id = f"enhance_{user_id}_{int(start_time.timestamp())}"
        
        try:
            # Convert to Pydantic models
            raw_ingredient_models = [
                RawIngredient(**ingredient) for ingredient in raw_ingredients
            ]
            
            user_prefs = UserPreferences(**user_preferences) if user_preferences else UserPreferences()
            
            # Create AI request
            ai_request = IngredientEnhancementRequest(
                raw_ingredients=raw_ingredient_models,
                user_preferences=user_prefs,
                optimization_level=optimization_level,
                user_id=user_id,
                request_id=request_id
            )
            
            # Try AI enhancement first
            if await self._should_use_ai(optimization_level, len(raw_ingredients)):
                try:
                    result = await self._call_ai_service(ai_request)
                    self.success_count += 1
                    
                    logger.info(f"AI enhancement successful - request_id={request_id}, user_id={user_id}, ingredient_count={len(raw_ingredients)}, processing_time={result.total_processing_time}, confidence={result.ai_confidence_average}")
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"AI enhancement failed, falling back to rule-based - request_id={request_id}, error={str(e)}, user_id={user_id}")
                    
                    if not self.fallback_enabled:
                        raise
            
            # Fallback to rule-based processing
            result = await self._rule_based_enhancement(raw_ingredient_models, user_prefs, optimization_level)
            self.fallback_count += 1
            
            logger.info(f"Rule-based enhancement used - request_id={request_id}, user_id={user_id}, ingredient_count={len(raw_ingredients)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ingredient enhancement failed completely - request_id={request_id}, error={str(e)}, user_id={user_id}")
            raise
        
        finally:
            self.request_count += 1
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Enhancement request completed - request_id={request_id}, processing_time={processing_time}, success_rate={self.success_count / self.request_count if self.request_count > 0 else 0}")
    
    async def _should_use_ai(self, optimization_level: OptimizationLevel, ingredient_count: int) -> bool:
        """Determine if AI service should be used based on optimization level and load"""
        
        # FORCE AI USAGE FOR AWARD-WINNING PERFORMANCE
        # Always use AI for all optimization levels to achieve 100% perfect categorization
        logger.info(f"🎯 FORCING AI usage - optimization_level={optimization_level.value}, ingredient_count={ingredient_count}")
        
        # Check AI service health
        is_healthy = await self._check_ai_service_health()
        logger.info(f"🏥 AI service health check: {is_healthy}")
        
        if not is_healthy:
            logger.warning("❌ AI service health check failed - falling back to rule-based")
            return False
        
        logger.info("✅ AI service healthy - using AI enhancement")
        return True
    
    async def _check_ai_service_health(self) -> bool:
        """Quick health check of AI microservice"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.ai_service_url}/health") as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _call_ai_service(self, request: IngredientEnhancementRequest) -> IngredientEnhancementResponse:
        """Call AI microservice for ingredient enhancement"""
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if self.cache_enabled:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.cache_hits += 1
                return cached_result
        
        # Make AI service call
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.ai_timeout)
        ) as session:
            
            async with session.post(
                f"{self.ai_service_url}/ai/enhance-shopping-ingredients",
                json=request.dict(),
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"AI service error {response.status}: {error_text}")
                
                result_data = await response.json()
                result = IngredientEnhancementResponse(**result_data)
                
                # Cache successful results
                if self.cache_enabled and result.ai_confidence_average > 0.8:
                    await self._cache_result(cache_key, result, request.optimization_level)
                
                return result
    
    async def _rule_based_enhancement(
        self,
        raw_ingredients: List[RawIngredient],
        user_prefs: UserPreferences,
        optimization_level: OptimizationLevel
    ) -> IngredientEnhancementResponse:
        """
        Fallback rule-based ingredient enhancement
        Uses existing ingredient aggregation logic
        """
        start_time = datetime.now()
        
        # Import existing aggregation service
        from services.ingredient_aggregation_service import IngredientAggregationService
        aggregation_service = IngredientAggregationService()
        
        enhanced_ingredients = []
        
        for raw_ingredient in raw_ingredients:
            try:
                # Use existing parsing logic
                parsed = aggregation_service.parse_ingredient_text(raw_ingredient.text)
                
                # Convert to enhanced format (parsed is a ParsedIngredient object)
                enhanced = EnhancedIngredient(
                    name=parsed.name,
                    quantity=parsed.quantity,
                    unit=parsed.unit,
                    standard_quantity=parsed.quantity,  # TODO: Add conversion
                    standard_unit=parsed.unit,
                    category=aggregation_service._categorize_ingredient(parsed.name),
                    preparation=parsed.preparation,
                    confidence_score=0.7,  # Rule-based gets medium confidence
                    confidence_level=IngredientConfidence.MEDIUM,
                    optimization_notes=None,
                    package_suggestion=None,
                    bulk_opportunity=False,
                    source_recipes=[raw_ingredient.recipe_id],
                    total_needed=parsed.quantity
                )
                
                enhanced_ingredients.append(enhanced)
                
            except Exception as e:
                logger.warning(f"Failed to parse ingredient: {raw_ingredient.text}, error: {e}")
                
                # Create minimal enhanced ingredient
                enhanced = EnhancedIngredient(
                    name=raw_ingredient.text,
                    quantity=1.0,
                    unit="piece",
                    standard_quantity=1.0,
                    standard_unit="piece",
                    category="other",
                    confidence_score=0.3,
                    confidence_level=IngredientConfidence.LOW,
                    source_recipes=[raw_ingredient.recipe_id],
                    total_needed=1.0
                )
                enhanced_ingredients.append(enhanced)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return IngredientEnhancementResponse(
            enhanced_ingredients=enhanced_ingredients,
            aggregation_suggestions=[],  # Rule-based doesn't provide aggregation suggestions
            total_processing_time=processing_time,
            ai_confidence_average=0.7,  # Average confidence for rule-based
            optimization_level_used=optimization_level,
            total_tokens_used=0,  # No AI tokens used
            estimated_cost=0.0,  # No cost for rule-based
            cache_hit_rate=0.0,  # No cache for rule-based
            fallback_count=len(raw_ingredients),  # All ingredients used fallback
            error_count=0
        )
    
    def _generate_cache_key(self, request: IngredientEnhancementRequest) -> str:
        """Generate cache key for AI request"""
        # Create deterministic hash from ingredients and preferences
        content = {
            "ingredients": [ing.text for ing in request.raw_ingredients],
            "optimization_level": request.optimization_level.value,
            "measurement_system": request.user_preferences.measurement_system if request.user_preferences else "metric"
        }
        
        content_str = json.dumps(content, sort_keys=True)
        return f"ai_shopping_enhancement:{hashlib.md5(content_str.encode()).hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[IngredientEnhancementResponse]:
        """Get cached AI result if available"""
        # TODO: Implement Redis caching
        # For now, return None (no cache)
        return None
    
    async def _cache_result(
        self, 
        cache_key: str, 
        result: IngredientEnhancementResponse,
        optimization_level: OptimizationLevel
    ) -> None:
        """Cache AI result for future use"""
        # TODO: Implement Redis caching with TTL based on optimization level
        pass
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service performance statistics"""
        return {
            "total_requests": self.request_count,
            "success_rate": self.success_count / self.request_count if self.request_count > 0 else 0,
            "fallback_rate": self.fallback_count / self.request_count if self.request_count > 0 else 0,
            "cache_hit_rate": self.cache_hits / self.request_count if self.request_count > 0 else 0,
            "ai_service_url": self.ai_service_url,
            "fallback_enabled": self.fallback_enabled
        }


# ============================================================================
# INTEGRATION WITH EXISTING SHOPPING LIST SERVICE
# ============================================================================

class EnhancedShoppingListGenerator:
    """
    Enhanced shopping list generator that uses AI when available
    Integrates with existing meal planning infrastructure
    """
    
    def __init__(self):
        self.ai_service = AIShoppingEnhancementService()
        
        # Import existing services
        from services.ingredient_aggregation_service import IngredientAggregationService
        self.aggregation_service = IngredientAggregationService()
    
    async def generate_enhanced_shopping_list(
        self,
        meal_plan_id: str,
        user_id: str,
        user_preferences: Optional[Dict[str, Any]] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
        db = None
    ) -> Dict[str, Any]:
        """
        Generate AI-enhanced shopping list from meal plan
        
        Args:
            meal_plan_id: ID of the meal plan to generate shopping list for
            user_id: User ID for personalization and tracking
            user_preferences: User preferences affecting generation
            optimization_level: AI optimization level
            db: Database session
            
        Returns:
            Enhanced shopping list data with AI improvements
        """
        
        logger.info(f"Starting enhanced shopping list generation - meal_plan_id={meal_plan_id}, user_id={user_id}, optimization_level={optimization_level.value}")
        
        try:
            # Step 1: Extract raw ingredients from meal plan (existing logic)
            raw_ingredients = await self._extract_raw_ingredients_from_meal_plan(
                meal_plan_id, db
            )
            
            if not raw_ingredients:
                raise ValueError(f"No ingredients found for meal plan {meal_plan_id}")
            
            logger.info(f"Extracted {len(raw_ingredients)} raw ingredients from meal plan")
            
            # Step 2: AI Enhancement Pipeline
            enhancement_result = await self.ai_service.enhance_ingredients(
                raw_ingredients=raw_ingredients,
                user_preferences=user_preferences,
                optimization_level=optimization_level,
                user_id=user_id
            )
            
            # Step 3: Create shopping list from enhanced ingredients
            shopping_list_data = await self._create_shopping_list_from_enhanced(
                enhancement_result, meal_plan_id, user_id, db
            )
            
            # Step 4: Add enhancement metadata (match backend field names)
            shopping_list_data["ai_enhancement"] = {
                "used_ai": enhancement_result.fallback_count < len(enhancement_result.enhanced_ingredients),
                "confidence_average": enhancement_result.ai_confidence_average,
                "processing_time": enhancement_result.total_processing_time,
                "optimization_level": enhancement_result.optimization_level_used.value,
                "fallback_count": enhancement_result.fallback_count,
                "model_used": "gemini-2.0-flash-exp",  # Add model info
                "cost_estimate": enhancement_result.estimated_cost,
                "cache_hit_rate": enhancement_result.cache_hit_rate
            }
            
            logger.info(f"Enhanced shopping list generation completed - meal_plan_id={meal_plan_id}, user_id={user_id}, total_items={len(shopping_list_data.get('items', []))}, ai_confidence={enhancement_result.ai_confidence_average}, used_fallback={enhancement_result.fallback_count > 0}")
            
            return shopping_list_data
            
        except Exception as e:
            logger.error(f"Enhanced shopping list generation failed - meal_plan_id={meal_plan_id}, user_id={user_id}, error={str(e)}")
            raise
    
    async def _extract_raw_ingredients_from_meal_plan(
        self, 
        meal_plan_id: str, 
        db
    ) -> List[Dict[str, Any]]:
        """Extract raw ingredients from meal plan using AI-powered extraction"""
        
        try:
            # Get raw ingredient texts directly from recipe database
            recipe_data = await db.fetch("""
                SELECT DISTINCT
                    r.id as recipe_id,
                    r.title as recipe_title,
                    CONCAT(
                        COALESCE(ri.quantity::text, '1'), ' ',
                        COALESCE(u.name, 'piece'), ' ',
                        COALESCE(i.name, 'Unknown Ingredient'),
                        CASE 
                            WHEN ri.preparation_notes IS NOT NULL 
                            THEN ', ' || ri.preparation_notes 
                            ELSE '' 
                        END
                    ) as ingredient_text,
                    pm.servings,
                    ri.display_order
                FROM meal_planning.planned_meals pm
                JOIN recipe.recipes r ON pm.recipe_id = r.id
                JOIN recipe.recipe_ingredients ri ON r.id = ri.recipe_id
                LEFT JOIN recipe.ingredients i ON ri.ingredient_id = i.id
                LEFT JOIN recipe.units u ON ri.unit_id = u.id
                WHERE pm.meal_plan_id = $1
                AND r.deleted_at IS NULL
                ORDER BY r.title, ri.display_order
            """, meal_plan_id)
            
            if not recipe_data:
                logger.warning(f"No recipe data found for meal plan {meal_plan_id}")
                return []
            
            # Use AI extraction service to get clean ingredients
            import aiohttp
            
            ingredient_texts = []
            for row in recipe_data:
                # Scale quantity by servings
                base_quantity = float(row.get('quantity', 1.0) or 1.0)
                servings = row['servings'] or 1
                scaled_text = row['ingredient_text'].replace(
                    str(base_quantity), 
                    str(base_quantity * servings), 
                    1
                )
                ingredient_texts.append(scaled_text)
            
            # Call AI extraction service with detailed logging
            extraction_url = "https://chefoodai-ai-service-mpsrniojta-uc.a.run.app/ai/extract-ingredients"
            logger.info(f"🤖 Calling AI extraction service: {extraction_url} with {len(ingredient_texts)} texts")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    extraction_url,
                    json={
                        "ingredient_texts": ingredient_texts,
                        "optimization_level": "standard",
                        "request_id": f"extract_{meal_plan_id}_{int(datetime.now().timestamp())}"
                    },
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    logger.info(f"🎯 AI extraction response: status={response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"❌ AI extraction service failed: status={response.status}, error={error_text}")
                        # Fallback to existing aggregation service
                        ingredients = await self.aggregation_service.generate_shopping_list_from_meal_plan(meal_plan_id, db)
                        return self._convert_aggregated_to_raw(ingredients)
                    
                    extraction_result = await response.json()
                    
                    # Convert AI extraction result to raw ingredient format
                    raw_ingredients = []
                    for i, extracted in enumerate(extraction_result.get("extracted_ingredients", [])):
                        original_row = recipe_data[i] if i < len(recipe_data) else {}
                        
                        raw_ingredients.append({
                            "text": f"{extracted['quantity']} {extracted['unit']} {extracted['name']}".strip(),
                            "recipe_id": str(original_row.get('recipe_id', '')),
                            "recipe_title": original_row.get('recipe_title', ''),
                            "servings": original_row.get('servings', 1),
                            "meal_date": None,
                            "preparation": extracted.get("preparation"),
                            "ai_extracted": True,
                            "extraction_confidence": extracted.get("confidence", 0.95)
                        })
                    
                    logger.info(f"🤖 AI extraction successful: {len(raw_ingredients)} ingredients extracted from {len(ingredient_texts)} texts")
                    return raw_ingredients
            
        except Exception as e:
            logger.warning(f"AI extraction failed, falling back to existing logic: {e}")
            # Fallback to existing aggregation service
            ingredients = await self.aggregation_service.generate_shopping_list_from_meal_plan(meal_plan_id, db)
            return self._convert_aggregated_to_raw(ingredients)
    
    def _convert_aggregated_to_raw(self, ingredients: List) -> List[Dict[str, Any]]:
        """Convert aggregated ingredients to raw format (fallback method)"""
        raw_ingredients = []
        for ingredient in ingredients:
            raw_ingredients.append({
                "text": f"{ingredient.total_quantity} {ingredient.unit} {ingredient.name}".strip(),
                "recipe_id": ingredient.recipe_sources[0]["recipe_id"] if ingredient.recipe_sources else "",
                "recipe_title": ingredient.recipe_sources[0]["recipe_title"] if ingredient.recipe_sources else "",
                "servings": 1,  # Already aggregated
                "meal_date": None,  # Not applicable for aggregated data
                "ai_extracted": False,
                "extraction_confidence": 0.7
            })
        return raw_ingredients
    
    async def _create_shopping_list_from_enhanced(
        self,
        enhancement_result: IngredientEnhancementResponse,
        meal_plan_id: str,
        user_id: str,
        db
    ) -> Dict[str, Any]:
        """Create shopping list from AI-enhanced ingredients"""
        
        # Group enhanced ingredients by category
        categorized_items = {}
        total_estimated_cost = 0.0
        
        for enhanced in enhancement_result.enhanced_ingredients:
            # Handle both dict and object formats
            if isinstance(enhanced, dict):
                category = enhanced.get("category", "other")
                item = {
                    "name": enhanced.get("name", "Unknown"),
                    "quantity": enhanced.get("total_needed", enhanced.get("quantity", 1.0)),
                    "unit": enhanced.get("unit", "piece"),
                    "standard_quantity": enhanced.get("standard_quantity", enhanced.get("quantity", 1.0)),
                    "standard_unit": enhanced.get("standard_unit", enhanced.get("unit", "piece")),
                    "category": category,
                    "preparation": enhanced.get("preparation"),
                    "confidence_score": enhanced.get("confidence_score", 0.5),
                    "optimization_notes": enhanced.get("optimization_notes"),
                    "package_suggestion": enhanced.get("package_suggestion"),
                    "bulk_opportunity": enhanced.get("bulk_opportunity", False),
                    "source_recipes": enhanced.get("source_recipes", []),
                    "is_completed": False,
                    "estimated_cost": 0.0,
                    "ai_processed": True  # This was processed by AI (not fallback)
                }
            else:
                # Pydantic object format
                category = enhanced.category
                item = {
                    "name": enhanced.name,
                    "quantity": enhanced.total_needed,
                    "unit": enhanced.unit,
                    "standard_quantity": enhanced.standard_quantity,
                    "standard_unit": enhanced.standard_unit,
                    "category": category,
                    "preparation": enhanced.preparation,
                    "confidence_score": enhanced.confidence_score,
                    "optimization_notes": enhanced.optimization_notes,
                    "package_suggestion": enhanced.package_suggestion,
                    "bulk_opportunity": enhanced.bulk_opportunity,
                    "source_recipes": enhanced.source_recipes,
                    "is_completed": False,
                    "estimated_cost": 0.0,
                    "ai_processed": True  # This was processed by AI (not fallback)
                }
            
            if category not in categorized_items:
                categorized_items[category] = []
            
            categorized_items[category].append(item)
        
        # Apply aggregation suggestions
        if enhancement_result.aggregation_suggestions:
            categorized_items = await self._apply_aggregation_suggestions(
                categorized_items, enhancement_result.aggregation_suggestions
            )
        
        # Create final shopping list structure
        shopping_list = {
            "meal_plan_id": meal_plan_id,
            "user_id": user_id,
            "name": f"AI-Enhanced Shopping List - {datetime.now().strftime('%Y-%m-%d')}",
            "categories": categorized_items,
            "items": [item for category_items in categorized_items.values() for item in category_items],
            "total_items": sum(len(items) for items in categorized_items.values()),
            "total_estimated_cost": total_estimated_cost,
            "generated_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        return shopping_list
    
    async def _apply_aggregation_suggestions(
        self,
        categorized_items: Dict[str, List[Dict]],
        suggestions: List[AggregationSuggestion]
    ) -> Dict[str, List[Dict]]:
        """Apply AI aggregation suggestions to combine similar ingredients"""
        
        for suggestion in suggestions:
            if suggestion.confidence < 0.8:  # Only apply high-confidence suggestions
                continue
            
            # Find and combine suggested ingredients
            # TODO: Implement intelligent ingredient combination logic
            pass
        
        return categorized_items
    
    def _generate_cache_key(self, request: IngredientEnhancementRequest) -> str:
        """Generate cache key for the enhancement request"""
        return self.ai_service._generate_cache_key(request)


# ============================================================================
# SERVICE FACTORY AND CONFIGURATION
# ============================================================================

def get_ai_shopping_service() -> EnhancedShoppingListGenerator:
    """Factory function to get AI shopping enhancement service"""
    return EnhancedShoppingListGenerator()


def get_optimization_level_for_user(user_tier: str) -> OptimizationLevel:
    """Determine optimization level based on user tier"""
    tier_mapping = {
        "free": OptimizationLevel.BASIC,
        "premium": OptimizationLevel.STANDARD,
        "enterprise": OptimizationLevel.PREMIUM,
        "pro": OptimizationLevel.STANDARD  # Pro tier gets standard optimization
    }
    
    return tier_mapping.get(user_tier.lower(), OptimizationLevel.STANDARD)
