"""
ChefoodAI AI Service Client
Calls the AI microservice for all AI-powered functionality
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import structlog
import httpx
import os

from core.config import settings
from core.redis import cache
from middleware.logging import log_business_event

logger = structlog.get_logger()

# AI Microservice configuration
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "https://chefoodai-ai-service-1074761757006.us-central1.run.app")


class AIRequestType(Enum):
    """Types of AI requests for different features"""
    RECIPE_GENERATION = "recipe_generation"
    MEAL_PLANNING = "meal_planning"
    NUTRITION_ANALYSIS = "nutrition_analysis"
    SHOPPING_LIST = "shopping_list"
    INGREDIENT_PARSING = "ingredient_parsing"
    UNIT_CONVERSION = "unit_conversion"
    QUANTITY_OPTIMIZATION = "quantity_optimization"
    INGREDIENT_SUBSTITUTION = "ingredient_substitution"
    RECIPE_ENHANCEMENT = "recipe_enhancement"
    MEAL_SUGGESTIONS = "meal_suggestions"


class AIServiceClient:
    """Client for communicating with the AI microservice"""
    
    def __init__(self):
        self.base_url = AI_SERVICE_URL
        self.client = httpx.AsyncClient(timeout=60.0)
        self.cache_ttl = 3600  # 1 hour cache
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def generate_recipe(
        self,
        ingredients: List[str],
        cuisine: Optional[str] = None,
        difficulty: Optional[str] = None,
        dietary_restrictions: Optional[List[str]] = None,
        cooking_time: Optional[int] = None,
        servings: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate a recipe using the AI service"""
        
        # Create cache key
        cache_key = f"recipe:{':'.join(sorted(ingredients))}:{cuisine}:{difficulty}"
        
        # Check cache
        if cache:
            cached = await cache.get(cache_key)
            if cached:
                logger.info("Recipe cache hit", cache_key=cache_key)
                return json.loads(cached)
        
        # Call AI service
        try:
            response = await self.client.post(
                f"{self.base_url}/api/ai/recipe/generate",
                json={
                    "ingredients": ingredients,
                    "cuisine": cuisine,
                    "difficulty": difficulty,
                    "dietary_restrictions": dietary_restrictions,
                    "cooking_time": cooking_time,
                    "servings": servings
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache result
            if cache and result.get("success"):
                await cache.set(cache_key, json.dumps(result), ttl=self.cache_ttl)
            
            return result
            
        except httpx.RequestError as e:
            logger.error("AI service request failed", error=str(e))
            raise Exception(f"Failed to generate recipe: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error("AI service returned error", status=e.response.status_code)
            raise Exception(f"AI service error: {e.response.text}")
    
    async def generate_meal_plan(
        self,
        days: int = 7,
        people: int = 1,
        dietary_restrictions: Optional[List[str]] = None,
        budget_level: Optional[str] = None,
        cuisine_preferences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate a meal plan using the AI service"""
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/ai/meal-plan/generate",
                json={
                    "days": days,
                    "people": people,
                    "dietary_restrictions": dietary_restrictions,
                    "budget_level": budget_level,
                    "cuisine_preferences": cuisine_preferences
                }
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.RequestError as e:
            logger.error("AI service request failed", error=str(e))
            raise Exception(f"Failed to generate meal plan: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error("AI service returned error", status=e.response.status_code)
            raise Exception(f"AI service error: {e.response.text}")
    
    async def analyze_nutrition(
        self,
        recipe_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze nutrition using the AI service"""
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/ai/nutrition/analyze",
                json=recipe_data
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.RequestError as e:
            logger.error("AI service request failed", error=str(e))
            raise Exception(f"Failed to analyze nutrition: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error("AI service returned error", status=e.response.status_code)
            raise Exception(f"AI service error: {e.response.text}")
    
    async def parse_ingredients(
        self,
        ingredient_text: str
    ) -> List[Dict[str, Any]]:
        """Parse ingredient text using the AI service"""
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/ai/ingredients/parse",
                json={"text": ingredient_text}
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.RequestError as e:
            logger.error("AI service request failed", error=str(e))
            raise Exception(f"Failed to parse ingredients: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error("AI service returned error", status=e.response.status_code)
            raise Exception(f"AI service error: {e.response.text}")
    
    async def optimize_shopping_list(
        self,
        items: List[Dict[str, Any]],
        budget: Optional[float] = None,
        store_preferences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Optimize shopping list using the AI service"""
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/ai/shopping/optimize",
                json={
                    "items": items,
                    "budget": budget,
                    "store_preferences": store_preferences
                }
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.RequestError as e:
            logger.error("AI service request failed", error=str(e))
            raise Exception(f"Failed to optimize shopping list: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error("AI service returned error", status=e.response.status_code)
            raise Exception(f"AI service error: {e.response.text}")
    
    async def enhance_recipe(
        self,
        recipe: Dict[str, Any],
        enhancement_type: str = "full"
    ) -> Dict[str, Any]:
        """Enhance a recipe with additional details using the AI service"""
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/ai/recipe/enhance",
                json={
                    "recipe": recipe,
                    "enhancement_type": enhancement_type
                }
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.RequestError as e:
            logger.error("AI service request failed", error=str(e))
            raise Exception(f"Failed to enhance recipe: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error("AI service returned error", status=e.response.status_code)
            raise Exception(f"AI service error: {e.response.text}")
    
    async def generate_image(
        self,
        prompt: str,
        style: str = "photorealistic"
    ) -> Dict[str, Any]:
        """Generate an image using the AI service"""
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/ai/image/generate",
                json={
                    "prompt": prompt,
                    "style": style
                }
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.RequestError as e:
            logger.error("AI service request failed", error=str(e))
            raise Exception(f"Failed to generate image: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error("AI service returned error", status=e.response.status_code)
            raise Exception(f"AI service error: {e.response.text}")
    
    async def analyze_food_image(
        self,
        image_data: bytes,
        analysis_focus: Optional[str] = None,
        dietary_restrictions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze a food image using the AI service"""
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/ai/image/analyze",
                json={
                    "image_data": image_data.hex() if image_data else None,
                    "analysis_focus": analysis_focus,
                    "dietary_restrictions": dietary_restrictions
                }
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.RequestError as e:
            logger.error("AI service request failed", error=str(e))
            raise Exception(f"Failed to analyze image: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error("AI service returned error", status=e.response.status_code)
            raise Exception(f"AI service error: {e.response.text}")
    
    async def stream_cooking_guidance(
        self,
        question: str,
        recipe_data: Optional[Dict[str, Any]] = None,
        current_step: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Stream cooking guidance from the AI service"""
        
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/ai/cooking/guidance",
                json={
                    "question": question,
                    "recipe_data": recipe_data,
                    "current_step": current_step
                }
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    yield chunk
                    
        except httpx.RequestError as e:
            logger.error("AI service request failed", error=str(e))
            yield f"Error: Failed to get guidance - {str(e)}"
        except httpx.HTTPStatusError as e:
            logger.error("AI service returned error", status=e.response.status_code)
            yield f"Error: AI service error - {e.response.text}"
    
    async def generate_meal_plan_name(
        self,
        duration_days: int,
        theme: Optional[str] = None,
        dietary_restrictions: Optional[List[str]] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a creative meal plan name using the AI service"""
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/ai/meal-plan/name",
                json={
                    "duration_days": duration_days,
                    "theme": theme,
                    "dietary_restrictions": dietary_restrictions,
                    "preferences": preferences
                }
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.RequestError as e:
            logger.error("AI service request failed", error=str(e))
            raise Exception(f"Failed to generate meal plan name: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error("AI service returned error", status=e.response.status_code)
            raise Exception(f"AI service error: {e.response.text}")


# Global AI service client instance
ai_service = AIServiceClient()


# Backward compatibility functions
async def generate_recipe_with_ai(
    ingredients: List[str],
    cuisine: Optional[str] = None,
    difficulty: Optional[str] = None,
    dietary_restrictions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Generate a recipe using AI (backward compatibility)"""
    return await ai_service.generate_recipe(
        ingredients=ingredients,
        cuisine=cuisine,
        difficulty=difficulty,
        dietary_restrictions=dietary_restrictions
    )


async def generate_meal_plan_with_ai(
    days: int = 7,
    people: int = 1,
    dietary_restrictions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Generate a meal plan using AI (backward compatibility)"""
    return await ai_service.generate_meal_plan(
        days=days,
        people=people,
        dietary_restrictions=dietary_restrictions
    )


async def analyze_nutrition_with_ai(recipe_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze nutrition using AI (backward compatibility)"""
    return await ai_service.analyze_nutrition(recipe_data)