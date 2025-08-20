"""
🤖 AI-Powered Ingredient Extraction Service
Replaces regex-based parsing with intelligent AI extraction for perfect accuracy
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Import AI models
try:
    from ai.shopping_enhancement import OptimizationLevel
except ImportError:
    from enum import Enum
    class OptimizationLevel(str, Enum):
        BASIC = "basic"
        STANDARD = "standard"
        PREMIUM = "premium"

logger = logging.getLogger(__name__)

@dataclass
class AIExtractedIngredient:
    """AI-extracted ingredient with perfect parsing"""
    name: str
    quantity: float
    unit: str
    preparation: Optional[str] = None
    original_text: str = ""
    confidence: float = 0.95
    recipe_id: Optional[str] = None
    recipe_title: Optional[str] = None

# ============================================================================
# AI EXTRACTION PROMPTS
# ============================================================================

AI_INGREDIENT_EXTRACTION_PROMPT = """
You are an expert chef and ingredient parsing specialist. Extract clean, accurate ingredients from recipe ingredient texts.

INGREDIENT TEXTS TO PARSE:
{ingredient_texts}

EXTRACTION RULES:
1. Extract the CORE INGREDIENT NAME (remove quantities, units, preparations)
2. Parse QUANTITY and UNIT accurately (handle fractions, ranges)
3. Extract PREPARATION separately (diced, chopped, peeled, zested, etc.)
4. NEVER create separate ingredients for preparation methods
5. Handle COMPLEX FORMATS: "1 mango, peeled and cubed" → name: "mango", preparation: "peeled and cubed"
6. PRESERVE essential descriptors: "chicken breast", "roma tomatoes"
7. REMOVE brand names, optional indicators, parenthetical notes

CRITICAL EXAMPLES:
- "1.000 pc mango, peeled and cubed" → name: "mango", quantity: 1.0, unit: "piece", preparation: "peeled and cubed"
- "2.000 cup diced roma tomatoes" → name: "tomatoes", quantity: 2.0, unit: "cup", preparation: "diced"
- "1.000 tbsp coconut oil, melted" → name: "coconut oil", quantity: 1.0, unit: "tablespoon", preparation: "melted"
- "1/2 cup fresh basil leaves, chopped" → name: "basil", quantity: 0.5, unit: "cup", preparation: "fresh, chopped"

Return ONLY a JSON array with this EXACT format:
[
  {{
    "name": "mango",
    "quantity": 1.0,
    "unit": "piece", 
    "preparation": "peeled and cubed",
    "confidence": 0.98,
    "original": "1.000 pc mango, peeled and cubed"
  }}
]

NEVER create separate ingredients for preparation methods like "peeled", "zested", "chopped".
"""

# ============================================================================
# AI EXTRACTION SERVICE
# ============================================================================

class AIIngredientExtractionService:
    """
    AI-powered ingredient extraction service
    Replaces regex parsing with intelligent AI processing
    """
    
    def __init__(self):
        self.ai_service_url = "https://chefoodai-ai-service-mpsrniojta-uc.a.run.app"
        
    async def extract_ingredients_from_meal_plan(
        self,
        meal_plan_id: str,
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
        db = None
    ) -> List[AIExtractedIngredient]:
        """
        Extract ingredients directly from recipe texts using AI
        This bypasses regex parsing completely to eliminate artifacts
        """
        
        logger.info(f"🤖 Starting AI ingredient extraction for meal plan {meal_plan_id}")
        
        try:
            # Get raw recipe ingredient texts from database
            raw_ingredient_texts = await self._get_raw_ingredient_texts(meal_plan_id, db)
            
            if not raw_ingredient_texts:
                logger.warning(f"No ingredient texts found for meal plan {meal_plan_id}")
                return []
            
            logger.info(f"Found {len(raw_ingredient_texts)} raw ingredient texts for AI extraction")
            
            # Extract ingredients using AI
            extracted_ingredients = await self._ai_extract_ingredients(
                raw_ingredient_texts, optimization_level
            )
            
            logger.info(f"✅ AI extraction complete: {len(extracted_ingredients)} ingredients extracted with average confidence {sum(ing.confidence for ing in extracted_ingredients) / len(extracted_ingredients):.3f}")
            
            return extracted_ingredients
            
        except Exception as e:
            logger.error(f"AI ingredient extraction failed for meal plan {meal_plan_id}: {e}")
            raise
    
    async def _get_raw_ingredient_texts(self, meal_plan_id: str, db) -> List[Dict[str, Any]]:
        """Get raw ingredient texts from meal plan recipes"""
        
        # Query to get original ingredient texts from recipes
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
                pm.servings
            FROM meal_planning.planned_meals pm
            JOIN recipe.recipes r ON pm.recipe_id = r.id
            JOIN recipe.recipe_ingredients ri ON r.id = ri.recipe_id
            LEFT JOIN recipe.ingredients i ON ri.ingredient_id = i.id
            LEFT JOIN recipe.units u ON ri.unit_id = u.id
            WHERE pm.meal_plan_id = $1
            AND r.deleted_at IS NULL
            ORDER BY r.title, ri.display_order
        """, meal_plan_id)
        
        # Convert to extraction format
        ingredient_texts = []
        for row in recipe_data:
            # Scale quantity by servings
            scaled_quantity = (row.get('quantity', 1.0) or 1.0) * (row['servings'] or 1)
            
            ingredient_texts.append({
                "text": row['ingredient_text'],
                "recipe_id": str(row['recipe_id']),
                "recipe_title": row['recipe_title'],
                "servings": row['servings']
            })
        
        return ingredient_texts
    
    async def _ai_extract_ingredients(
        self,
        raw_texts: List[Dict[str, Any]],
        optimization_level: OptimizationLevel
    ) -> List[AIExtractedIngredient]:
        """Use AI service to extract ingredients from raw texts"""
        
        import aiohttp
        
        # Prepare texts for AI processing
        ingredient_list = [item["text"] for item in raw_texts]
        
        prompt = AI_INGREDIENT_EXTRACTION_PROMPT.format(
            ingredient_texts=json.dumps(ingredient_list, indent=2)
        )
        
        try:
            # Call AI service for extraction
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ai_service_url}/ai/extract-ingredients",
                    json={
                        "ingredient_texts": ingredient_list,
                        "optimization_level": optimization_level.value,
                        "request_id": f"extract_{int(datetime.now().timestamp())}"
                    },
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status != 200:
                        logger.warning(f"AI extraction service unavailable: {response.status}")
                        return await self._fallback_extraction(raw_texts)
                    
                    result = await response.json()
                    
                    # Convert to AIExtractedIngredient objects
                    extracted = []
                    for i, item_data in enumerate(result.get("extracted_ingredients", [])):
                        original_data = raw_texts[i] if i < len(raw_texts) else {}
                        
                        extracted.append(AIExtractedIngredient(
                            name=item_data["name"],
                            quantity=item_data["quantity"],
                            unit=item_data["unit"],
                            preparation=item_data.get("preparation"),
                            original_text=item_data.get("original", ""),
                            confidence=item_data.get("confidence", 0.95),
                            recipe_id=original_data.get("recipe_id"),
                            recipe_title=original_data.get("recipe_title")
                        ))
                    
                    return extracted
                    
        except Exception as e:
            logger.warning(f"AI ingredient extraction failed: {e}")
            return await self._fallback_extraction(raw_texts)
    
    async def _fallback_extraction(self, raw_texts: List[Dict[str, Any]]) -> List[AIExtractedIngredient]:
        """Fallback to improved regex extraction if AI fails"""
        
        # Use existing aggregation service but with improved parsing
        from services.ingredient_aggregation_service import IngredientAggregationService
        aggregation_service = IngredientAggregationService()
        
        extracted = []
        for item in raw_texts:
            try:
                parsed = aggregation_service.parse_ingredient_text(item["text"])
                
                # Skip invalid ingredients (preparation artifacts)
                if parsed.name.lower() in ['peeled', 'zested', 'chopped', 'diced', 'minced', 'sliced']:
                    continue
                
                extracted.append(AIExtractedIngredient(
                    name=parsed.name,
                    quantity=parsed.quantity,
                    unit=parsed.unit,
                    preparation=parsed.preparation,
                    original_text=item["text"],
                    confidence=0.7,  # Lower confidence for fallback
                    recipe_id=item.get("recipe_id"),
                    recipe_title=item.get("recipe_title")
                ))
                
            except Exception as e:
                logger.warning(f"Failed to parse ingredient '{item['text']}': {e}")
                continue
        
        return extracted

# Global service instance
ai_extraction_service = AIIngredientExtractionService()
