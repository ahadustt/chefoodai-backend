"""
⚖️ AI-Enhanced Unit Conversion Service
Context-aware unit conversions considering ingredient properties
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import vertexai
from vertexai.generative_models import GenerativeModel

from core.config import get_settings
try:
    from ai.shopping_enhancement import OptimizationLevel
except ImportError:
    from enum import Enum
    class OptimizationLevel(str, Enum):
        BASIC = "basic"
        STANDARD = "standard" 
        PREMIUM = "premium"

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize Vertex AI
vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT, location=settings.VERTEX_AI_LOCATION)


@dataclass
class ConversionContext:
    """Context information for intelligent unit conversion"""
    ingredient_name: str
    ingredient_type: str  # liquid, solid, powder, etc.
    density_category: str  # light, medium, heavy
    cooking_context: str  # baking, cooking, seasoning
    measurement_preference: str  # metric, imperial


@dataclass
class ConversionResult:
    """Result of AI-enhanced unit conversion"""
    original_quantity: float
    original_unit: str
    converted_quantity: float
    converted_unit: str
    confidence_score: float
    conversion_factor: float
    context_notes: Optional[str]
    is_approximate: bool


class IngredientDensityCategory(Enum):
    """Density categories for accurate volume-to-weight conversions"""
    LIGHT = "light"      # flour, powdered sugar
    MEDIUM = "medium"    # water, milk, most liquids
    HEAVY = "heavy"      # honey, molasses, nut butters


class ConversionType(Enum):
    """Types of unit conversions"""
    VOLUME_TO_VOLUME = "volume_to_volume"      # cups to ml
    WEIGHT_TO_WEIGHT = "weight_to_weight"      # oz to grams
    VOLUME_TO_WEIGHT = "volume_to_weight"      # cups flour to grams
    WEIGHT_TO_VOLUME = "weight_to_volume"      # grams to cups
    COUNT_TO_WEIGHT = "count_to_weight"        # cloves to grams
    STANDARDIZATION = "standardization"        # normalize similar units


# ============================================================================
# AI PROMPTS FOR UNIT CONVERSION
# ============================================================================

UNIT_CONVERSION_PROMPT = """
You are a culinary expert specializing in cooking measurements. Convert these ingredients to standardized units considering their physical properties.

INGREDIENTS TO CONVERT:
{ingredients_with_context}

CONVERSION RULES:
1. Consider ingredient DENSITY and TYPE (flour vs water vs honey)
2. Use COOKING CONTEXT (baking needs precision, seasoning is flexible)
3. Prefer STANDARD UNITS: grams for solids, ml for liquids, pieces for count
4. Handle COOKING-SPECIFIC conversions (cloves garlic vs garlic powder)
5. Consider REGIONAL PREFERENCES (metric vs imperial)
6. Provide CONFIDENCE SCORE based on conversion accuracy

INGREDIENT DENSITY GUIDE:
- LIGHT (flour, powdered sugar): 1 cup ≈ 120-130g
- MEDIUM (water, milk): 1 cup = 240ml = 240g  
- HEAVY (honey, nut butter): 1 cup ≈ 280-340g
- SPICES: 1 tsp ≈ 2-5g depending on density

COOKING CONVERSIONS:
- 1 clove garlic ≈ 1/2 tsp minced ≈ 1/8 tsp powder
- 1 medium onion ≈ 1 cup diced ≈ 150g
- 1 lemon ≈ 3 tbsp juice ≈ 45ml

Return ONLY a JSON array with this EXACT format:
[
  {{
    "ingredient": "flour",
    "original_quantity": 2.0,
    "original_unit": "cups",
    "converted_quantity": 250.0,
    "converted_unit": "grams", 
    "confidence": 0.95,
    "conversion_factor": 125.0,
    "notes": "Flour density: 125g per cup",
    "is_approximate": false
  }}
]
"""

DENSITY_ANALYSIS_PROMPT = """
Analyze these ingredients to determine their density category for accurate unit conversions.

INGREDIENTS TO ANALYZE:
{ingredient_names}

DENSITY CATEGORIES:
- LIGHT: Flour, powdered sugar, cocoa powder, spices (dry powders)
- MEDIUM: Water, milk, most liquids, eggs, yogurt 
- HEAVY: Honey, molasses, nut butters, thick sauces, oils

COOKING CONTEXT:
- BAKING ingredients need precise conversions
- SEASONING ingredients are more flexible
- LIQUID ingredients follow standard density
- SOLID ingredients vary significantly

Return ONLY a JSON array:
[{{"ingredient": "flour", "density_category": "light", "ingredient_type": "powder", "confidence": 0.98}}]
"""


# ============================================================================
# AI UNIT CONVERSION SERVICE
# ============================================================================

class AIUnitConverter:
    """AI-enhanced unit conversion with cooking context awareness"""
    
    def __init__(self):
        self.model_cache = {}
        self.conversion_cache = {}
        
        # Standard conversion factors (fallback)
        self.standard_conversions = {
            # Volume conversions
            ("cup", "ml"): 240.0,
            ("tablespoon", "ml"): 15.0,
            ("teaspoon", "ml"): 5.0,
            ("ounce", "ml"): 29.5735,  # fluid ounce
            
            # Weight conversions
            ("pound", "gram"): 453.592,
            ("ounce", "gram"): 28.3495,  # weight ounce
            ("kilogram", "gram"): 1000.0,
            
            # Cooking-specific
            ("clove", "teaspoon"): 0.5,  # garlic clove to minced
            ("medium_onion", "cup"): 1.0,  # diced
        }
        
        # Ingredient density database (for volume-to-weight conversions)
        self.ingredient_densities = {
            # Light ingredients (per cup)
            "flour": {"grams": 125, "category": "light"},
            "powdered sugar": {"grams": 120, "category": "light"},
            "cocoa powder": {"grams": 85, "category": "light"},
            
            # Medium ingredients (per cup)
            "water": {"grams": 240, "category": "medium"},
            "milk": {"grams": 245, "category": "medium"},
            "yogurt": {"grams": 245, "category": "medium"},
            
            # Heavy ingredients (per cup)
            "honey": {"grams": 340, "category": "heavy"},
            "peanut butter": {"grams": 250, "category": "heavy"},
            "olive oil": {"grams": 220, "category": "heavy"},
        }
    
    async def convert_ingredients_batch(
        self,
        ingredients_with_units: List[Dict[str, Any]],
        target_system: str = "metric",
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    ) -> List[ConversionResult]:
        """
        Convert ingredient units using AI-enhanced context awareness
        
        Args:
            ingredients_with_units: List of ingredients with quantity/unit info
            target_system: Target measurement system (metric/imperial)
            optimization_level: AI optimization level
            
        Returns:
            List of ConversionResult objects with standardized units
        """
        
        logger.info(
            f"Starting AI unit conversion",
            ingredient_count=len(ingredients_with_units),
            target_system=target_system,
            optimization_level=optimization_level.value
        )
        
        try:
            # Step 1: Analyze ingredient densities using AI
            density_analysis = await self._analyze_ingredient_densities(
                [ing["name"] for ing in ingredients_with_units],
                optimization_level
            )
            
            # Step 2: Prepare conversion context
            conversion_requests = []
            for i, ingredient in enumerate(ingredients_with_units):
                density_info = density_analysis[i] if i < len(density_analysis) else {}
                
                context = ConversionContext(
                    ingredient_name=ingredient["name"],
                    ingredient_type=density_info.get("ingredient_type", "unknown"),
                    density_category=density_info.get("density_category", "medium"),
                    cooking_context=self._determine_cooking_context(ingredient["name"]),
                    measurement_preference=target_system
                )
                
                conversion_requests.append({
                    "ingredient": ingredient["name"],
                    "quantity": ingredient["quantity"],
                    "unit": ingredient["unit"],
                    "context": context
                })
            
            # Step 3: Perform AI-enhanced conversions
            conversion_results = await self._convert_with_ai_context(
                conversion_requests, optimization_level
            )
            
            logger.info(
                f"AI unit conversion completed",
                conversions_performed=len(conversion_results),
                avg_confidence=sum(r.confidence_score for r in conversion_results) / len(conversion_results)
            )
            
            return conversion_results
            
        except Exception as e:
            logger.error(f"AI unit conversion failed: {e}")
            # Fallback to rule-based conversion
            return await self._fallback_unit_conversion(ingredients_with_units, target_system)
    
    async def _analyze_ingredient_densities(
        self,
        ingredient_names: List[str],
        optimization_level: OptimizationLevel
    ) -> List[Dict[str, Any]]:
        """Use AI to analyze ingredient densities for accurate conversions"""
        
        model = await self._get_model_for_optimization(optimization_level)
        
        prompt = DENSITY_ANALYSIS_PROMPT.format(
            ingredient_names=json.dumps(ingredient_names, indent=2)
        )
        
        try:
            response = await model.generate_content_async(prompt)
            density_data = json.loads(response.text.strip())
            return density_data
            
        except Exception as e:
            logger.warning(f"AI density analysis failed: {e}")
            # Fallback to predefined densities
            return [self._get_fallback_density_info(name) for name in ingredient_names]
    
    async def _convert_with_ai_context(
        self,
        conversion_requests: List[Dict[str, Any]],
        optimization_level: OptimizationLevel
    ) -> List[ConversionResult]:
        """Perform AI-enhanced unit conversions with cooking context"""
        
        model = await self._get_model_for_optimization(optimization_level)
        
        # Prepare ingredients for AI conversion
        ingredients_for_ai = []
        for req in conversion_requests:
            ingredients_for_ai.append({
                "ingredient": req["ingredient"],
                "quantity": req["quantity"],
                "unit": req["unit"],
                "density_category": req["context"].density_category,
                "ingredient_type": req["context"].ingredient_type,
                "target_system": req["context"].measurement_preference
            })
        
        prompt = UNIT_CONVERSION_PROMPT.format(
            ingredients_with_context=json.dumps(ingredients_for_ai, indent=2)
        )
        
        try:
            response = await model.generate_content_async(prompt)
            conversion_data = json.loads(response.text.strip())
            
            # Convert to ConversionResult objects
            results = []
            for i, req in enumerate(conversion_requests):
                if i < len(conversion_data):
                    data = conversion_data[i]
                    result = ConversionResult(
                        original_quantity=req["quantity"],
                        original_unit=req["unit"],
                        converted_quantity=float(data.get("converted_quantity", req["quantity"])),
                        converted_unit=data.get("converted_unit", req["unit"]),
                        confidence_score=float(data.get("confidence", 0.8)),
                        conversion_factor=float(data.get("conversion_factor", 1.0)),
                        context_notes=data.get("notes"),
                        is_approximate=data.get("is_approximate", False)
                    )
                    results.append(result)
                else:
                    # Fallback conversion
                    results.append(self._fallback_convert_single(req))
            
            return results
            
        except Exception as e:
            logger.warning(f"AI unit conversion failed: {e}")
            return [self._fallback_convert_single(req) for req in conversion_requests]
    
    def _determine_cooking_context(self, ingredient_name: str) -> str:
        """Determine cooking context for ingredient"""
        
        baking_ingredients = ["flour", "sugar", "baking powder", "vanilla", "butter"]
        seasoning_ingredients = ["salt", "pepper", "herbs", "spices", "garlic"]
        liquid_ingredients = ["milk", "water", "broth", "oil", "vinegar"]
        
        name_lower = ingredient_name.lower()
        
        if any(baking in name_lower for baking in baking_ingredients):
            return "baking"
        elif any(seasoning in name_lower for seasoning in seasoning_ingredients):
            return "seasoning"
        elif any(liquid in name_lower for liquid in liquid_ingredients):
            return "liquid"
        else:
            return "general"
    
    def _get_fallback_density_info(self, ingredient_name: str) -> Dict[str, Any]:
        """Get fallback density information for ingredient"""
        
        name_lower = ingredient_name.lower()
        
        # Check predefined densities
        if name_lower in self.ingredient_densities:
            info = self.ingredient_densities[name_lower]
            return {
                "ingredient": ingredient_name,
                "density_category": info["category"],
                "ingredient_type": "known",
                "confidence": 0.9
            }
        
        # Classify by type
        if any(word in name_lower for word in ["flour", "powder", "sugar"]):
            return {
                "ingredient": ingredient_name,
                "density_category": "light",
                "ingredient_type": "powder",
                "confidence": 0.7
            }
        elif any(word in name_lower for word in ["oil", "honey", "butter"]):
            return {
                "ingredient": ingredient_name,
                "density_category": "heavy", 
                "ingredient_type": "liquid",
                "confidence": 0.7
            }
        else:
            return {
                "ingredient": ingredient_name,
                "density_category": "medium",
                "ingredient_type": "unknown",
                "confidence": 0.5
            }
    
    def _fallback_convert_single(self, conversion_request: Dict[str, Any]) -> ConversionResult:
        """Fallback unit conversion using standard factors"""
        
        original_qty = conversion_request["quantity"]
        original_unit = conversion_request["unit"].lower()
        ingredient_name = conversion_request["ingredient"]
        
        # Try standard conversions
        target_unit = self._get_target_unit(original_unit, ingredient_name)
        conversion_key = (original_unit, target_unit)
        
        if conversion_key in self.standard_conversions:
            factor = self.standard_conversions[conversion_key]
            converted_qty = original_qty * factor
            confidence = 0.8
        else:
            # No conversion available, keep original
            converted_qty = original_qty
            target_unit = original_unit
            factor = 1.0
            confidence = 0.6
        
        return ConversionResult(
            original_quantity=original_qty,
            original_unit=conversion_request["unit"],
            converted_quantity=converted_qty,
            converted_unit=target_unit,
            confidence_score=confidence,
            conversion_factor=factor,
            context_notes="Rule-based conversion",
            is_approximate=True
        )
    
    def _get_target_unit(self, original_unit: str, ingredient_name: str) -> str:
        """Determine target unit for standardization"""
        
        unit_lower = original_unit.lower()
        name_lower = ingredient_name.lower()
        
        # Volume units → ml
        if unit_lower in ["cup", "cups", "tablespoon", "tbsp", "teaspoon", "tsp"]:
            return "ml"
        
        # Weight units → gram
        elif unit_lower in ["pound", "lb", "ounce", "oz", "kilogram", "kg"]:
            return "gram"
        
        # Count units stay as count
        elif unit_lower in ["piece", "clove", "slice"]:
            return "piece"
        
        else:
            return original_unit
    
    async def _get_model_for_optimization(self, optimization_level: OptimizationLevel) -> GenerativeModel:
        """Get appropriate AI model based on optimization level"""
        
        model_mapping = {
            OptimizationLevel.BASIC: "gemini-1.5-flash",
            OptimizationLevel.STANDARD: "gemini-1.5-pro",
            OptimizationLevel.PREMIUM: "gemini-2.0-flash-thinking"
        }
        
        model_name = model_mapping.get(optimization_level, "gemini-1.5-pro")
        
        if model_name not in self.model_cache:
            self.model_cache[model_name] = GenerativeModel(model_name)
        
        return self.model_cache[model_name]
    
    async def _fallback_unit_conversion(
        self,
        ingredients_with_units: List[Dict[str, Any]],
        target_system: str
    ) -> List[ConversionResult]:
        """Fallback rule-based unit conversion"""
        
        results = []
        for ingredient in ingredients_with_units:
            conversion_request = {
                "ingredient": ingredient["name"],
                "quantity": ingredient["quantity"],
                "unit": ingredient["unit"]
            }
            result = self._fallback_convert_single(conversion_request)
            results.append(result)
        
        return results


# ============================================================================
# SPECIALIZED CONVERSION FUNCTIONS
# ============================================================================

class CookingUnitConverter:
    """Specialized converter for cooking-specific units"""
    
    @staticmethod
    def convert_garlic_units(quantity: float, from_unit: str, to_unit: str) -> Tuple[float, float]:
        """Convert between different garlic measurements"""
        
        # Garlic conversion factors
        garlic_conversions = {
            ("clove", "teaspoon"): 0.5,      # 1 clove = 1/2 tsp minced
            ("clove", "tablespoon"): 0.167,   # 1 clove = 1/6 tbsp minced
            ("teaspoon", "clove"): 2.0,       # 1 tsp = 2 cloves
            ("clove", "gram"): 3.0,           # 1 clove ≈ 3g
        }
        
        key = (from_unit.lower(), to_unit.lower())
        if key in garlic_conversions:
            factor = garlic_conversions[key]
            return quantity * factor, 0.9  # High confidence
        
        return quantity, 0.5  # Low confidence, no conversion
    
    @staticmethod
    def convert_onion_units(quantity: float, from_unit: str, to_unit: str) -> Tuple[float, float]:
        """Convert between onion measurements"""
        
        onion_conversions = {
            ("medium", "cup"): 1.0,       # 1 medium onion = 1 cup diced
            ("large", "cup"): 1.5,        # 1 large onion = 1.5 cups diced
            ("small", "cup"): 0.5,        # 1 small onion = 0.5 cup diced
            ("medium", "gram"): 150,      # 1 medium onion ≈ 150g
        }
        
        key = (from_unit.lower(), to_unit.lower())
        if key in onion_conversions:
            factor = onion_conversions[key]
            return quantity * factor, 0.85
        
        return quantity, 0.5


# ============================================================================
# SERVICE FACTORY
# ============================================================================

def get_ai_unit_converter() -> AIUnitConverter:
    """Factory function to get AI unit converter"""
    return AIUnitConverter()


async def convert_shopping_list_units(
    shopping_ingredients: List[Dict[str, Any]],
    user_preferences: Dict[str, Any],
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
) -> List[Dict[str, Any]]:
    """
    Main entry point for AI-enhanced unit conversion in shopping lists
    
    Args:
        shopping_ingredients: Ingredients with quantity/unit info
        user_preferences: User preferences including measurement system
        optimization_level: AI optimization level
        
    Returns:
        Ingredients with standardized units and conversion metadata
    """
    
    converter = get_ai_unit_converter()
    target_system = user_preferences.get("measurement_system", "metric")
    
    # Perform conversions
    conversion_results = await converter.convert_ingredients_batch(
        shopping_ingredients, target_system, optimization_level
    )
    
    # Apply conversions to shopping ingredients
    enhanced_ingredients = []
    for ingredient, conversion in zip(shopping_ingredients, conversion_results):
        enhanced = ingredient.copy()
        
        # Update with converted values
        enhanced["standard_quantity"] = conversion.converted_quantity
        enhanced["standard_unit"] = conversion.converted_unit
        enhanced["conversion_confidence"] = conversion.confidence_score
        enhanced["conversion_notes"] = conversion.context_notes
        enhanced["is_approximate"] = conversion.is_approximate
        
        # Keep original for reference
        enhanced["original_quantity"] = conversion.original_quantity
        enhanced["original_unit"] = conversion.original_unit
        
        enhanced_ingredients.append(enhanced)
    
    return enhanced_ingredients
