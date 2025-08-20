"""
🧠 AI-Powered Ingredient Name Cleaning & Parsing Service
Handles complex ingredient formats with intelligent parsing
"""

import re
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
class ParsedIngredient:
    """Result of AI ingredient parsing"""
    original_text: str
    cleaned_name: str
    quantity: float
    unit: str
    preparation: Optional[str]
    brand_info: Optional[str]
    package_info: Optional[str]
    quality_descriptors: List[str]
    confidence_score: float
    parsing_notes: Optional[str]


class IngredientComplexity(Enum):
    """Classification of ingredient parsing complexity"""
    SIMPLE = "simple"        # "2 cups flour"
    MODERATE = "moderate"    # "1 large yellow onion, diced"
    COMPLEX = "complex"      # "boneless skinless chicken breast"
    VERY_COMPLEX = "very_complex"  # "(4-6 oz) artisanal crackers, such as rosemary"


# ============================================================================
# AI PROMPT TEMPLATES FOR INGREDIENT PARSING
# ============================================================================

INGREDIENT_PARSING_PROMPT = """
You are an expert chef and grocery shopping assistant. Parse these ingredient texts into clean, standardized components.

INGREDIENTS TO PARSE:
{ingredient_list}

PARSING RULES:
1. Extract the CORE INGREDIENT NAME (remove quantities, units, preparations)
2. Identify QUANTITY and UNIT separately
3. Extract PREPARATION METHOD if specified (diced, chopped, minced, etc.)
4. Remove BRAND NAMES and unnecessary adjectives
5. Handle INCOMPLETE PARSING (e.g., "boneless" → "chicken breast")
6. Clean COMPLEX FORMATS like "(4-6 oz) artisanal crackers"
7. Preserve ESSENTIAL DESCRIPTORS (e.g., "chicken breast" not just "chicken")

EXAMPLES:
- "2 cups diced roma tomatoes" → name: "tomatoes", quantity: 2, unit: "cups", preparation: "diced"
- "1 large yellow onion, diced" → name: "onion", quantity: 1, unit: "piece", preparation: "diced"
- "boneless skinless chicken breast" → name: "chicken breast", quantity: 1, unit: "piece", preparation: null
- "(4-6 oz) artisanal crackers" → name: "crackers", quantity: 5, unit: "oz", preparation: null
- "1 tsp ground cinnamon" → name: "cinnamon", quantity: 1, unit: "teaspoon", preparation: "ground"
- "fresh basil leaves, for garnish" → name: "basil", quantity: 1, unit: "bunch", preparation: "fresh leaves"

Return ONLY a JSON array with this EXACT format:
[
  {{
    "original": "2 cups diced roma tomatoes",
    "name": "tomatoes", 
    "quantity": 2.0,
    "unit": "cups",
    "preparation": "diced",
    "brand_info": null,
    "package_info": null,
    "quality_descriptors": ["roma"],
    "confidence": 0.95,
    "notes": "Removed variety descriptor, preserved preparation"
  }}
]
"""

COMPLEX_INGREDIENT_PROMPT = """
You are an expert at parsing complex ingredient descriptions. Handle these challenging cases:

COMPLEX INGREDIENTS:
{complex_ingredients}

SPECIAL HANDLING RULES:
1. INCOMPLETE NAMES: "boneless" → "chicken breast", "skinless" → "chicken breast"
2. PACKAGE FORMATS: "(4-6 oz)" → extract average quantity: 5 oz
3. BRAND/ARTISANAL: Remove "artisanal", "premium", brand names
4. COMPOUND DESCRIPTIONS: "boneless skinless chicken breast" → "chicken breast"
5. OPTIONAL ITEMS: "for garnish (optional)" → remove optional indicators
6. RANGE QUANTITIES: "4-6 oz" → use middle value: 5 oz
7. MULTIPLE DESCRIPTORS: Keep only essential ones

CRITICAL: If the ingredient name seems incomplete (like "boneless", "skinless", "fresh"), 
infer the complete ingredient from context and cooking knowledge.

Return ONLY a JSON array with complete ingredient information.
"""


# ============================================================================
# AI INGREDIENT PARSING SERVICE
# ============================================================================

class AIIngredientParser:
    """Advanced AI-powered ingredient parsing service"""
    
    def __init__(self):
        self.model_cache = {}
        self.parsing_stats = {
            "total_parsed": 0,
            "complex_handled": 0,
            "fallback_used": 0,
            "avg_confidence": 0.0
        }
    
    async def parse_ingredients_batch(
        self,
        raw_ingredients: List[str],
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    ) -> List[ParsedIngredient]:
        """
        Parse multiple ingredients using AI with intelligent complexity detection
        
        Args:
            raw_ingredients: List of raw ingredient text strings
            optimization_level: AI optimization level affecting model choice
            
        Returns:
            List of ParsedIngredient objects with cleaned data
        """
        
        logger.info(
            f"Starting AI ingredient parsing",
            ingredient_count=len(raw_ingredients),
            optimization_level=optimization_level.value
        )
        
        try:
            # Step 1: Classify ingredients by complexity
            complexity_groups = self._classify_ingredients_by_complexity(raw_ingredients)
            
            # Step 2: Process each complexity group with appropriate strategy
            all_parsed = []
            
            for complexity, ingredients in complexity_groups.items():
                if not ingredients:
                    continue
                
                logger.info(f"Processing {len(ingredients)} {complexity.value} ingredients")
                
                if complexity in [IngredientComplexity.COMPLEX, IngredientComplexity.VERY_COMPLEX]:
                    # Use specialized complex parsing
                    parsed = await self._parse_complex_ingredients(ingredients, optimization_level)
                    self.parsing_stats["complex_handled"] += len(parsed)
                else:
                    # Use standard parsing
                    parsed = await self._parse_standard_ingredients(ingredients, optimization_level)
                
                all_parsed.extend(parsed)
            
            # Step 3: Post-processing and validation
            validated_parsed = self._validate_and_enhance_parsed(all_parsed)
            
            # Update stats
            self.parsing_stats["total_parsed"] += len(validated_parsed)
            avg_conf = sum(p.confidence_score for p in validated_parsed) / len(validated_parsed)
            self.parsing_stats["avg_confidence"] = avg_conf
            
            logger.info(
                f"AI ingredient parsing completed",
                total_parsed=len(validated_parsed),
                avg_confidence=avg_conf,
                complex_handled=self.parsing_stats["complex_handled"]
            )
            
            return validated_parsed
            
        except Exception as e:
            logger.error(f"AI ingredient parsing failed: {e}")
            # Fallback to rule-based parsing
            return await self._fallback_parse_ingredients(raw_ingredients)
    
    def _classify_ingredients_by_complexity(
        self, 
        ingredients: List[str]
    ) -> Dict[IngredientComplexity, List[str]]:
        """Classify ingredients by parsing complexity"""
        
        groups = {
            IngredientComplexity.SIMPLE: [],
            IngredientComplexity.MODERATE: [],
            IngredientComplexity.COMPLEX: [],
            IngredientComplexity.VERY_COMPLEX: []
        }
        
        for ingredient in ingredients:
            complexity = self._assess_ingredient_complexity(ingredient)
            groups[complexity].append(ingredient)
        
        return groups
    
    def _assess_ingredient_complexity(self, ingredient_text: str) -> IngredientComplexity:
        """Assess parsing complexity of an ingredient"""
        
        text = ingredient_text.lower()
        
        # Very complex patterns
        very_complex_patterns = [
            r'\([^)]*\)',  # Parentheses with content
            r'\d+-\d+',    # Range quantities like "4-6"
            r'such as|like|or',  # Alternative suggestions
            r'artisanal|premium|organic|free-range',  # Quality descriptors
        ]
        
        # Complex patterns  
        complex_patterns = [
            r'boneless|skinless',  # Incomplete meat descriptions
            r'fresh.*leaves|for garnish',  # Preparation contexts
            r'high-quality|preferably|if possible',  # Quality preferences
            r'aged|cold|room temperature',  # State descriptors
        ]
        
        # Moderate patterns
        moderate_patterns = [
            r'diced|chopped|minced|sliced',  # Preparation methods
            r'large|medium|small',  # Size descriptors
            r'red|yellow|green|white',  # Color descriptors
        ]
        
        # Check complexity
        if any(re.search(pattern, text) for pattern in very_complex_patterns):
            return IngredientComplexity.VERY_COMPLEX
        elif any(re.search(pattern, text) for pattern in complex_patterns):
            return IngredientComplexity.COMPLEX
        elif any(re.search(pattern, text) for pattern in moderate_patterns):
            return IngredientComplexity.MODERATE
        else:
            return IngredientComplexity.SIMPLE
    
    async def _parse_complex_ingredients(
        self,
        complex_ingredients: List[str],
        optimization_level: OptimizationLevel
    ) -> List[ParsedIngredient]:
        """Use specialized AI parsing for complex ingredient formats"""
        
        # Get appropriate AI model
        model = await self._get_model_for_optimization(optimization_level)
        
        # Create specialized prompt for complex ingredients
        prompt = COMPLEX_INGREDIENT_PROMPT.format(
            complex_ingredients=json.dumps(complex_ingredients, indent=2)
        )
        
        try:
            # Call AI model
            response = await model.generate_content_async(prompt)
            parsed_data = json.loads(response.text.strip())
            
            # Convert to ParsedIngredient objects
            parsed_ingredients = []
            for i, ingredient_text in enumerate(complex_ingredients):
                if i < len(parsed_data):
                    data = parsed_data[i]
                    parsed = ParsedIngredient(
                        original_text=ingredient_text,
                        cleaned_name=data.get("name", ingredient_text),
                        quantity=float(data.get("quantity", 1.0)),
                        unit=data.get("unit", "piece"),
                        preparation=data.get("preparation"),
                        brand_info=data.get("brand_info"),
                        package_info=data.get("package_info"),
                        quality_descriptors=data.get("quality_descriptors", []),
                        confidence_score=float(data.get("confidence", 0.8)),
                        parsing_notes=data.get("notes")
                    )
                    parsed_ingredients.append(parsed)
                else:
                    # Fallback for missing AI results
                    parsed_ingredients.append(self._create_fallback_parsed(ingredient_text))
            
            return parsed_ingredients
            
        except Exception as e:
            logger.warning(f"Complex ingredient AI parsing failed: {e}")
            return [self._create_fallback_parsed(ing) for ing in complex_ingredients]
    
    async def _parse_standard_ingredients(
        self,
        standard_ingredients: List[str],
        optimization_level: OptimizationLevel
    ) -> List[ParsedIngredient]:
        """Parse standard ingredients using main AI parsing prompt"""
        
        model = await self._get_model_for_optimization(optimization_level)
        
        prompt = INGREDIENT_PARSING_PROMPT.format(
            ingredient_list=json.dumps(standard_ingredients, indent=2)
        )
        
        try:
            response = await model.generate_content_async(prompt)
            parsed_data = json.loads(response.text.strip())
            
            parsed_ingredients = []
            for i, ingredient_text in enumerate(standard_ingredients):
                if i < len(parsed_data):
                    data = parsed_data[i]
                    parsed = ParsedIngredient(
                        original_text=ingredient_text,
                        cleaned_name=data.get("name", ingredient_text),
                        quantity=float(data.get("quantity", 1.0)),
                        unit=data.get("unit", "piece"),
                        preparation=data.get("preparation"),
                        brand_info=data.get("brand_info"),
                        package_info=data.get("package_info"),
                        quality_descriptors=data.get("quality_descriptors", []),
                        confidence_score=float(data.get("confidence", 0.85)),
                        parsing_notes=data.get("notes")
                    )
                    parsed_ingredients.append(parsed)
                else:
                    parsed_ingredients.append(self._create_fallback_parsed(ingredient_text))
            
            return parsed_ingredients
            
        except Exception as e:
            logger.warning(f"Standard ingredient AI parsing failed: {e}")
            return [self._create_fallback_parsed(ing) for ing in standard_ingredients]
    
    def _create_fallback_parsed(self, ingredient_text: str) -> ParsedIngredient:
        """Create fallback parsed ingredient using rule-based parsing"""
        
        # Basic regex parsing for fallback
        quantity, unit = self._extract_quantity_unit(ingredient_text)
        name = self._extract_ingredient_name(ingredient_text)
        preparation = self._extract_preparation(ingredient_text)
        
        return ParsedIngredient(
            original_text=ingredient_text,
            cleaned_name=name,
            quantity=quantity,
            unit=unit,
            preparation=preparation,
            brand_info=None,
            package_info=None,
            quality_descriptors=[],
            confidence_score=0.6,  # Lower confidence for fallback
            parsing_notes="Rule-based fallback parsing"
        )
    
    def _extract_quantity_unit(self, text: str) -> Tuple[float, str]:
        """Extract quantity and unit using regex patterns"""
        
        # Comprehensive patterns for quantity and unit extraction
        patterns = [
            # Fraction patterns
            (r'(\d+)\s*-\s*(\d+)\s*(oz|ounces)', lambda m: (float(m.group(1)) + float(m.group(2))) / 2, m.group(3)),
            (r'(\d+)\s*/\s*(\d+)\s*(cups?|cup)', lambda m: float(m.group(1)) / float(m.group(2)), m.group(3)),
            
            # Standard patterns
            (r'(\d+(?:\.\d+)?)\s*(cups?|cup|c\b)', lambda m: (float(m.group(1)), m.group(2))),
            (r'(\d+(?:\.\d+)?)\s*(tablespoons?|tbsp|tbs)', lambda m: (float(m.group(1)), m.group(2))),
            (r'(\d+(?:\.\d+)?)\s*(teaspoons?|tsp|t\b)', lambda m: (float(m.group(1)), m.group(2))),
            (r'(\d+(?:\.\d+)?)\s*(ounces?|oz)', lambda m: (float(m.group(1)), m.group(2))),
            (r'(\d+(?:\.\d+)?)\s*(pounds?|lbs?|lb)', lambda m: (float(m.group(1)), m.group(2))),
            (r'(\d+(?:\.\d+)?)\s*(grams?|g\b)', lambda m: (float(m.group(1)), m.group(2))),
            (r'(\d+(?:\.\d+)?)\s*(kilograms?|kg)', lambda m: (float(m.group(1)), m.group(2))),
            (r'(\d+(?:\.\d+)?)\s*(milliliters?|ml)', lambda m: (float(m.group(1)), m.group(2))),
            (r'(\d+(?:\.\d+)?)\s*(liters?|l\b)', lambda m: (float(m.group(1)), m.group(2))),
            
            # Count patterns
            (r'(\d+(?:\.\d+)?)\s*(pieces?|pcs?|pc)', lambda m: (float(m.group(1)), "piece")),
            (r'(\d+(?:\.\d+)?)\s*(cloves?)', lambda m: (float(m.group(1)), "clove")),
            (r'(\d+(?:\.\d+)?)\s*(slices?)', lambda m: (float(m.group(1)), "slice")),
            
            # Size descriptors
            (r'(\d+(?:\.\d+)?)\s*(large|medium|small)', lambda m: (float(m.group(1)), "piece")),
            
            # Container patterns
            (r'(\d+(?:\.\d+)?)\s*(cans?|boxes?|packages?)', lambda m: (float(m.group(1)), m.group(2))),
            (r'(\d+(?:\.\d+)?)\s*(pints?|quarts?)', lambda m: (float(m.group(1)), m.group(2))),
        ]
        
        for pattern_info in patterns:
            if len(pattern_info) == 3:
                pattern, quantity_func, unit_func = pattern_info
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if callable(quantity_func):
                        quantity, unit = quantity_func(match)
                        return quantity, unit
                    else:
                        return quantity_func, unit_func
            else:
                pattern, extract_func = pattern_info
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return extract_func(match)
        
        # Default if no pattern matches
        return 1.0, "piece"
    
    def _extract_ingredient_name(self, text: str) -> str:
        """Extract core ingredient name using rule-based logic"""
        
        # Remove quantities and units first
        cleaned = re.sub(r'\d+(?:\.\d+)?\s*(?:cups?|tbsp|tsp|oz|lbs?|g|kg|ml|l|pieces?|cloves?)', '', text, flags=re.IGNORECASE)
        
        # Remove parentheses and their contents
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        
        # Remove common preparation words
        prep_words = ['diced', 'chopped', 'minced', 'sliced', 'fresh', 'dried', 'ground', 'whole', 'crushed']
        for word in prep_words:
            cleaned = re.sub(rf'\b{word}\b', '', cleaned, flags=re.IGNORECASE)
        
        # Remove quality descriptors
        quality_words = ['organic', 'local', 'artisanal', 'premium', 'high-quality', 'preferably', 'if possible']
        for word in quality_words:
            cleaned = re.sub(rf'\b{word}\b', '', cleaned, flags=re.IGNORECASE)
        
        # Handle incomplete names
        incomplete_mappings = {
            'boneless': 'chicken breast',
            'skinless': 'chicken breast', 
            'boneless skinless': 'chicken breast'
        }
        
        cleaned_lower = cleaned.lower().strip()
        for incomplete, complete in incomplete_mappings.items():
            if incomplete in cleaned_lower:
                return complete
        
        # Clean up whitespace and punctuation
        cleaned = re.sub(r'[,;]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned if cleaned else text
    
    def _extract_preparation(self, text: str) -> Optional[str]:
        """Extract preparation method from ingredient text"""
        
        prep_patterns = [
            r'\b(diced|chopped|minced|sliced|julienned|grated|shredded)\b',
            r'\b(fresh|dried|ground|whole|crushed|torn)\b',
            r'\b(pounded|marinated|seasoned|cooked)\b'
        ]
        
        preparations = []
        for pattern in prep_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            preparations.extend(matches)
        
        return ', '.join(preparations) if preparations else None
    
    async def _get_model_for_optimization(self, optimization_level: OptimizationLevel) -> GenerativeModel:
        """Get appropriate AI model based on optimization level"""
        
        model_mapping = {
            OptimizationLevel.BASIC: "gemini-1.5-flash",
            OptimizationLevel.STANDARD: "gemini-1.5-pro", 
            OptimizationLevel.PREMIUM: "gemini-2.0-flash-thinking"
        }
        
        model_name = model_mapping.get(optimization_level, "gemini-1.5-pro")
        
        # Cache models for efficiency
        if model_name not in self.model_cache:
            self.model_cache[model_name] = GenerativeModel(model_name)
        
        return self.model_cache[model_name]
    
    def _validate_and_enhance_parsed(self, parsed_ingredients: List[ParsedIngredient]) -> List[ParsedIngredient]:
        """Post-process and validate parsed ingredients"""
        
        enhanced = []
        
        for parsed in parsed_ingredients:
            # Validate ingredient name isn't empty
            if not parsed.cleaned_name or parsed.cleaned_name.strip() == "":
                parsed.cleaned_name = parsed.original_text
                parsed.confidence_score = min(parsed.confidence_score, 0.3)
            
            # Validate quantity is reasonable
            if parsed.quantity <= 0 or parsed.quantity > 1000:
                parsed.quantity = 1.0
                parsed.confidence_score = min(parsed.confidence_score, 0.5)
            
            # Standardize unit names
            parsed.unit = self._standardize_unit_name(parsed.unit)
            
            # Clean up preparation text
            if parsed.preparation:
                parsed.preparation = parsed.preparation.strip().lower()
            
            enhanced.append(parsed)
        
        return enhanced
    
    def _standardize_unit_name(self, unit: str) -> str:
        """Standardize unit names to consistent format"""
        
        unit_mappings = {
            'cups': 'cup', 'c': 'cup',
            'tablespoons': 'tablespoon', 'tbsp': 'tablespoon', 'tbs': 'tablespoon',
            'teaspoons': 'teaspoon', 'tsp': 'teaspoon', 't': 'teaspoon',
            'ounces': 'ounce', 'oz': 'ounce',
            'pounds': 'pound', 'lbs': 'pound', 'lb': 'pound',
            'grams': 'gram', 'g': 'gram',
            'kilograms': 'kilogram', 'kg': 'kilogram',
            'milliliters': 'milliliter', 'ml': 'milliliter',
            'liters': 'liter', 'l': 'liter',
            'pieces': 'piece', 'pcs': 'piece', 'pc': 'piece',
            'cloves': 'clove',
            'slices': 'slice'
        }
        
        return unit_mappings.get(unit.lower(), unit)
    
    async def _fallback_parse_ingredients(self, ingredients: List[str]) -> List[ParsedIngredient]:
        """Fallback rule-based ingredient parsing"""
        
        self.parsing_stats["fallback_used"] += len(ingredients)
        
        parsed = []
        for ingredient in ingredients:
            quantity, unit = self._extract_quantity_unit(ingredient)
            name = self._extract_ingredient_name(ingredient)
            preparation = self._extract_preparation(ingredient)
            
            parsed.append(ParsedIngredient(
                original_text=ingredient,
                cleaned_name=name,
                quantity=quantity,
                unit=unit,
                preparation=preparation,
                brand_info=None,
                package_info=None,
                quality_descriptors=[],
                confidence_score=0.6,
                parsing_notes="Rule-based fallback"
            ))
        
        return parsed
    
    async def get_parsing_stats(self) -> Dict[str, Any]:
        """Get parsing performance statistics"""
        return {
            "total_parsed": self.parsing_stats["total_parsed"],
            "complex_handled": self.parsing_stats["complex_handled"],
            "fallback_used": self.parsing_stats["fallback_used"],
            "avg_confidence": self.parsing_stats["avg_confidence"],
            "complex_success_rate": (
                (self.parsing_stats["complex_handled"] / self.parsing_stats["total_parsed"]) 
                if self.parsing_stats["total_parsed"] > 0 else 0
            )
        }


# ============================================================================
# INTEGRATION FUNCTIONS
# ============================================================================

def get_ai_ingredient_parser() -> AIIngredientParser:
    """Factory function to get AI ingredient parser"""
    return AIIngredientParser()


async def parse_ingredients_for_shopping_list(
    raw_ingredient_texts: List[str],
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
) -> List[Dict[str, Any]]:
    """
    Main entry point for AI ingredient parsing
    Returns parsed ingredients in format compatible with shopping list generation
    """
    
    parser = get_ai_ingredient_parser()
    parsed_ingredients = await parser.parse_ingredients_batch(
        raw_ingredient_texts, optimization_level
    )
    
    # Convert to shopping list format
    shopping_ingredients = []
    for parsed in parsed_ingredients:
        shopping_ingredients.append({
            "name": parsed.cleaned_name,
            "quantity": parsed.quantity,
            "unit": parsed.unit,
            "preparation": parsed.preparation,
            "confidence_score": parsed.confidence_score,
            "original_text": parsed.original_text,
            "parsing_notes": parsed.parsing_notes,
            "quality_descriptors": parsed.quality_descriptors
        })
    
    return shopping_ingredients
