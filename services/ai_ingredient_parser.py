"""
🧠 AI-Powered Ingredient Name Cleaning & Parsing Service
Handles complex ingredient formats using the AI microservice
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from services.ai_service import ai_service
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class OptimizationLevel(str, Enum):
    BASIC = "basic"
    STANDARD = "standard" 
    PREMIUM = "premium"


class AIIngredientParser:
    """Parser for cleaning and structuring ingredient names using AI service"""
    
    def __init__(self):
        self.ai_client = ai_service
    
    async def parse_ingredient(self, ingredient_text: str) -> Dict[str, Any]:
        """Parse a single ingredient using the AI service"""
        try:
            # Call AI service for parsing
            result = await self.ai_client.parse_ingredients(ingredient_text)
            
            # Return first result if list, otherwise return as-is
            if isinstance(result, list) and len(result) > 0:
                return result[0]
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse ingredient: {e}")
            # Fallback to basic parsing
            return self._basic_parse(ingredient_text)
    
    def _basic_parse(self, ingredient_text: str) -> Dict[str, Any]:
        """Basic fallback parsing without AI"""
        # Remove extra whitespace
        cleaned = ' '.join(ingredient_text.split())
        
        # Basic pattern matching for quantity and unit
        quantity_pattern = r'^([\d\.\s/]+)\s*([a-zA-Z]+)?\s+(.+)'
        match = re.match(quantity_pattern, cleaned)
        
        if match:
            quantity, unit, name = match.groups()
            return {
                "name": name.strip(),
                "quantity": quantity.strip() if quantity else "1",
                "unit": unit.strip() if unit else "",
                "original_text": ingredient_text
            }
        
        return {
            "name": cleaned,
            "quantity": "1",
            "unit": "",
            "original_text": ingredient_text
        }
    
    async def parse_multiple(self, ingredients: List[str]) -> List[Dict[str, Any]]:
        """Parse multiple ingredients"""
        results = []
        for ingredient in ingredients:
            parsed = await self.parse_ingredient(ingredient)
            results.append(parsed)
        return results
    
    async def clean_ingredient_name(
        self,
        ingredient_name: str,
        optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    ) -> str:
        """Clean and normalize ingredient name"""
        
        # Basic cleaning
        cleaned = ingredient_name.strip().lower()
        
        # Remove common non-essential words
        non_essential = ['fresh', 'organic', 'large', 'small', 'medium', 'chopped', 'diced', 'sliced']
        for word in non_essential:
            cleaned = cleaned.replace(word, '').strip()
        
        # Remove extra spaces
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
    def extract_core_ingredient(self, ingredient_text: str) -> str:
        """Extract the core ingredient name from complex text"""
        
        # Remove parenthetical information
        core = re.sub(r'\([^)]*\)', '', ingredient_text)
        
        # Remove brand names and descriptors
        core = re.sub(r'\b(brand|style|type)\b.*', '', core, flags=re.IGNORECASE)
        
        # Clean up
        core = ' '.join(core.split())
        
        return core.strip()


# Global instance
ai_ingredient_parser = AIIngredientParser()