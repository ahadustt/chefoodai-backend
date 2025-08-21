"""
⚖️ AI-Enhanced Unit Conversion Service
Context-aware unit conversions using AI microservice
"""

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


class AIUnitConverter:
    """Unit converter with AI enhancements"""
    
    def __init__(self):
        self.ai_client = ai_service
        
        # Basic conversion rates for fallback
        self.basic_conversions = {
            # Volume
            ('cup', 'ml'): 236.588,
            ('ml', 'cup'): 0.00422675,
            ('tbsp', 'ml'): 14.787,
            ('ml', 'tbsp'): 0.067628,
            ('tsp', 'ml'): 4.929,
            ('ml', 'tsp'): 0.202884,
            ('l', 'ml'): 1000,
            ('ml', 'l'): 0.001,
            ('gal', 'l'): 3.78541,
            ('l', 'gal'): 0.264172,
            
            # Weight
            ('oz', 'g'): 28.3495,
            ('g', 'oz'): 0.035274,
            ('lb', 'kg'): 0.453592,
            ('kg', 'lb'): 2.20462,
            ('kg', 'g'): 1000,
            ('g', 'kg'): 0.001,
        }
    
    async def convert_unit(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        ingredient: Optional[str] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    ) -> Dict[str, Any]:
        """Convert units with optional AI enhancement for ingredient-specific conversions"""
        
        # Normalize units
        from_unit_lower = from_unit.lower().strip()
        to_unit_lower = to_unit.lower().strip()
        
        # Check for same unit
        if from_unit_lower == to_unit_lower:
            return {
                "success": True,
                "value": value,
                "unit": to_unit,
                "method": "same_unit"
            }
        
        # Try basic conversion first
        conversion_key = (from_unit_lower, to_unit_lower)
        if conversion_key in self.basic_conversions:
            converted_value = value * self.basic_conversions[conversion_key]
            return {
                "success": True,
                "value": round(converted_value, 2),
                "unit": to_unit,
                "method": "basic_conversion"
            }
        
        # For complex conversions (e.g., volume to weight), use AI service if premium
        if optimization_level == OptimizationLevel.PREMIUM and ingredient:
            try:
                # This would call the AI service for ingredient-specific conversion
                # For now, use fallback
                logger.info(f"Would use AI for converting {value} {from_unit} of {ingredient} to {to_unit}")
            except Exception as e:
                logger.error(f"AI conversion failed: {e}")
        
        # Fallback: return unconverted with warning
        return {
            "success": False,
            "value": value,
            "unit": from_unit,
            "message": f"Cannot convert from {from_unit} to {to_unit}",
            "method": "no_conversion"
        }
    
    async def standardize_units(
        self,
        items: List[Dict[str, Any]],
        target_system: str = "metric"
    ) -> List[Dict[str, Any]]:
        """Standardize all units to a consistent system"""
        
        standardized = []
        for item in items:
            unit = item.get('unit', '').lower()
            value = item.get('quantity', 1)
            
            # Determine target unit based on type
            if target_system == "metric":
                if unit in ['oz', 'lb']:
                    target_unit = 'g' if value < 500 else 'kg'
                elif unit in ['cup', 'tbsp', 'tsp', 'fl oz']:
                    target_unit = 'ml' if value < 1000 else 'l'
                else:
                    target_unit = unit
            else:
                target_unit = unit
            
            # Convert if needed
            if unit != target_unit:
                result = await self.convert_unit(value, unit, target_unit, item.get('name'))
                if result['success']:
                    item['quantity'] = result['value']
                    item['unit'] = result['unit']
            
            standardized.append(item)
        
        return standardized
    
    def get_compatible_units(self, unit: str) -> List[str]:
        """Get list of units that can be converted to/from the given unit"""
        
        unit_lower = unit.lower()
        compatible = set()
        
        for (from_unit, to_unit) in self.basic_conversions.keys():
            if from_unit == unit_lower:
                compatible.add(to_unit)
            elif to_unit == unit_lower:
                compatible.add(from_unit)
        
        return list(compatible)


# Global instance
ai_unit_converter = AIUnitConverter()