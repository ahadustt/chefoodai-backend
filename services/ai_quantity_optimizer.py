"""
📦 AI-Enhanced Quantity Optimization Service
Optimizes shopping quantities using AI microservice
"""

import json
import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from services.ai_service import ai_service
from services.ai_unit_converter import ai_unit_converter
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class OptimizationLevel(str, Enum):
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"


@dataclass
class PackageSize:
    """Standard package size information"""
    size: float
    unit: str
    price: Optional[float] = None
    is_bulk: bool = False


class AIQuantityOptimizer:
    """Quantity optimizer with AI enhancements via microservice"""
    
    def __init__(self):
        self.ai_client = ai_service
        self.unit_converter = ai_unit_converter
        
        # Common package sizes for reference
        self.common_packages = {
            'milk': [PackageSize(1, 'l'), PackageSize(2, 'l')],
            'eggs': [PackageSize(6, 'count'), PackageSize(12, 'count'), PackageSize(18, 'count')],
            'bread': [PackageSize(1, 'loaf')],
            'rice': [PackageSize(1, 'kg'), PackageSize(2, 'kg'), PackageSize(5, 'kg', is_bulk=True)],
            'pasta': [PackageSize(500, 'g'), PackageSize(1, 'kg')],
            'flour': [PackageSize(1, 'kg'), PackageSize(2, 'kg'), PackageSize(5, 'kg', is_bulk=True)],
        }
    
    async def optimize_quantities(
        self,
        items: List[Dict[str, Any]],
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
        household_size: int = 2
    ) -> Dict[str, Any]:
        """Optimize quantities for practical shopping"""
        
        # Group and merge similar items first
        merged_items = await self._merge_similar_items(items)
        
        # Optimize each item
        optimized_items = []
        total_savings = 0
        
        for item in merged_items:
            optimized = await self._optimize_single_item(item, optimization_level, household_size)
            optimized_items.append(optimized)
            
            if 'savings' in optimized:
                total_savings += optimized['savings']
        
        # Get bulk suggestions if premium
        bulk_suggestions = []
        if optimization_level == OptimizationLevel.PREMIUM:
            bulk_suggestions = await self._suggest_bulk_purchases(optimized_items, household_size)
        
        return {
            'items': optimized_items,
            'total_items': len(optimized_items),
            'optimization_level': optimization_level.value,
            'estimated_savings': total_savings,
            'bulk_suggestions': bulk_suggestions
        }
    
    async def _merge_similar_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge items with the same name"""
        
        grouped = defaultdict(list)
        for item in items:
            name = self._normalize_name(item.get('name', ''))
            grouped[name].append(item)
        
        merged = []
        for name, group_items in grouped.items():
            if len(group_items) == 1:
                merged.append(group_items[0])
            else:
                # Merge quantities
                merged_item = group_items[0].copy()
                
                # Standardize units and sum
                standardized = await self.unit_converter.standardize_units(group_items)
                total_quantity = sum(item.get('quantity', 0) for item in standardized)
                
                merged_item['quantity'] = total_quantity
                merged_item['unit'] = standardized[0].get('unit', '')
                merged_item['merged_count'] = len(group_items)
                
                merged.append(merged_item)
        
        return merged
    
    async def _optimize_single_item(
        self,
        item: Dict[str, Any],
        optimization_level: OptimizationLevel,
        household_size: int
    ) -> Dict[str, Any]:
        """Optimize a single item's quantity"""
        
        name = item.get('name', '')
        quantity = item.get('quantity', 0)
        unit = item.get('unit', '')
        
        # Round to practical amounts
        optimized_quantity = self._round_to_practical(quantity, unit)
        
        # Adjust for household size if premium
        if optimization_level == OptimizationLevel.PREMIUM:
            optimized_quantity = self._adjust_for_household(
                optimized_quantity, unit, household_size, name
            )
        
        # Find best package size if available
        package_suggestion = self._find_best_package(name, optimized_quantity, unit)
        
        result = item.copy()
        result['optimized_quantity'] = optimized_quantity
        result['original_quantity'] = quantity
        
        if package_suggestion:
            result['package_suggestion'] = package_suggestion
        
        return result
    
    def _round_to_practical(self, quantity: float, unit: str) -> float:
        """Round to practical shopping amounts"""
        
        unit_lower = unit.lower()
        
        if unit_lower in ['g', 'ml']:
            if quantity < 100:
                return math.ceil(quantity / 25) * 25
            elif quantity < 500:
                return math.ceil(quantity / 50) * 50
            else:
                return math.ceil(quantity / 100) * 100
        
        elif unit_lower in ['kg', 'l']:
            if quantity < 1:
                return math.ceil(quantity * 10) / 10
            elif quantity < 2:
                return math.ceil(quantity * 4) / 4
            else:
                return math.ceil(quantity * 2) / 2
        
        elif unit_lower in ['cup', 'tbsp', 'tsp']:
            if unit_lower == 'cup':
                return math.ceil(quantity * 4) / 4
            else:
                return math.ceil(quantity)
        
        else:
            return math.ceil(quantity)
    
    def _adjust_for_household(
        self,
        quantity: float,
        unit: str,
        household_size: int,
        item_name: str
    ) -> float:
        """Adjust quantities based on household size"""
        
        # Base adjustment factor
        factor = 1.0
        
        if household_size == 1:
            factor = 0.6
        elif household_size == 2:
            factor = 1.0
        elif household_size <= 4:
            factor = 1.5
        else:
            factor = 2.0
        
        # Adjust based on perishability
        if self._is_perishable(item_name):
            factor = min(factor, 1.2)  # Don't over-buy perishables
        
        return quantity * factor
    
    def _find_best_package(
        self,
        item_name: str,
        quantity: float,
        unit: str
    ) -> Optional[Dict[str, Any]]:
        """Find the best package size for an item"""
        
        normalized_name = self._normalize_name(item_name)
        
        # Check if we have package info for this item
        for key, packages in self.common_packages.items():
            if key in normalized_name:
                # Find best matching package
                best_package = None
                min_waste = float('inf')
                
                for package in packages:
                    # Convert units if needed
                    if package.unit.lower() == unit.lower():
                        waste = abs(package.size - quantity)
                        if waste < min_waste:
                            min_waste = waste
                            best_package = package
                
                if best_package:
                    return {
                        'size': best_package.size,
                        'unit': best_package.unit,
                        'is_bulk': best_package.is_bulk
                    }
        
        return None
    
    async def _suggest_bulk_purchases(
        self,
        items: List[Dict[str, Any]],
        household_size: int
    ) -> List[Dict[str, Any]]:
        """Suggest bulk purchases for cost savings"""
        
        suggestions = []
        
        for item in items:
            name = item.get('name', '')
            
            # Check if item is good for bulk buying
            if self._is_bulk_friendly(name) and household_size >= 2:
                suggestions.append({
                    'item': name,
                    'reason': 'Non-perishable item suitable for bulk buying',
                    'estimated_savings': '10-20%'
                })
        
        return suggestions
    
    def _is_perishable(self, item_name: str) -> bool:
        """Check if an item is perishable"""
        
        perishables = [
            'milk', 'yogurt', 'cheese', 'meat', 'chicken', 'fish',
            'vegetable', 'fruit', 'bread', 'egg', 'fresh'
        ]
        
        item_lower = item_name.lower()
        return any(p in item_lower for p in perishables)
    
    def _is_bulk_friendly(self, item_name: str) -> bool:
        """Check if an item is suitable for bulk buying"""
        
        bulk_friendly = [
            'rice', 'pasta', 'flour', 'sugar', 'salt', 'oil',
            'canned', 'dried', 'beans', 'lentils', 'oats',
            'toilet paper', 'paper towel', 'detergent'
        ]
        
        item_lower = item_name.lower()
        return any(bf in item_lower for bf in bulk_friendly)
    
    def _normalize_name(self, name: str) -> str:
        """Normalize item name for comparison"""
        return name.lower().strip()


# Global instance
ai_quantity_optimizer = AIQuantityOptimizer()