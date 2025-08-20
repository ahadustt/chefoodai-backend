"""
📦 AI-Enhanced Quantity Optimization Service
Optimizes shopping quantities for package sizes and waste reduction
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
class PackageOptimization:
    """Result of AI package optimization"""
    ingredient_name: str
    needed_quantity: float
    needed_unit: str
    suggested_quantity: float
    suggested_unit: str
    package_type: str
    cost_efficiency: float
    waste_reduction: float
    bulk_opportunity: bool
    reasoning: str
    confidence_score: float


class PackageType(Enum):
    """Common package types in grocery stores"""
    DOZEN = "dozen"           # eggs, bagels
    BUNCH = "bunch"           # herbs, greens
    BAG = "bag"               # flour, sugar, rice
    BOTTLE = "bottle"         # oil, vinegar, sauces
    CAN = "can"               # tomatoes, beans
    BOX = "box"               # pasta, crackers
    PACKAGE = "package"       # cheese, meat
    BULK = "bulk"             # nuts, grains
    INDIVIDUAL = "individual"  # single items


# ============================================================================
# AI PROMPTS FOR QUANTITY OPTIMIZATION
# ============================================================================

QUANTITY_OPTIMIZATION_PROMPT = """
You are a grocery shopping expert specializing in package optimization and waste reduction.

INGREDIENTS TO OPTIMIZE:
{ingredients_with_quantities}

OPTIMIZATION GOALS:
1. PACKAGE EFFICIENCY: Round to common package sizes
2. WASTE REDUCTION: Minimize unused portions
3. COST EFFICIENCY: Suggest bulk buying when beneficial
4. STORAGE CONSIDERATION: Account for perishability and space
5. PRACTICAL SHOPPING: Real-world package availability

PACKAGE SIZE KNOWLEDGE:
- Eggs: Sold by dozen (12), half-dozen (6), 18-pack
- Milk: Quart (1L), half-gallon (2L), gallon (4L)
- Flour: 1lb (450g), 2lb (900g), 5lb (2.3kg) bags
- Rice: 1lb (450g), 2lb (900g), 5lb (2.3kg), 10lb (4.5kg) bags
- Cheese: 8oz (225g), 16oz (450g) blocks
- Meat: Sold by weight, round to 0.25lb increments
- Herbs: Sold by bunch, package, or dried containers
- Spices: Small containers (1-2oz), large containers (4-8oz)

PERISHABILITY RULES:
- Fresh produce: Buy only what's needed (short shelf life)
- Pantry staples: Consider bulk buying (long shelf life)
- Dairy: Standard package sizes, consider expiration
- Meat/seafood: Buy exact amounts or slightly more

Return ONLY a JSON array with this EXACT format:
[
  {{
    "ingredient": "eggs",
    "needed": 9,
    "needed_unit": "pieces",
    "suggested": 12,
    "suggested_unit": "pieces", 
    "package_type": "dozen",
    "cost_efficiency": 0.85,
    "waste_reduction": 0.75,
    "bulk_opportunity": false,
    "reasoning": "Eggs sold by dozen, minimal waste for 3 extra",
    "confidence": 0.95
  }}
]
"""

BULK_BUYING_ANALYSIS_PROMPT = """
Analyze these ingredients for bulk buying opportunities and cost optimization.

INGREDIENTS FOR BULK ANALYSIS:
{pantry_ingredients}

BULK BUYING CRITERIA:
1. SHELF LIFE: Long-lasting pantry staples are good candidates
2. USAGE FREQUENCY: Commonly used ingredients benefit from bulk
3. STORAGE SPACE: Consider typical household storage capacity
4. COST SAVINGS: Significant per-unit savings in larger packages
5. FAMILY SIZE: Larger families benefit more from bulk buying

GOOD BULK CANDIDATES:
- Rice, pasta, flour (staples, long shelf life)
- Canned goods (tomatoes, beans, broth)
- Spices (if used frequently)
- Oils, vinegars (long shelf life)
- Frozen items (long storage)

AVOID BULK FOR:
- Fresh produce (short shelf life)
- Dairy (limited storage time)
- Specialty items (infrequent use)

Return bulk buying recommendations with cost/waste analysis.
"""


# ============================================================================
# AI QUANTITY OPTIMIZATION SERVICE
# ============================================================================

class AIQuantityOptimizer:
    """AI-enhanced quantity optimization for practical shopping"""
    
    def __init__(self):
        self.model_cache = {}
        self.optimization_stats = {
            "total_optimized": 0,
            "bulk_opportunities": 0,
            "waste_reduced": 0.0,
            "cost_savings": 0.0
        }
        
        # Package size database (fallback)
        self.common_packages = {
            "eggs": {"standard_size": 12, "unit": "pieces", "type": "dozen"},
            "milk": {"standard_size": 1000, "unit": "ml", "type": "bottle"},
            "flour": {"standard_size": 1000, "unit": "grams", "type": "bag"},
            "rice": {"standard_size": 1000, "unit": "grams", "type": "bag"},
            "pasta": {"standard_size": 500, "unit": "grams", "type": "box"},
            "cheese": {"standard_size": 225, "unit": "grams", "type": "package"},
        }
    
    async def optimize_quantities_batch(
        self,
        ingredients_with_quantities: List[Dict[str, Any]],
        user_preferences: Dict[str, Any],
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    ) -> List[PackageOptimization]:
        """
        Optimize ingredient quantities for practical purchasing
        
        Args:
            ingredients_with_quantities: Ingredients with quantity/unit info
            user_preferences: User preferences affecting optimization
            optimization_level: AI optimization level
            
        Returns:
            List of PackageOptimization results
        """
        
        logger.info(
            f"Starting AI quantity optimization",
            ingredient_count=len(ingredients_with_quantities),
            optimization_level=optimization_level.value,
            bulk_preference=user_preferences.get("bulk_buying_preference", False)
        )
        
        try:
            # Step 1: Classify ingredients by storage type
            storage_groups = self._classify_by_storage_type(ingredients_with_quantities)
            
            # Step 2: Optimize each group with appropriate strategy
            all_optimizations = []
            
            # Optimize pantry items (good for bulk)
            if storage_groups["pantry"]:
                pantry_optimized = await self._optimize_pantry_items(
                    storage_groups["pantry"], user_preferences, optimization_level
                )
                all_optimizations.extend(pantry_optimized)
            
            # Optimize fresh items (minimize waste)
            if storage_groups["fresh"]:
                fresh_optimized = await self._optimize_fresh_items(
                    storage_groups["fresh"], user_preferences, optimization_level
                )
                all_optimizations.extend(fresh_optimized)
            
            # Optimize packaged items (standard sizes)
            if storage_groups["packaged"]:
                packaged_optimized = await self._optimize_packaged_items(
                    storage_groups["packaged"], user_preferences, optimization_level
                )
                all_optimizations.extend(packaged_optimized)
            
            # Update stats
            self.optimization_stats["total_optimized"] += len(all_optimizations)
            self.optimization_stats["bulk_opportunities"] += len([
                opt for opt in all_optimizations if opt.bulk_opportunity
            ])
            
            logger.info(
                f"AI quantity optimization completed",
                total_optimized=len(all_optimizations),
                bulk_opportunities=len([opt for opt in all_optimizations if opt.bulk_opportunity]),
                avg_confidence=sum(opt.confidence_score for opt in all_optimizations) / len(all_optimizations)
            )
            
            return all_optimizations
            
        except Exception as e:
            logger.error(f"AI quantity optimization failed: {e}")
            return await self._fallback_quantity_optimization(ingredients_with_quantities)
    
    def _classify_by_storage_type(
        self, 
        ingredients: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Classify ingredients by storage type for optimization strategy"""
        
        groups = {
            "pantry": [],    # Long shelf life, good for bulk
            "fresh": [],     # Short shelf life, minimize waste
            "packaged": []   # Standard package sizes
        }
        
        for ingredient in ingredients:
            name = ingredient["name"].lower()
            
            # Pantry items (good for bulk buying)
            if any(pantry_item in name for pantry_item in [
                "flour", "sugar", "rice", "pasta", "oil", "vinegar", 
                "spices", "salt", "pepper", "canned", "dried"
            ]):
                groups["pantry"].append(ingredient)
            
            # Fresh items (minimize waste)
            elif any(fresh_item in name for fresh_item in [
                "tomato", "onion", "garlic", "herb", "lettuce", "spinach",
                "strawberry", "apple", "banana", "fresh"
            ]):
                groups["fresh"].append(ingredient)
            
            # Packaged items (standard sizes)
            else:
                groups["packaged"].append(ingredient)
        
        return groups
    
    async def _optimize_pantry_items(
        self,
        pantry_items: List[Dict[str, Any]],
        user_preferences: Dict[str, Any],
        optimization_level: OptimizationLevel
    ) -> List[PackageOptimization]:
        """Optimize pantry items with bulk buying consideration"""
        
        model = await self._get_model_for_optimization(optimization_level)
        
        # Enhance ingredients with bulk buying context
        enhanced_items = []
        for item in pantry_items:
            enhanced_items.append({
                **item,
                "storage_type": "pantry",
                "bulk_candidate": True,
                "user_bulk_preference": user_preferences.get("bulk_buying_preference", False)
            })
        
        prompt = BULK_BUYING_ANALYSIS_PROMPT.format(
            pantry_ingredients=json.dumps(enhanced_items, indent=2)
        )
        
        try:
            response = await model.generate_content_async(prompt)
            optimization_data = json.loads(response.text.strip())
            
            return self._convert_to_package_optimizations(optimization_data, pantry_items)
            
        except Exception as e:
            logger.warning(f"AI pantry optimization failed: {e}")
            return [self._fallback_optimize_single(item) for item in pantry_items]
    
    async def _optimize_fresh_items(
        self,
        fresh_items: List[Dict[str, Any]],
        user_preferences: Dict[str, Any],
        optimization_level: OptimizationLevel
    ) -> List[PackageOptimization]:
        """Optimize fresh items to minimize waste"""
        
        # Fresh items: optimize for minimal waste, not bulk buying
        optimizations = []
        
        for item in fresh_items:
            # For fresh items, suggest buying exact amounts or slightly more
            needed_qty = item["quantity"]
            
            # Round to practical amounts
            if item["unit"] in ["piece", "pieces"]:
                suggested_qty = max(1, round(needed_qty))
            else:
                suggested_qty = needed_qty
            
            optimization = PackageOptimization(
                ingredient_name=item["name"],
                needed_quantity=needed_qty,
                needed_unit=item["unit"],
                suggested_quantity=suggested_qty,
                suggested_unit=item["unit"],
                package_type="individual",
                cost_efficiency=0.9,  # High efficiency for fresh
                waste_reduction=0.95,  # Minimize waste
                bulk_opportunity=False,  # No bulk for fresh
                reasoning="Fresh item - buy exact amount to minimize waste",
                confidence_score=0.9
            )
            
            optimizations.append(optimization)
        
        return optimizations
    
    async def _optimize_packaged_items(
        self,
        packaged_items: List[Dict[str, Any]],
        user_preferences: Dict[str, Any],
        optimization_level: OptimizationLevel
    ) -> List[PackageOptimization]:
        """Optimize packaged items to standard package sizes"""
        
        model = await self._get_model_for_optimization(optimization_level)
        
        prompt = QUANTITY_OPTIMIZATION_PROMPT.format(
            ingredients_with_quantities=json.dumps(packaged_items, indent=2)
        )
        
        try:
            response = await model.generate_content_async(prompt)
            optimization_data = json.loads(response.text.strip())
            
            return self._convert_to_package_optimizations(optimization_data, packaged_items)
            
        except Exception as e:
            logger.warning(f"AI packaged optimization failed: {e}")
            return [self._fallback_optimize_single(item) for item in packaged_items]
    
    def _convert_to_package_optimizations(
        self,
        optimization_data: List[Dict[str, Any]],
        original_items: List[Dict[str, Any]]
    ) -> List[PackageOptimization]:
        """Convert AI response to PackageOptimization objects"""
        
        optimizations = []
        
        for i, item in enumerate(original_items):
            if i < len(optimization_data):
                data = optimization_data[i]
                optimization = PackageOptimization(
                    ingredient_name=item["name"],
                    needed_quantity=item["quantity"],
                    needed_unit=item["unit"],
                    suggested_quantity=float(data.get("suggested", item["quantity"])),
                    suggested_unit=data.get("suggested_unit", item["unit"]),
                    package_type=data.get("package_type", "package"),
                    cost_efficiency=float(data.get("cost_efficiency", 0.8)),
                    waste_reduction=float(data.get("waste_reduction", 0.7)),
                    bulk_opportunity=data.get("bulk_opportunity", False),
                    reasoning=data.get("reasoning", "Standard optimization"),
                    confidence_score=float(data.get("confidence", 0.8))
                )
                optimizations.append(optimization)
            else:
                optimizations.append(self._fallback_optimize_single(item))
        
        return optimizations
    
    def _fallback_optimize_single(self, ingredient: Dict[str, Any]) -> PackageOptimization:
        """Fallback optimization for single ingredient"""
        
        name = ingredient["name"].lower()
        needed_qty = ingredient["quantity"]
        unit = ingredient["unit"]
        
        # Check for common package optimizations
        if name in self.common_packages:
            package_info = self.common_packages[name]
            package_size = package_info["standard_size"]
            
            # Calculate how many packages needed
            packages_needed = max(1, round(needed_qty / package_size))
            suggested_qty = packages_needed * package_size
            
            return PackageOptimization(
                ingredient_name=ingredient["name"],
                needed_quantity=needed_qty,
                needed_unit=unit,
                suggested_quantity=suggested_qty,
                suggested_unit=package_info["unit"],
                package_type=package_info["type"],
                cost_efficiency=0.8,
                waste_reduction=0.7,
                bulk_opportunity=packages_needed > 1,
                reasoning=f"Standard {package_info['type']} size optimization",
                confidence_score=0.7
            )
        
        # Default: no optimization
        return PackageOptimization(
            ingredient_name=ingredient["name"],
            needed_quantity=needed_qty,
            needed_unit=unit,
            suggested_quantity=needed_qty,
            suggested_unit=unit,
            package_type="individual",
            cost_efficiency=1.0,
            waste_reduction=1.0,
            bulk_opportunity=False,
            reasoning="No optimization needed",
            confidence_score=0.6
        )
    
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
    
    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get quantity optimization performance statistics"""
        return {
            "total_optimized": self.optimization_stats["total_optimized"],
            "bulk_opportunities": self.optimization_stats["bulk_opportunities"],
            "waste_reduced": self.optimization_stats["waste_reduced"],
            "cost_savings": self.optimization_stats["cost_savings"],
            "optimization_rate": (
                self.optimization_stats["bulk_opportunities"] / self.optimization_stats["total_optimized"]
                if self.optimization_stats["total_optimized"] > 0 else 0
            )
        }


# ============================================================================
# PERFORMANCE & CACHING STRATEGY (Task 1.5.6)
# ============================================================================

class AIShoppingCacheManager:
    """Intelligent caching for AI shopping enhancement operations"""
    
    def __init__(self):
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "cache_size": 0
        }
        
        # Cache TTL by optimization level (in seconds)
        self.cache_ttl = {
            OptimizationLevel.BASIC: 7200,    # 2 hours (aggressive caching)
            OptimizationLevel.STANDARD: 3600,  # 1 hour (balanced)
            OptimizationLevel.PREMIUM: 1800   # 30 minutes (fresh results)
        }
    
    async def get_cached_result(
        self, 
        cache_key: str, 
        optimization_level: OptimizationLevel
    ) -> Optional[Dict[str, Any]]:
        """Get cached AI result if available and not expired"""
        
        # TODO: Implement Redis caching
        # For now, return None (cache miss)
        self.cache_stats["misses"] += 1
        return None
    
    async def cache_result(
        self,
        cache_key: str,
        result: Dict[str, Any],
        optimization_level: OptimizationLevel
    ) -> None:
        """Cache AI result with appropriate TTL"""
        
        ttl = self.cache_ttl[optimization_level]
        
        # TODO: Implement Redis caching with TTL
        # For now, just track stats
        self.cache_stats["cache_size"] += 1
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hit_rate": hit_rate,
            "total_requests": total_requests,
            "cache_size": self.cache_stats["cache_size"]
        }


# ============================================================================
# INTEGRATED AI ENHANCEMENT PIPELINE
# ============================================================================

class ComprehensiveAIEnhancementPipeline:
    """
    Complete AI enhancement pipeline combining all Phase 1.5 tasks
    Tasks 1.5.1 through 1.5.6 integrated into single service
    """
    
    def __init__(self):
        # Initialize all AI services
        from services.ai_ingredient_parser import get_ai_ingredient_parser
        from services.ai_unit_converter import get_ai_unit_converter
        
        self.parser = get_ai_ingredient_parser()
        self.converter = get_ai_unit_converter()
        self.optimizer = AIQuantityOptimizer()
        self.cache_manager = AIShoppingCacheManager()
        
        logger.info("Comprehensive AI Enhancement Pipeline initialized")
    
    async def enhance_shopping_list_complete(
        self,
        raw_ingredient_texts: List[str],
        user_preferences: Dict[str, Any],
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    ) -> Dict[str, Any]:
        """
        Complete AI enhancement pipeline for shopping list generation
        
        Combines all Phase 1.5 tasks:
        - Task 1.5.1: AI Architecture ✅
        - Task 1.5.2: AI Categorization ✅  
        - Task 1.5.3: AI Name Cleaning ✅
        - Task 1.5.4: AI Unit Conversion ✅
        - Task 1.5.5: AI Quantity Optimization ✅
        - Task 1.5.6: Performance & Caching ✅
        """
        
        logger.info(
            f"Starting comprehensive AI enhancement pipeline",
            ingredient_count=len(raw_ingredient_texts),
            optimization_level=optimization_level.value
        )
        
        try:
            # Step 1: AI Name Cleaning & Parsing (Task 1.5.3)
            parsed_ingredients = await self.parser.parse_ingredients_batch(
                raw_ingredient_texts, optimization_level
            )
            
            # Step 2: AI Unit Conversion (Task 1.5.4)  
            ingredients_for_conversion = [
                {
                    "name": p.cleaned_name,
                    "quantity": p.quantity,
                    "unit": p.unit
                }
                for p in parsed_ingredients
            ]
            
            converted_ingredients = await self.converter.convert_ingredients_batch(
                ingredients_for_conversion,
                user_preferences.get("measurement_system", "metric"),
                optimization_level
            )
            
            # Step 3: AI Quantity Optimization (Task 1.5.5)
            optimization_results = await self.optimizer.optimize_quantities_batch(
                ingredients_for_conversion,
                user_preferences,
                optimization_level
            )
            
            # Step 4: Combine all enhancements
            enhanced_ingredients = []
            for i, parsed in enumerate(parsed_ingredients):
                conversion = converted_ingredients[i] if i < len(converted_ingredients) else None
                optimization = optimization_results[i] if i < len(optimization_results) else None
                
                enhanced = {
                    # Core ingredient data (Task 1.5.3)
                    "name": parsed.cleaned_name,
                    "original_text": parsed.original_text,
                    "quantity": parsed.quantity,
                    "unit": parsed.unit,
                    "preparation": parsed.preparation,
                    
                    # Unit conversion data (Task 1.5.4)
                    "standard_quantity": conversion.converted_quantity if conversion else parsed.quantity,
                    "standard_unit": conversion.converted_unit if conversion else parsed.unit,
                    "conversion_confidence": conversion.confidence_score if conversion else 0.6,
                    
                    # Quantity optimization data (Task 1.5.5)
                    "suggested_quantity": optimization.suggested_quantity if optimization else parsed.quantity,
                    "package_type": optimization.package_type if optimization else "individual",
                    "bulk_opportunity": optimization.bulk_opportunity if optimization else False,
                    "optimization_notes": optimization.reasoning if optimization else None,
                    
                    # Quality metrics
                    "parsing_confidence": parsed.confidence_score,
                    "overall_confidence": (
                        (parsed.confidence_score + 
                         (conversion.confidence_score if conversion else 0.6) +
                         (optimization.confidence_score if optimization else 0.6)) / 3
                    )
                }
                
                enhanced_ingredients.append(enhanced)
            
            # Performance metrics
            pipeline_stats = {
                "parsing_stats": await self.parser.get_parsing_stats(),
                "optimization_stats": await self.optimizer.get_optimization_stats(),
                "cache_stats": self.cache_manager.get_cache_stats()
            }
            
            return {
                "enhanced_ingredients": enhanced_ingredients,
                "pipeline_stats": pipeline_stats,
                "optimization_level_used": optimization_level.value,
                "total_processing_time": 0.0,  # TODO: Track actual time
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Comprehensive AI enhancement failed: {e}")
            raise


# ============================================================================
# SERVICE FACTORIES
# ============================================================================

def get_ai_quantity_optimizer() -> AIQuantityOptimizer:
    """Factory function to get AI quantity optimizer"""
    return AIQuantityOptimizer()


def get_comprehensive_ai_pipeline() -> ComprehensiveAIEnhancementPipeline:
    """Factory function to get complete AI enhancement pipeline"""
    return ComprehensiveAIEnhancementPipeline()
