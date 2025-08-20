"""
ChefoodAI Cost Optimization Service
Intelligent cost management for AI requests with premium features
"""

import time
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

from core.config import settings
from core.redis import cache
from models.users import UserTier

logger = structlog.get_logger()


class CostTier(Enum):
    """Cost tiers for different optimization strategies"""
    AGGRESSIVE = "aggressive"  # Maximum cost savings
    BALANCED = "balanced"     # Balance cost and quality
    PREMIUM = "premium"       # Quality first


@dataclass
class CostMetrics:
    """Cost tracking metrics"""
    estimated_cost: float
    tokens_estimated: int
    model_recommended: str
    optimization_applied: str
    cache_hit_rate: float
    monthly_savings: float


class CostOptimizer:
    """
    Advanced cost optimization for AI requests
    Implements intelligent caching, model selection, and request batching
    """
    
    def __init__(self):
        # Model pricing (per 1K tokens) - approximate Vertex AI pricing
        self.model_pricing = {
            "gemini-2.0-flash-thinking": {"input": 0.000125, "output": 0.000375},
            "gemini-1.5-pro": {"input": 0.00125, "output": 0.00375},
            "gemini-1.5-flash": {"input": 0.000025, "output": 0.000075}
        }
        
        # User tier quotas and cost limits
        self.tier_limits = {
            UserTier.FREE: {
                "daily_requests": 5,
                "monthly_cost_limit": 0,  # Free tier
                "optimization_tier": CostTier.AGGRESSIVE
            },
            UserTier.PREMIUM: {
                "daily_requests": 1000,
                "monthly_cost_limit": 50,  # $50/month for AI
                "optimization_tier": CostTier.BALANCED
            },
            UserTier.ENTERPRISE: {
                "daily_requests": 10000,
                "monthly_cost_limit": 200,  # $200/month for AI
                "optimization_tier": CostTier.PREMIUM
            }
        }
        
        # Optimization strategies by tier
        self.optimization_strategies = {
            CostTier.AGGRESSIVE: {
                "cache_ttl_multiplier": 3.0,
                "prefer_cheaper_models": True,
                "batch_requests": True,
                "compress_prompts": True,
                "max_tokens_reduction": 0.7
            },
            CostTier.BALANCED: {
                "cache_ttl_multiplier": 2.0,
                "prefer_cheaper_models": True,
                "batch_requests": True,
                "compress_prompts": False,
                "max_tokens_reduction": 0.85
            },
            CostTier.PREMIUM: {
                "cache_ttl_multiplier": 1.0,
                "prefer_cheaper_models": False,
                "batch_requests": False,
                "compress_prompts": False,
                "max_tokens_reduction": 1.0
            }
        }
    
    async def optimize_request(self, request) -> Any:
        """
        Optimize AI request for cost efficiency while maintaining quality
        """
        try:
            # Get user cost tier
            user_tier = await self._get_user_tier(request.user_id)
            cost_tier = self.tier_limits[user_tier]["optimization_tier"]
            strategy = self.optimization_strategies[cost_tier]
            
            # Check daily quota
            if not await self._check_user_quota(request.user_id, user_tier):
                raise ValueError("Daily AI quota exceeded")
            
            # Check monthly cost limit
            if not await self._check_monthly_cost_limit(request.user_id, user_tier):
                raise ValueError("Monthly AI cost limit reached")
            
            # Optimize model selection
            optimized_model = await self._select_optimal_model(
                request.request_type,
                request.model_type,
                strategy
            )
            
            # Optimize token usage
            optimized_tokens = int(request.max_tokens * strategy["max_tokens_reduction"])
            
            # Optimize temperature for cost efficiency
            optimized_temperature = request.temperature
            if strategy["prefer_cheaper_models"]:
                optimized_temperature = min(request.temperature, 0.7)  # Lower temp = more predictable = cheaper
            
            # Create optimized request
            request.model_type = optimized_model
            request.max_tokens = optimized_tokens
            request.temperature = optimized_temperature
            
            # Log optimization
            logger.info(
                "Request optimized",
                user_id=request.user_id,
                original_model=request.model_type.value,
                optimized_model=optimized_model.value,
                cost_tier=cost_tier.value,
                token_reduction=1 - strategy["max_tokens_reduction"]
            )
            
            return request
            
        except Exception as e:
            logger.error(f"Request optimization failed: {str(e)}")
            return request  # Return original request if optimization fails
    
    async def calculate_request_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate estimated cost for AI request"""
        if model not in self.model_pricing:
            model = "gemini-1.5-flash"  # Default to cheapest model
        
        pricing = self.model_pricing[model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    async def track_usage(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached: bool = False
    ):
        """Track user AI usage for cost monitoring"""
        if cached:
            return  # No cost for cached responses
        
        cost = await self.calculate_request_cost(model, input_tokens, output_tokens)
        
        # Update daily usage
        today = time.strftime("%Y-%m-%d")
        daily_key = f"ai_usage:daily:{user_id}:{today}"
        
        current_usage = await cache.get_json(daily_key, "usage_tracking") or {
            "requests": 0,
            "tokens": 0,
            "cost": 0.0
        }
        
        current_usage["requests"] += 1
        current_usage["tokens"] += input_tokens + output_tokens
        current_usage["cost"] += cost
        
        await cache.set_json(daily_key, current_usage, 86400, "usage_tracking")
        
        # Update monthly usage
        month = time.strftime("%Y-%m")
        monthly_key = f"ai_usage:monthly:{user_id}:{month}"
        
        monthly_usage = await cache.get_json(monthly_key, "usage_tracking") or {
            "requests": 0,
            "tokens": 0,
            "cost": 0.0
        }
        
        monthly_usage["requests"] += 1
        monthly_usage["tokens"] += input_tokens + output_tokens
        monthly_usage["cost"] += cost
        
        await cache.set_json(monthly_key, monthly_usage, 86400 * 31, "usage_tracking")
        
        logger.info(
            "AI usage tracked",
            user_id=user_id,
            daily_cost=current_usage["cost"],
            monthly_cost=monthly_usage["cost"],
            model=model
        )
    
    async def get_usage_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get user's AI usage analytics"""
        today = time.strftime("%Y-%m-%d")
        month = time.strftime("%Y-%m")
        
        daily_key = f"ai_usage:daily:{user_id}:{today}"
        monthly_key = f"ai_usage:monthly:{user_id}:{month}"
        
        daily_usage = await cache.get_json(daily_key, "usage_tracking") or {}
        monthly_usage = await cache.get_json(monthly_key, "usage_tracking") or {}
        
        user_tier = await self._get_user_tier(user_id)
        limits = self.tier_limits[user_tier]
        
        return {
            "daily": {
                "requests": daily_usage.get("requests", 0),
                "limit": limits["daily_requests"],
                "remaining": max(0, limits["daily_requests"] - daily_usage.get("requests", 0)),
                "cost": round(daily_usage.get("cost", 0), 4)
            },
            "monthly": {
                "requests": monthly_usage.get("requests", 0),
                "cost": round(monthly_usage.get("cost", 0), 2),
                "limit": limits["monthly_cost_limit"],
                "remaining": max(0, limits["monthly_cost_limit"] - monthly_usage.get("cost", 0))
            },
            "tier": user_tier.value,
            "optimization_level": limits["optimization_tier"].value
        }
    
    async def predict_monthly_cost(self, user_id: str) -> Dict[str, float]:
        """Predict monthly AI costs based on current usage"""
        month = time.strftime("%Y-%m")
        monthly_key = f"ai_usage:monthly:{user_id}:{month}"
        
        monthly_usage = await cache.get_json(monthly_key, "usage_tracking") or {}
        current_cost = monthly_usage.get("cost", 0)
        
        # Calculate days elapsed in month
        import datetime
        today = datetime.date.today()
        days_elapsed = today.day
        days_in_month = (datetime.date(today.year, today.month + 1, 1) - datetime.timedelta(days=1)).day
        
        # Predict based on current usage rate
        if days_elapsed > 0:
            daily_average = current_cost / days_elapsed
            predicted_monthly = daily_average * days_in_month
        else:
            predicted_monthly = 0
        
        return {
            "current_month_cost": round(current_cost, 2),
            "predicted_month_cost": round(predicted_monthly, 2),
            "days_elapsed": days_elapsed,
            "days_remaining": days_in_month - days_elapsed
        }
    
    async def get_cost_savings_report(self, user_id: str) -> Dict[str, Any]:
        """Generate cost savings report for user"""
        # Get usage data
        usage = await self.get_usage_analytics(user_id)
        
        # Calculate potential costs without optimization
        monthly_requests = usage["monthly"]["requests"]
        if monthly_requests == 0:
            return {"savings": 0, "optimization_rate": 0}
        
        # Estimate costs without optimization (using most expensive model)
        unoptimized_cost = monthly_requests * 0.02  # Rough estimate
        actual_cost = usage["monthly"]["cost"]
        
        savings = max(0, unoptimized_cost - actual_cost)
        optimization_rate = (savings / unoptimized_cost) if unoptimized_cost > 0 else 0
        
        return {
            "estimated_savings": round(savings, 2),
            "optimization_rate": round(optimization_rate * 100, 1),
            "actual_cost": actual_cost,
            "estimated_unoptimized_cost": round(unoptimized_cost, 2),
            "cache_efficiency": await self._calculate_cache_efficiency(user_id)
        }
    
    # Private helper methods
    
    async def _get_user_tier(self, user_id: str) -> UserTier:
        """Get user's subscription tier"""
        # TODO: Implement actual user tier lookup from database
        # For now, return default tier
        return UserTier.PREMIUM
    
    async def _check_user_quota(self, user_id: str, user_tier: UserTier) -> bool:
        """Check if user has remaining daily quota"""
        today = time.strftime("%Y-%m-%d")
        daily_key = f"ai_usage:daily:{user_id}:{today}"
        
        daily_usage = await cache.get_json(daily_key, "usage_tracking") or {}
        requests_made = daily_usage.get("requests", 0)
        daily_limit = self.tier_limits[user_tier]["daily_requests"]
        
        return requests_made < daily_limit
    
    async def _check_monthly_cost_limit(self, user_id: str, user_tier: UserTier) -> bool:
        """Check if user has remaining monthly cost budget"""
        month = time.strftime("%Y-%m")
        monthly_key = f"ai_usage:monthly:{user_id}:{month}"
        
        monthly_usage = await cache.get_json(monthly_key, "usage_tracking") or {}
        cost_spent = monthly_usage.get("cost", 0)
        cost_limit = self.tier_limits[user_tier]["monthly_cost_limit"]
        
        if cost_limit == 0:  # Free tier
            return True  # Quota handled separately
        
        return cost_spent < cost_limit
    
    async def _select_optimal_model(self, request_type, preferred_model, strategy):
        """Select optimal model based on request type and cost strategy"""
        from services.ai_service import AIModelType, AIRequestType
        
        if not strategy["prefer_cheaper_models"]:
            return preferred_model
        
        # Model selection logic based on request type
        if request_type == AIRequestType.RECIPE_GENERATION:
            return AIModelType.GEMINI_1_5_FLASH  # Cheaper for simple recipes
        elif request_type == AIRequestType.IMAGE_ANALYSIS:
            return AIModelType.GEMINI_2_FLASH_THINKING  # Need multimodal
        elif request_type == AIRequestType.MEAL_PLANNING:
            return AIModelType.GEMINI_1_5_PRO  # Balance of capability and cost
        else:
            return AIModelType.GEMINI_1_5_FLASH  # Default to cheapest
    
    async def _calculate_cache_efficiency(self, user_id: str) -> float:
        """Calculate cache hit rate for user"""
        # TODO: Implement cache efficiency tracking
        return 0.75  # Placeholder: 75% cache hit rate


# Global cost optimizer instance
cost_optimizer = CostOptimizer()