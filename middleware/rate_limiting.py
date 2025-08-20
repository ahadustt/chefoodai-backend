"""
ChefoodAI Rate Limiting Middleware
Implements tiered rate limiting based on user subscription and endpoint sensitivity
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
import time
from typing import Dict, Tuple, Optional

from core.config import settings
from core.redis import rate_limiter
from models.users import UserTier

logger = structlog.get_logger()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Tiered rate limiting middleware with different limits for:
    - Free tier users
    - Premium users  
    - Enterprise users
    - Anonymous users
    - AI-specific endpoints
    """
    
    def __init__(self, app):
        super().__init__(app)
        
        # Rate limit configurations by endpoint type
        self.rate_limits = {
            # Global API limits (per hour)
            "global": {
                UserTier.FREE: {"requests": 100, "window": 3600},
                UserTier.PREMIUM: {"requests": 1000, "window": 3600},
                UserTier.ENTERPRISE: {"requests": 10000, "window": 3600},
                "anonymous": {"requests": 20, "window": 3600}
            },
            
            # AI endpoint limits (per day)
            "ai": {
                UserTier.FREE: {"requests": settings.FREE_TIER_AI_REQUESTS, "window": 86400},
                UserTier.PREMIUM: {"requests": settings.PREMIUM_TIER_AI_REQUESTS, "window": 86400},
                UserTier.ENTERPRISE: {"requests": settings.ENTERPRISE_TIER_AI_REQUESTS, "window": 86400},
                "anonymous": {"requests": 2, "window": 86400}
            },
            
            # Auth endpoint limits (per hour) - stricter to prevent abuse
            "auth": {
                UserTier.FREE: {"requests": 10, "window": 3600},
                UserTier.PREMIUM: {"requests": 50, "window": 3600},
                UserTier.ENTERPRISE: {"requests": 100, "window": 3600},
                "anonymous": {"requests": 5, "window": 3600}
            },
            
            # File upload limits (per hour)
            "upload": {
                UserTier.FREE: {"requests": 10, "window": 3600},
                UserTier.PREMIUM: {"requests": 100, "window": 3600},
                UserTier.ENTERPRISE: {"requests": 500, "window": 3600},
                "anonymous": {"requests": 2, "window": 3600}
            },
            
            # Recipe creation limits (per day)
            "recipe_create": {
                UserTier.FREE: {"requests": 5, "window": 86400},
                UserTier.PREMIUM: {"requests": 50, "window": 86400},
                UserTier.ENTERPRISE: {"requests": 500, "window": 86400},
                "anonymous": {"requests": 1, "window": 86400}
            }
        }
        
        # IP-based rate limiting for additional protection
        self.ip_limits = {
            "global": {"requests": 1000, "window": 3600},  # Per IP per hour
            "auth": {"requests": 20, "window": 3600},      # Stricter for auth
            "ai": {"requests": 100, "window": 86400}        # Per IP per day for AI
        }
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting based on endpoint and user tier"""
        try:
            # Skip rate limiting for health checks
            if request.url.path.startswith("/health") or request.url.path.startswith("/ready"):
                return await call_next(request)
            
            # Get user information
            user_id, user_tier = await self._get_user_info(request)
            client_ip = self._get_client_ip(request)
            
            # Determine endpoint type
            endpoint_type = self._classify_endpoint(request.url.path)
            
            # Check user-based rate limit
            if user_id:
                user_allowed, user_headers = await self._check_user_rate_limit(
                    user_id, user_tier, endpoint_type
                )
                if not user_allowed:
                    return self._create_rate_limit_response(user_headers)
            
            # Check IP-based rate limit
            ip_allowed, ip_headers = await self._check_ip_rate_limit(
                client_ip, endpoint_type
            )
            if not ip_allowed:
                return self._create_rate_limit_response(ip_headers)
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to response
            headers = user_headers if user_id else ip_headers
            for key, value in headers.items():
                response.headers[f"X-RateLimit-{key}"] = str(value)
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limit middleware error: {str(e)}")
            # Fail open - allow request if rate limiting fails
            return await call_next(request)
    
    async def _get_user_info(self, request: Request) -> Tuple[Optional[str], Optional[UserTier]]:
        """Extract user ID and tier from request"""
        try:
            # Try to get from Authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return None, None
            
            # This would normally decode JWT and get user info
            # For now, we'll simulate it - in real implementation,
            # this would use the JWT service
            token = auth_header.split(" ")[1]
            
            # TODO: Implement JWT decoding to get user_id and tier
            # user_data = jwt_service.decode_token(token)
            # return user_data.get("user_id"), user_data.get("tier")
            
            # Placeholder return
            return None, None
            
        except Exception as e:
            logger.error(f"Error extracting user info: {str(e)}")
            return None, None
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check for forwarded headers
        forwarded = request.headers.get("CF-Connecting-IP")
        if forwarded:
            return forwarded
        
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        forwarded = request.headers.get("X-Real-IP")
        if forwarded:
            return forwarded
        
        return request.client.host if request.client else "unknown"
    
    def _classify_endpoint(self, path: str) -> str:
        """Classify endpoint type for rate limiting"""
        if path.startswith("/api/v1/ai"):
            return "ai"
        elif path.startswith("/api/v1/auth"):
            return "auth"
        elif "upload" in path or "file" in path:
            return "upload"
        elif path.startswith("/api/v1/recipes") and any(method in path for method in ["POST", "PUT"]):
            return "recipe_create"
        else:
            return "global"
    
    async def _check_user_rate_limit(
        self, 
        user_id: str, 
        user_tier: UserTier, 
        endpoint_type: str
    ) -> Tuple[bool, Dict[str, int]]:
        """Check rate limit for authenticated user"""
        try:
            # Get rate limit configuration
            tier_limits = self.rate_limits.get(endpoint_type, self.rate_limits["global"])
            limit_config = tier_limits.get(user_tier, tier_limits[UserTier.FREE])
            
            # Check rate limit
            is_allowed, rate_info = await rate_limiter.is_allowed(
                identifier=user_id,
                limit=limit_config["requests"],
                window=limit_config["window"],
                endpoint=endpoint_type
            )
            
            return is_allowed, {
                "Limit": rate_info["limit"],
                "Remaining": rate_info["remaining"],
                "Reset": rate_info["reset"],
                "Window": limit_config["window"]
            }
            
        except Exception as e:
            logger.error(f"User rate limit check error: {str(e)}")
            return True, {}
    
    async def _check_ip_rate_limit(
        self, 
        client_ip: str, 
        endpoint_type: str
    ) -> Tuple[bool, Dict[str, int]]:
        """Check rate limit for client IP"""
        try:
            # Get IP rate limit configuration
            limit_config = self.ip_limits.get(endpoint_type, self.ip_limits["global"])
            
            # Check rate limit
            is_allowed, rate_info = await rate_limiter.is_allowed(
                identifier=f"ip:{client_ip}",
                limit=limit_config["requests"],
                window=limit_config["window"],
                endpoint=f"ip_{endpoint_type}"
            )
            
            return is_allowed, {
                "Limit": rate_info["limit"],
                "Remaining": rate_info["remaining"],
                "Reset": rate_info["reset"],
                "Window": limit_config["window"]
            }
            
        except Exception as e:
            logger.error(f"IP rate limit check error: {str(e)}")
            return True, {}
    
    def _create_rate_limit_response(self, headers: Dict[str, int]) -> JSONResponse:
        """Create rate limit exceeded response"""
        reset_time = headers.get("Reset", int(time.time()) + 3600)
        
        response = JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Try again after {reset_time}",
                "retry_after": reset_time
            }
        )
        
        # Add rate limit headers
        for key, value in headers.items():
            response.headers[f"X-RateLimit-{key}"] = str(value)
        
        response.headers["Retry-After"] = str(reset_time - int(time.time()))
        
        return response


class AIRateLimitException(HTTPException):
    """Custom exception for AI rate limit exceeded"""
    
    def __init__(self, remaining_quota: int, reset_time: int):
        super().__init__(
            status_code=429,
            detail={
                "error": "AI quota exceeded",
                "message": "You have exceeded your AI request quota for today",
                "remaining_quota": remaining_quota,
                "reset_time": reset_time,
                "upgrade_url": "/pricing"
            }
        )