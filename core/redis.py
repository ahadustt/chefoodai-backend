"""
ChefoodAI Redis Configuration
Async Redis setup for caching, sessions, and rate limiting
"""

try:
    from redis import asyncio as redis_async
except ImportError:
    redis_async = None
import json
import pickle
import structlog
from datetime import timedelta
from typing import Any, Optional, Union, Dict, List
from contextlib import asynccontextmanager

from core.config import settings

logger = structlog.get_logger()

# Global Redis connection pool
redis_pool = None


async def init_redis() -> None:
    """Initialize Redis connection pool"""
    global redis_pool
    
    if redis_async is None:
        logger.warning("redis async not available - Redis functionality disabled")
        return
    
    try:
        redis_pool = redis_async.ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_POOL_SIZE,
            retry_on_timeout=True,
            health_check_interval=30,
            decode_responses=False  # We'll handle encoding manually
        )
        
        # Test connection
        redis = redis_async.Redis(connection_pool=redis_pool)
        await redis.ping()
        await redis.close()
        
        logger.info("Redis connection pool initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {str(e)}")
        raise


async def close_redis() -> None:
    """Close Redis connection pool"""
    global redis_pool
    
    if redis_pool:
        await redis_pool.disconnect()
        logger.info("Redis connection pool closed")


@asynccontextmanager
async def get_redis():
    """Get Redis connection from pool"""
    if redis_async is None:
        logger.warning("redis async not available - returning None")
        yield None
        return
        
    if not redis_pool:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    
    redis = redis_async.Redis(connection_pool=redis_pool)
    try:
        yield redis
    finally:
        await redis.close()


class RedisCache:
    """Redis cache utility class with different serialization strategies"""
    
    def __init__(self):
        self.prefix = f"{settings.APP_NAME}:cache:"
    
    def _make_key(self, key: str, namespace: str = "default") -> str:
        """Create a namespaced cache key"""
        return f"{self.prefix}{namespace}:{key}"
    
    async def get_json(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get JSON serialized value from cache"""
        try:
            async with get_redis() as redis:
                value = await redis.get(self._make_key(key, namespace))
                if value:
                    return json.loads(value.decode('utf-8'))
                return None
        except Exception as e:
            logger.error(f"Redis get_json error: {str(e)}")
            return None
    
    async def set_json(
        self, 
        key: str, 
        value: Any, 
        ttl: int = settings.CACHE_TTL_MEDIUM,
        namespace: str = "default"
    ) -> bool:
        """Set JSON serialized value in cache"""
        try:
            async with get_redis() as redis:
                json_value = json.dumps(value, default=str)
                return await redis.setex(
                    self._make_key(key, namespace),
                    ttl,
                    json_value
                )
        except Exception as e:
            logger.error(f"Redis set_json error: {str(e)}")
            return False
    
    async def get_pickle(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get pickle serialized value from cache"""
        try:
            async with get_redis() as redis:
                value = await redis.get(self._make_key(key, namespace))
                if value:
                    return pickle.loads(value)
                return None
        except Exception as e:
            logger.error(f"Redis get_pickle error: {str(e)}")
            return None
    
    async def set_pickle(
        self, 
        key: str, 
        value: Any, 
        ttl: int = settings.CACHE_TTL_MEDIUM,
        namespace: str = "default"
    ) -> bool:
        """Set pickle serialized value in cache"""
        try:
            async with get_redis() as redis:
                pickled_value = pickle.dumps(value)
                return await redis.setex(
                    self._make_key(key, namespace),
                    ttl,
                    pickled_value
                )
        except Exception as e:
            logger.error(f"Redis set_pickle error: {str(e)}")
            return False
    
    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete key from cache"""
        try:
            async with get_redis() as redis:
                result = await redis.delete(self._make_key(key, namespace))
                return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {str(e)}")
            return False
    
    async def exists(self, key: str, namespace: str = "default") -> bool:
        """Check if key exists in cache"""
        try:
            async with get_redis() as redis:
                return await redis.exists(self._make_key(key, namespace))
        except Exception as e:
            logger.error(f"Redis exists error: {str(e)}")
            return False
    
    async def clear_namespace(self, namespace: str) -> bool:
        """Clear all keys in a namespace"""
        try:
            async with get_redis() as redis:
                pattern = f"{self.prefix}{namespace}:*"
                keys = await redis.keys(pattern)
                if keys:
                    return await redis.delete(*keys) > 0
                return True
        except Exception as e:
            logger.error(f"Redis clear_namespace error: {str(e)}")
            return False


class RedisRateLimiter:
    """Redis-based rate limiter"""
    
    def __init__(self):
        self.prefix = f"{settings.APP_NAME}:ratelimit:"
    
    def _make_key(self, identifier: str, endpoint: str = "global") -> str:
        """Create rate limit key"""
        return f"{self.prefix}{endpoint}:{identifier}"
    
    async def is_allowed(
        self,
        identifier: str,
        limit: int,
        window: int,
        endpoint: str = "global"
    ) -> tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed and return rate limit info
        Returns: (is_allowed, {"remaining": int, "reset": int, "limit": int})
        """
        try:
            async with get_redis() as redis:
                key = self._make_key(identifier, endpoint)
                
                # Use sliding window counter
                current_time = int(timedelta().total_seconds())
                window_start = current_time - window
                
                # Remove old entries
                await redis.zremrangebyscore(key, 0, window_start)
                
                # Count current requests
                current_requests = await redis.zcard(key)
                
                if current_requests < limit:
                    # Add current request
                    await redis.zadd(key, {str(current_time): current_time})
                    await redis.expire(key, window)
                    
                    remaining = limit - current_requests - 1
                    reset_time = current_time + window
                    
                    return True, {
                        "remaining": remaining,
                        "reset": reset_time,
                        "limit": limit
                    }
                else:
                    # Rate limit exceeded
                    oldest_score = await redis.zrange(key, 0, 0, withscores=True)
                    reset_time = int(oldest_score[0][1]) + window if oldest_score else current_time + window
                    
                    return False, {
                        "remaining": 0,
                        "reset": reset_time,
                        "limit": limit
                    }
                    
        except Exception as e:
            logger.error(f"Rate limiter error: {str(e)}")
            # Fail open - allow request if Redis is down
            return True, {"remaining": limit, "reset": 0, "limit": limit}


class RedisSession:
    """Redis-based session management"""
    
    def __init__(self):
        self.prefix = f"{settings.APP_NAME}:session:"
        self.default_ttl = 86400 * settings.REFRESH_TOKEN_EXPIRE_DAYS  # Same as refresh token
    
    def _make_key(self, session_id: str) -> str:
        """Create session key"""
        return f"{self.prefix}{session_id}"
    
    async def create_session(self, session_id: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Create a new session"""
        try:
            async with get_redis() as redis:
                ttl = ttl or self.default_ttl
                session_data = json.dumps(data, default=str)
                return await redis.setex(self._make_key(session_id), ttl, session_data)
        except Exception as e:
            logger.error(f"Session create error: {str(e)}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        try:
            async with get_redis() as redis:
                data = await redis.get(self._make_key(session_id))
                if data:
                    return json.loads(data.decode('utf-8'))
                return None
        except Exception as e:
            logger.error(f"Session get error: {str(e)}")
            return None
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update existing session data"""
        try:
            async with get_redis() as redis:
                key = self._make_key(session_id)
                ttl = await redis.ttl(key)
                if ttl > 0:
                    session_data = json.dumps(data, default=str)
                    return await redis.setex(key, ttl, session_data)
                return False
        except Exception as e:
            logger.error(f"Session update error: {str(e)}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        try:
            async with get_redis() as redis:
                return await redis.delete(self._make_key(session_id)) > 0
        except Exception as e:
            logger.error(f"Session delete error: {str(e)}")
            return False
    
    async def extend_session(self, session_id: str, ttl: Optional[int] = None) -> bool:
        """Extend session TTL"""
        try:
            async with get_redis() as redis:
                ttl = ttl or self.default_ttl
                return await redis.expire(self._make_key(session_id), ttl)
        except Exception as e:
            logger.error(f"Session extend error: {str(e)}")
            return False


class RedisHealthCheck:
    """Redis health check utilities"""
    
    @staticmethod
    async def check_connection() -> bool:
        """Check if Redis connection is healthy"""
        try:
            async with get_redis() as redis:
                await redis.ping()
                return True
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False
    
    @staticmethod
    async def get_info() -> Dict[str, Any]:
        """Get Redis server information"""
        try:
            async with get_redis() as redis:
                info = await redis.info()
                return {
                    "status": "healthy",
                    "version": info.get("redis_version"),
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "uptime": info.get("uptime_in_seconds")
                }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {str(e)}")
            return {"status": "error", "error": str(e)}


# Create global instances
cache = RedisCache()
rate_limiter = RedisRateLimiter() 
session_manager = RedisSession()

# Export commonly used items
__all__ = [
    "init_redis",
    "close_redis",
    "get_redis",
    "cache",
    "rate_limiter",
    "session_manager",
    "RedisHealthCheck"
]