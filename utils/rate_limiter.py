"""
ChefoodAI Rate Limiter
Advanced rate limiting with Redis backend for security protection
"""

import asyncio
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import json
import redis.asyncio as redis
from core.config import get_settings

settings = get_settings()

class RateLimiter:
    def __init__(self):
        self.redis_client = None
        self._memory_store: Dict[str, Dict[str, Any]] = {}
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    async def _get_redis_client(self):
        """Get or create Redis client"""
        if self.redis_client is None:
            try:
                self.redis_client = redis.Redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                # Test connection
                await self.redis_client.ping()
            except Exception as e:
                print(f"Redis connection failed, using memory store: {e}")
                self.redis_client = None
        return self.redis_client
    
    async def check_rate_limit(
        self,
        key: str,
        max_attempts: int,
        window_minutes: int,
        identifier: Optional[str] = None
    ) -> bool:
        """
        Check if request is within rate limit
        
        Args:
            key: Unique identifier for the rate limit (e.g., "login:user@email.com")
            max_attempts: Maximum attempts allowed in the window
            window_minutes: Time window in minutes
            identifier: Optional additional identifier
            
        Returns:
            True if within limit, False if exceeded
        """
        try:
            redis_client = await self._get_redis_client()
            
            if redis_client:
                return await self._check_rate_limit_redis(
                    redis_client, key, max_attempts, window_minutes, identifier
                )
            else:
                return await self._check_rate_limit_memory(
                    key, max_attempts, window_minutes, identifier
                )
                
        except Exception as e:
            print(f"Rate limiting error: {e}")
            # On error, allow the request (fail open)
            return True
    
    async def _check_rate_limit_redis(
        self,
        redis_client: redis.Redis,
        key: str,
        max_attempts: int,
        window_minutes: int,
        identifier: Optional[str] = None
    ) -> bool:
        """Redis-based rate limiting using sliding window"""
        
        # Create full key with identifier if provided
        full_key = f"rate_limit:{key}"
        if identifier:
            full_key += f":{identifier}"
        
        current_time = int(time.time())
        window_start = current_time - (window_minutes * 60)
        
        # Use Redis pipeline for atomic operations
        pipe = redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(full_key, 0, window_start)
        
        # Count current entries
        pipe.zcard(full_key)
        
        # Add current request
        pipe.zadd(full_key, {str(current_time): current_time})
        
        # Set expiration
        pipe.expire(full_key, window_minutes * 60 + 60)  # Extra minute for cleanup
        
        results = await pipe.execute()
        current_count = results[1]  # Count after cleanup
        
        return current_count < max_attempts
    
    async def _check_rate_limit_memory(
        self,
        key: str,
        max_attempts: int,
        window_minutes: int,
        identifier: Optional[str] = None
    ) -> bool:
        """Memory-based rate limiting (fallback)"""
        
        # Periodic cleanup
        await self._cleanup_memory_store()
        
        # Create full key
        full_key = key
        if identifier:
            full_key += f":{identifier}"
        
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        # Get or create entry
        if full_key not in self._memory_store:
            self._memory_store[full_key] = {
                'attempts': [],
                'created_at': current_time
            }
        
        entry = self._memory_store[full_key]
        
        # Remove old attempts
        entry['attempts'] = [
            attempt_time for attempt_time in entry['attempts']
            if attempt_time > window_start
        ]
        
        # Check if within limit
        if len(entry['attempts']) >= max_attempts:
            return False
        
        # Add current attempt
        entry['attempts'].append(current_time)
        
        return True
    
    async def _cleanup_memory_store(self):
        """Clean up old entries from memory store"""
        current_time = time.time()
        
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        # Remove entries older than 1 hour
        cutoff_time = current_time - 3600
        keys_to_remove = []
        
        for key, entry in self._memory_store.items():
            if entry['created_at'] < cutoff_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._memory_store[key]
        
        self._last_cleanup = current_time
    
    async def get_rate_limit_status(
        self,
        key: str,
        window_minutes: int,
        identifier: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current rate limit status
        
        Returns:
            Dictionary with current count, window info, etc.
        """
        try:
            redis_client = await self._get_redis_client()
            
            if redis_client:
                return await self._get_status_redis(
                    redis_client, key, window_minutes, identifier
                )
            else:
                return await self._get_status_memory(
                    key, window_minutes, identifier
                )
                
        except Exception as e:
            print(f"Rate limit status error: {e}")
            return {
                'current_count': 0,
                'window_minutes': window_minutes,
                'window_start': datetime.utcnow(),
                'reset_time': datetime.utcnow() + timedelta(minutes=window_minutes)
            }
    
    async def _get_status_redis(
        self,
        redis_client: redis.Redis,
        key: str,
        window_minutes: int,
        identifier: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get rate limit status from Redis"""
        
        full_key = f"rate_limit:{key}"
        if identifier:
            full_key += f":{identifier}"
        
        current_time = int(time.time())
        window_start = current_time - (window_minutes * 60)
        
        # Clean up old entries and get count
        pipe = redis_client.pipeline()
        pipe.zremrangebyscore(full_key, 0, window_start)
        pipe.zcard(full_key)
        
        results = await pipe.execute()
        current_count = results[1]
        
        return {
            'current_count': current_count,
            'window_minutes': window_minutes,
            'window_start': datetime.fromtimestamp(window_start),
            'reset_time': datetime.fromtimestamp(current_time + (window_minutes * 60))
        }
    
    async def _get_status_memory(
        self,
        key: str,
        window_minutes: int,
        identifier: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get rate limit status from memory"""
        
        full_key = key
        if identifier:
            full_key += f":{identifier}"
        
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        current_count = 0
        if full_key in self._memory_store:
            entry = self._memory_store[full_key]
            # Count attempts in current window
            current_count = len([
                attempt_time for attempt_time in entry['attempts']
                if attempt_time > window_start
            ])
        
        return {
            'current_count': current_count,
            'window_minutes': window_minutes,
            'window_start': datetime.fromtimestamp(window_start),
            'reset_time': datetime.fromtimestamp(current_time + (window_minutes * 60))
        }
    
    async def reset_rate_limit(
        self,
        key: str,
        identifier: Optional[str] = None
    ) -> bool:
        """
        Reset rate limit for a specific key
        
        Args:
            key: Rate limit key to reset
            identifier: Optional identifier
            
        Returns:
            True if reset successful
        """
        try:
            redis_client = await self._get_redis_client()
            
            if redis_client:
                full_key = f"rate_limit:{key}"
                if identifier:
                    full_key += f":{identifier}"
                
                await redis_client.delete(full_key)
                return True
            else:
                full_key = key
                if identifier:
                    full_key += f":{identifier}"
                
                if full_key in self._memory_store:
                    del self._memory_store[full_key]
                return True
                
        except Exception as e:
            print(f"Rate limit reset error: {e}")
            return False
    
    async def add_to_whitelist(
        self,
        key: str,
        expiry_minutes: int = 60
    ) -> bool:
        """
        Add key to whitelist (bypasses rate limiting)
        
        Args:
            key: Key to whitelist
            expiry_minutes: How long to whitelist (minutes)
            
        Returns:
            True if added successfully
        """
        try:
            redis_client = await self._get_redis_client()
            
            if redis_client:
                whitelist_key = f"whitelist:{key}"
                await redis_client.setex(
                    whitelist_key, 
                    expiry_minutes * 60, 
                    "1"
                )
                return True
            else:
                # For memory store, we'd need a separate whitelist dict
                # Simplified implementation for now
                return True
                
        except Exception as e:
            print(f"Whitelist add error: {e}")
            return False
    
    async def is_whitelisted(self, key: str) -> bool:
        """
        Check if key is whitelisted
        
        Args:
            key: Key to check
            
        Returns:
            True if whitelisted
        """
        try:
            redis_client = await self._get_redis_client()
            
            if redis_client:
                whitelist_key = f"whitelist:{key}"
                result = await redis_client.get(whitelist_key)
                return result is not None
            else:
                return False
                
        except Exception as e:
            print(f"Whitelist check error: {e}")
            return False
    
    async def get_rate_limit_info(
        self,
        key: str,
        max_attempts: int,
        window_minutes: int,
        identifier: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive rate limit information
        
        Returns detailed information about rate limit status
        """
        try:
            # Check if whitelisted
            is_whitelisted = await self.is_whitelisted(key)
            if is_whitelisted:
                return {
                    'whitelisted': True,
                    'current_count': 0,
                    'max_attempts': max_attempts,
                    'remaining_attempts': max_attempts,
                    'window_minutes': window_minutes,
                    'reset_time': None,
                    'blocked': False
                }
            
            # Get current status
            status = await self.get_rate_limit_status(key, window_minutes, identifier)
            current_count = status['current_count']
            
            return {
                'whitelisted': False,
                'current_count': current_count,
                'max_attempts': max_attempts,
                'remaining_attempts': max(0, max_attempts - current_count),
                'window_minutes': window_minutes,
                'window_start': status['window_start'],
                'reset_time': status['reset_time'],
                'blocked': current_count >= max_attempts
            }
            
        except Exception as e:
            print(f"Rate limit info error: {e}")
            return {
                'error': str(e),
                'whitelisted': False,
                'current_count': 0,
                'max_attempts': max_attempts,
                'remaining_attempts': max_attempts,
                'window_minutes': window_minutes,
                'blocked': False
            }
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()

# Create singleton instance
rate_limiter = RateLimiter()