"""
ChefoodAI Health Check Endpoints
System health monitoring and diagnostics
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import asyncio
import time
from typing import Dict, Any

from core.database import DatabaseHealthCheck
from core.redis import RedisHealthCheck
from core.config import settings

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "ChefoodAI",
        "version": settings.VERSION,
        "timestamp": time.time()
    }


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint"""
    return {"status": "alive"}


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint
    Checks all critical dependencies
    """
    try:
        # Run health checks in parallel
        db_task = asyncio.create_task(DatabaseHealthCheck.check_connection())
        redis_task = asyncio.create_task(RedisHealthCheck.check_connection())
        
        # Wait for all checks with timeout
        db_healthy, redis_healthy = await asyncio.wait_for(
            asyncio.gather(db_task, redis_task),
            timeout=5.0
        )
        
        if db_healthy and redis_healthy:
            return {
                "status": "ready",
                "database": "connected",
                "redis": "connected",
                "timestamp": time.time()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "not_ready",
                    "database": "connected" if db_healthy else "disconnected",
                    "redis": "connected" if redis_healthy else "disconnected"
                }
            )
            
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "error": "Health check timeout"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "error": str(e)}
        )


@router.get("/detailed")
async def detailed_health_check():
    """
    Detailed health check with comprehensive system information
    Only available in development environment
    """
    if not settings.is_development:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Endpoint not available in production"
        )
    
    try:
        # Get detailed information from all systems
        db_info_task = asyncio.create_task(DatabaseHealthCheck.get_connection_info())
        redis_info_task = asyncio.create_task(RedisHealthCheck.get_info())
        
        db_info, redis_info = await asyncio.gather(db_info_task, redis_info_task)
        
        return {
            "status": "healthy",
            "service": "ChefoodAI",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "timestamp": time.time(),
            "components": {
                "database": db_info,
                "redis": redis_info,
                "ai_service": {
                    "status": "available",
                    "model": settings.GEMINI_MODEL,
                    "fallback_model": settings.GEMINI_MODEL_FALLBACK
                }
            },
            "configuration": {
                "debug": settings.DEBUG,
                "ai_enabled": settings.ENABLE_PREMIUM_FEATURES,
                "multimodal_enabled": settings.ENABLE_MULTIMODAL_AI,
                "cache_ttl": {
                    "short": settings.CACHE_TTL_SHORT,
                    "medium": settings.CACHE_TTL_MEDIUM,
                    "long": settings.CACHE_TTL_LONG
                }
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )


@router.get("/metrics")
async def metrics_endpoint():
    """
    Prometheus-compatible metrics endpoint
    Returns basic application metrics
    """
    if not settings.ENABLE_METRICS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics not enabled"
        )
    
    # TODO: Implement Prometheus metrics collection
    # This would integrate with prometheus_client library
    
    return {
        "status": "metrics_available",
        "endpoint": f"http://localhost:{settings.METRICS_PORT}/metrics",
        "note": "Full Prometheus metrics available on dedicated port"
    }


@router.get("/version")
async def version_info():
    """Application version information"""
    return {
        "service": "ChefoodAI", 
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "python_version": "3.11+",
        "fastapi_version": "0.104+",
        "features": {
            "premium_features": settings.ENABLE_PREMIUM_FEATURES,
            "multimodal_ai": settings.ENABLE_MULTIMODAL_AI,
            "social_features": settings.ENABLE_SOCIAL_FEATURES,
            "marketplace": settings.ENABLE_MARKETPLACE
        }
    }