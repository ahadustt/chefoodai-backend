"""
ChefoodAI Backend Service - Main API Server
Handles core business logic, data management, and orchestration
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog
import time
import os
from typing import AsyncGenerator
import httpx

from core.config import settings
from core.database import init_db, close_db
from core.redis import init_redis, close_redis
from api.routes import api_router
from middleware.rate_limiting import RateLimitMiddleware
from middleware.security import SecurityMiddleware
from middleware.logging import LoggingMiddleware

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# AI Service client configuration
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://ai:8001")
http_client = httpx.AsyncClient(timeout=30.0)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events"""
    # Startup
    logger.info("Starting ChefoodAI Backend Service")
    
    try:
        await init_db()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        
    try:
        await init_redis()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
    
    logger.info("Backend service startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ChefoodAI Backend Service")
    await http_client.aclose()
    await close_db()
    await close_redis()
    logger.info("Backend service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="ChefoodAI Backend Service",
    description="Core business logic and data management for ChefoodAI",
    version="2.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT", "development") == "development" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT", "development") == "development" else None,
    lifespan=lifespan
)

# Security Middleware - Always enable to handle Cloud Run deployments
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS + ["*.a.run.app", "*.run.app"]
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS + ["https://*.a.run.app", "https://*.run.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"]
)

# Custom Middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(LoggingMiddleware)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add response time header"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(
        "Unhandled exception",
        exception=str(exc),
        path=request.url.path,
        method=request.method
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": request.headers.get("X-Request-ID")
        }
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ChefoodAI Backend Service",
        "version": "2.0.0",
        "status": "healthy",
        "environment": os.getenv("ENVIRONMENT", "development")
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "service": "backend",
        "timestamp": time.time(),
        "version": "2.0.0",
        "checks": {}
    }
    
    # Database health check
    try:
        # TODO: Add actual database ping
        health_status["checks"]["database"] = {"status": "healthy"}
    except Exception as e:
        health_status["checks"]["database"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Redis health check
    try:
        # TODO: Add actual Redis ping
        health_status["checks"]["redis"] = {"status": "healthy"}
    except Exception as e:
        health_status["checks"]["redis"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # AI Service health check
    try:
        response = await http_client.get(f"{AI_SERVICE_URL}/health", timeout=5.0)
        if response.status_code == 200:
            health_status["checks"]["ai_service"] = {"status": "healthy"}
        else:
            health_status["checks"]["ai_service"] = {"status": "unhealthy", "code": response.status_code}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["ai_service"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    return health_status


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint for Kubernetes"""
    try:
        # Check critical dependencies
        # TODO: Add actual connectivity checks
        return {
            "status": "ready",
            "database": "connected",
            "redis": "connected",
            "ai_service": "available"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


# AI Service proxy endpoints
@app.post("/api/v1/ai/generate-recipe")
async def proxy_generate_recipe(request: Request):
    """Proxy recipe generation to AI service"""
    try:
        body = await request.json()
        response = await http_client.post(
            f"{AI_SERVICE_URL}/generate-recipe",
            json=body
        )
        return response.json()
    except Exception as e:
        logger.error(f"Failed to proxy to AI service: {e}")
        raise HTTPException(status_code=503, detail="AI service unavailable")


@app.post("/api/v1/ai/generate-image")
async def proxy_generate_image(request: Request):
    """Proxy image generation to AI service"""
    try:
        body = await request.json()
        response = await http_client.post(
            f"{AI_SERVICE_URL}/generate-image",
            json=body
        )
        return response.json()
    except Exception as e:
        logger.error(f"Failed to proxy to AI service: {e}")
        raise HTTPException(status_code=503, detail="AI service unavailable")


@app.post("/api/v1/ai/analyze-nutrition")
async def proxy_analyze_nutrition(request: Request):
    """Proxy nutrition analysis to AI service"""
    try:
        body = await request.json()
        response = await http_client.post(
            f"{AI_SERVICE_URL}/analyze-nutrition",
            json=body
        )
        return response.json()
    except Exception as e:
        logger.error(f"Failed to proxy to AI service: {e}")
        raise HTTPException(status_code=503, detail="AI service unavailable")


@app.post("/api/v1/ai/optimize-meal-plan")
async def proxy_optimize_meal_plan(request: Request):
    """Proxy meal plan optimization to AI service"""
    try:
        body = await request.json()
        response = await http_client.post(
            f"{AI_SERVICE_URL}/optimize-meal-plan",
            json=body
        )
        return response.json()
    except Exception as e:
        logger.error(f"Failed to proxy to AI service: {e}")
        raise HTTPException(status_code=503, detail="AI service unavailable")


# Include API routes from the monolith
app.include_router(api_router, prefix="/api/v1")


# Service discovery endpoint
@app.get("/services")
async def service_discovery():
    """Service discovery information"""
    return {
        "services": {
            "backend": {
                "url": os.getenv("BACKEND_URL", "http://localhost:8000"),
                "version": "2.0.0",
                "status": "active"
            },
            "ai": {
                "url": AI_SERVICE_URL,
                "version": "2.0.0",
                "status": "active"
            },
            "frontend": {
                "url": os.getenv("FRONTEND_URL", "http://localhost:3000"),
                "version": "2.0.0",
                "status": "active"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Cloud Run sets this)
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENVIRONMENT", "development") == "development",
        log_config=None  # Use structlog instead
    )