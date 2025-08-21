"""
ChefoodAI Backend Service - Simplified Main API Server
Minimal configuration for Cloud Run deployment debugging
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import time

# Create FastAPI app
app = FastAPI(
    title="ChefoodAI Backend",
    version="1.0.0",
    description="Backend API for ChefoodAI"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ChefoodAI Backend",
        "status": "running",
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ChefoodAI Backend",
        "port": os.environ.get("PORT", 8000),
        "environment": os.environ.get("ENVIRONMENT", "production"),
        "timestamp": time.time()
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    return {
        "status": "ready",
        "service": "ChefoodAI Backend",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)