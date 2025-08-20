"""
ChefoodAI Monitoring and Health Check Endpoints
Production monitoring, health checks, and metrics exposure
"""

import asyncio
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from sqlalchemy import text
import redis
from prometheus_client import CONTENT_TYPE_LATEST

from core.database import get_db
from core.monitoring import (
    get_metrics, health_check,
    active_users_total, db_connections_active,
    memory_usage_percent, cpu_usage_percent,
    external_service_up, daily_cost,
    record_business_metric
)
from core.config import get_settings
from core.dependencies import CurrentUser, AdminUser
from services.ai_service import ai_service

settings = get_settings()
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

@router.get("/health", include_in_schema=False)
async def health_endpoint():
    """
    Basic health check endpoint for load balancers and uptime monitoring
    Returns 200 OK if service is healthy
    """
    try:
        health_data = health_check()
        return health_data
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@router.get("/health/detailed")
async def detailed_health_check(
    db: Session = Depends(get_db),
    current_user: AdminUser = Depends()
):
    """
    Comprehensive health check with dependency status
    Requires admin authentication
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "checks": {}
    }
    
    overall_healthy = True
    
    # Database Health Check
    try:
        start_time = time.time()
        db.execute(text("SELECT 1"))
        db_latency = (time.time() - start_time) * 1000
        
        # Check connection pool
        active_connections = db.execute(
            text("SELECT count(*) FROM pg_stat_activity WHERE datname = :db_name"),
            {"db_name": settings.DATABASE_NAME}
        ).scalar()
        
        health_status["checks"]["database"] = {
            "status": "healthy",
            "latency_ms": round(db_latency, 2),
            "active_connections": active_connections,
            "max_connections": 100  # From your DB config
        }
        
        # Update metrics
        db_connections_active.set(active_connections)
        
    except Exception as e:
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_healthy = False
    
    # Redis Health Check
    try:
        if settings.REDIS_URL:
            import redis
            redis_client = redis.from_url(settings.REDIS_URL)
            
            start_time = time.time()
            redis_client.ping()
            redis_latency = (time.time() - start_time) * 1000
            
            redis_info = redis_client.info()
            
            health_status["checks"]["redis"] = {
                "status": "healthy",
                "latency_ms": round(redis_latency, 2),
                "connected_clients": redis_info.get("connected_clients", 0),
                "used_memory": redis_info.get("used_memory_human", "0B"),
                "uptime_seconds": redis_info.get("uptime_in_seconds", 0)
            }
        else:
            health_status["checks"]["redis"] = {
                "status": "disabled",
                "note": "Redis URL not configured"
            }
            
    except Exception as e:
        health_status["checks"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_healthy = False
    
    # AI Service Health Check
    try:
        # Test AI service with a simple request
        start_time = time.time()
        test_response = await ai_service.health_check()
        ai_latency = (time.time() - start_time) * 1000
        
        health_status["checks"]["ai_service"] = {
            "status": "healthy" if test_response.get("status") == "ok" else "degraded",
            "latency_ms": round(ai_latency, 2),
            "model_status": test_response.get("models", {}),
            "quota_remaining": test_response.get("quota_remaining", "unknown")
        }
        
        # Update external service metric
        external_service_up.labels(service="vertex_ai").set(1)
        
    except Exception as e:
        health_status["checks"]["ai_service"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        external_service_up.labels(service="vertex_ai").set(0)
        overall_healthy = False
    
    # System Resource Check
    try:
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_info = psutil.disk_usage('/')
        
        health_status["checks"]["system"] = {
            "status": "healthy",
            "memory": {
                "total": memory_info.total,
                "available": memory_info.available,
                "percent": memory_info.percent,
                "status": "healthy" if memory_info.percent < 85 else "warning"
            },
            "cpu": {
                "percent": cpu_percent,
                "status": "healthy" if cpu_percent < 80 else "warning"
            },
            "disk": {
                "total": disk_info.total,
                "free": disk_info.free,
                "percent": round((disk_info.used / disk_info.total) * 100, 2),
                "status": "healthy" if (disk_info.used / disk_info.total) < 0.85 else "warning"
            }
        }
        
        # Update system metrics
        memory_usage_percent.labels(service="backend").set(memory_info.percent)
        cpu_usage_percent.labels(service="backend").set(cpu_percent)
        
    except Exception as e:
        health_status["checks"]["system"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Set overall status
    health_status["status"] = "healthy" if overall_healthy else "unhealthy"
    
    # Return appropriate HTTP status
    if not overall_healthy:
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status

@router.get("/metrics", include_in_schema=False)
async def metrics_endpoint():
    """
    Prometheus metrics endpoint
    Returns metrics in Prometheus exposition format
    """
    try:
        # Update real-time metrics before serving
        await update_runtime_metrics()
        
        metrics_data = get_metrics()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST,
            headers={"Cache-Control": "no-cache"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate metrics: {str(e)}")

@router.get("/status")
async def service_status():
    """
    Service status page information
    Public endpoint for status page
    """
    return {
        "service": "ChefoodAI Backend",
        "status": "operational",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": get_service_uptime(),
        "region": settings.GOOGLE_CLOUD_REGION or "us-central1"
    }

@router.get("/dashboard")
async def monitoring_dashboard(current_user: AdminUser = Depends()):
    """
    Admin monitoring dashboard data
    Comprehensive metrics for admin interface
    """
    try:
        # Get system metrics
        system_metrics = await get_system_metrics()
        
        # Get business metrics
        business_metrics = await get_business_metrics()
        
        # Get recent alerts
        recent_alerts = await get_recent_alerts()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": system_metrics,
            "business": business_metrics,
            "alerts": recent_alerts,
            "costs": {
                "daily": daily_cost._value._value,
                "monthly_estimate": daily_cost._value._value * 30,
                "budget_utilization": (daily_cost._value._value / 200) * 100  # $200 daily budget
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard data: {str(e)}")

@router.post("/alerts/webhook", include_in_schema=False)
async def alertmanager_webhook(alert_data: Dict[str, Any]):
    """
    Webhook endpoint for Alertmanager
    Receives and processes alert notifications
    """
    try:
        alerts = alert_data.get("alerts", [])
        
        for alert in alerts:
            alert_name = alert.get("labels", {}).get("alertname", "Unknown")
            severity = alert.get("labels", {}).get("severity", "unknown")
            status = alert.get("status", "unknown")
            
            # Log alert for audit trail
            print(f"Alert received: {alert_name} - {severity} - {status}")
            
            # Process specific alert types
            if alert_name == "ServiceDown":
                await handle_service_down_alert(alert)
            elif alert_name == "HighErrorRate":
                await handle_high_error_rate_alert(alert)
            elif severity == "critical":
                await handle_critical_alert(alert)
        
        return {"status": "received", "processed_alerts": len(alerts)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process alerts: {str(e)}")

@router.get("/logs/search")
async def search_logs(
    query: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
    current_user: AdminUser = Depends()
):
    """
    Search application logs
    Admin endpoint for log analysis
    """
    try:
        # This would integrate with your logging system (Loki, ELK, etc.)
        # For now, return a placeholder response
        
        logs = {
            "query": query,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "total_matches": 0,
            "logs": [],
            "note": "Log search not yet implemented - integrate with Loki/ELK"
        }
        
        return logs
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Log search failed: {str(e)}")

@router.post("/maintenance")
async def toggle_maintenance_mode(
    enable: bool,
    reason: str = "Scheduled maintenance",
    current_user: AdminUser = Depends()
):
    """
    Toggle maintenance mode
    Admin endpoint to enable/disable maintenance mode
    """
    try:
        # This would integrate with your load balancer or feature flags
        maintenance_status = {
            "maintenance_mode": enable,
            "reason": reason,
            "enabled_by": current_user.email,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store maintenance status (Redis, database, etc.)
        # For now, just return the status
        
        return maintenance_status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to toggle maintenance mode: {str(e)}")

# Helper Functions
async def update_runtime_metrics():
    """Update metrics that need real-time calculation"""
    try:
        # Update active users (would query from your session store)
        # active_users_total.set(get_active_user_count())
        
        # Update cost metrics (would query from billing APIs)
        # update_cost_metrics()
        
        pass
    except Exception as e:
        print(f"Failed to update runtime metrics: {e}")

async def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics"""
    memory_info = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    return {
        "memory_percent": memory_info.percent,
        "cpu_percent": cpu_percent,
        "active_connections": db_connections_active._value._value,
        "active_users": active_users_total._value._value
    }

async def get_business_metrics() -> Dict[str, Any]:
    """Get business-related metrics"""
    return {
        "recipes_generated_today": 150,  # Would query from database
        "meal_plans_created_today": 45,  # Would query from database
        "new_users_today": 12,           # Would query from database
        "revenue_today": 450.00          # Would query from billing
    }

async def get_recent_alerts() -> List[Dict[str, Any]]:
    """Get recent alerts"""
    # This would query from Alertmanager API or database
    return [
        {
            "name": "HighLatency",
            "severity": "warning",
            "timestamp": (datetime.utcnow() - timedelta(minutes=15)).isoformat(),
            "status": "resolved"
        }
    ]

def get_service_uptime() -> str:
    """Calculate service uptime"""
    # This would track actual service start time
    # For now, return a placeholder
    return "99.9% (last 30 days)"

async def handle_service_down_alert(alert: Dict[str, Any]):
    """Handle service down alerts"""
    service_name = alert.get("labels", {}).get("service", "unknown")
    print(f"SERVICE DOWN: {service_name}")
    # Add specific handling logic here

async def handle_high_error_rate_alert(alert: Dict[str, Any]):
    """Handle high error rate alerts"""
    service_name = alert.get("labels", {}).get("service", "unknown")
    error_rate = alert.get("labels", {}).get("error_rate", "unknown")
    print(f"HIGH ERROR RATE: {service_name} - {error_rate}")
    # Add specific handling logic here

async def handle_critical_alert(alert: Dict[str, Any]):
    """Handle critical alerts"""
    alert_name = alert.get("labels", {}).get("alertname", "unknown")
    print(f"CRITICAL ALERT: {alert_name}")
    # Add escalation logic here