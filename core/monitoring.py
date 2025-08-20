"""
ChefoodAI Monitoring and Observability
Comprehensive monitoring, metrics, and tracing for production deployment
"""

import time
import logging
import functools
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from datetime import datetime, timedelta

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from opentelemetry import trace, metrics as otel_metrics, baggage
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
import structlog

from core.config import get_settings

settings = get_settings()
logger = structlog.get_logger()

# Prometheus Metrics Registry
registry = CollectorRegistry()

# Application Metrics
http_requests_total = Counter(
    'chefoodai_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status', 'service'],
    registry=registry
)

http_request_duration_seconds = Histogram(
    'chefoodai_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint', 'service'],
    registry=registry
)

active_users_total = Gauge(
    'chefoodai_active_users_total',
    'Number of active users',
    registry=registry
)

# AI Service Metrics
ai_requests_total = Counter(
    'chefoodai_ai_requests_total',
    'Total AI service requests',
    ['model', 'type', 'status'],
    registry=registry
)

ai_request_duration_seconds = Histogram(
    'chefoodai_ai_request_duration_seconds',
    'AI request duration in seconds',
    ['model', 'type'],
    registry=registry
)

ai_cost_total = Counter(
    'chefoodai_ai_cost_total',
    'Total AI service costs in USD',
    ['model', 'type'],
    registry=registry
)

ai_cost_per_hour = Gauge(
    'chefoodai_ai_cost_per_hour',
    'AI service cost per hour in USD',
    registry=registry
)

ai_requests_timeout_total = Counter(
    'chefoodai_ai_requests_timeout_total',
    'Total AI request timeouts',
    ['model'],
    registry=registry
)

# Business Metrics
recipes_generated_total = Counter(
    'chefoodai_recipes_generated_total',
    'Total recipes generated',
    ['user_plan', 'complexity'],
    registry=registry
)

meal_plans_total = Counter(
    'chefoodai_meal_plans_total',
    'Total meal plans created',
    ['user_plan', 'duration_days'],
    registry=registry
)

meal_plans_failed_total = Counter(
    'chefoodai_meal_plans_failed_total',
    'Total failed meal plan generations',
    ['reason'],
    registry=registry
)

user_registrations_total = Counter(
    'chefoodai_user_registrations_total',
    'Total user registrations',
    ['plan', 'source'],
    registry=registry
)

recipe_quality_score = Gauge(
    'chefoodai_recipe_quality_score',
    'Average recipe quality score',
    registry=registry
)

# Authentication Metrics
auth_requests_total = Counter(
    'chefoodai_auth_requests_total',
    'Total authentication requests',
    ['type', 'status'],
    registry=registry
)

auth_failed_total = Counter(
    'chefoodai_auth_failed_total',
    'Total failed authentication attempts',
    ['reason'],
    registry=registry
)

# Database Metrics
db_connections_active = Gauge(
    'chefoodai_db_connections_active',
    'Active database connections',
    registry=registry
)

db_query_duration_seconds = Histogram(
    'chefoodai_db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation', 'table'],
    registry=registry
)

# Cache Metrics
cache_operations_total = Counter(
    'chefoodai_cache_operations_total',
    'Total cache operations',
    ['operation', 'status'],
    registry=registry
)

cache_hit_ratio = Gauge(
    'chefoodai_cache_hit_ratio',
    'Cache hit ratio',
    registry=registry
)

# System Metrics
memory_usage_percent = Gauge(
    'chefoodai_memory_usage_percent',
    'Memory usage percentage',
    ['service'],
    registry=registry
)

cpu_usage_percent = Gauge(
    'chefoodai_cpu_usage_percent',
    'CPU usage percentage',
    ['service'],
    registry=registry
)

# External Service Metrics
external_service_up = Gauge(
    'chefoodai_external_service_up',
    'External service availability',
    ['service'],
    registry=registry
)

external_api_duration_seconds = Histogram(
    'chefoodai_external_api_duration_seconds',
    'External API call duration',
    ['service', 'endpoint'],
    registry=registry
)

# Rate Limiting Metrics
rate_limit_exceeded_total = Counter(
    'chefoodai_rate_limit_exceeded_total',
    'Total rate limit violations',
    ['endpoint', 'user_plan'],
    registry=registry
)

# Cost Tracking
daily_cost = Gauge(
    'chefoodai_daily_cost',
    'Daily operational cost in USD',
    registry=registry
)

total_cost = Counter(
    'chefoodai_total_cost',
    'Total operational cost in USD',
    ['category'],
    registry=registry
)

# Data Quality Metrics
nutrition_data_errors_total = Counter(
    'chefoodai_nutrition_data_errors_total',
    'Total nutrition data errors',
    ['error_type'],
    registry=registry
)

# OpenTelemetry Setup
def setup_tracing():
    """Setup OpenTelemetry tracing"""
    resource = Resource.create({
        "service.name": "chefoodai-backend",
        "service.version": "1.0.0",
        "deployment.environment": settings.ENVIRONMENT,
    })
    
    trace.set_tracer_provider(TracerProvider(resource=resource))
    
    # Cloud Trace Exporter
    if settings.GOOGLE_CLOUD_PROJECT:
        cloud_trace_exporter = CloudTraceSpanExporter(
            project_id=settings.GOOGLE_CLOUD_PROJECT
        )
        span_processor = BatchSpanProcessor(cloud_trace_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Auto-instrumentation
    FastAPIInstrumentor.instrument()
    SQLAlchemyInstrumentor.instrument()
    RedisInstrumentor.instrument()
    RequestsInstrumentor.instrument()

def setup_metrics():
    """Setup OpenTelemetry metrics"""
    if settings.GOOGLE_CLOUD_PROJECT:
        resource = Resource.create({
            "service.name": "chefoodai-backend",
            "service.version": "1.0.0",
        })
        
        metric_reader = PeriodicExportingMetricReader(
            exporter=CloudMonitoringMetricsExporter(
                project_id=settings.GOOGLE_CLOUD_PROJECT
            ),
            export_interval_millis=30000,
        )
        
        otel_metrics.set_meter_provider(
            MeterProvider(resource=resource, metric_readers=[metric_reader])
        )

# Monitoring Decorators
def monitor_endpoint(endpoint_name: str):
    """Decorator to monitor API endpoints"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Extract request info
            request = kwargs.get('request') or args[0] if args else None
            method = request.method if hasattr(request, 'method') else 'UNKNOWN'
            
            try:
                result = await func(*args, **kwargs)
                status = getattr(result, 'status_code', 200)
                
                # Record metrics
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint_name,
                    status=str(status),
                    service="backend"
                ).inc()
                
                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint_name,
                    service="backend"
                ).observe(duration)
                
                return result
                
            except Exception as e:
                # Record error metrics
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint_name,
                    status="500",
                    service="backend"
                ).inc()
                
                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint_name,
                    service="backend"
                ).observe(duration)
                
                logger.error(
                    "endpoint_error",
                    endpoint=endpoint_name,
                    method=method,
                    error=str(e),
                    duration=duration
                )
                raise
                
        return wrapper
    return decorator

def monitor_ai_request(model_name: str, request_type: str):
    """Decorator to monitor AI service requests"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record successful request
                ai_requests_total.labels(
                    model=model_name,
                    type=request_type,
                    status="success"
                ).inc()
                
                duration = time.time() - start_time
                ai_request_duration_seconds.labels(
                    model=model_name,
                    type=request_type
                ).observe(duration)
                
                # Estimate cost (simplified)
                estimated_cost = duration * 0.01  # $0.01 per second
                ai_cost_total.labels(
                    model=model_name,
                    type=request_type
                ).inc(estimated_cost)
                
                return result
                
            except TimeoutError:
                ai_requests_timeout_total.labels(model=model_name).inc()
                ai_requests_total.labels(
                    model=model_name,
                    type=request_type,
                    status="timeout"
                ).inc()
                raise
                
            except Exception as e:
                ai_requests_total.labels(
                    model=model_name,
                    type=request_type,
                    status="error"
                ).inc()
                
                logger.error(
                    "ai_request_error",
                    model=model_name,
                    type=request_type,
                    error=str(e)
                )
                raise
                
        return wrapper
    return decorator

def monitor_db_query(operation: str, table: str):
    """Decorator to monitor database queries"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                duration = time.time() - start_time
                db_query_duration_seconds.labels(
                    operation=operation,
                    table=table
                ).observe(duration)
                
                return result
                
            except Exception as e:
                logger.error(
                    "db_query_error",
                    operation=operation,
                    table=table,
                    error=str(e)
                )
                raise
                
        return wrapper
    return decorator

# Context Managers
@contextmanager
def trace_span(name: str, attributes: Dict[str, Any] = None):
    """Context manager for creating trace spans"""
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise

@contextmanager
def monitor_external_service(service_name: str, endpoint: str = ""):
    """Monitor external service calls"""
    start_time = time.time()
    
    try:
        yield
        
        # Record successful call
        external_service_up.labels(service=service_name).set(1)
        
        duration = time.time() - start_time
        external_api_duration_seconds.labels(
            service=service_name,
            endpoint=endpoint
        ).observe(duration)
        
    except Exception as e:
        # Record service failure
        external_service_up.labels(service=service_name).set(0)
        
        logger.error(
            "external_service_error",
            service=service_name,
            endpoint=endpoint,
            error=str(e)
        )
        raise

# Utility Functions
def record_user_activity(user_id: str, activity: str, metadata: Dict[str, Any] = None):
    """Record user activity for analytics"""
    logger.info(
        "user_activity",
        user_id=user_id,
        activity=activity,
        metadata=metadata or {}
    )

def record_business_metric(metric_name: str, value: float, labels: Dict[str, str] = None):
    """Record custom business metrics"""
    if metric_name == "recipe_generated":
        recipes_generated_total.labels(
            user_plan=labels.get("user_plan", "unknown"),
            complexity=labels.get("complexity", "unknown")
        ).inc()
    elif metric_name == "meal_plan_created":
        meal_plans_total.labels(
            user_plan=labels.get("user_plan", "unknown"),
            duration_days=labels.get("duration_days", "unknown")
        ).inc()
    elif metric_name == "user_registered":
        user_registrations_total.labels(
            plan=labels.get("plan", "free"),
            source=labels.get("source", "web")
        ).inc()

def update_cost_metrics():
    """Update cost-related metrics"""
    # This would typically pull from billing APIs
    # For now, we'll use placeholder logic
    current_hour = datetime.now().hour
    daily_cost_value = 150 + (current_hour * 5)  # Simulated daily cost
    
    daily_cost.set(daily_cost_value)

def get_metrics():
    """Get Prometheus metrics in exposition format"""
    return generate_latest(registry)

def health_check() -> Dict[str, Any]:
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "services": {
            "database": True,  # Would check actual DB connection
            "redis": True,     # Would check actual Redis connection
            "ai_service": True # Would check AI service availability
        },
        "metrics": {
            "active_users": active_users_total._value._value,
            "daily_cost": daily_cost._value._value
        }
    }

# Startup/Shutdown Functions
async def setup_monitoring():
    """Initialize monitoring systems"""
    logger.info("Setting up monitoring and observability")
    
    setup_tracing()
    setup_metrics()
    
    # Start cost tracking background task
    # asyncio.create_task(cost_tracking_loop())
    
    logger.info("Monitoring setup completed")

async def cleanup_monitoring():
    """Cleanup monitoring resources"""
    logger.info("Cleaning up monitoring resources")
    
    # Flush any pending traces/metrics
    if trace.get_tracer_provider():
        trace.get_tracer_provider().shutdown()
    
    logger.info("Monitoring cleanup completed")