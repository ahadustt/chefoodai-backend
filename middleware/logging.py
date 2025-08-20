"""
ChefoodAI Logging Middleware
Structured logging with request/response tracking and performance monitoring
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
import time
import uuid
import json
from typing import Dict, Any, Optional
from contextvars import ContextVar

from core.config import settings

logger = structlog.get_logger()

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logging middleware that provides:
    - Request/response logging with unique IDs
    - Performance monitoring
    - Error tracking
    - User activity logging
    - Structured logging format
    """
    
    def __init__(self, app):
        super().__init__(app)
        
        # Paths to exclude from detailed logging
        self.exclude_paths = {
            "/health", "/ready", "/metrics", "/favicon.ico"
        }
        
        # Sensitive headers to mask in logs
        self.sensitive_headers = {
            "authorization", "cookie", "x-api-key", "x-auth-token"
        }
        
        # Large response types to truncate
        self.truncate_content_types = {
            "application/octet-stream",
            "image/",
            "video/",
            "audio/"
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request with comprehensive logging"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Set context variables
        request_id_var.set(request_id)
        
        try:
            # Skip detailed logging for excluded paths
            if any(request.url.path.startswith(path) for path in self.exclude_paths):
                response = await call_next(request)
                response.headers["X-Request-ID"] = request_id
                return response
            
            # Extract request information
            request_info = await self._extract_request_info(request, request_id)
            
            # Log incoming request
            logger.info(
                "Request started",
                **request_info,
                event_type="request_start"
            )
            
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Extract response information
            response_info = self._extract_response_info(response, process_time)
            
            # Log completed request
            log_level = self._determine_log_level(response.status_code)
            logger.log(
                log_level,
                "Request completed",
                **request_info,
                **response_info,
                event_type="request_complete"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            # Log performance metrics if enabled
            if settings.ENABLE_METRICS:
                await self._log_performance_metrics(request_info, response_info, process_time)
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            
            logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                client_ip=self._get_client_ip(request),
                process_time=process_time,
                error=str(e),
                error_type=type(e).__name__,
                event_type="request_error"
            )
            
            # Re-raise the exception
            raise
    
    async def _extract_request_info(self, request: Request, request_id: str) -> Dict[str, Any]:
        """Extract comprehensive request information"""
        try:
            # Basic request info
            info = {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", ""),
                "referer": request.headers.get("referer", ""),
                "content_type": request.headers.get("content-type", ""),
                "content_length": request.headers.get("content-length", 0),
            }
            
            # Add filtered headers
            info["headers"] = self._filter_headers(dict(request.headers))
            
            # Extract user ID if available
            user_id = await self._extract_user_id(request)
            if user_id:
                info["user_id"] = user_id
                user_id_var.set(user_id)
            
            # Add request body for POST/PUT requests (if not too large)
            if request.method in ["POST", "PUT", "PATCH"]:
                content_length = int(request.headers.get("content-length", 0))
                if content_length > 0 and content_length < 1024:  # Only log small payloads
                    try:
                        body = await request.body()
                        if body:
                            # Try to parse as JSON for better logging
                            try:
                                info["request_body"] = json.loads(body.decode('utf-8'))
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                info["request_body"] = body.decode('utf-8', errors='ignore')[:500]
                    except Exception:
                        pass
            
            return info
            
        except Exception as e:
            logger.error(f"Error extracting request info: {str(e)}")
            return {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "error": "Failed to extract request info"
            }
    
    def _extract_response_info(self, response: Response, process_time: float) -> Dict[str, Any]:
        """Extract response information"""
        info = {
            "status_code": response.status_code,
            "process_time": round(process_time, 4),
            "response_size": len(response.body) if hasattr(response, 'body') else 0,
        }
        
        # Add response headers (filtered)
        info["response_headers"] = self._filter_headers(dict(response.headers))
        
        # Add response body for small responses and errors
        if (response.status_code >= 400 or settings.DEBUG) and hasattr(response, 'body'):
            try:
                body_size = len(response.body) if response.body else 0
                if body_size > 0 and body_size < 1024:  # Only log small responses
                    content_type = response.headers.get("content-type", "")
                    if not any(ct in content_type for ct in self.truncate_content_types):
                        try:
                            if isinstance(response.body, bytes):
                                body_text = response.body.decode('utf-8')
                                info["response_body"] = json.loads(body_text)
                            else:
                                info["response_body"] = response.body
                        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                            info["response_body"] = str(response.body)[:500]
            except Exception:
                pass
        
        return info
    
    async def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request (JWT token, session, etc.)"""
        try:
            # Try to get from Authorization header
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                # TODO: Implement JWT decoding to get user ID
                # token = auth_header.split(" ")[1]
                # user_data = jwt_service.decode_token(token)
                # return user_data.get("user_id")
                pass
            
            # Try to get from session cookie
            session_cookie = request.cookies.get("session_id")
            if session_cookie:
                # TODO: Implement session lookup
                # session_data = await session_manager.get_session(session_cookie)
                # return session_data.get("user_id")
                pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting user ID: {str(e)}")
            return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check for forwarded headers
        forwarded = request.headers.get("cf-connecting-ip")
        if forwarded:
            return forwarded
        
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        forwarded = request.headers.get("x-real-ip")
        if forwarded:
            return forwarded
        
        return request.client.host if request.client else "unknown"
    
    def _filter_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Filter sensitive headers from logs"""
        filtered = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                filtered[key] = "***MASKED***"
            else:
                filtered[key] = value
        return filtered
    
    def _determine_log_level(self, status_code: int) -> int:
        """Determine appropriate log level based on status code"""
        if status_code >= 500:
            return 40  # ERROR
        elif status_code >= 400:
            return 30  # WARNING
        elif status_code >= 300:
            return 20  # INFO
        else:
            return 20  # INFO
    
    async def _log_performance_metrics(
        self,
        request_info: Dict[str, Any],
        response_info: Dict[str, Any],
        process_time: float
    ):
        """Log performance metrics for monitoring"""
        try:
            metrics = {
                "metric_type": "performance",
                "endpoint": f"{request_info['method']} {request_info['path']}",
                "response_time": process_time,
                "status_code": response_info["status_code"],
                "response_size": response_info.get("response_size", 0),
                "user_id": request_info.get("user_id"),
                "timestamp": time.time()
            }
            
            # Log slow requests
            if process_time > 2.0:  # Slow request threshold
                logger.warning(
                    "Slow request detected",
                    **metrics,
                    event_type="slow_request"
                )
            
            # TODO: Send metrics to monitoring system (Prometheus, etc.)
            # metrics_collector.record_request_duration(
            #     method=request_info['method'],
            #     endpoint=request_info['path'],
            #     status_code=response_info['status_code'],
            #     duration=process_time
            # )
            
        except Exception as e:
            logger.error(f"Error logging performance metrics: {str(e)}")


# Utility functions for structured logging
def get_request_id() -> str:
    """Get current request ID from context"""
    return request_id_var.get()


def get_user_id() -> str:
    """Get current user ID from context"""
    return user_id_var.get()


def log_user_activity(activity: str, details: Dict[str, Any] = None):
    """Log user activity with context"""
    logger.info(
        "User activity",
        request_id=get_request_id(),
        user_id=get_user_id(),
        activity=activity,
        details=details or {},
        event_type="user_activity"
    )


def log_business_event(event: str, data: Dict[str, Any] = None):
    """Log business events for analytics"""
    logger.info(
        "Business event",
        request_id=get_request_id(),
        user_id=get_user_id(),
        event=event,
        data=data or {},
        event_type="business_event"
    )