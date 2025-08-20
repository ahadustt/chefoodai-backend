"""
ChefoodAI Security Middleware
Implements security headers, request validation, and threat protection
"""

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
import time
import hashlib
import re
from typing import List, Pattern
from urllib.parse import urlparse

from core.config import settings

logger = structlog.get_logger()


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware that adds:
    - Security headers
    - Request validation
    - Input sanitization
    - Basic threat detection
    """
    
    def __init__(self, app):
        super().__init__(app)
        
        # Compile regex patterns for performance
        self.sql_injection_patterns = [
            re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)", re.IGNORECASE),
            re.compile(r"(--|#|/\*|\*/)", re.IGNORECASE),
            re.compile(r"(\b(OR|AND)\s+\d+\s*=\s*\d+)", re.IGNORECASE),
            re.compile(r"('|(\\x27)|(\\x2D)|(\\x23))", re.IGNORECASE)
        ]
        
        self.xss_patterns = [
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),
            re.compile(r"<iframe[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL)
        ]
        
        # Suspicious user agents
        self.suspicious_user_agents = [
            re.compile(r"sqlmap", re.IGNORECASE),
            re.compile(r"nikto", re.IGNORECASE),
            re.compile(r"nessus", re.IGNORECASE),
            re.compile(r"acunetix", re.IGNORECASE),
            re.compile(r"havij", re.IGNORECASE)
        ]
        
        # Rate limiting for security events
        self.security_violations = {}
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next):
        """Process request through security checks"""
        start_time = time.time()
        
        try:
            # Get client IP
            client_ip = self._get_client_ip(request)
            
            # Check if IP is temporarily blocked
            if self._is_ip_blocked(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"error": "Too many security violations. Try again later."}
                )
            
            # Validate request
            security_violation = await self._validate_request(request, client_ip)
            if security_violation:
                return security_violation
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response, request)
            
            # Log request
            process_time = time.time() - start_time
            logger.info(
                "Request processed",
                method=request.method,
                path=request.url.path,
                client_ip=client_ip,
                process_time=process_time,
                status_code=response.status_code
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address considering proxies"""
        # Check for forwarded headers (CloudFlare, Load Balancer)
        forwarded = request.headers.get("CF-Connecting-IP")
        if forwarded:
            return forwarded
        
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        forwarded = request.headers.get("X-Real-IP")
        if forwarded:
            return forwarded
        
        # Fall back to direct connection
        return request.client.host if request.client else "unknown"
    
    async def _validate_request(self, request: Request, client_ip: str) -> JSONResponse:
        """Validate request for security threats"""
        try:
            # Check user agent
            user_agent = request.headers.get("user-agent", "")
            if self._is_suspicious_user_agent(user_agent):
                self._record_security_violation(client_ip, "suspicious_user_agent")
                return JSONResponse(
                    status_code=403,
                    content={"error": "Forbidden"}
                )
            
            # Check request size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > settings.MAX_FILE_SIZE:
                self._record_security_violation(client_ip, "oversized_request")
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request too large"}
                )
            
            # Validate URL path
            if self._has_path_traversal(request.url.path):
                self._record_security_violation(client_ip, "path_traversal")
                return JSONResponse(
                    status_code=403,
                    content={"error": "Forbidden"}
                )
            
            # Validate query parameters
            if await self._has_malicious_input(str(request.query_params)):
                self._record_security_violation(client_ip, "malicious_query")
                return JSONResponse(
                    status_code=403,
                    content={"error": "Forbidden"}
                )
            
            # Validate request body (for POST/PUT requests)
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body and await self._has_malicious_input(body.decode('utf-8', errors='ignore')):
                        self._record_security_violation(client_ip, "malicious_body")
                        return JSONResponse(
                            status_code=403,
                            content={"error": "Forbidden"}
                        )
                except Exception:
                    # Body might be binary or malformed
                    pass
            
            return None
            
        except Exception as e:
            logger.error(f"Request validation error: {str(e)}")
            return None
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is suspicious"""
        if not user_agent or len(user_agent) < 5:
            return True
        
        for pattern in self.suspicious_user_agents:
            if pattern.search(user_agent):
                return True
        
        return False
    
    def _has_path_traversal(self, path: str) -> bool:
        """Check for path traversal attempts"""
        dangerous_patterns = ["../", "..\\", "..%2f", "..%5c", "%2e%2e%2f", "%2e%2e%5c"]
        path_lower = path.lower()
        
        return any(pattern in path_lower for pattern in dangerous_patterns)
    
    async def _has_malicious_input(self, input_text: str) -> bool:
        """Check for SQL injection and XSS attempts"""
        if not input_text:
            return False
        
        # Check for SQL injection
        for pattern in self.sql_injection_patterns:
            if pattern.search(input_text):
                return True
        
        # Check for XSS
        for pattern in self.xss_patterns:
            if pattern.search(input_text):
                return True
        
        return False
    
    def _record_security_violation(self, client_ip: str, violation_type: str):
        """Record security violation for rate limiting"""
        current_time = time.time()
        
        # Cleanup old violations
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_violations()
            self.last_cleanup = current_time
        
        # Record violation
        if client_ip not in self.security_violations:
            self.security_violations[client_ip] = []
        
        self.security_violations[client_ip].append({
            "type": violation_type,
            "timestamp": current_time
        })
        
        logger.warning(
            "Security violation detected",
            client_ip=client_ip,
            violation_type=violation_type,
            total_violations=len(self.security_violations[client_ip])
        )
    
    def _is_ip_blocked(self, client_ip: str) -> bool:
        """Check if IP should be temporarily blocked"""
        if client_ip not in self.security_violations:
            return False
        
        # Count recent violations (last 1 hour)
        current_time = time.time()
        recent_violations = [
            v for v in self.security_violations[client_ip]
            if current_time - v["timestamp"] < 3600
        ]
        
        # Block if more than 5 violations in the last hour
        return len(recent_violations) > 5
    
    def _cleanup_violations(self):
        """Remove old security violations"""
        current_time = time.time()
        cutoff_time = current_time - self.cleanup_interval * 2  # Keep 2 hours of history
        
        for ip in list(self.security_violations.keys()):
            self.security_violations[ip] = [
                v for v in self.security_violations[ip]
                if current_time - v["timestamp"] < cutoff_time
            ]
            
            # Remove IPs with no recent violations
            if not self.security_violations[ip]:
                del self.security_violations[ip]
    
    def _add_security_headers(self, response: Response, request: Request):
        """Add security headers to response"""
        # Basic security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        
        # HSTS (only for HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # CSP for HTML responses
        if response.headers.get("content-type", "").startswith("text/html"):
            csp_policy = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://apis.google.com; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "img-src 'self' data: https:; "
                "font-src 'self' https://fonts.gstatic.com; "
                "connect-src 'self' https://api.chefoodai.com; "
                "frame-ancestors 'none';"
            )
            response.headers["Content-Security-Policy"] = csp_policy
        
        # API-specific headers
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"