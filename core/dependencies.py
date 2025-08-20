"""
ChefoodAI Core Dependencies
FastAPI dependencies for authentication, authorization, and common functionality
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional, Generator, Annotated
from datetime import datetime
import logging

from core.database import get_db
from services.auth_service import auth_service, AuthenticationError
from models.users import User
from utils.rate_limiter import RateLimiter
from utils.request_utils import get_client_ip, extract_request_context

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)
rate_limiter = RateLimiter()

async def get_current_user(
    request: Request,
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
    db: Session = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token
    
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not credentials:
        raise credentials_exception
    
    try:
        # Extract token from credentials
        token = credentials.credentials
        
        # Get user from auth service
        user = await auth_service.get_current_user(token, db)
        
        if not user:
            raise credentials_exception
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled"
            )
        
        # Log the authentication for security monitoring
        request_context = extract_request_context(request)
        logger.info(f"User {user.id} authenticated successfully", extra={
            "user_id": user.id,
            "email": user.email,
            "ip": request_context["ip"],
            "user_agent": request_context["user_agent"]["raw"]
        })
        
        return user
        
    except AuthenticationError as e:
        logger.warning(f"Authentication failed: {str(e)}", extra={
            "ip": get_client_ip(request),
            "error": str(e)
        })
        raise credentials_exception
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}", extra={
            "ip": get_client_ip(request),
            "error": str(e)
        })
        raise credentials_exception

async def get_optional_user(
    request: Request,
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise
    
    Used for endpoints that work for both authenticated and anonymous users
    """
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        user = await auth_service.get_current_user(token, db)
        
        if user and user.is_active:
            return user
            
    except Exception as e:
        logger.debug(f"Optional authentication failed: {str(e)}")
    
    return None

def require_plan(required_plan: str):
    """
    Dependency factory for requiring specific user plan
    
    Args:
        required_plan: Required plan ('free', 'premium', 'enterprise')
    """
    def plan_dependency(current_user: User = Depends(get_current_user)) -> User:
        plan_hierarchy = {'free': 0, 'premium': 1, 'enterprise': 2}
        
        user_plan_level = plan_hierarchy.get(current_user.plan, 0)
        required_plan_level = plan_hierarchy.get(required_plan, 0)
        
        if user_plan_level < required_plan_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires {required_plan} plan or higher"
            )
        
        return current_user
    
    return plan_dependency

def require_premium(current_user: User = Depends(get_current_user)) -> User:
    """Dependency for premium features"""
    return require_plan('premium')(current_user)

def require_enterprise(current_user: User = Depends(get_current_user)) -> User:
    """Dependency for enterprise features"""
    return require_plan('enterprise')(current_user)

def require_verified_email(current_user: User = Depends(get_current_user)) -> User:
    """Dependency for requiring verified email"""
    if not current_user.is_email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required"
        )
    
    return current_user

async def check_rate_limit(
    request: Request,
    key_suffix: str = "",
    max_attempts: int = 60,
    window_minutes: int = 1
):
    """
    Rate limiting dependency
    
    Args:
        key_suffix: Additional key suffix for rate limiting
        max_attempts: Maximum attempts allowed
        window_minutes: Time window in minutes
    """
    ip_address = get_client_ip(request)
    rate_limit_key = f"api:{ip_address}"
    
    if key_suffix:
        rate_limit_key += f":{key_suffix}"
    
    if not await rate_limiter.check_rate_limit(
        rate_limit_key, max_attempts, window_minutes
    ):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )

def rate_limit(max_attempts: int = 60, window_minutes: int = 1, key_suffix: str = ""):
    """
    Rate limiting dependency factory
    
    Args:
        max_attempts: Maximum attempts allowed
        window_minutes: Time window in minutes
        key_suffix: Additional key suffix
    """
    async def rate_limit_dependency(request: Request):
        await check_rate_limit(request, key_suffix, max_attempts, window_minutes)
    
    return rate_limit_dependency

async def check_ai_quota(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> User:
    """
    Check if user has remaining AI quota
    
    Raises:
        HTTPException: If quota exceeded
    """
    # Check daily quota
    if current_user.ai_quota_daily != -1:  # -1 means unlimited
        if current_user.ai_usage_daily >= current_user.ai_quota_daily:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Daily AI quota exceeded. Upgrade your plan for more usage."
            )
    
    # Check monthly quota
    if current_user.ai_quota_monthly != -1:  # -1 means unlimited
        if current_user.ai_usage_monthly >= current_user.ai_quota_monthly:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Monthly AI quota exceeded. Upgrade your plan for more usage."
            )
    
    return current_user

async def increment_ai_usage(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Increment user's AI usage counters
    
    Should be called after successful AI operations
    """
    try:
        current_user.ai_usage_daily = (current_user.ai_usage_daily or 0) + 1
        current_user.ai_usage_monthly = (current_user.ai_usage_monthly or 0) + 1
        current_user.updated_at = datetime.utcnow()
        
        db.commit()
        
    except Exception as e:
        logger.error(f"Failed to increment AI usage for user {current_user.id}: {e}")
        db.rollback()

def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Dependency for admin-only endpoints"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user

async def get_pagination_params(
    page: int = 1,
    limit: int = 20,
    max_limit: int = 100
) -> dict:
    """
    Get pagination parameters with validation
    
    Args:
        page: Page number (1-based)
        limit: Items per page
        max_limit: Maximum allowed limit
    
    Returns:
        Dictionary with offset, limit, page
    """
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page must be greater than 0"
        )
    
    if limit < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be greater than 0"
        )
    
    if limit > max_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Limit cannot exceed {max_limit}"
        )
    
    offset = (page - 1) * limit
    
    return {
        "offset": offset,
        "limit": limit,
        "page": page
    }

class SecurityHeaders:
    """Middleware for adding security headers"""
    
    @staticmethod
    def get_security_headers() -> dict:
        """Get security headers"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "Permissions-Policy": "camera=(), microphone=(), location=()"
        }

def get_request_context(request: Request) -> dict:
    """
    Extract request context for logging/monitoring
    
    Returns:
        Dictionary with request metadata
    """
    return extract_request_context(request)

# Type aliases for common dependencies
CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[Optional[User], Depends(get_optional_user)]
PremiumUser = Annotated[User, Depends(require_premium)]
EnterpriseUser = Annotated[User, Depends(require_enterprise)]
VerifiedUser = Annotated[User, Depends(require_verified_email)]
AdminUser = Annotated[User, Depends(require_admin)]
AIQuotaUser = Annotated[User, Depends(check_ai_quota)]
PaginationParams = Annotated[dict, Depends(get_pagination_params)]
RequestContext = Annotated[dict, Depends(get_request_context)]

# Common rate limits
LoginRateLimit = Depends(rate_limit(max_attempts=5, window_minutes=15, key_suffix="login"))
RegisterRateLimit = Depends(rate_limit(max_attempts=3, window_minutes=60, key_suffix="register"))
APIRateLimit = Depends(rate_limit(max_attempts=100, window_minutes=1, key_suffix="api"))
AIRateLimit = Depends(rate_limit(max_attempts=10, window_minutes=1, key_suffix="ai"))