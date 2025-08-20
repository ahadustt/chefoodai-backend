"""
ChefoodAI Authentication Endpoints
Comprehensive authentication API with advanced security features
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
import logging

from core.database import get_db
from core.dependencies import get_current_user, get_optional_user
from services.auth_service import auth_service, AuthenticationError, AuthorizationError
from schemas.auth_schemas import (
    UserCreate, UserLogin, TokenResponse, TokenRefresh,
    PasswordChange, PasswordResetRequest, PasswordReset,
    EmailVerification, User, UserUpdate, AuthResponse,
    LogoutResponse, EmailVerificationResponse, PasswordChangeResponse,
    PasswordResetResponse, UserSessions, SecurityLogs, ApiKeyCreate,
    ApiKeyResponse, TwoFactorSetup, TwoFactorVerify, TwoFactorEnable,
    TwoFactorStatus, OAuthProvider, OAuthCallback
)
from utils.rate_limiter import RateLimiter
from utils.request_utils import get_client_ip, get_user_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)
rate_limiter = RateLimiter()

@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Register a new user account
    
    Features:
    - Email and password validation
    - Plan selection (free/premium/enterprise)
    - Terms acceptance validation
    - Rate limiting protection
    - Security logging
    - Welcome email sending
    """
    try:
        ip_address = get_client_ip(request)
        user_agent = get_user_agent(request)
        
        # Rate limiting
        if not await rate_limiter.check_rate_limit(
            f"register:{ip_address}", max_attempts=5, window_minutes=60
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many registration attempts. Please try again later."
            )
        
        user, tokens = await auth_service.register_user(
            user_data, db, user_agent, ip_address
        )
        
        return AuthResponse(
            user=User.from_orm(user),
            tokens=tokens,
            message="Registration successful! Please check your email to verify your account."
        )
        
    except AuthenticationError as e:
        logger.warning(f"Registration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed. Please try again."
        )

@router.post("/login", response_model=AuthResponse)
async def login(
    login_data: UserLogin,
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return access tokens
    
    Features:
    - Email/password authentication
    - Remember me functionality
    - Rate limiting protection
    - Security logging
    - Session management
    - Login notifications for new devices
    """
    try:
        ip_address = get_client_ip(request)
        user_agent = get_user_agent(request)
        
        # Rate limiting
        if not await rate_limiter.check_rate_limit(
            f"login:{ip_address}:{login_data.email}", max_attempts=5, window_minutes=15
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts. Please try again later."
            )
        
        user, tokens = await auth_service.authenticate_user(
            login_data, db, user_agent, ip_address
        )
        
        # Set secure HTTP-only cookie for refresh token
        response.set_cookie(
            key="refresh_token",
            value=tokens.refresh_token,
            httponly=True,
            secure=True,
            samesite="strict",
            max_age=30 * 24 * 60 * 60 if login_data.remember_me else 24 * 60 * 60  # 30 days or 1 day
        )
        
        return AuthResponse(
            user=User.from_orm(user),
            tokens=tokens,
            message="Login successful!"
        )
        
    except AuthenticationError as e:
        logger.warning(f"Login failed for {login_data.email}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed. Please try again."
        )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    token_data: TokenRefresh,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Refresh access token using refresh token
    
    Features:
    - Validates refresh token
    - Issues new access token
    - Session validation
    - Security logging
    """
    try:
        ip_address = get_client_ip(request)
        user_agent = get_user_agent(request)
        
        # Rate limiting
        if not await rate_limiter.check_rate_limit(
            f"refresh:{ip_address}", max_attempts=10, window_minutes=15
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many refresh attempts. Please try again later."
            )
        
        tokens = await auth_service.refresh_access_token(
            token_data.refresh_token, db, user_agent, ip_address
        )
        
        return tokens
        
    except AuthenticationError as e:
        logger.warning(f"Token refresh failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed. Please try again."
        )

@router.post("/logout", response_model=LogoutResponse)
async def logout(
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Logout user and invalidate session
    
    Features:
    - Invalidates current session
    - Clears HTTP-only cookies
    - Security logging
    """
    try:
        ip_address = get_client_ip(request)
        user_agent = get_user_agent(request)
        
        # Get refresh token from cookie or header
        refresh_token = request.cookies.get("refresh_token")
        if not refresh_token:
            # Try to get from request body if not in cookie
            authorization: HTTPAuthorizationCredentials = await security(request)
            if authorization:
                refresh_token = authorization.credentials
        
        if refresh_token:
            await auth_service.logout_user(
                refresh_token, db, user_agent, ip_address
            )
        
        # Clear cookies
        response.delete_cookie("refresh_token")
        
        return LogoutResponse(message="Logged out successfully")
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        # Still return success to avoid revealing errors
        return LogoutResponse(message="Logged out successfully")

@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user information
    
    Returns:
    - User profile data
    - Preferences and settings
    - Usage statistics
    """
    return current_user

@router.put("/me", response_model=User)
async def update_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update user profile information
    
    Features:
    - Updates profile data
    - Validates input data
    - Preserves sensitive fields
    """
    try:
        # Update user fields
        for field, value in user_update.dict(exclude_unset=True).items():
            if hasattr(current_user, field) and value is not None:
                setattr(current_user, field, value)
        
        current_user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(current_user)
        
        return User.from_orm(current_user)
        
    except Exception as e:
        logger.error(f"Profile update error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )

@router.post("/change-password", response_model=PasswordChangeResponse)
async def change_password(
    password_change: PasswordChange,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Change user password
    
    Features:
    - Validates current password
    - Enforces password strength
    - Invalidates all other sessions
    - Sends notification email
    """
    try:
        ip_address = get_client_ip(request)
        user_agent = get_user_agent(request)
        
        # Rate limiting
        if not await rate_limiter.check_rate_limit(
            f"password_change:{current_user.id}", max_attempts=3, window_minutes=60
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many password change attempts. Please try again later."
            )
        
        await auth_service.change_password(
            current_user.id, password_change, db, user_agent, ip_address
        )
        
        return PasswordChangeResponse(message="Password changed successfully")
        
    except AuthenticationError as e:
        logger.warning(f"Password change failed for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Password change error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )

@router.post("/forgot-password")
async def forgot_password(
    reset_request: PasswordResetRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Request password reset token
    
    Features:
    - Generates secure reset token
    - Sends reset email
    - Rate limiting protection
    - Always returns success for security
    """
    try:
        ip_address = get_client_ip(request)
        
        # Rate limiting
        if not await rate_limiter.check_rate_limit(
            f"forgot_password:{ip_address}", max_attempts=5, window_minutes=60
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many password reset requests. Please try again later."
            )
        
        await auth_service.request_password_reset(reset_request.email, db)
        
        return {"message": "If an account with that email exists, we've sent a password reset link."}
        
    except Exception as e:
        logger.error(f"Forgot password error: {str(e)}")
        # Always return success for security
        return {"message": "If an account with that email exists, we've sent a password reset link."}

@router.post("/reset-password", response_model=PasswordResetResponse)
async def reset_password(
    password_reset: PasswordReset,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Reset password using reset token
    
    Features:
    - Validates reset token
    - Enforces password strength
    - Invalidates all sessions
    - Sends confirmation email
    """
    try:
        ip_address = get_client_ip(request)
        
        # Rate limiting
        if not await rate_limiter.check_rate_limit(
            f"reset_password:{ip_address}", max_attempts=5, window_minutes=60
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many password reset attempts. Please try again later."
            )
        
        await auth_service.reset_password(password_reset, db)
        
        return PasswordResetResponse(message="Password reset successfully")
        
    except AuthenticationError as e:
        logger.warning(f"Password reset failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Password reset error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset failed"
        )