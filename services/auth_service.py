"""
ChefoodAI Premium Authentication Service
Comprehensive JWT authentication with advanced security features
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from core.config import get_settings
from core.database import get_db
from models.users import User, UserSession
from schemas.auth_schemas import (
    UserCreate, UserLogin, TokenResponse, 
    PasswordReset, PasswordChange
)
from services.email_service import EmailService
from utils.security import SecurityUtils
from utils.rate_limiter import RateLimiter

settings = get_settings()

class AuthenticationError(Exception):
    """Custom authentication error"""
    pass

class AuthorizationError(Exception):
    """Custom authorization error"""
    pass

class AuthService:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.email_service = EmailService()
        self.security_utils = SecurityUtils()
        self.rate_limiter = RateLimiter()
        
        # JWT settings
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return self.pwd_context.hash(password)
    
    def create_access_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
            
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": secrets.token_urlsafe(32)  # JWT ID for revocation
        })
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != token_type:
                raise JWTError("Invalid token type")
                
            return payload
            
        except JWTError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
    
    async def register_user(
        self, 
        user_data: UserCreate, 
        db: Session,
        user_agent: str,
        ip_address: str
    ) -> Tuple[User, TokenResponse]:
        """Register a new user with comprehensive validation"""
        
        # Rate limiting for registration
        if not await self.rate_limiter.check_rate_limit(
            f"register:{ip_address}", max_attempts=5, window_minutes=60
        ):
            raise AuthenticationError("Too many registration attempts. Please try again later.")
        
        # Check if user already exists
        existing_user = db.query(User).filter(
            or_(
                User.email == user_data.email.lower(),
                User.username == user_data.username if hasattr(user_data, 'username') else False
            )
        ).first()
        
        if existing_user:
            # Log failed registration attempt
            await self._log_security_event(
                db, None, "registration_failed", 
                {"reason": "user_exists", "email": user_data.email}, 
                ip_address, user_agent
            )
            raise AuthenticationError("User with this email already exists")
        
        # Validate password strength
        if not self.security_utils.validate_password_strength(user_data.password):
            raise AuthenticationError(
                "Password must be at least 8 characters with uppercase, lowercase, and number"
            )
        
        # Create user
        hashed_password = self.get_password_hash(user_data.password)
        
        user = User(
            email=user_data.email.lower(),
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            password_hash=hashed_password,
            plan=user_data.plan or "free",
            is_email_verified=False,
            accept_terms=user_data.accept_terms,
            marketing_emails=user_data.marketing_emails,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Set user preferences based on plan
        if user_data.plan == "premium":
            user.ai_quota_daily = 100
            user.ai_quota_monthly = 3000
        elif user_data.plan == "enterprise":
            user.ai_quota_daily = -1  # Unlimited
            user.ai_quota_monthly = -1
        else:  # free plan
            user.ai_quota_daily = 5
            user.ai_quota_monthly = 150
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Create tokens
        access_token = self.create_access_token(
            data={"sub": str(user.id), "email": user.email}
        )
        refresh_token = self.create_refresh_token(
            data={"sub": str(user.id), "email": user.email}
        )
        
        # Create user session
        session = UserSession(
            user_id=user.id,
            session_token=hashlib.sha256(refresh_token.encode()).hexdigest(),
            user_agent=user_agent,
            ip_address=ip_address,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        )
        
        db.add(session)
        db.commit()
        
        # Send welcome email with email verification
        try:
            await self.email_service.send_welcome_email(user)
            await self.email_service.send_email_verification(user)
        except Exception as e:
            # Log email error but don't fail registration
            print(f"Failed to send welcome email: {e}")
        
        # Log successful registration
        await self._log_security_event(
            db, user.id, "registration_success", 
            {"plan": user.plan}, ip_address, user_agent
        )
        
        token_response = TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=self.access_token_expire_minutes * 60
        )
        
        return user, token_response
    
    async def authenticate_user(
        self, 
        login_data: UserLogin, 
        db: Session,
        user_agent: str,
        ip_address: str
    ) -> Tuple[User, TokenResponse]:
        """Authenticate user with advanced security checks"""
        
        # Rate limiting for login attempts
        if not await self.rate_limiter.check_rate_limit(
            f"login:{ip_address}:{login_data.email}", max_attempts=5, window_minutes=15
        ):
            raise AuthenticationError("Too many login attempts. Please try again later.")
        
        # Find user
        user = db.query(User).filter(User.email == login_data.email.lower()).first()
        
        if not user or not self.verify_password(login_data.password, user.password_hash):
            # Log failed login attempt
            await self._log_security_event(
                db, user.id if user else None, "login_failed", 
                {"reason": "invalid_credentials", "email": login_data.email}, 
                ip_address, user_agent
            )
            raise AuthenticationError("Invalid email or password")
        
        # Check if user is active
        if not user.is_active:
            await self._log_security_event(
                db, user.id, "login_failed", 
                {"reason": "account_disabled"}, ip_address, user_agent
            )
            raise AuthenticationError("Account is disabled. Please contact support.")
        
        # Check for suspicious login patterns
        await self._check_suspicious_activity(db, user, ip_address, user_agent)
        
        # Create tokens
        access_token = self.create_access_token(
            data={"sub": str(user.id), "email": user.email}
        )
        refresh_token = self.create_refresh_token(
            data={"sub": str(user.id), "email": user.email}
        )
        
        # Update user login info
        user.last_login = datetime.utcnow()
        user.login_count = (user.login_count or 0) + 1
        
        # Create or update session
        existing_session = db.query(UserSession).filter(
            and_(
                UserSession.user_id == user.id,
                UserSession.user_agent == user_agent,
                UserSession.is_active == True
            )
        ).first()
        
        if existing_session and not login_data.remember_me:
            # Invalidate existing session if not remember me
            existing_session.is_active = False
            existing_session.ended_at = datetime.utcnow()
        
        # Create new session
        session = UserSession(
            user_id=user.id,
            session_token=hashlib.sha256(refresh_token.encode()).hexdigest(),
            user_agent=user_agent,
            ip_address=ip_address,
            remember_me=login_data.remember_me,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(
                days=self.refresh_token_expire_days if login_data.remember_me else 1
            )
        )
        
        db.add(session)
        db.commit()
        db.refresh(user)
        
        # Log successful login
        await self._log_security_event(
            db, user.id, "login_success", 
            {"remember_me": login_data.remember_me}, ip_address, user_agent
        )
        
        # Send login notification if from new device/location
        if await self._is_new_login_location(db, user, ip_address, user_agent):
            try:
                await self.email_service.send_login_notification(user, ip_address, user_agent)
            except Exception as e:
                print(f"Failed to send login notification: {e}")
        
        token_response = TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=self.access_token_expire_minutes * 60
        )
        
        return user, token_response
    
    async def refresh_access_token(
        self, 
        refresh_token: str, 
        db: Session,
        user_agent: str,
        ip_address: str
    ) -> TokenResponse:
        """Refresh access token using refresh token"""
        
        try:
            payload = self.verify_token(refresh_token, "refresh")
            user_id = payload.get("sub")
            jti = payload.get("jti")
            
            if not user_id:
                raise AuthenticationError("Invalid refresh token")
            
            # Check if session exists and is valid
            session_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
            session = db.query(UserSession).filter(
                and_(
                    UserSession.session_token == session_token_hash,
                    UserSession.is_active == True,
                    UserSession.expires_at > datetime.utcnow()
                )
            ).first()
            
            if not session:
                raise AuthenticationError("Invalid or expired session")
            
            # Get user
            user = db.query(User).filter(User.id == user_id).first()
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")
            
            # Create new access token
            access_token = self.create_access_token(
                data={"sub": str(user.id), "email": user.email}
            )
            
            # Update session last used
            session.last_used = datetime.utcnow()
            db.commit()
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,  # Keep same refresh token
                token_type="bearer",
                expires_in=self.access_token_expire_minutes * 60
            )
            
        except JWTError:
            raise AuthenticationError("Invalid refresh token")
    
    async def logout_user(
        self, 
        refresh_token: str, 
        db: Session,
        user_agent: str,
        ip_address: str
    ) -> bool:
        """Logout user and invalidate session"""
        
        try:
            session_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
            session = db.query(UserSession).filter(
                UserSession.session_token == session_token_hash
            ).first()
            
            if session:
                session.is_active = False
                session.ended_at = datetime.utcnow()
                
                # Log logout
                await self._log_security_event(
                    db, session.user_id, "logout", 
                    {}, ip_address, user_agent
                )
                
                db.commit()
                return True
                
        except Exception:
            pass
        
        return False
    
    async def logout_all_sessions(
        self, 
        user_id: int, 
        db: Session,
        current_session_token: str = None
    ) -> bool:
        """Logout user from all sessions except current"""
        
        query = db.query(UserSession).filter(
            and_(
                UserSession.user_id == user_id,
                UserSession.is_active == True
            )
        )
        
        if current_session_token:
            current_hash = hashlib.sha256(current_session_token.encode()).hexdigest()
            query = query.filter(UserSession.session_token != current_hash)
        
        sessions = query.all()
        
        for session in sessions:
            session.is_active = False
            session.ended_at = datetime.utcnow()
        
        db.commit()
        return True
    
    async def change_password(
        self, 
        user_id: int,
        password_change: PasswordChange,
        db: Session,
        user_agent: str,
        ip_address: str
    ) -> bool:
        """Change user password with validation"""
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise AuthenticationError("User not found")
        
        # Verify current password
        if not self.verify_password(password_change.current_password, user.password_hash):
            await self._log_security_event(
                db, user_id, "password_change_failed", 
                {"reason": "invalid_current_password"}, ip_address, user_agent
            )
            raise AuthenticationError("Current password is incorrect")
        
        # Validate new password
        if not self.security_utils.validate_password_strength(password_change.new_password):
            raise AuthenticationError(
                "New password must be at least 8 characters with uppercase, lowercase, and number"
            )
        
        # Check if new password is different from current
        if self.verify_password(password_change.new_password, user.password_hash):
            raise AuthenticationError("New password must be different from current password")
        
        # Update password
        user.password_hash = self.get_password_hash(password_change.new_password)
        user.password_changed_at = datetime.utcnow()
        user.updated_at = datetime.utcnow()
        
        # Invalidate all other sessions
        await self.logout_all_sessions(user_id, db)
        
        db.commit()
        
        # Log password change
        await self._log_security_event(
            db, user_id, "password_changed", {}, ip_address, user_agent
        )
        
        # Send notification email
        try:
            await self.email_service.send_password_change_notification(user)
        except Exception as e:
            print(f"Failed to send password change notification: {e}")
        
        return True
    
    async def request_password_reset(
        self, 
        email: str, 
        db: Session
    ) -> bool:
        """Request password reset token"""
        
        user = db.query(User).filter(User.email == email.lower()).first()
        if not user:
            # Don't reveal if email exists or not
            return True
        
        # Generate reset token
        reset_token = secrets.token_urlsafe(32)
        reset_token_hash = hashlib.sha256(reset_token.encode()).hexdigest()
        
        # Store reset token
        user.reset_token_hash = reset_token_hash
        user.reset_token_expires = datetime.utcnow() + timedelta(hours=1)
        user.updated_at = datetime.utcnow()
        
        db.commit()
        
        # Send reset email
        try:
            await self.email_service.send_password_reset_email(user, reset_token)
        except Exception as e:
            print(f"Failed to send password reset email: {e}")
        
        return True
    
    async def reset_password(
        self, 
        password_reset: PasswordReset, 
        db: Session
    ) -> bool:
        """Reset password using reset token"""
        
        # Hash the provided token to compare
        token_hash = hashlib.sha256(password_reset.token.encode()).hexdigest()
        
        user = db.query(User).filter(
            and_(
                User.reset_token_hash == token_hash,
                User.reset_token_expires > datetime.utcnow()
            )
        ).first()
        
        if not user:
            raise AuthenticationError("Invalid or expired reset token")
        
        # Validate new password
        if not self.security_utils.validate_password_strength(password_reset.new_password):
            raise AuthenticationError(
                "Password must be at least 8 characters with uppercase, lowercase, and number"
            )
        
        # Update password and clear reset tokens
        user.password_hash = self.get_password_hash(password_reset.new_password)
        user.password_changed_at = datetime.utcnow()
        user.reset_token_hash = None
        user.reset_token_expires = None
        user.updated_at = datetime.utcnow()
        
        # Invalidate all sessions
        await self.logout_all_sessions(user.id, db)
        
        db.commit()
        
        # Send confirmation email
        try:
            await self.email_service.send_password_reset_confirmation(user)
        except Exception as e:
            print(f"Failed to send password reset confirmation: {e}")
        
        return True
    
    async def get_current_user(self, token: str, db: Session) -> User:
        """Get current user from JWT token"""
        
        try:
            payload = self.verify_token(token)
            user_id = payload.get("sub")
            
            if not user_id:
                raise AuthenticationError("Invalid token payload")
            
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise AuthenticationError("User not found")
            
            if not user.is_active:
                raise AuthenticationError("User account is disabled")
            
            return user
            
        except JWTError:
            raise AuthenticationError("Could not validate credentials")
    
    async def _log_security_event(
        self, 
        db: Session, 
        user_id: Optional[int], 
        event_type: str,
        details: Dict[str, Any],
        ip_address: str,
        user_agent: str
    ):
        """Log security events for monitoring"""
        
        # TODO: Implement UserSecurityLog model
        pass
    
    async def _check_suspicious_activity(
        self, 
        db: Session, 
        user: User, 
        ip_address: str, 
        user_agent: str
    ):
        """Check for suspicious login activity"""
        
        # TODO: Implement UserSecurityLog model for rate limiting
        recent_failures = 0
        
        if recent_failures >= 3:
            # Log suspicious activity
            await self._log_security_event(
                db, user.id, "suspicious_activity", 
                {"reason": "multiple_failed_attempts", "count": recent_failures}, 
                ip_address, user_agent
            )
    
    async def _is_new_login_location(
        self, 
        db: Session, 
        user: User, 
        ip_address: str, 
        user_agent: str
    ) -> bool:
        """Check if login is from a new location/device"""
        
        recent_logins = db.query(UserSession).filter(
            and_(
                UserSession.user_id == user.id,
                UserSession.created_at > datetime.utcnow() - timedelta(days=30)
            )
        ).all()
        
        # Check if we've seen this IP or user agent before
        for session in recent_logins:
            if session.ip_address == ip_address and session.user_agent == user_agent:
                return False
        
        return True

# Create singleton instance
auth_service = AuthService()