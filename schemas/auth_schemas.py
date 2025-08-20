"""
ChefoodAI Authentication Schemas
Pydantic models for authentication requests and responses
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, validator, Field

class UserBase(BaseModel):
    """Base user schema with common fields"""
    email: EmailStr
    first_name: str = Field(..., min_length=2, max_length=50)
    last_name: str = Field(..., min_length=2, max_length=50)

class UserCreate(UserBase):
    """Schema for user registration"""
    password: str = Field(..., min_length=8, max_length=128)
    plan: Optional[str] = Field(default="free", pattern="^(free|premium|enterprise)$")
    accept_terms: bool = Field(..., description="User must accept terms and conditions")
    marketing_emails: bool = Field(default=False)
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        
        if not (has_upper and has_lower and has_digit):
            raise ValueError('Password must contain uppercase, lowercase, and number')
        
        return v
    
    @validator('accept_terms')
    def validate_terms(cls, v):
        """Ensure terms are accepted"""
        if not v:
            raise ValueError('You must accept the terms and conditions')
        return v

class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr
    password: str = Field(..., min_length=1)
    remember_me: bool = Field(default=False)

class TokenResponse(BaseModel):
    """Schema for authentication token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    
class TokenRefresh(BaseModel):
    """Schema for token refresh request"""
    refresh_token: str

class PasswordChange(BaseModel):
    """Schema for password change"""
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)
    
    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength"""
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        
        if not (has_upper and has_lower and has_digit):
            raise ValueError('Password must contain uppercase, lowercase, and number')
        
        return v

class PasswordResetRequest(BaseModel):
    """Schema for password reset request"""
    email: EmailStr

class PasswordReset(BaseModel):
    """Schema for password reset"""
    token: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)
    
    @validator('new_password')
    def validate_password(cls, v):
        """Validate password strength"""
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        
        if not (has_upper and has_lower and has_digit):
            raise ValueError('Password must contain uppercase, lowercase, and number')
        
        return v

class EmailVerification(BaseModel):
    """Schema for email verification"""
    token: str = Field(..., min_length=1)

class User(UserBase):
    """Schema for user response"""
    id: int
    plan: str
    is_active: bool
    is_email_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    login_count: Optional[int] = 0
    
    # User preferences
    dietary_preferences: Optional[List[str]] = []
    cuisine_preferences: Optional[List[str]] = []
    skill_level: Optional[str] = "intermediate"
    cooking_goals: Optional[List[str]] = []
    
    # AI usage stats
    ai_quota_daily: Optional[int] = 5
    ai_quota_monthly: Optional[int] = 150
    ai_usage_daily: Optional[int] = 0
    ai_usage_monthly: Optional[int] = 0
    
    # Notification preferences
    notification_settings: Optional[Dict[str, bool]] = {
        "email": True,
        "push": True,
        "marketing": False,
        "recipe_recommendations": True,
        "meal_plan_reminders": True
    }
    
    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    """Schema for user profile updates"""
    first_name: Optional[str] = Field(None, min_length=2, max_length=50)
    last_name: Optional[str] = Field(None, min_length=2, max_length=50)
    phone: Optional[str] = Field(None, max_length=20)
    bio: Optional[str] = Field(None, max_length=500)
    location: Optional[str] = Field(None, max_length=100)
    date_of_birth: Optional[datetime] = None
    
    # Preferences
    dietary_preferences: Optional[List[str]] = None
    cuisine_preferences: Optional[List[str]] = None
    skill_level: Optional[str] = Field(None, pattern="^(beginner|intermediate|advanced|expert)$")
    cooking_goals: Optional[List[str]] = None
    
    # Notification settings
    notification_settings: Optional[Dict[str, bool]] = None

class UserSession(BaseModel):
    """Schema for user session"""
    id: int
    user_id: int
    user_agent: str
    ip_address: str
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: datetime
    is_active: bool
    remember_me: bool = False
    
    class Config:
        from_attributes = True

class UserSessions(BaseModel):
    """Schema for user sessions list"""
    sessions: List[UserSession]
    total: int

class SecurityLog(BaseModel):
    """Schema for security log entry"""
    id: int
    user_id: Optional[int] = None
    event_type: str
    ip_address: str
    user_agent: str
    details: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class SecurityLogs(BaseModel):
    """Schema for security logs list"""
    logs: List[SecurityLog]
    total: int

class AuthResponse(BaseModel):
    """Schema for authentication response"""
    user: User
    tokens: TokenResponse
    message: str = "Authentication successful"

class LogoutResponse(BaseModel):
    """Schema for logout response"""
    message: str = "Logged out successfully"

class EmailVerificationResponse(BaseModel):
    """Schema for email verification response"""
    message: str = "Email verified successfully"
    is_verified: bool = True

class PasswordChangeResponse(BaseModel):
    """Schema for password change response"""
    message: str = "Password changed successfully"

class PasswordResetResponse(BaseModel):
    """Schema for password reset response"""
    message: str = "Password reset successfully"

class ApiKey(BaseModel):
    """Schema for API key"""
    id: int
    name: str
    key_prefix: str
    permissions: List[str]
    expires_at: Optional[datetime] = None
    created_at: datetime
    last_used: Optional[datetime] = None
    is_active: bool = True
    
    class Config:
        from_attributes = True

class ApiKeyCreate(BaseModel):
    """Schema for API key creation"""
    name: str = Field(..., min_length=3, max_length=50, description="Name for the API key")
    permissions: List[str] = Field(..., description="List of permissions for the API key")
    expires_at: Optional[datetime] = Field(None, description="Expiration date for the API key")

class ApiKeyResponse(BaseModel):
    """Schema for API key creation response"""
    api_key: ApiKey
    key: str  # Full API key (only returned once)
    message: str = "API key created successfully"

class TwoFactorSetup(BaseModel):
    """Schema for 2FA setup"""
    secret: str
    qr_code: str
    backup_codes: List[str]

class TwoFactorVerify(BaseModel):
    """Schema for 2FA verification"""
    token: str = Field(..., min_length=6, max_length=6, description="6-digit TOTP code")

class TwoFactorEnable(BaseModel):
    """Schema for enabling 2FA"""
    token: str = Field(..., min_length=6, max_length=6)
    backup_codes: List[str]

class TwoFactorStatus(BaseModel):
    """Schema for 2FA status"""
    is_enabled: bool
    backup_codes_remaining: int

# OAuth schemas
class OAuthProvider(BaseModel):
    """Schema for OAuth provider"""
    provider: str = Field(..., pattern="^(google|github|twitter)$")
    client_id: str
    redirect_uri: str

class OAuthCallback(BaseModel):
    """Schema for OAuth callback"""
    provider: str = Field(..., pattern="^(google|github|twitter)$")
    code: str
    state: Optional[str] = None

class OAuthLink(BaseModel):
    """Schema for linking OAuth account"""
    provider: str = Field(..., pattern="^(google|github|twitter)$")
    provider_user_id: str
    provider_email: Optional[str] = None

class LinkedAccount(BaseModel):
    """Schema for linked OAuth account"""
    id: int
    provider: str
    provider_user_id: str
    provider_email: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class LinkedAccounts(BaseModel):
    """Schema for linked accounts list"""
    accounts: List[LinkedAccount]

# Admin schemas
class AdminUserUpdate(BaseModel):
    """Schema for admin user updates"""
    is_active: Optional[bool] = None
    plan: Optional[str] = Field(None, pattern="^(free|premium|enterprise)$")
    ai_quota_daily: Optional[int] = None
    ai_quota_monthly: Optional[int] = None
    is_email_verified: Optional[bool] = None
    notes: Optional[str] = Field(None, max_length=1000)

class UserStats(BaseModel):
    """Schema for user statistics"""
    total_users: int
    active_users: int
    new_users_today: int
    new_users_this_week: int
    new_users_this_month: int
    users_by_plan: Dict[str, int]
    users_by_status: Dict[str, int]

class AuthStats(BaseModel):
    """Schema for authentication statistics"""
    total_logins_today: int
    total_logins_this_week: int
    total_logins_this_month: int
    failed_logins_today: int
    unique_active_users_today: int
    average_session_duration: float  # in minutes