"""
User model for authentication
"""

from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
from core.database import Base


class User(Base):
    """User model matching the auth.users table schema"""
    __tablename__ = "users"
    __table_args__ = {"schema": "auth"}
    
    # Primary key
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # User credentials
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # Profile information
    full_name = Column(String(255), nullable=False)
    phone_number = Column(String(50))
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_premium = Column(Boolean, default=False, nullable=False)
    
    # Subscription info
    subscription_type = Column(String(50), default='free')
    subscription_expires_at = Column(DateTime(timezone=True))
    
    # OAuth info (nullable for regular registration)
    google_id = Column(String(255))
    apple_id = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime(timezone=True))
    
    # Verification
    email_verification_token = Column(String(255))
    email_verified_at = Column(DateTime(timezone=True))
    
    # Password reset
    password_reset_token = Column(String(255))
    password_reset_expires_at = Column(DateTime(timezone=True))
    
    # Additional fields
    preferences = Column(Text)  # JSON string for user preferences
    failed_login_attempts = Column(Integer, default=0)
    account_locked_until = Column(DateTime(timezone=True))

    def __repr__(self):
        return f"<User(user_id={self.user_id}, email={self.email})>"