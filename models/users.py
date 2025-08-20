"""
ChefoodAI User Models
Database models for user management with premium features
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Enum, ForeignKey, JSON, Numeric
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum as PyEnum
import uuid

from core.database import Base


class UserTier(PyEnum):
    """User subscription tiers"""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class UserStatus(PyEnum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class DietaryRestrictionType(PyEnum):
    """Types of dietary restrictions"""
    ALLERGY = "allergy"
    INTOLERANCE = "intolerance"
    PREFERENCE = "preference"
    MEDICAL = "medical"
    RELIGIOUS = "religious"
    ETHICAL = "ethical"


class HealthGoalType(PyEnum):
    """Types of health goals"""
    WEIGHT_LOSS = "weight_loss"
    WEIGHT_GAIN = "weight_gain"
    MUSCLE_GAIN = "muscle_gain"
    MAINTENANCE = "maintenance"
    ATHLETIC_PERFORMANCE = "athletic_performance"
    HEART_HEALTH = "heart_health"
    DIABETES_MANAGEMENT = "diabetes_management"
    GENERAL_WELLNESS = "general_wellness"


class User(Base):
    """Main user account model"""
    __tablename__ = "users"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Account status
    tier: Mapped[UserTier] = mapped_column(Enum(UserTier), default=UserTier.FREE, nullable=False)
    status: Mapped[UserStatus] = mapped_column(Enum(UserStatus), default=UserStatus.PENDING_VERIFICATION, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Profile information
    first_name: Mapped[str] = mapped_column(String(100), nullable=True)
    last_name: Mapped[str] = mapped_column(String(100), nullable=True)
    display_name: Mapped[str] = mapped_column(String(100), nullable=True)
    avatar_url: Mapped[str] = mapped_column(String(500), nullable=True)
    bio: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Contact information
    phone: Mapped[str] = mapped_column(String(20), nullable=True)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC", nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)
    
    # Subscription information
    subscription_id: Mapped[str] = mapped_column(String(100), nullable=True)
    subscription_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Usage tracking
    ai_requests_used_today: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    ai_requests_reset_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), nullable=False)
    total_recipes_created: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_meal_plans_created: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Privacy settings
    profile_public: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    allow_social_features: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    marketing_emails: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    dietary_restrictions = relationship("UserDietaryRestriction", back_populates="user", cascade="all, delete-orphan")
    health_goals = relationship("UserHealthGoal", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete-orphan")
    addresses = relationship("UserAddress", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, tier={self.tier.value})>"
    
    @property
    def full_name(self) -> str:
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.display_name or self.email.split("@")[0]
    
    @property
    def is_premium(self) -> bool:
        """Check if user has premium features"""
        return self.tier in [UserTier.PREMIUM, UserTier.ENTERPRISE]
    
    @property
    def is_enterprise(self) -> bool:
        """Check if user has enterprise features"""
        return self.tier == UserTier.ENTERPRISE
    
    def can_use_ai(self, requests_needed: int = 1) -> bool:
        """Check if user can make AI requests"""
        if self.tier == UserTier.FREE:
            return self.ai_requests_used_today + requests_needed <= 5
        elif self.tier == UserTier.PREMIUM:
            return self.ai_requests_used_today + requests_needed <= 1000
        else:  # Enterprise
            return self.ai_requests_used_today + requests_needed <= 10000


class UserProfile(Base):
    """Extended user profile information"""
    __tablename__ = "user_profiles"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, unique=True)
    
    # Physical attributes
    age: Mapped[int] = mapped_column(Integer, nullable=True)
    height_cm: Mapped[int] = mapped_column(Integer, nullable=True)
    weight_kg: Mapped[float] = mapped_column(Numeric(5, 2), nullable=True)
    gender: Mapped[str] = mapped_column(String(20), nullable=True)
    activity_level: Mapped[str] = mapped_column(String(20), nullable=True)  # sedentary, light, moderate, active, very_active
    
    # Cooking experience
    cooking_skill_level: Mapped[str] = mapped_column(String(20), default="beginner", nullable=False)  # beginner, intermediate, advanced, expert
    favorite_cuisines: Mapped[list] = mapped_column(ARRAY(String), nullable=True)
    cooking_time_preference: Mapped[str] = mapped_column(String(20), nullable=True)  # quick, moderate, no_preference
    
    # Kitchen setup
    kitchen_equipment: Mapped[list] = mapped_column(ARRAY(String), nullable=True)
    dietary_style: Mapped[str] = mapped_column(String(50), nullable=True)  # omnivore, vegetarian, vegan, pescatarian, etc.
    
    # Nutritional preferences
    target_calories: Mapped[int] = mapped_column(Integer, nullable=True)
    target_protein_g: Mapped[int] = mapped_column(Integer, nullable=True)
    target_carbs_g: Mapped[int] = mapped_column(Integer, nullable=True)
    target_fat_g: Mapped[int] = mapped_column(Integer, nullable=True)
    target_fiber_g: Mapped[int] = mapped_column(Integer, nullable=True)
    
    # Family information
    household_size: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    children_ages: Mapped[list] = mapped_column(ARRAY(Integer), nullable=True)
    
    # Budget preferences
    grocery_budget_weekly: Mapped[float] = mapped_column(Numeric(8, 2), nullable=True)
    price_sensitivity: Mapped[str] = mapped_column(String(20), nullable=True)  # low, medium, high
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="profile")
    
    def __repr__(self):
        return f"<UserProfile(user_id={self.user_id}, skill_level={self.cooking_skill_level})>"


class UserDietaryRestriction(Base):
    """User dietary restrictions and allergies"""
    __tablename__ = "user_dietary_restrictions"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    
    restriction_type: Mapped[DietaryRestrictionType] = mapped_column(Enum(DietaryRestrictionType), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # mild, moderate, severe, life_threatening
    
    # For allergies - specific ingredients to avoid
    avoided_ingredients: Mapped[list] = mapped_column(ARRAY(String), nullable=True)
    
    # Medical information
    medical_condition: Mapped[str] = mapped_column(String(100), nullable=True)
    doctor_recommended: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="dietary_restrictions")
    
    def __repr__(self):
        return f"<UserDietaryRestriction(user_id={self.user_id}, name={self.name}, severity={self.severity})>"


class UserHealthGoal(Base):
    """User health and fitness goals"""
    __tablename__ = "user_health_goals"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    
    goal_type: Mapped[HealthGoalType] = mapped_column(Enum(HealthGoalType), nullable=False)
    target_value: Mapped[float] = mapped_column(Numeric(8, 2), nullable=True)
    current_value: Mapped[float] = mapped_column(Numeric(8, 2), nullable=True)
    target_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    description: Mapped[str] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=1, nullable=False)  # 1=high, 2=medium, 3=low
    
    # Progress tracking
    progress_percentage: Mapped[float] = mapped_column(Numeric(5, 2), default=0, nullable=False)
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="health_goals")
    
    def __repr__(self):
        return f"<UserHealthGoal(user_id={self.user_id}, goal_type={self.goal_type.value}, target={self.target_value})>"


class UserPreference(Base):
    """User application preferences and settings"""
    __tablename__ = "user_preferences"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    
    category: Mapped[str] = mapped_column(String(50), nullable=False)  # ui, notifications, ai, privacy
    key: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    data_type: Mapped[str] = mapped_column(String(20), default="string", nullable=False)  # string, integer, boolean, json
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="preferences")
    
    def __repr__(self):
        return f"<UserPreference(user_id={self.user_id}, category={self.category}, key={self.key})>"


class UserAddress(Base):
    """User addresses for delivery"""
    __tablename__ = "user_addresses"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    
    label: Mapped[str] = mapped_column(String(50), nullable=False)  # home, work, other
    street_address: Mapped[str] = mapped_column(String(255), nullable=False)
    apartment: Mapped[str] = mapped_column(String(50), nullable=True)
    city: Mapped[str] = mapped_column(String(100), nullable=False)
    state: Mapped[str] = mapped_column(String(50), nullable=False)
    postal_code: Mapped[str] = mapped_column(String(20), nullable=False)
    country: Mapped[str] = mapped_column(String(50), default="US", nullable=False)
    
    # Location data for delivery optimization
    latitude: Mapped[float] = mapped_column(Numeric(10, 8), nullable=True)
    longitude: Mapped[float] = mapped_column(Numeric(11, 8), nullable=True)
    
    # Delivery preferences
    delivery_instructions: Mapped[str] = mapped_column(Text, nullable=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="addresses")
    
    def __repr__(self):
        return f"<UserAddress(user_id={self.user_id}, label={self.label}, city={self.city})>"


class UserSession(Base):
    """User session tracking"""
    __tablename__ = "user_sessions"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    
    session_token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    refresh_token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    
    # Session metadata
    device_type: Mapped[str] = mapped_column(String(50), nullable=True)
    device_name: Mapped[str] = mapped_column(String(100), nullable=True)
    browser: Mapped[str] = mapped_column(String(100), nullable=True)
    os: Mapped[str] = mapped_column(String(50), nullable=True)
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False)  # IPv6 support
    user_agent: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Location data (approximate)
    country: Mapped[str] = mapped_column(String(50), nullable=True)
    city: Mapped[str] = mapped_column(String(100), nullable=True)
    
    # Session status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_activity_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    def __repr__(self):
        return f"<UserSession(user_id={self.user_id}, device_type={self.device_type}, is_active={self.is_active})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.utcnow() > self.expires_at