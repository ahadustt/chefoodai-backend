"""
ChefoodAI Configuration Settings
Manages all application configuration with environment-based overrides
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "ChefoodAI"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    VERSION: str = "1.0.0"
    
    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Security
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    JWT_SECRET_KEY: str = Field(default="dev-jwt-secret-change-in-production", env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000", 
            "http://localhost:5173",
            "https://chefoodai-frontend-1074761757006.us-central1.run.app",
            "https://*.run.app"
        ],
        env="CORS_ORIGINS"
    )
    ALLOWED_HOSTS: List[str] = Field(
        default=["localhost", "127.0.0.1", "chefoodai-backend-mpsrniojta-uc.a.run.app", "*.run.app"],
        env="ALLOWED_HOSTS"
    )
    
    # Database
    DATABASE_URL: str = Field(default="postgresql://user:pass@localhost/chefoodai", env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = Field(default=10, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    DATABASE_POOL_TIMEOUT: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    REDIS_POOL_SIZE: int = Field(default=10, env="REDIS_POOL_SIZE")
    
    # AI Microservice Configuration
    AI_SERVICE_URL: str = Field(default="https://chefoodai-ai-service-mpsrniojta-uc.a.run.app", env="AI_SERVICE_URL")
    AI_TIMEOUT: int = Field(default=30, env="AI_TIMEOUT")
    AI_ENHANCEMENT_ENABLED: bool = Field(default=True, env="AI_ENHANCEMENT_ENABLED")
    FALLBACK_ENABLED: bool = Field(default=True, env="FALLBACK_ENABLED")
    CACHE_ENABLED: bool = Field(default=True, env="CACHE_ENABLED")
    
    # Google Cloud
    GOOGLE_CLOUD_PROJECT: str = Field(default="mychef-467404", env="GOOGLE_CLOUD_PROJECT")
    GOOGLE_CLOUD_REGION: str = Field(default="us-central1", env="GOOGLE_CLOUD_REGION")
    
    # Vertex AI
    VERTEX_AI_LOCATION: str = Field(default="us-central1", env="VERTEX_AI_LOCATION")
    GEMINI_MODEL: str = Field(default="gemini-2.0-flash-thinking", env="GEMINI_MODEL")
    GEMINI_MODEL_FALLBACK: str = Field(default="gemini-1.5-flash", env="GEMINI_MODEL_FALLBACK")
    
    # AI Configuration
    AI_MAX_TOKENS: int = Field(default=8192, env="AI_MAX_TOKENS")
    AI_TEMPERATURE: float = Field(default=0.7, env="AI_TEMPERATURE")
    AI_TOP_P: float = Field(default=0.9, env="AI_TOP_P")
    AI_TOP_K: int = Field(default=40, env="AI_TOP_K")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    
    # User Quotas (per day)
    FREE_TIER_AI_REQUESTS: int = Field(default=5, env="FREE_TIER_AI_REQUESTS")
    PREMIUM_TIER_AI_REQUESTS: int = Field(default=1000, env="PREMIUM_TIER_AI_REQUESTS")
    ENTERPRISE_TIER_AI_REQUESTS: int = Field(default=10000, env="ENTERPRISE_TIER_AI_REQUESTS")
    
    # Cache Settings
    CACHE_TTL_SHORT: int = Field(default=300, env="CACHE_TTL_SHORT")  # 5 minutes
    CACHE_TTL_MEDIUM: int = Field(default=3600, env="CACHE_TTL_MEDIUM")  # 1 hour
    CACHE_TTL_LONG: int = Field(default=86400, env="CACHE_TTL_LONG")  # 24 hours
    
    # File Storage
    STORAGE_BUCKET: str = Field(default="chefoodai-storage", env="STORAGE_BUCKET")
    MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=["image/jpeg", "image/png", "image/webp", "video/mp4"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    
    # Email Configuration
    SMTP_SERVER: str = Field(default="smtp.gmail.com", env="SMTP_SERVER")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    SMTP_USERNAME: str = Field(default="", env="SMTP_USERNAME")
    SMTP_PASSWORD: str = Field(default="", env="SMTP_PASSWORD")
    FROM_EMAIL: str = Field(default="noreply@chefoodai.com", env="FROM_EMAIL")
    FROM_NAME: str = Field(default="ChefoodAI", env="FROM_NAME")
    
    # External APIs
    NUTRITION_API_KEY: Optional[str] = Field(default=None, env="NUTRITION_API_KEY")
    GROCERY_API_KEY: Optional[str] = Field(default=None, env="GROCERY_API_KEY")
    
    # Premium Features
    ENABLE_PREMIUM_FEATURES: bool = Field(default=True, env="ENABLE_PREMIUM_FEATURES")
    ENABLE_MULTIMODAL_AI: bool = Field(default=True, env="ENABLE_MULTIMODAL_AI")
    ENABLE_SOCIAL_FEATURES: bool = Field(default=True, env="ENABLE_SOCIAL_FEATURES")
    ENABLE_MARKETPLACE: bool = Field(default=True, env="ENABLE_MARKETPLACE")
    
    # Business Logic
    RECIPE_GENERATION_TIMEOUT: int = Field(default=30, env="RECIPE_GENERATION_TIMEOUT")
    MEAL_PLAN_MAX_DAYS: int = Field(default=30, env="MEAL_PLAN_MAX_DAYS")
    SHOPPING_LIST_MAX_ITEMS: int = Field(default=100, env="SHOPPING_LIST_MAX_ITEMS")
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("ALLOWED_HOSTS", pre=True)
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @validator("ALLOWED_FILE_TYPES", pre=True)
    def parse_allowed_file_types(cls, v):
        if isinstance(v, str):
            return [file_type.strip() for file_type in v.split(",")]
        return v
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    @property
    def database_url_async(self) -> str:
        """Convert sync database URL to async"""
        return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance"""
    return settings


# Environment-specific configurations
if settings.is_production:
    # Production optimizations
    settings.DEBUG = False
    settings.LOG_LEVEL = "WARNING"
    settings.DATABASE_POOL_SIZE = 20
    settings.REDIS_POOL_SIZE = 20
    settings.RATE_LIMIT_REQUESTS = 1000
    
elif settings.is_development:
    # Development configurations
    settings.DEBUG = True
    settings.LOG_LEVEL = "DEBUG"
    settings.DATABASE_POOL_SIZE = 5
    settings.REDIS_POOL_SIZE = 5