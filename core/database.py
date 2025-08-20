"""
ChefoodAI Database Configuration
Async PostgreSQL database setup with SQLAlchemy 2.0
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, declared_attr
from sqlalchemy.pool import QueuePool
from sqlalchemy import MetaData, event, text
from contextlib import asynccontextmanager
import structlog
from typing import AsyncGenerator

from core.config_simple import settings

logger = structlog.get_logger()

# Database engine
engine = None
async_session_factory = None


class Base(DeclarativeBase):
    """Base class for all database models"""
    
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s"
        }
    )
    
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()


async def init_db() -> None:
    """Initialize database connection and create tables"""
    global engine, async_session_factory
    
    try:
        # Create async engine with optimized settings
        engine = create_async_engine(
            settings.database_url_async,
            poolclass=QueuePool,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
            pool_timeout=settings.DATABASE_POOL_TIMEOUT,
            pool_pre_ping=True,
            pool_recycle=3600,  # 1 hour
            echo=settings.DEBUG,
            echo_pool=settings.DEBUG,
        )
        
        # Create session factory
        async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False
        )
        
        # Test connection
        async with engine.begin() as conn:
            await conn.run_sync(lambda _: None)
        
        logger.info("Database connection initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise


async def close_db() -> None:
    """Close database connections"""
    global engine
    
    if engine:
        await engine.dispose()
        logger.info("Database connections closed")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions
    Provides automatic transaction management and cleanup
    """
    if not async_session_factory:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI to get database session
    """
    async with get_db_session() as session:
        yield session


# Connection event handlers will be set up after engine is created


class DatabaseHealthCheck:
    """Database health check utilities"""
    
    @staticmethod
    async def check_connection() -> bool:
        """Check if database connection is healthy"""
        try:
            async with get_db_session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    @staticmethod
    async def get_connection_info() -> dict:
        """Get database connection information"""
        try:
            if not engine:
                return {"status": "not_initialized"}
            
            pool = engine.pool
            return {
                "status": "healthy",
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid()
            }
        except Exception as e:
            logger.error(f"Failed to get connection info: {str(e)}")
            return {"status": "error", "error": str(e)}


# Export commonly used items
__all__ = [
    "Base",
    "engine",
    "async_session_factory",
    "init_db",
    "close_db",
    "get_db_session",
    "get_db",
    "DatabaseHealthCheck"
]