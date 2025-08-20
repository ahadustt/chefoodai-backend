"""
ChefoodAI Alembic Environment
Database migration environment configuration
"""

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import settings
from core.database import Base

# Import all models to ensure they're registered with SQLAlchemy
from models import *

# Alembic Config object
config = context.config

# Setup logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate support
target_metadata = Base.metadata

# Override database URL from environment
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    This configures the context with just a URL and not an Engine.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    Creates an Engine and associates a connection with the context.
    """
    # Override URL with async version for migration compatibility
    database_url = settings.DATABASE_URL
    
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = database_url
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_schemas=True,
            # Custom compare functions for better migration generation
            compare_server_default=True,
            render_as_batch=False,
        )

        with context.begin_transaction():
            context.run_migrations()


# Run migrations based on context
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()