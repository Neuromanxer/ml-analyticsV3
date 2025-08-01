# ────────────────────────────────────────────────────────────────
# Standard library imports
# ────────────────────────────────────────────────────────────────
import os
import logging
import secrets
from datetime import datetime, timedelta, date
from contextlib import contextmanager
from typing import Optional, List, Dict, Generator, Any
from fastapi import APIRouter, Depends, HTTPException, status, Form, Path
# ────────────────────────────────────────────────────────────────
# Third‐party imports
# ────────────────────────────────────────────────────────────────
import pandas as pd
import psycopg2
from jose import JWTError
import jwt                            # If you are actually using PyJWT
from passlib.context import CryptContext

from fastapi import (
    FastAPI,
    APIRouter,
    HTTPException,
    Depends,
    status,
    Body,
    Form,
    Path as FPath,   # rename FastAPI’s Path to FPath to avoid conflicts with pathlib.Path if needed
    Query,
)
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware



from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    LargeBinary,
    ForeignKey,
    JSON,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    sessionmaker,
    Session as SQLAlchemySession,
)

# ────────────────────────────────────────────────────────────────
# Local‐app imports (from your own modules)
# ────────────────────────────────────────────────────────────────

# from .auth import (
#     # Authentication & token utilities
#     get_current_active_user,
#     create_access_token,
#     authenticate_user,
#     User,
#     # Pydantic schemas for auth
#     UserCreate,
#     UserResponse,
#     Token,
#     AuthTokenResponse,
#     RegisterResponse,
#     Session,
#     # Database/session helpers
#     get_user_session,
#     get_user_session_direct,
#     get_user_engine,
#     get_user_by_email,

# )
from auth import (
    # Authentication & token utilities
    get_current_active_user,
    create_access_token,
    authenticate_user,
    User,
    # Pydantic schemas for auth
    UserCreate,
    UserResponse,
    Token,
    AuthTokenResponse,
    RegisterResponse,
    Session,
    # Database/session helpers
    get_user_session,
    get_user_session_direct,
    get_user_engine,
    get_user_by_email,

)

# Base configuration
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "ethanhong")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "printing")
MASTER_DB_NAME = os.environ.get("MASTER_DB_NAME", "master_ml_insights")
REFRESH_TOKEN_EXPIRE_DAYS = 7  # Example: Refresh token expires after 7 days
# Master database for user management
MASTER_DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{MASTER_DB_NAME}"
master_engine = create_engine(MASTER_DB_URL)
MasterSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=master_engine)
# Master database for user management
MASTER_DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{MASTER_DB_NAME}"
master_engine = create_engine(MASTER_DB_URL)
MasterSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=master_engine)

# Dictionary to store user-specific engines and sessionmakers
user_engines = {}
user_sessionmakers: dict[str, sessionmaker] = {}

Base = declarative_base()
logger = logging.getLogger(__name__)
# Auth configuration - move to environment variables in production
SECRET_KEY = "printing"  # CHANGE THIS IN PRODUCTION!
ALGORITHM = "HS256"
# Password hashing utilities
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
@property
def password(self):
    return self.hashed_password
# Router setup
d_router = APIRouter(prefix="/datasets", tags=["datasets"])


@contextmanager
def get_user_db(current_user: User):
    """
    Context manager to get the session for the current user's database.
    """
    db = None
    try:
        engine = get_user_engine(current_user)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()  # Create a session
        yield db  # Yield the session to the caller
    except Exception as e:
        logger.error(f"Error creating session for user {current_user.email}: {str(e)}")
        raise  # Raise exception if anything goes wrong
    finally:
        if db is not None:
            db.close()  # Ensure the session is closed when done

@contextmanager
def master_db_cm() -> Generator[Session, None, None]:
    db = MasterSessionLocal()
    try:
        yield db
        db.commit()
    except:
        db.rollback()
        raise
    finally:
        db.close()
def init_master_db():
    """Initialize the master database that stores user information."""
    logger.info("Initializing master database")
     
    # Connect to postgres as superuser
    conn_str = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{MASTER_DB_NAME}"
    "?sslmode=require"
    )

    superuser_engine = create_engine(conn_str, isolation_level="AUTOCOMMIT")

    try:
        with superuser_engine.connect() as conn:
            result = conn.execute(text("SELECT 1 FROM pg_database WHERE datname='master_ml_insights'"))
            exists = result.fetchone()
            if exists is None:
                conn.execute(text("CREATE DATABASE master_ml_insights"))
                logger.info("Created master_ml_insights database")
                # Set owner
                conn.execute(text(f"ALTER DATABASE master_ml_insights OWNER TO {POSTGRES_USER}"))
            
            # Grant connect privileges 
            conn.execute(text(f"GRANT ALL PRIVILEGES ON DATABASE master_ml_insights TO {POSTGRES_USER}"))
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL: {str(e)}")
        raise

    try:
        Base.metadata.create_all(bind=master_engine)
        
        with master_engine.connect() as conn:
            conn.execute(text(f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {POSTGRES_USER}"))
            conn.execute(text(f"GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {POSTGRES_USER}"))
            conn.execute(text(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {POSTGRES_USER}"))

        logger.info("Master database initialization complete")
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        raise
def init_db():
    """Initialize the master database tables."""
    init_master_db()
    logger.info("Database system initialization complete")

if __name__ == "__main__":
    init_db()