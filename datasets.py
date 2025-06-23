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

from pydantic import BaseModel, Field, EmailStr

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

# These names should match exactly what you export from auth.py
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
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "securepassword")
REFRESH_TOKEN_EXPIRE_DAYS = 7  # Example: Refresh token expires after 7 days
# Master database for user management
MASTER_DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/master_ml_insights"
master_engine = create_engine(MASTER_DB_URL)
MasterSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=master_engine)
# Master database for user management
MASTER_DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/master_ml_insights"
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

class Dataset(Base):
    """
    Dataset model representing a CSV dataset.
    This class is used for ORM mapping in user-specific databases.
    """
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    table_name = Column(String, unique=True, index=True)  # Actual table name in the DB
    file_path = Column(String)  # Path to the original CSV file
    row_count = Column(Integer)
    column_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Schema information stored as JSON
    schema = Column(JSONB)
    
    def __repr__(self):
        return f"Dataset(id={self.id}, name={self.name}, table_name={self.table_name})"
class DatasetResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    created_by: str
    created_at: datetime

    class Config:
        orm_mode = True
class DatasetCreate(BaseModel):
    name: str = Field(..., max_length=100, description="Name of the dataset")
    description: str | None = Field(None, description="Optional description of the dataset")
    
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
def register_dataset(
    db: Session,
    name: str,
    description: str,
    table_name: str,
    file_path: str,
    row_count: int,
    column_count: int,
    schema: dict,
    overwrite_existing: bool = False
) -> Dataset:
    try:
        existing = db.query(Dataset).filter(Dataset.table_name == table_name).first()

        if existing:
            if not overwrite_existing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset '{table_name}' already exists. Use `overwrite_existing=true` to replace it."
                )

            # 🔁 Clean up table and metadata
            logger.info(f"Overwriting dataset '{table_name}'")
            db.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))
            db.delete(existing)
            db.flush()

        dataset = Dataset(
            name=name,
            description=description,
            table_name=table_name,
            file_path=file_path,
            row_count=row_count,
            column_count=column_count,
            schema=schema
        )
        db.add(dataset)
        db.flush()

        logger.info(f"✅ Dataset '{name}' registered with ID {dataset.id}")
        return dataset

    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"SQLAlchemy error: {e}")
        raise HTTPException(status_code=500, detail="Error registering dataset.")

from sqlalchemy import text

@d_router.get(
    "/{dataset_id}/download",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK
)
async def download_dataset(
    dataset_id: int = Path(..., gt=0),
    current_user = Depends(get_current_active_user)
):
    """
    Stream the raw CSV file for a user's dataset.
    """
    try:
        # 1) Look up the Dataset in the user's DB
        with get_user_db(current_user) as db:
            ds = (
                db
                .query(Dataset)
                .filter(Dataset.id == dataset_id)
                .first()
            )
            if not ds:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Dataset not found"
                )

        # 2) Verify the file exists on disk
        if not ds.file_path or not os.path.isfile(ds.file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="CSV file not found on server"
            )

        # 3) Return the file for download
        return FileResponse(
            path=ds.file_path,
            filename=f"{ds.name}.csv",
            media_type="text/csv"
        )

    except HTTPException:
        # Re-raise 404 or auth errors
        raise
    except Exception as e:
        logger.error(f"Error in download_dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while preparing download"
        )

def init_master_db():
    """Initialize the master database that stores user information."""
    logger.info("Initializing master database")
     
    # Connect to postgres as superuser
    conn_str = f"postgresql://postgres:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/postgres"
    superuser_engine = create_engine(conn_str, isolation_level="AUTOCOMMIT")

    try:
        with superuser_engine.connect() as conn:
            result = conn.execute(text("SELECT 1 FROM pg_database WHERE datname='master_ml_insights'"))
            exists = result.fetchone()
            if not exists:
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

# def create_user_database(user_email: str) -> str:
#     """Create a dedicated PostgreSQL database for a user."""
#     clean_name = ''.join(c for c in user_email.split('@')[0] if c.isalnum() or c == '_')
#     db_name = f"ml_user_{clean_name}_{int(datetime.now().timestamp())}"

#     with psycopg2.connect(
#         dbname='postgres',
#         user=POSTGRES_USER,
#         hashed_password=POSTGRES_PASSWORD,
#         host=POSTGRES_HOST,
#         port=POSTGRES_PORT
#     ) as conn:
#         conn.autocommit = True
#         with conn.cursor() as cursor:
#             cursor.execute(f"CREATE DATABASE {db_name}")
    
#     logger.info(f"Created new database '{db_name}' for user {user_email}")
#     return db_name

def create_user_database(email: str) -> str:
    """Create a new database for the user and return the database name."""
    safe_email = email.replace('@', '_at_').replace('.', '_dot_')
    db_name = f"ml_user_{safe_email}"

    # Connect to the master engine and create the new database for the user
    with master_engine.connect() as conn:
        conn.execute(f"CREATE DATABASE {db_name}")

    # Return the database name (this will be used when connecting to the user's DB)
    return db_name

def get_dataset_by_id(
    user_email: str,
    dataset_id: int
) -> Optional[Dataset]:
    """
    Master-DB lookup to find the User, then per-user DB lookup for the Dataset.
    """
    # 1) Master-DB lookup for the User
    with master_db_cm() as master_db:
        user: User = get_user_by_email(master_db, user_email)
        if not user:
            raise HTTPException(status_code=404, detail=f"User {user_email} not found")

    # 2) Per-user lookup
    with get_user_db(user) as user_db:
        return (
            user_db
            .query(Dataset)
            .filter(Dataset.id == dataset_id)
            .first()
        )
def get_dataset_data(
    user_email: str,
    dataset_id: int,
    limit: int = 100
) -> Optional[Dict[str, Any]]:
    """
    Retrieve up to `limit` rows from a user's dataset table.
    Returns a dict with:
      - "dataset": the Dataset object
      - "columns": list of column names
      - "data": list of row-dicts
    """
    # 1) Look up User in the master DB
    with get_master_db_session() as master_db:
        user = get_user_by_email(master_db, user_email)
        if not user:
            raise ValueError(f"User {user_email} not found")

    try:
        # 2) Session to user's DB
        with get_user_session(user.db_name) as db:
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                return None

        # 3) Use the user-specific engine to SELECT * from their table
        engine = get_user_engine(user)
        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT * FROM {dataset.table_name} LIMIT :limit"),
                {"limit": limit}
            )
            cols = result.keys()
            rows = result.fetchall()

        return {
            "dataset": dataset,
            "columns": cols,
            "data": [dict(zip(cols, row)) for row in rows],
        }

    except Exception as e:
        logger.error(f"Error retrieving dataset data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving dataset data: {e}"
        )
def delete_dataset_crud(
    user_email: str,
    dataset_id: int
) -> Optional[Dataset]:
    # 1) Find user in the *master* DB
    with MasterSessionLocal() as master_db:
        user: User = get_user_by_email(master_db, user_email)
        if not user:
            logger.warning(f"User {user_email} not found")
            raise HTTPException(404, f"User {user_email} not found")
        db_name = user.db_name

    # 2) Build or retrieve the per‐user SessionLocal
    engine = get_user_engine(user)
    SessionLocal = user_sessionmakers.get(db_name)
    if SessionLocal is None:
        from sqlalchemy.orm import sessionmaker
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        user_sessionmakers[db_name] = SessionLocal

    # 3) Use a real SQLAlchemy Session context manager
    try:
        with SessionLocal() as db:
            ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not ds:
                return None

            # Drop the actual data table
            with engine.connect() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS "{ds.table_name}"'))

            # Copy what we need to return
            deleted = Dataset(id=ds.id, name=ds.name, created_at=ds.created_at)

            # Delete the record
            db.delete(ds)
            db.commit()

            logger.info(f"Deleted dataset {dataset_id} for user {user_email}")
            return deleted

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset {dataset_id}: {e}")
        raise HTTPException(500, f"Error deleting dataset: {e}")
def query_dataset(
    user_email: str,
    dataset_id: int,
    query: str
) -> Optional[Dict[str, Any]]:
    """
    Run a custom SELECT query against a dataset in the user's database.
    Returns a dict with keys "dataset", "columns", "data", "row_count",
    or None if the dataset doesn't exist.
    """
    # 1) Master‐DB lookup
    with get_master_db_session() as master_db:
        user = get_user_by_email(master_db, user_email)
        if not user:
            raise HTTPException(status_code=404, detail=f"User {user_email} not found")

    # 2) Validate and fetch Dataset record
    with get_user_session(user.db_name) as db:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return None

    # 3) Only allow SELECT statements
    sql = query.strip()
    if not sql.lower().startswith("select"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")

    try:
        # 4) Execute on the user's engine
        engine = get_user_engine(user)
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            cols = result.keys()
            rows = result.fetchall()

        return {
            "dataset": dataset,
            "columns": cols,
            "data": [dict(zip(cols, row)) for row in rows],
            "row_count": len(rows)
        }

    except Exception as e:
        logger.error(f"Error executing query on dataset {dataset_id} for user {user_email}: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing query: {e}")
def init_db():
    """Initialize the master database tables."""
    init_master_db()
    logger.info("Database system initialization complete")

if __name__ == "__main__":
    init_db()