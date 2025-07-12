from fastapi import APIRouter, Depends, HTTPException, status, Form, Path
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional, List
import jwt
from passlib.context import CryptContext
import logging
from sqlalchemy.orm import Session
import os
from contextlib import contextmanager
import secrets
from jose import JWTError
from sqlalchemy import create_engine, Column, Integer, String, Float, Table, MetaData, Boolean, DateTime, ForeignKey, Text, text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from sqlalchemy.dialects.postgresql import JSONB
import pandas as pd
from datetime import datetime
import os
import logging
import psycopg2
import contextlib
from typing import Optional, Dict, Generator, Any
from fastapi import APIRouter, HTTPException, Depends, status, Body
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.responses import FileResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy import LargeBinary, DateTime, ForeignKey
from datetime import datetime
from sqlalchemy import BigInteger, Float
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "ethanhong")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "printing")
REFRESH_TOKEN_EXPIRE_DAYS = 7  # Example: Refresh token expires after 7 days
MASTER_DB_NAME = os.environ.get("MASTER_DB_NAME", "master_ml_insights")

# Master database for user management
MASTER_DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{MASTER_DB_NAME}"

master_engine = create_engine(MASTER_DB_URL)

SUPERUSER_DB_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{MASTER_DB_NAME}"
    "?sslmode=require"
)

superuser_engine = create_engine(SUPERUSER_DB_URL, isolation_level="AUTOCOMMIT")

MasterSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=master_engine)

# Dictionary to store user-specific engines and sessionmakers
user_engines = {}
user_sessionmakers: dict[str, sessionmaker] = {}

Base = declarative_base()
Base.metadata.create_all(bind=master_engine)
IMAGES_DIR = "images"
# Auth configuration - move to environment variables in production
SECRET_KEY = "printing"  # CHANGE THIS IN PRODUCTION!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Time for access token expiry
ACCESS_TOKEN_EXPIRE_MINUTES = 30
# Password hashing utilities
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
@property
def password(self):
    return self.hashed_password
# Router setup
router = APIRouter(prefix="/api/auth", tags=["auth"])
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
    
# Models
class TokenData(BaseModel):
    email: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    hashed_password: str


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    is_active: bool
    created_at: datetime
    tokens: float

    class Config:
        orm_mode = True
class AuthTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class RegisterResponse(BaseModel):
    user: UserResponse
    token: AuthTokenResponse
class RefreshTokenRequest(BaseModel):
    refresh_token: str
class User(Base):
    """
    User model representing a user in the master database.
    We’ve added all of the profile fields that your endpoint expects.
    """
    __tablename__ = "users"

    id                = Column(Integer, primary_key=True, index=True)
    email             = Column(String, unique=True, index=True, nullable=False)
    hashed_password   = Column(String, nullable=False)
    is_active         = Column(Boolean, default=True)
    created_at        = Column(DateTime, default=datetime.utcnow)
    db_name           = Column(String, unique=True, nullable=False)
    db_user           = Column(String, unique=True, nullable=False)
    db_password       = Column(String, unique=True, nullable=False)
    # ───────── Profile fields ─────────
    first_name        = Column(String, nullable=True)
    last_name         = Column(String, nullable=True)
    phone             = Column(String, nullable=True)
    company           = Column(String, nullable=True)
    role              = Column(String, nullable=True, default="user")
    bio               = Column(Text, nullable=True)
    notifications_pref= Column(String, default="all")   # e.g. 'all', 'important', 'none'
    timezone          = Column(String, default="utc")
    billing = relationship("BillingTable", back_populates="user", uselist=False)
    # ───────── Usage / billing fields ─────────
    storage_used      = Column(Float, default=0.0)    # e.g. in MB or GB
    # … existing columns …
    total_bytes_processed   = Column(BigInteger, default=0)      # in bytes
    total_compute_seconds   = Column(Float,      default=0.0)    # in seconds
    prod_api_key      = Column(String, nullable=True)
    dev_api_key       = Column(String, nullable=True)
    total_cost_dollars = Column(Float, default=0.0)
    # ───────── Subscription / plan fields ─────────
    subscription      = Column(String, default="Free")  # e.g. 'Free', 'Pro', 'Enterprise'
    stripe_customer_id          = Column(String, nullable=True)
    stripe_subscription_id      = Column(String, nullable=True)
    stripe_data_item_id         = Column(String, nullable=True)
    stripe_compute_item_id      = Column(String, nullable=True)
        # <-- define this to match back_populates on ActivityLog
    activity_logs = relationship(
        "ActivityLog",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    tokens = Column(Integer, default=0)  # or 0, or however you initialize
    token_usage_logs = relationship("TokenUsageLog", back_populates="user")
    def __repr__(self):
        return (
            f"User(id={self.id}, email={self.email}, "
            f"first_name={self.first_name}, last_name={self.last_name}, "
            f"subscription={self.subscription})"
        )



# OAuth2 scheme - Update tokenUrl to include the full path
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")
# Single unified function to get master DB session


def require_role(allowed_roles: list[str]):
    def role_checker(current_user: User = Depends(get_current_active_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {allowed_roles}"
            )
        return current_user
    return role_checker

from typing import Generator
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
def get_master_db_session() -> Generator[Session, None, None]:
    db = MasterSessionLocal()
    try:
        yield db
        db.commit()
    except:
        db.rollback()
        raise
    finally:
        db.close()
# Password handling utilities
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)
# Function to create a new access token
def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict, expires_delta: timedelta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Update the refresh token function to be more robust
def refresh_access_token(refresh_token: str):
    """Validate a refresh token and create a new access token."""
    try:
        # Decode the refresh token and verify the payload
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        
        if email is None:
            return None
        
        # Create a new access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        new_access_token = create_access_token(
            data={"sub": email}, 
            expires_delta=access_token_expires
        )
        
        return new_access_token

    except (ExpiredSignatureError, PyJWTError):
        return None

    except JWTError as e:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def authenticate_user(email: str, password: str, db: Session):
    user = get_user_by_email(db, email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user



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

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import ExpiredSignatureError, PyJWTError
from sqlalchemy.orm import Session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_master_db_session),
) -> User:
    # 1) First, try API‐key lookup
    user = (
        db.query(User)
          .filter(
            (User.prod_api_key == token) |
            (User.dev_api_key  == token)
          )
          .first()
    )
    if user:
        return user

    # 2) Then treat it as a JWT
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise credentials_exception
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except PyJWTError:
        raise credentials_exception

    # 3) Look up the user by email in the DB
    user = get_user_by_email(db, email)
    if not user:
        raise credentials_exception

    return user
async def get_current_active_user(current_user: User = Depends(get_current_user),) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
def create_password_reset_token(email: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=1)  # 1 hour validity
    to_encode = {"sub": email, "exp": expire}
    reset_token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return reset_token
def verify_password_reset_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise ValueError("Invalid token payload")
        return email
    except (jwt.ExpiredSignatureError, jwt.PyJWTError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )
from pydantic import BaseModel, EmailStr

class PasswordResetRequest(BaseModel):
    email: EmailStr

@router.post("/request-password-reset")
async def request_password_reset(
    req: PasswordResetRequest,
    db: Session = Depends(get_master_db_session)
):
    user = get_user_by_email(db, req.email)
    if not user:
        # Do not reveal if email exists
        return {"message": "If this email exists, you will receive reset instructions."}

    reset_token = create_password_reset_token(user.email)

    # You would normally email this token to user
    # For now we just return it directly
    return {"reset_token": reset_token}
class PasswordResetSubmit(BaseModel):
    reset_token: str
    new_password: str

@router.post("/reset-password")
async def reset_password(
    req: PasswordResetSubmit,
    db: Session = Depends(get_master_db_session)
):
    email = verify_password_reset_token(req.reset_token)
    user = get_user_by_email(db, email)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid user")

    user.hashed_password = get_password_hash(req.new_password)
    db.commit()
    db.refresh(user)

    return {"message": "Password has been successfully reset."}


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_user_endpoint(user: UserCreate, db: Session = Depends(get_master_db_session)):
    # Check if the user already exists in the database
    db_user = get_user_by_email(db, user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    try:
        # Hash the user's password
        hashed_password = get_password_hash(user.hashed_password)

        # Register the user and create a new user database/schema
        db_user = register_user(user.name, user.email, hashed_password, db)

      
        # Create access and refresh tokens for the user
        access_token = create_access_token(data={"sub": db_user.email})
        refresh_token = create_refresh_token(data={"sub": db_user.email})

        # Return the response with the user info and generated tokens
        return RegisterResponse(
            user=UserResponse(
                id=db_user.id,
                name=db_user.first_name,
                email=db_user.email,
                is_active=db_user.is_active,
                created_at=db_user.created_at,
                tokens=db_user.tokens  # <-- this fixes it
            ),
            token=AuthTokenResponse(
                access_token=access_token,
                refresh_token=refresh_token
            )
        )

    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed.")

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return UserResponse(
        id=current_user.id,
        name=current_user.first_name,
        email=current_user.email,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        tokens=current_user.tokens or 0.0
    )

@router.delete("/me")
async def delete_user_account(current_user = Depends(get_current_active_user)):
    """
    Delete the current user's account and all associated data.
    This action is irreversible.
    """
    try:
        success = delete_user(current_user.email)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete user")
        
        return {"message": "User account deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")

# First, let's fix the token endpoint to properly return both tokens
@router.post("/token", response_model=AuthTokenResponse)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_master_db_session)
):
    """Get access token and refresh token for authentication."""
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        logger.warning(f"Failed login attempt for email: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info(f"User logged in successfully: {form_data.username}")

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    # Create refresh token
    refresh_token_expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    refresh_token = create_refresh_token(data={"sub": user.email}, expires_delta=refresh_token_expires)

    return {
        "access_token": access_token, 
        "refresh_token": refresh_token, 
        "token_type": "bearer"
    }


# Now, let's add a proper refresh token endpoint
@router.post("/refresh-token", response_model=Token)
async def refresh_token_endpoint(
    refresh_data: RefreshTokenRequest,
    db: Session = Depends(get_master_db_session)
):
    """Refresh access token using a valid refresh token."""
    try:
        # Decode and verify the refresh token
        payload = jwt.decode(
            refresh_data.refresh_token, 
            SECRET_KEY, 
            algorithms=[ALGORITHM]
        )
        email = payload.get("sub")
        
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # Get the user from the database
        user = get_user_by_email(db, email)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # Create new access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, 
            expires_delta=access_token_expires
        )
        
        refresh_token_expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        new_refresh_token = create_refresh_token(
            data={"sub": user.email}, 
            expires_delta=refresh_token_expires
        )
        
        logger.info(f"Token refreshed successfully for: {email}")
        
        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer"
        }
        
    except JWTError:
        logger.warning("JWT decode error during token refresh")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Error during token refresh: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed",
        )




def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()

@router.post("/reinit-user", status_code=status.HTTP_200_OK)
async def reinitialize_user(
    email: str = Form(...),
    db: Session = Depends(get_master_db_session)
):
    # Retrieve the user based on email
    db_user_record = get_user_by_email(db, email)
    if not db_user_record:
        raise HTTPException(status_code=404, detail="User not found")

    # Create role name and a new password
    uid = db_user_record.id  # Assuming the user already has an ID
    safe_email = email.replace("@", "_at_").replace(".", "_dot_")
    db_user = f"ml_user_{safe_email}_{uid}"
    db_password = secrets.token_urlsafe(16)

    try:
        with master_engine.connect() as conn:
            # Create role if it doesn't exist
            result = conn.execute(
                text("SELECT 1 FROM pg_roles WHERE rolname = :role_name"),
                {"role_name": db_user}
            ).fetchone()

            if not result:
                logger.info(f"Creating role {db_user}")
                conn.execute(
                    text(f"CREATE ROLE \"{db_user}\" LOGIN PASSWORD :password"),
                    {"password": db_password}
                )
            else:
                logger.info(f"Role {db_user} already exists. Skipping creation.")

            # Grant INSERT, SELECT on datasets table
            conn.execute(
                text(f"GRANT INSERT, SELECT ON public.datasets TO \"{db_user}\"")
            )

            # Confirm permissions
            permissions_check = conn.execute(
                text("""
                    SELECT grantee, privilege_type
                    FROM information_schema.role_table_grants
                    WHERE table_name = 'datasets' AND grantee = :role_name
                """),
                {"role_name": db_user}
            ).fetchall()

            if permissions_check:
                logger.info(f"Permissions granted: {permissions_check}")
            else:
                logger.warning(f"No permissions found for {db_user} on 'datasets'.")

        # Update user's db_user and db_password fields
        db_user_record.db_user = db_user
        db_user_record.db_password = db_password
        db.commit()

        logger.info(f"Reinitialized user: {email}")
        return {
            "message": f"User {email} reinitialized successfully",
            "db_user": db_user,
            "db_password": db_password
        }

    except Exception as e:
        logger.error(f"Error reinitializing user: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Reinitialization failed")
import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path as PathL  # Your existing alias
from fastapi import HTTPException, Depends, Query
import uuid
from io import BytesIO
from sqlalchemy import text
def _get_meta_path(user_id: str) -> PathL:
    """Get the metadata file path for a user."""
    meta_dir = PathL("data") / "visualizations"
    meta_dir.mkdir(parents=True, exist_ok=True)
    safe_user_id = str(user_id).replace("/", "_").replace("\\", "_")
    return meta_dir / f"{safe_user_id}.json"
from sqlalchemy import text
def _save_metadata(user_id: str, data: List[Dict[str, Any]], db: Session) -> None:
    """Overwrite all metadata for a user (not common, but preserved)."""
    try:
        db.execute(text("DELETE FROM visualizations_metadata WHERE user_id = :user_id"), {"user_id": user_id})
        for entry in data:
            db.execute(text("""
                INSERT INTO visualizations_metadata (id, user_id, type, created_at, metadata)
                VALUES (:id, :user_id, :type, :created_at, :metadata)
            """), {
                "id": str(uuid4()),
                "user_id": str(user_id),
                "type": entry.get("type", "other"),
                "created_at": entry.get("created_at", datetime.utcnow().isoformat()),
                "metadata": json.dumps(entry)
            })
        db.commit()
        logging.info(f"✅ Saved {len(data)} metadata entries for user {user_id}")
    except Exception as e:
        logging.error(f"❌ Error saving metadata for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error saving metadata")


def _load_metadata(user_id: str, db: Session) -> List[Dict[str, Any]]:
    """Load metadata for a user from PostgreSQL."""
    try:
        result = db.execute(text("""
            SELECT metadata FROM visualizations_metadata
            WHERE user_id = :user_id
            ORDER BY created_at DESC
        """), {"user_id": str(user_id)})
        metadata_list = [json.loads(row[0]) for row in result.fetchall()]
        return metadata_list
    except Exception as e:
        logging.error(f"❌ Error loading metadata for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading metadata")


def _append_metadata(user_id: str, new_entry: Dict[str, Any], db: Session) -> None:
    """Append a new metadata entry to the user's record."""
    try:
        metadata_id = new_entry.get("id") or str(uuid4())
        new_entry["id"] = metadata_id  # Ensure ID is set in metadata too

        db.execute(text("""
            INSERT INTO visualizations_metadata (id, user_id, type, created_at, metadata)
            VALUES (:id, :user_id, :type, :created_at, :metadata)
        """), {
            "id": metadata_id,
            "user_id": str(user_id),
            "type": new_entry.get("type", "other"),
            "created_at": new_entry.get("created_at", datetime.utcnow().isoformat()),
            "metadata": json.dumps(new_entry)
        })
        db.commit()
        logging.info(f"✅ Appended metadata entry {metadata_id} for user {user_id}")

    except Exception as e:
        logging.error(f"❌ Error appending metadata for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error appending metadata")

def _append_limited_metadata(user_id: str, new_entry: Dict[str, Any], db: Session, max_entries: int = 5):
    """Append a new entry and keep only the most recent ones for that entry's type."""
    try:
        current_type = new_entry.get("type", "other")

        # Fetch entries of same type
        result = db.execute(text("""
            SELECT id, created_at FROM visualizations_metadata
            WHERE user_id = :user_id AND type = :type
            ORDER BY created_at DESC
        """), {"user_id": str(user_id), "type": current_type})
        entries = result.fetchall()

        # Delete oldest if over the limit
        if len(entries) >= max_entries:
            to_delete = entries[max_entries - 1:]
            for row in to_delete:
                db.execute(text("""
                    DELETE FROM visualizations_metadata WHERE id = :id
                """), {"id": row[0]})

        # Insert the new entry
        db.execute(text("""
            INSERT INTO visualizations_metadata (id, user_id, type, created_at, metadata)
            VALUES (:id, :user_id, :type, :created_at, :metadata)
        """), {
            "id": new_entry["id"],
            "user_id": str(user_id),
            "type": current_type,
            "created_at": new_entry.get("created_at", datetime.utcnow().isoformat()),
            "metadata": json.dumps(new_entry)
        })

        db.commit()
        logging.info(f"✅ Saved metadata entry for user {user_id} [type={current_type}]")

    except Exception as e:
        logging.error(f"❌ Failed to append limited metadata for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error appending limited metadata")

@contextmanager
def get_user_session_direct(db_name: str):
    """
    Get a session for the user's database using a context manager.
    """
    engine = get_user_engine(db_name)  # assume this just needs db_name
    if db_name not in user_sessionmakers:
        user_sessionmakers[db_name] = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )

    session = user_sessionmakers[db_name]()
    try:
        yield session
    finally:
        session.close()


def get_user_engine(current_user):
    """Create a SQLAlchemy engine for the current user's database using their email as username."""
    # Use a secure method to fetch the password (e.g., environment variable or secure vault)
    password = current_user.hashed_password  # Ensure this is securely fetched
    
    # It's a security risk to include passwords directly in the URL, consider using a safer approach
    engine_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{current_user.db_name}"
    
    return create_engine(engine_url)


# Single unified function to get user DB session
@contextlib.contextmanager
def get_user_session(db_name):
    """Get a session for the user's database using context manager pattern."""
    engine = get_user_engine(db_name)
    if db_name not in user_sessionmakers:
        user_sessionmakers[db_name] = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    session = user_sessionmakers[db_name]()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def register_user(name: str, email: str, hashed_password: str, db: Session) -> User:
    try:
        # Step 1: Create a per-user database (or schema) for isolation
        db_name = create_user_database(email)

        # Generate a safe db_user name and password for the user
        safe_email = email.replace('@', '_at_').replace('.', '_dot_')
        uid = secrets.token_urlsafe(8)
        db_user = f"ml_user_{safe_email}_{uid}"
        db_password = secrets.token_urlsafe(16)

        # Step 2: Register user in the master DB
        user = User(
            first_name=name,
            email=email,
            hashed_password=hashed_password,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password  # Store the generated password safely
        )
        db.add(user)
        db.flush()  # To assign user.id before commit
        db.refresh(user)
        # Step 3: Grant INSERT and SELECT on 'datasets' to the user's role
        grant_sql = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT FROM pg_roles WHERE rolname = '{db_user}'
            ) THEN
                CREATE ROLE "{db_user}" LOGIN PASSWORD '{db_password}';
            END IF;

            GRANT INSERT, SELECT ON TABLE datasets TO "{db_user}";
        END $$;
        """


        try:
            # Execute the SQL for role creation and granting privileges
            with master_engine.connect() as conn:
                conn.execute(text(grant_sql))
        except Exception as e:
            db.rollback()
            raise Exception(f"Error creating role or granting privileges: {str(e)}")

        # Step 4: Get the engine for the user's database and create tables
        user_engine = get_user_engine(user)  # Get engine using the new user's info

        # Create the tables in the user's database
        Base.metadata.create_all(user_engine)

        db.commit()
        return user
    except Exception as e:
        db.rollback()
        raise Exception(f"Error during user registration: {str(e)}")

def delete_user(user_email: str) -> bool:
    """
    Delete a user and their entire database.
    Use with caution as this is irreversible.
    """
    # 1) Look up the User in the master DB
    with get_master_db_session() as master_db:
        user = get_user_by_email(master_db, user_email)
        if not user:
            raise ValueError(f"User {user_email} not found")
        db_name = user.db_name

        # 2) Delete the User record
        master_db.delete(user)
        master_db.commit()
        logger.info(f"Deleted user record for {user_email}")

    # 3) Drop the user's database via superuser connection
    conn_str = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{MASTER_DB_NAME}"
    "?sslmode=require"
    )

    engine = create_engine(conn_str, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}"'))
            logger.info(f"Dropped database {db_name}")
    except Exception as e:
        logger.error(f"Error dropping database {db_name}: {e}")
        raise
    finally:
        engine.dispose()

    # 4) Clean up any cached engines or sessionmakers
    if db_name in user_engines:
        user_engines[db_name].dispose()
        del user_engines[db_name]
    if db_name in user_sessionmakers:
        del user_sessionmakers[db_name]

    logger.info(f"User {user_email} and database {db_name} deleted successfully")
    return True

def create_user_database(email: str) -> str:
    """Create a new database for the user and return the database name."""

    # Sanitize email into valid DB name
    safe_email = email.replace('@', '_at_').replace('.', '_dot_')
    db_name = f"ml_user_{safe_email}"

    # Connect to the postgres system database using superuser privileges
    with superuser_engine.connect() as conn:
        # Check if database already exists
        result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"))
        if result.fetchone():
            logger.info(f"Database {db_name} already exists")
        else:
            conn.execute(text(f"CREATE DATABASE {db_name}"))
            logger.info(f"✅ Successfully created database: {db_name}")

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

@router.get(
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
