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
import tempfile
import numpy as np
# from .storage import supabase, download_file_from_supabase

from storage import supabase, download_file_from_supabase



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
MASTER_DB_NAME = os.environ.get("MASTER_DB_NAME", "ml_insights_db")

DB_COMMON_ARGS = {
    "pool_pre_ping": True,        # check connection before using
    "pool_recycle": 300,          # recycle every 5 minutes
    "pool_size": 5,               # base pool (tune for load)
    "max_overflow": 10,           # extra bursts
    "connect_args": {
        "sslmode": "require",     # Supabase requires SSL
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    },
}

# Master database for user management
MASTER_DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{MASTER_DB_NAME}"

master_engine = create_engine(
    MASTER_DB_URL,
    **DB_COMMON_ARGS,
)


SUPERUSER_DB_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{MASTER_DB_NAME}"
    "?sslmode=require"
)

superuser_engine = create_engine(
    SUPERUSER_DB_URL,
    isolation_level="AUTOCOMMIT",
    **DB_COMMON_ARGS,
)
MasterSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=master_engine)

# Dictionary to store user-specific engines and sessionmakers
user_engines = {}
user_sessionmakers: dict[str, sessionmaker] = {}

Base = declarative_base()
Base.metadata.create_all(bind=master_engine)
IMAGES_DIR = "images"
# Auth configuration - move to environment variables in production
SECRET_KEY = os.getenv("SECRET_KEY", "")  # fallback for local testing
ALGORITHM = os.getenv("ALGORITHM", "HS256")

# Time for access token expiry
ACCESS_TOKEN_EXPIRE_MINUTES = 60
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
    agreed_to_terms: bool
    policy_version: Optional[str] = "v1.0"


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    is_active: bool
    created_at: datetime
    tokens: float

    class Config:
        orm_mode = True
class DataLocationOut(BaseModel):
    db_name: Optional[str]
    storage_bucket: Optional[str]
    storage_region: Optional[str]
    file_storage_path: Optional[str]  # e.g., "user-uploads/user123/"
    
    class Config:
        orm_mode = True
class AuthTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str


class RegisterResponse(BaseModel):
    user: UserResponse
    token: AuthTokenResponse
class RefreshTokenRequest(BaseModel):
    refresh_token: str

class TermsStatus(BaseModel):
    agreed_to_terms: bool
    agreed_at: datetime | None
    policy_version: str | None
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
    agreed_to_terms = Column(Boolean, default=False)
    agreed_at = Column(DateTime)
    policy_version = Column(String, default="v1.0")

    storage_bucket = Column(String, default="user-uploads")
    storage_region = Column(String, default="us-east-1")  # or auto-fetch

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
def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        # e.g., malformed/legacy hashes
        return False

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
# For decoding in middleware or routes
from jose import jwt, JWTError
from fastapi import Request
def decode_user_from_request(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        return None
    token = auth.split(" ", 1)[1].strip()
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except (ExpiredSignatureError, JWTError):
        return None

# --- Authentication ---
def authenticate_user(email: str, password: str, db: Session):
    """Return user object if credentials are valid, else None."""
    email = (email or "").strip().lower()
    user = get_user_by_email(db, email)
    if not user:
        # Optional: sleep a bit here to make user-not-found timing similar to bad password
        return None

    # DO NOT hash the incoming password yourself for comparison
    if not verify_password(password, user.hashed_password):
        return None

    # Optional: block inactive/suspended accounts
    # if not user.is_active:
    #     return None

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
from jose import jwt, JWTError, ExpiredSignatureError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_master_db_session),
) -> User:
    # 1) Check for API key match first
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

    # 2) Otherwise, treat token as JWT
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid JWT payload: missing email.",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="⏰ Session expired. Please reload and log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="⚠️ Invalid token. Please try again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3) Look up user by email from JWT
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    
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

@contextmanager
def get_user_db(current_user: User):
    import re
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
from pydantic import BaseModel, EmailStr

class PasswordResetRequest(BaseModel):
    email: EmailStr
import smtplib
from email.message import EmailMessage
import os
import smtplib
import ssl
from email.message import EmailMessage

async def send_email(to_email: str, subject: str, body: str):
    print("🚀 Attempting to send email via SMTP...")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.getenv("SMTP_USERNAME")  # use the same email you log in with
    msg["To"] = to_email
    msg.set_content(body)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(os.getenv("SMTP_HOST"), port=465, context=context) as server:
            server.login(os.getenv("SMTP_USERNAME"), os.getenv("SMTP_PASSWORD"))
            server.send_message(msg)
        print("✅ SMTP email sent.")
    except Exception as e:
        print(f"❌ SMTP Error: {e}")
        raise

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

async def sendgrid_email(to_email: str, subject: str, body: str):
    message = Mail(
        from_email=os.getenv("SENDGRID_FROM_EMAIL"),
        to_emails=to_email,
        subject=subject,
        plain_text_content=body
    )

    try:
        sg = SendGridAPIClient(api_key=os.getenv("SENDGRID_API_KEY"))
        response = sg.send(message)
        print(f"✅ SendGrid email sent to {to_email} - Status Code: {response.status_code}")
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(f"❌ SendGrid Error: {str(e)}")
        raise

@router.post("/request-password-reset")
async def request_password_reset(
    req: PasswordResetRequest,
    db: Session = Depends(get_master_db_session)
):
    sanitized_email = str(req.email).lower().strip()
    user = get_user_by_email(db, sanitized_email)

    if not user:
        # Return the same generic response to avoid revealing valid emails
        return {"message": "If this email exists, you will receive reset instructions."}

    reset_token = create_password_reset_token(user.email)
    reset_url = f"{os.getenv('FRONTEND_BASE_URL')}/reset-password?token={reset_token}"

    subject = "Reset Your Password"
    body = f"""
    Hello,

    You (or someone else) requested to reset your password. If it was you, click the link below:

    {reset_url}

    If you didn’t request this, you can safely ignore this email.

    – MLAnalytics
    """
    try:
        print("📧 Sending email via SMTP...")
        await send_email(to_email=user.email, subject=subject, body=body)
    except Exception as smtp_error:
        print(f"❌ SMTP failed: {smtp_error}")

    try:
        print("📧 Sending email via SendGrid...")
        await sendgrid_email(to_email=user.email, subject=subject, body=body)
    except Exception as sendgrid_error:
        print(f"❌ SendGrid failed: {sendgrid_error}")

    # Return success regardless of email result
    return {"message": "If this email exists, you will receive reset instructions."}
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
        db_user = register_user(
            name=user.name,
            email=user.email,
            hashed_password=hashed_password,
            db=db,
            agreed_to_terms=user.agreed_to_terms,
            policy_version=user.policy_version
        )


      
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
# @router.get("/user/data-location", response_model=DataLocationOut)
# def get_data_location(current_user: User = Depends(get_current_active_user)):
#     return {
#         "db_name": current_user.db_name,
#         "storage_bucket": current_user.storage_bucket,
#         "storage_region": current_user.storage_region,
#         "file_storage_path": f"{SUPABASE_BUCKET}/{current_user.id}/"
#     }
@router.delete("/admin/delete-user/{user_id}", tags=["admin"])
async def admin_delete_user(
    user_id: str,
    db: Session = Depends(get_master_db_session),
    current_user: User = Depends(require_role(["admin"]))
):
    # Do not allow deleting yourself via admin route
    if current_user.id == user_id:
        raise HTTPException(status_code=403, detail="Admins cannot delete themselves using this route.")
    
    user_to_delete = db.query(User).filter(User.id == user_id).first()

    if not user_to_delete:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user_to_delete.role == "admin":
        raise HTTPException(status_code=403, detail="Cannot delete other admins via API")

    db.delete(user_to_delete)
    db.commit()
    logger.info(f"[ADMIN ACTION] {current_user.email} deleted user {user_to_delete.email} at {datetime.utcnow()}")

    
    return {"message": f"User '{user_to_delete.email}' deleted successfully by admin."}
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

@router.post("/token", response_model=AuthTokenResponse)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_master_db_session)
):
    """Authenticate user and return access + refresh tokens."""
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
    refresh_token = create_refresh_token(
        data={"sub": user.email}, expires_delta=refresh_token_expires
    )

    # Return both tokens as JSON
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
    # ────── Step 1: Retrieve user ──────
    db_user_record = get_user_by_email(db, email)
    if not db_user_record:
        raise HTTPException(status_code=404, detail="User not found")

    uid = db_user_record.id
    safe_email = email.replace("@", "_at_").replace(".", "_dot_")
    db_user = f"ml_user_{safe_email}_{uid}"
    db_password = secrets.token_urlsafe(16)

    try:
        with master_engine.connect() as conn:
            # ────── Step 2: Create role if it doesn't exist ──────
            result = conn.execute(
                text("SELECT 1 FROM pg_roles WHERE rolname = :role_name"),
                {"role_name": db_user}
            ).fetchone()

            if not result:
                logger.info(f"🔧 Creating role {db_user}")
                conn.execute(
                    text(f"CREATE ROLE \"{db_user}\" LOGIN PASSWORD :password"),
                    {"password": db_password}
                )
            else:
                logger.info(f"✅ Role {db_user} already exists. Skipping creation.")

            # ────── Step 3: Revoke all privileges first ──────
            conn.execute(text(f"REVOKE ALL ON SCHEMA public FROM \"{db_user}\""))
            conn.execute(text(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM \"{db_user}\""))

            # ────── Step 4: Grant minimum required permissions ──────
            conn.execute(text(f"GRANT USAGE ON SCHEMA public TO \"{db_user}\""))
            conn.execute(text(f"GRANT INSERT, SELECT ON public.datasets TO \"{db_user}\""))
            conn.execute(text(f"GRANT INSERT, SELECT ON public.visualizations_metadata TO \"{db_user}\""))

            # ────── Step 5: Optional – Audit permissions ──────
            permissions_check = conn.execute(
                text("""
                    SELECT grantee, privilege_type, table_name
                    FROM information_schema.role_table_grants
                    WHERE grantee = :role_name
                """),
                {"role_name": db_user}
            ).fetchall()

            if permissions_check:
                logger.info(f"🔐 Permissions granted: {permissions_check}")
            else:
                logger.warning(f"⚠️ No permissions found for {db_user} on any table.")

        # ────── Step 6: Save db credentials to user record ──────
        db_user_record.db_user = db_user
        db_user_record.db_password = db_password
        db.commit()

        logger.info(f"✅ Reinitialized user: {email}")
        return {
            "message": f"User {email} reinitialized successfully",
            "db_user": db_user,
            "db_password": db_password
        }

    except Exception as e:
        logger.error(f"❌ Error reinitializing user: {str(e)}")
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
from uuid import uuid4
from sqlalchemy import text
from datetime import datetime

def _save_metadata(user_id: str, data: List[Dict[str, Any]], db: Session) -> None:
    """Overwrite all metadata for a user (not common, but preserved)."""
    try:
        # Delete previous metadata entries
        db.execute(
            text("DELETE FROM visualizations_metadata WHERE user_id = :user_id"),
            {"user_id": user_id}
        )

        for entry in data:
            entry_id = entry.get("id") or str(uuid4())
            entry["id"] = entry_id  # ensure ID is injected into the entry itself

            db.execute(
                text("""
                INSERT INTO visualizations_metadata (id, user_id, type, created_at, metadata)
                VALUES (:id, :user_id, :type, :created_at, :metadata)
                """),
                {
                    "id": entry_id,
                    "user_id": str(user_id),
                    "type": entry.get("type", "other"),
                    "created_at": entry.get("created_at", datetime.utcnow().isoformat()),
                    "metadata": json.dumps(entry, default=str)
                }
            )

        db.commit()
        logging.info(f"✅ Saved {len(data)} metadata entries for user {user_id}")

    except Exception as e:
        logging.error(f"❌ Error saving metadata for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error saving metadata")

import logging
import traceback
def _load_metadata(user_id: str, db: Session) -> List[Dict[str, Any]]:
    """Load metadata for a user from PostgreSQL."""
    try:
        logging.info(f"📥 Attempting to load metadata for user_id: {user_id}")
        
        result = db.execute(text("""
            SELECT metadata FROM visualizations_metadata
            WHERE user_id = :user_id
            ORDER BY created_at DESC
        """), {"user_id": str(user_id)})
        
        # Do NOT use json.loads since it's already a dict
        metadata_list = [row[0] for row in result.fetchall()]
        logging.info(f"✅ Loaded {len(metadata_list)} metadata entries for user {user_id}")
        
        return metadata_list

    except Exception as e:
        logging.error(f"❌ Error loading metadata for user {user_id}: {str(e)}")
        logging.error(traceback.format_exc())
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
import traceback
def _append_limited_metadata(user_id: str, new_entry: Dict[str, Any], db: Session, max_entries: int = 5):
    """Append a new entry and keep only the most recent ones for that entry's type in SQL."""
    try:
        current_type = new_entry.get("type", "other")
        new_entry_id = new_entry.get("id") or str(uuid4())
        new_entry["id"] = new_entry_id

        # Fetch existing entries of same type
        result = db.execute(text("""
            SELECT id, created_at FROM visualizations_metadata
            WHERE user_id = :user_id AND type = :type
            ORDER BY created_at DESC
        """), {"user_id": str(user_id), "type": current_type})
        same_type_entries = result.fetchall()

        # Delete oldest if exceeding max_entries
        if len(same_type_entries) >= max_entries:
            to_delete = same_type_entries[max_entries - 1:]
            for row in to_delete:
                db.execute(
                    text("DELETE FROM visualizations_metadata WHERE id = :id"),
                    {"id": row[0]}
                )

        # Insert the new entry
        db.execute(text("""
            INSERT INTO visualizations_metadata (id, user_id, type, created_at, metadata)
            VALUES (:id, :user_id, :type, :created_at, :metadata)
        """), {
            "id": new_entry_id,
            "user_id": str(user_id),
            "type": current_type,
            "created_at": new_entry.get("created_at", datetime.utcnow().isoformat()),
            "metadata": json.dumps(new_entry)
        })

        db.commit()
        logging.info(f"✅ Saved metadata entry {new_entry_id} for user {user_id} [type={current_type}]")

    except Exception as e:
        logging.error(f"❌ Failed to append limited metadata for {user_id}: {e}")
        traceback.print_exc()  # <-- Full traceback in logs
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
def register_user(
    name: str,
    email: str,
    hashed_password: str,
    db: Session,
    agreed_to_terms: bool = False,
    policy_version: str = "v1.0"
) -> User:
    try:
        # Step 1: Create a per-user database (or schema) for isolation
        db_name = create_user_database(email)

        # Step 2: Generate safe DB user credentials
        safe_email = email.replace('@', '_at_').replace('.', '_dot_')
        uid = secrets.token_urlsafe(8)
        db_user = f"ml_user_{safe_email}_{uid}"
        db_password = secrets.token_urlsafe(16)

        # Step 3: Register user in the master DB
        user = User(
            first_name=name,
            email=email,
            hashed_password=hashed_password,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password,
            agreed_to_terms=agreed_to_terms,
            agreed_at=datetime.utcnow(),
            policy_version=policy_version,
            tokens=10  # 🎁 Give 10 tokens on registration
        )
        db.add(user)
        db.flush()
        db.refresh(user)

        # Step 4: Role Creation & Granting Privileges via db: Session
        role_exists = db.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = :role_name"),
            {"role_name": db_user}
        ).fetchone()

        if not role_exists:
            db.execute(
                text(f'CREATE ROLE "{db_user}" LOGIN PASSWORD :password'),
                {"password": db_password}
            )

        # Revoke all privileges first to ensure clean state
        db.execute(text(f'REVOKE ALL ON SCHEMA public FROM "{db_user}"'))
        db.execute(text(f'REVOKE ALL ON ALL TABLES IN SCHEMA public FROM "{db_user}"'))

        # Grant minimum access
        db.execute(text(f'GRANT USAGE ON SCHEMA public TO "{db_user}"'))
        db.execute(text(f'GRANT INSERT, SELECT ON TABLE public.datasets TO "{db_user}"'))
        db.execute(text(f'GRANT INSERT, SELECT ON TABLE public.visualizations_metadata TO "{db_user}"'))

        # Step 5: Create tables in user's private DB
        user_engine = get_user_engine(user)
        Base.metadata.create_all(user_engine)

        db.commit()
        return user

    except Exception as e:
        db.rollback()
        raise Exception(f"❌ Error during user registration: {str(e)}")

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

@router.get("/terms-status", response_model=TermsStatus)
async def get_terms_status(current_user: User = Depends(get_current_active_user)):
    return {
        "agreed_to_terms": current_user.agreed_to_terms,
        "agreed_at": current_user.agreed_at,
        "policy_version": current_user.policy_version
    }

from datetime import datetime, timezone

class AgreementData(BaseModel):
    termsAccepted: bool
    privacyAccepted: bool
 
@router.post("/user/agreements")
async def update_user_agreements(
    agreement_data: AgreementData,
    current_user: User = Depends(get_current_active_user),  # Get the currently authenticated user
    db: Session = Depends(get_master_db_session)  # Use the master DB session
):
    try:
        # 1) Fetch the user from the master database
        user = db.query(User).filter(User.id == current_user.id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # 2) Update the user record in the per-user database using the user's specific session
        with get_user_db(current_user) as user_db:
            user_data = user_db.query(User).filter(User.id == user.id).first()
            if not user_data:
                raise HTTPException(status_code=404, detail=f"User data not found for {current_user.email}")

            # Update the user's agreement status and timestamp
            user_data.agreed_to_terms = agreement_data.termsAccepted
            user_data.privacy_accepted = agreement_data.privacyAccepted
            user_data.agreed_at = datetime.now(timezone.utc) if (agreement_data.termsAccepted or agreement_data.privacyAccepted) else None

            # Commit the changes to the per-user database
            user_db.commit()

        return {"message": "✅ Agreement status updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update agreement status: {str(e)}")
def get_admin_user(current_user: User = Depends(get_current_active_user)):
    if not current_user.is_admin:  # or use role == 'admin', is_dev, etc.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can perform this action.",
        )
    return current_user