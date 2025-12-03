import logging
import time
import io
import joblib
import pandas as pd
import os
import jwt
import uvicorn
import shutil
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from sklearn.model_selection import train_test_split
import numpy as np
from sqlalchemy import MetaData, Table, text
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Request, Path, Depends, APIRouter, HTTPException, Depends, status, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from passlib.context import CryptContext
from sqlalchemy.orm import Session
import traceback
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Optional, Dict, Any, Optional
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import timedelta
from sqlalchemy import LargeBinary, DateTime, ForeignKey
import shap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path as PathL
import json
import aiofiles
import uuid
from pydantic import EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Float, Table, MetaData, Boolean, DateTime, ForeignKey, Text, text
from sqlalchemy import Column
from sqlalchemy.ext.declarative import declarative_base
from fastapi.staticfiles import StaticFiles
import uuid, os, io
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pydantic import BaseModel
from sqlalchemy import Table, MetaData, Column, String, Integer, Float, Boolean
from sqlalchemy.dialects.postgresql import JSONB
import json
from io import StringIO
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import base64
from io import BytesIO



# from .ai import generate_insights
# from .worker import run_classification, run_clustering, run_segment_analysis, run_label_clusters, run_classification_predict, run_visualization, run_counterfactual
# from .worker import run_regression, run_risk_analysis, run_regression_predict, run_forecast, run_survival_analysis, run_what_if, run_decision_paths, run_ab_test
# from .ecs_launcher import launch_job_on_ecs
# from .worker import make_json_safe
# from .auth import Base, master_engine, decode_user_from_request
# from .tokens import TokenUsageLogResponse, TokenUsageLog
# from .classification import ModelClassifyingTrainer, lgb_params_c, cat_params_c, xgb_params_c
# from .preprocessing import preprocess_data
# from .survival import calculate_business_metrics
# from .anomaly_detection import train_best_anomaly_detection
# from .datasets import (
#     Base as DatasetBase,           # in case you want to do Base.metadata.create_all for per-user DBs
#     get_user_db, 

# )
#from datasets import router as d_router
# from .datasets import init_db as init_dataset_master_db
# from .activity import router as activity_router
# from .target import router as t_router
# from .auth import _load_metadata, _save_metadata
# from .account import APIStats, SubscriptionInfo, ProfileInfo, APIKeysInfo, BillingInfo, DashboardOut
# # These names should match exactly what you export from auth.py
# from .datasets import register_dataset,get_dataset_by_id, get_dataset_data, query_dataset

# from .actions import router as a_router
# from .auth import (
#     # Authentication & token utilities
#     get_current_active_user,
#     get_current_user,
#     create_access_token,
#     authenticate_user,
#     Dataset,
#     # Pydantic schemas for auth
#     UserCreate,
#     UserResponse,
#     Token,
#     AuthTokenResponse,
#     RegisterResponse,
    
#     # Database/session helpers
#     DatasetResponse,
#     DatasetCreate,

#     create_user_database,
#     get_user_session,
#     get_user_session_direct,
#     get_user_engine,
#     get_master_db_session,
#     get_user_by_email,
#     master_db_cm,
#     Base,
#     User

# )
# from .auth import get_master_db_session, delete_user
# from .auth import router as auth_router
# from .tokens import router as token_router
# from .storage import upload_file_to_supabase, download_file_from_supabase, handle_file_upload, download_file_from_supabase, list_user_files, delete_file_from_supabase, get_file_url
# from .planner_router import router as planner_router
# from .datasets import _key_only, _bytes_from_supabase_download, _build_supabase_key
# from .storage import _basename_from_key, supabase,  _strip_bucket_prefix, SUPABASE_BUCKET
# from .actions import router as actions_router
# from .ai import router as ai_router
# from .score import router as score_router
from score import router as score_router
from ai import generate_insights
from ai import router as ai_router
from worker import run_classification, run_clustering, run_segment_analysis, run_label_clusters, run_classification_predict, run_visualization, run_counterfactual
from worker import run_regression, run_risk_analysis, run_regression_predict, run_forecast, run_survival_analysis, run_what_if, run_decision_paths, run_ab_test
from ecs_launcher import launch_job_on_ecs
from worker import make_json_safe
from auth import Base, master_engine, decode_user_from_request, delete_user
from tokens import TokenUsageLogResponse, TokenUsageLog

from classification import ModelClassifyingTrainer, lgb_params_c, cat_params_c, xgb_params_c
from preprocessing import preprocess_data
from survival import calculate_business_metrics
from anomaly_detection import train_best_anomaly_detection
from datasets import (
    Base as DatasetBase,           # in case you want to do Base.metadata.create_all for per-user DBs
    get_user_db,

)
from datasets import router as d_router
from activity import router as activity_router
from account import router as a_router
from actions import router as actions_router
from target import router as t_router
from auth import _load_metadata, _save_metadata
from account import APIStats, SubscriptionInfo, ProfileInfo, APIKeysInfo, BillingInfo, DashboardOut
# These names should match exactly what you export from auth.py
from datasets import register_dataset,get_dataset_by_id, get_dataset_data, query_dataset
from datasets import _key_only, _bytes_from_supabase_download, _build_supabase_key
from storage import _basename_from_key, supabase,  _strip_bucket_prefix, SUPABASE_BUCKET

from auth import (
    # Authentication & token utilities
    get_current_active_user,
    get_current_user,
    create_access_token,
    authenticate_user,
    Dataset,
    # Pydantic schemas for auth
    UserCreate,
    UserResponse,
    Token,
    AuthTokenResponse,
    RegisterResponse,
    
    # Database/session helpers
    DatasetResponse,
    DatasetCreate,

    create_user_database,
    get_user_session,
    get_user_session_direct,
    get_user_engine,
    get_master_db_session,
    get_user_by_email,
    master_db_cm,
    Base,
    User

)
from auth import get_master_db_session
from auth import router as auth_router
from tokens import router as token_router
from storage import upload_file_to_supabase, download_file_from_supabase, handle_file_upload, download_file_from_supabase, list_user_files, delete_file_from_supabase, get_file_url
from planner_router import router as planner_router
from decision_card import router as dec_router
from datasets import init_db as init_dataset_master_db



IMAGES_DIR = "images"
# Auth configuration - move to environment variables in production
SECRET_KEY = "printing"  # CHANGE THIS IN PRODUCTION!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

init_dataset_master_db()
# Base configuration
print("🔐 Stripe Secret Key:", os.getenv("STRIPE_SECRET_KEY", "NOT SET"))
@property
def password(self):
    return self.hashed_password

class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None

# OAuth2 scheme
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Save logs to a file
        logging.StreamHandler()  # Print logs to console
    ],
)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI()
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://[::]:8080",
        "http://localhost:3000",  # Add any other origins you need
        "http://127.0.0.1:3000",
        'https://ml-insights-frontend.onrender.com'
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

@app.head("/", include_in_schema=False)
def read_root():
    print("✅ Root endpoint accessed")
    return {"status": "ok"}

# Middleware for logging requests and responses
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        body = await request.body()

        logger.info(f" Request: {request.method} {request.url} - Body: {body.decode('utf-8')}")
        
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(f"Response: {response.status_code} - Time: {process_time:.4f}s")
        
        return response
from fastapi.security.utils import get_authorization_scheme_param
async def try_get_current_user(request: Request) -> User | None:
    try:
        # Extract token manually
        auth: str = request.headers.get("Authorization")
        if not auth:
            return None
        scheme, token = get_authorization_scheme_param(auth)
        if scheme.lower() != "bearer":
            return None

        # Decode and verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            return None

        # Use generator-based session manually
        db_gen = get_master_db_session()
        db: Session = next(db_gen)

        user = get_user_by_email(db, email)
        if not user or not user.is_active:
            return None
        return user

    except ExpiredSignatureError:
        return None
    except Exception as e:
        print(f"Middleware get_current_user error: {e}")
        return None
    finally:
        try:
            next(db_gen)  # commit and close
        except Exception:
            pass
        
import math
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response, HTTPException
import math
import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response, HTTPException
import math
import time
import logging
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# Set up logging
logger = logging.getLogger(__name__)
# imports: keep ONE jwt lib. Here we standardize on python-jose.
from jose import jwt, JWTError, ExpiredSignatureError
from sqlalchemy.orm import Session

def resolve_user_from_request(request: Request, db: Session) -> "User | None":
    """Return a User either by API key (preferred) or by decoding JWT 'sub'."""
    auth = request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return None

    token = auth.split(" ", 1)[1].strip()

    # 1) API key match
    user = (
        db.query(User)
          .filter((User.prod_api_key == token) | (User.dev_api_key == token))
          .first()
    )
    if user:
        return user

    # 2) JWT decode
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            return None
        user = db.query(User).filter(User.email == email.strip().lower()).first()
        return user
    except ExpiredSignatureError:
        return None
    except JWTError:
        return None
class UsageTrackerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response: Response = await call_next(request)
        duration = time.monotonic() - start

        tracked_paths = [
            "/classification/", "/regression/", "/clustering/", "/forecast/",
            "/segment_analysis/", "/survival/", "/counterfactual/", "/visualize/",
            "/feature_impact/", "/risk_analysis/", "/decision_paths/",
            "/classification/predict/", "/regression/predict/", "/what_if/", "/ab_test/"
        ]

        # If not a tracked path, or if response is an error (401/403/4xx/5xx), skip billing
        if (request.url.path not in tracked_paths) or (response.status_code >= 400):
            return response

        # --- Concave-down pricing parameters ---
        DATA_PRICE_SCALE = 74.1   # scale factor for logarithmic pricing
        PRICE_PER_MINUTE = 0.50   # $0.50 per minute
        OVERDRAFT_LIMIT = -1.0

        # Estimate data volume
        content_length = request.headers.get("content-length", "0")
        bytes_estimated = int(content_length) if content_length.isdigit() else 0
        bytes_processed = getattr(request.state, "file_bytes", bytes_estimated)
        mb_used = bytes_processed / (1024 * 1024)

        # Calculate costs
        cost_data = DATA_PRICE_SCALE * math.log1p(mb_used)
        minutes_used = duration / 60
        cost_time = PRICE_PER_MINUTE * minutes_used
        cost_for_request = round(cost_data + cost_time, 2)

        logger.info(
            f"Usage tracking - Path: {request.url.path}, MB: {mb_used:.3f}, "
            f"Minutes: {minutes_used:.3f}, Data cost: ${cost_data:.2f}, "
            f"Time cost: ${cost_time:.2f}, Total: ${cost_for_request}"
        )

        # Billing update
        with master_db_cm() as db:
            user = resolve_user_from_request(request, db)
            if not user:
                logger.warning("No authenticated user found for billing (skipping).")
                return response

            # Initialize tokens if None
            if user.tokens is None:
                user.tokens = 0.0

            new_balance = user.tokens - cost_for_request
            if new_balance < OVERDRAFT_LIMIT:
                logger.warning(
                    f"User {user.email} exceeded usage limit. "
                    f"Current balance: ${user.tokens:.2f}, Request cost: ${cost_for_request:.2f}"
                )
                # Do NOT raise here; the request already completed successfully.
                # Consider tagging the user for soft-lock instead.
                return response

            # Update counters
            user.tokens = new_balance
            user.total_bytes_processed = (user.total_bytes_processed or 0) + bytes_processed
            user.total_compute_seconds = (user.total_compute_seconds or 0) + duration
            user.total_cost_dollars = (user.total_cost_dollars or 0) + cost_for_request

            logger.info(
                f"Billing successful - User: {user.email}, "
                f"New balance: ${user.tokens:.2f}, "
                f"Total cost: ${user.total_cost_dollars:.2f}"
            )

        return response

from fastapi import Request, HTTPException
import time
import openai
from fastapi import Request, HTTPException, Response
import time
class AITokenBillingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tracked_path_prefix = "/api/ai_insights"

        start_time = time.monotonic()
        response: Response = await call_next(request)
        duration = time.monotonic() - start_time

        # Only bill tracked paths and successful responses
        if (not request.url.path.startswith(tracked_path_prefix)) or (response.status_code >= 400):
            return response

        # ─── Pricing Configuration ─────────────────────
        OPENAI_COST_PER_1K = 0.04  # OpenAI base rate per 1K tokens
        MARKUP = 3.0               # 3x markup
        PRICE_PER_MB = 0.1875      # $/MB
        PRICE_PER_SECOND = 0.375   # $/sec
        OVERDRAFT_LIMIT = -1.0     # allow soft overdraft to -$1

        # ─── Usage Extraction ──────────────────────────
        # Prefer a value set by your handler; otherwise fall back to 0
        openai_tokens_used = (
            getattr(request.state, "openai_tokens_used", None)
            or getattr(response, "openai_tokens_used", None)
            or 0
        )

        # Size: prefer handler-provided bytes; else content-length header
        content_length = request.headers.get("content-length", "0")
        bytes_estimated = int(content_length) if content_length.isdigit() else 0
        bytes_processed = getattr(request.state, "file_bytes", bytes_estimated)
        mb_used = bytes_processed / (1024 * 1024)

        # ─── Charges ───────────────────────────────────
        ai_cost_usd = (openai_tokens_used / 1000.0) * OPENAI_COST_PER_1K
        ai_charge = round(ai_cost_usd * MARKUP, 3)

        time_charge = round(duration * PRICE_PER_SECOND, 3)
        size_charge = round(mb_used * PRICE_PER_MB, 3)
        additional_charge = round(time_charge + size_charge, 3)

        total_charge = round(ai_charge + additional_charge, 3)

        logger.info(
            f"AI billing - Path: {request.url.path}, "
            f"OpenAI tokens: {openai_tokens_used}, AI cost: ${ai_charge:.3f}, "
            f"MB: {mb_used:.3f}, Duration: {duration:.3f}s, "
            f"Additional: ${additional_charge:.3f}, Total: ${total_charge:.3f}"
        )

        # ─── Database billing update ───────────────────
        try:
            with master_db_cm() as db:
                user = resolve_user_from_request(request, db)
                if not user:
                    logger.warning("No authenticated user found for AI billing (skipping).")
                    return response

                # Initialize tokens if None
                if user.tokens is None:
                    user.tokens = 0.0

                new_balance = user.tokens - total_charge
                if new_balance < OVERDRAFT_LIMIT:
                    logger.warning(
                        f"AI billing - User {user.email} exceeded usage limit. "
                        f"Current balance: ${user.tokens:.2f}, "
                        f"Request cost: ${total_charge:.3f}"
                    )
                    # Do NOT raise here since the request already completed.
                    # Option: flag user for soft-lock instead of breaking the response.
                    return response

                # Update aggregates
                user.tokens = new_balance
                user.total_ai_tokens_used = (user.total_ai_tokens_used or 0) + int(openai_tokens_used)
                user.total_bytes_processed = (user.total_bytes_processed or 0) + int(bytes_processed)
                user.total_compute_seconds = (user.total_compute_seconds or 0) + float(duration)
                user.total_cost_dollars = (user.total_cost_dollars or 0) + float(total_charge)

                logger.info(
                    f"AI billing successful - User: {user.email}, "
                    f"New balance: ${user.tokens:.2f}, "
                    f"Total AI tokens used: {user.total_ai_tokens_used}, "
                    f"Total cost: ${user.total_cost_dollars:.2f}"
                )
        except Exception as e:
            # Never break the response due to billing
            logger.error(f"AI billing error for user "
                         f"{user.email if 'user' in locals() and user else 'unknown'}: {e}")

        return response
app.add_middleware(AITokenBillingMiddleware)
app.add_middleware(UsageTrackerMiddleware)
app.add_middleware(LoggingMiddleware)


routers = [
    auth_router,
    activity_router,
    actions_router,
    token_router,
    t_router,
    planner_router,
    ai_router,
    d_router,
    a_router,
    score_router,
    dec_router,
]

for r in routers:
    app.include_router(r)

# Dataset models
# Serve the frontend folder as static files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/app", StaticFiles(directory=frontend_path, html=True), name="frontend")

class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None

class DatasetQuery(BaseModel):
    query: str


# Initialize database on startup
@app.on_event("startup")
def startup_event():
    """Initialize database on startup"""
    init_dataset_master_db()
      # (make sure you import Base from wherever your User model lives)
    Base.metadata.create_all(bind=master_engine)
    logger.info("Application started and database initialized")
    

# Root endpoint
@app.get("/")
def read_root():
    logger.info("✅ Root endpoint accessed")
    return {"message": "FastAPI is running"}


@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return UserResponse(
        id=current_user.id,
        name=current_user.first_name,
        email=current_user.email,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        tokens=current_user.tokens or 0.0
    )

@app.delete("/users/me")
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


# API endpoint
@app.post("/eda")
async def eda(file: UploadFile = File(...), target_column: str = None):
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        
        eda_result = perform_eda(df)
        model_suggestion = {}
        
        if target_column:
            model_suggestion = suggest_models(df, target_column)
        
        result = {
            "EDA": eda_result,
            "Model Suggestion": model_suggestion
        }
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

import os
import io
import re
import math
import time
from typing import Dict, Any, List, Optional, Iterable, Tuple


# ---------- helpers ----------

def _read_csv_robust(path: str, delimiter: Optional[str] = None) -> pd.DataFrame:
    """
    CSV/TSV reader with delimiter sniffing + header heuristic.
    delimiter can be '\t', ',' or any one-char; if 'tab' is passed use '\t'.
    """
    sep = '\t' if delimiter == 'tab' else (delimiter if delimiter else None)
    try:
        df = pd.read_csv(path, sep=sep, engine="python")  # sep=None => sniff
    except Exception:
        df = pd.read_csv(path)  # fallback

    # Header heuristic: if most "headers" look like data (numeric/date-ish), treat as no header.
    cols = list(df.columns)
    def looks_like_data_token(x: Any) -> bool:
        s = str(x).strip()
        if not s:
            return False
        if s.replace('.', '', 1).isdigit():
            return True
        # dd-mm-YYYY or dd/mm/YYYY
        return bool(re.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$', s))

    if cols and sum(looks_like_data_token(c) for c in cols) / len(cols) >= 0.7:
        df = pd.read_csv(path, header=None, sep=sep, engine="python")
        df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
    return df

def _read_excel(path: str) -> pd.DataFrame:
    try:
        # openpyxl handles .xlsx; xlrd<2.0 handled .xls, but often better to use openpyxl for both if installed
        return pd.read_excel(path)  # pandas will pick engine
    except ImportError as e:
        raise HTTPException(status_code=400, detail="Excel support requires 'openpyxl' installed") from e

def _read_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)  # needs pyarrow or fastparquet
    except ImportError as e:
        raise HTTPException(status_code=400, detail="Parquet support requires 'pyarrow' or 'fastparquet'") from e

def _read_json_any(path: str, fmt: Optional[str]) -> pd.DataFrame:
    # If client says jsonl, prefer lines=True
    if fmt == 'jsonl':
        return pd.read_json(path, lines=True)
    # Try normal JSON first, fall back to JSONL
    try:
        return pd.read_json(path)
    except ValueError:
        return pd.read_json(path, lines=True)

def _read_any_table(path: str, fmt: str, delimiter: Optional[str]) -> pd.DataFrame:
    fmt = fmt.lower()
    if fmt in ('csv', 'tsv'):
        delim = delimiter or ('\t' if fmt == 'tsv' else None)
        # allow 'tab' keyword from client
        if delim == 'tab': delim = '\t'
        return _read_csv_robust(path, delimiter=delim)
    if fmt in ('xlsx', 'xls'):
        return _read_excel(path)
    if fmt == 'parquet':
        return _read_parquet(path)
    if fmt in ('json', 'jsonl'):
        return _read_json_any(path, fmt)
    # unknown
    raise HTTPException(status_code=400, detail=f"Unsupported format '{fmt}'")
import os, tempfile, shutil
from typing import Optional, List, Tuple
from sqlalchemy import inspect
from sqlalchemy.exc import NoSuchTableError, ProgrammingError, OperationalError
from sqlalchemy.sql.schema import quoted_name


SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "user-uploads")

def _download_object_to_temp(object_key: str, *, user_id: int, dataset_id: int) -> str:
    """Download object from Supabase to a temp file and return its path."""
    
    if not supabase or not object_key:
        raise FileNotFoundError("Supabase storage not configured or no object key provided")

    bucket = SUPABASE_BUCKET
    
    # Build the candidates for the storage key
    candidates = []
    
    # 1. Try the key as-is (stripped of bucket prefix)
    raw_key = _key_only(object_key, bucket)
    if raw_key:
        candidates.append(raw_key)
    
    # 2. Try building the correct key with current user_id
    filename = _basename_from_key(object_key)
    if filename:
        correct_key = _build_supabase_key(user_id, filename)
        if correct_key not in candidates:
            candidates.append(correct_key)
    
    # 3. If object_key looks like "old_user_id/filename", try with current user_id
    if "/" in raw_key:
        parts = raw_key.split("/", 1)
        if len(parts) == 2 and parts[0].isdigit():
            old_user_id, file_part = parts
            if old_user_id != str(user_id):
                corrected_key = f"{user_id}/{file_part}"
                if corrected_key not in candidates:
                    candidates.append(corrected_key)
    
    logger.info(f"Attempting Supabase download for user {user_id}, candidates: {candidates}")

    last_err = None
    for key in candidates:
        try:
            logger.info(f"Downloading from Supabase bucket={bucket} key={key}")
            resp = supabase.storage.from_(bucket).download(key)
            data_bytes = _bytes_from_supabase_download(resp)
            
            if not data_bytes:
                raise RuntimeError("Empty download payload")

            # Create temp file with appropriate extension
            file_ext = os.path.splitext(filename)[1] if filename else ".csv"
            fd, temp_path = tempfile.mkstemp(suffix=file_ext)
            os.close(fd)
            
            with open(temp_path, "wb") as f:
                f.write(data_bytes)
            
            # Verify the file was written correctly
            if os.path.getsize(temp_path) == 0:
                os.remove(temp_path)
                raise RuntimeError("Downloaded file is empty")
            
            logger.info(f"Successfully downloaded {len(data_bytes)} bytes to {temp_path}")
            return temp_path
            
        except Exception as e:
            last_err = e
            logger.warning(f"Supabase download failed for key '{key}': {e}")

    # If we got here, all candidates failed
    raise RuntimeError(f"Could not download '{object_key}' from bucket '{bucket}' for user {user_id}: {last_err}")


# Fix 2: Improve the key resolution logic
def _resolve_object_key(object_key: str, user_id: int) -> str:
    """
    Resolve the correct object key for download, handling different storage patterns.
    """
    if not object_key:
        return ""
    
    # If it's already a full path with user ID, use it
    if object_key.startswith(f"{user_id}/"):
        return object_key
    
    # If it's just a user ID and filename (like "1/Nike_Sales_Uncleaned.csv")
    # but the user_id doesn't match, prepend the correct user_id
    parts = object_key.split("/", 1)
    if len(parts) == 2:
        stored_user_id, filename = parts
        if stored_user_id != str(user_id):
            logger.info(f"Key user ID mismatch: stored={stored_user_id}, current={user_id}")
            # Try both the original key and the corrected one
            return object_key
    
    # Default: prepend user_id if not present
    if not object_key.startswith(f"{user_id}/"):
        return f"{user_id}/{object_key.lstrip('/')}"
    
    return object_key


@app.post("/upload_csv/")
async def upload_csv(
    file: UploadFile = File(...),
    drop_columns: str = Form(None),
    target_column: str = Form(None),
    feature_column: str = Form(None)
):
    logger.info("✅ /upload_csv/ endpoint called")
    try:
        # Attempt to read the file
        file_content = await file.read()

        os.makedirs("temp", exist_ok=True)
        file_path = f"temp/{file.filename}"

        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        # Process optional form fields
        drop_cols = drop_columns.split(",") if drop_columns else []
        target_col = target_column if target_column else None
        feature_cols = feature_column.split(",") if feature_column and not isinstance(feature_column, type(Form)) else []

        # Log for debugging
        logger.info(f"Drop Columns: {drop_cols}")
        logger.info(f"Target Column: {target_col}")
        logger.info(f"Feature Columns: {feature_cols}")

        return {"message": f"✅ CSV file '{file.filename}' processed successfully"}

    except ValueError as ve:
        logger.warning(f"⚠️ Value error in /upload_csv: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your message here"
        )
    except OSError as ioe:
        # Common for: I/O operation on closed file
        logger.error(f"❌ I/O error in /upload_csv: {str(ioe)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your message here"
        )

    except Exception as e:
        logger.error(f"❌ Unexpected error in /upload_csv: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your message here"
        )
@app.post("/classification/")
async def classification(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile = File(None),
    target_column: str = Form("target"),
    drop_columns: str = Form(""),
    current_user: User = Depends(get_current_active_user),
):
    # ───────── Token check ───────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id
    file_path = None
    train_path = None
    test_path = None

    try:
        # ───────── Save and upload files to Supabase ───────────
        if file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name

            file_path = upload_file_to_supabase(
                user_id=str(user_id),
                file_path=temp_path,
                filename=file.filename
            )
            os.remove(temp_path)

        elif train_file and test_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                shutil.copyfileobj(train_file.file, tmp)
                temp_train = tmp.name

            train_path = upload_file_to_supabase(
                user_id=str(user_id),
                file_path=temp_train,
                filename=train_file.filename
            )
            os.remove(temp_train)

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                shutil.copyfileobj(test_file.file, tmp)
                temp_test = tmp.name

            test_path = upload_file_to_supabase(
                user_id=str(user_id),
                file_path=temp_test,
                filename=test_file.filename
            )
            os.remove(temp_test)
        else:
            raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")

        task = run_classification.delay(
            user_id=user_id,
            current_user={
                "id": current_user.id,
                "email": current_user.email,
                "subscription": current_user.subscription
            },
            file_path=file_path,
            train_path=train_path,
            test_path=test_path,
            target_column=target_column,
            drop_columns=drop_columns
        )
        response_data = await run_in_threadpool(task.get)
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing classification: {str(e)}")

@app.post("/classification/predict/")
async def classification_predict(
    file: UploadFile = File(...),
    drop_columns: str = Form(""),
    output_predictions: bool = Form(True),
    current_user: User = Depends(get_current_active_user),
):
    """
    Predict using previously trained classification model.
    """
    # ───────── Token check ───────────
    MINIMUM_TOKENS = 0.05
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id

    # ───────── Validate trained model exists on Supabase ───────────
    model_supabase_path = f"{user_id}/{user_id}_best_classifier.pkl"
    try:
        _ = download_file_from_supabase(model_supabase_path)
    except Exception as e:
        raise HTTPException(404, f"No trained classification model found on Supabase. Please train a model first. Details: {str(e)}")

    # ───────── Validate and save uploaded file ───────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Prediction file must be a CSV.")


    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        # Upload to Supabase under user-specific path
        supabase_path = upload_file_to_supabase(
            user_id=str(user_id),
            file_path=temp_path,
            filename=f"predict_{file.filename}"
        )

        os.remove(temp_path)

        # ───────── Launch Prediction ───────────
        task = run_classification_predict.delay(
            user_id=user_id,
            current_user={
                "id": current_user.id,
                "email": current_user.email,
                "subscription": current_user.subscription
            },
            file_path=supabase_path,  # Supabase path
            drop_columns=drop_columns,
            output_predictions=output_predictions
        )

        response_data = await run_in_threadpool(task.get)

        # ───────── Deduct Tokens ───────────
        if response_data.get("status") == "success":
            try:
                tokens_to_deduct = min(MINIMUM_TOKENS, current_user.tokens)
                current_user.tokens -= tokens_to_deduct
                # Commit user token deduction in DB (you likely handle this elsewhere)
            except Exception as token_error:
                print(f"[⚠️] Token deduction failed: {token_error}")

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
@app.post("/cluster/")
async def clustering(
    file: Optional[UploadFile] = File(None),
    train_file: Optional[UploadFile] = File(None),
    test_file: Optional[UploadFile] = File(None),
    target_column: Optional[str] = Form(None),
    time_column: Optional[str] = Form(None),
    drop_columns: Optional[str] = Form(""),
    current_user: User = Depends(get_current_active_user)
):
    # ───────── Token check ───────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ───────── Handle file uploads to Supabase ───────────
    try:
        if file:
            file_path = await handle_file_upload(user_id, file)

        elif train_file and test_file:
            train_path = await handle_file_upload(user_id, train_file)
            test_path = await handle_file_upload(user_id, test_file)
        else:
            raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")
    except Exception as e:
        raise HTTPException(500, f"File upload failed: {str(e)}")


    # Free users = Celery worker
    task = run_clustering.delay(
        user_id=user_id,
        file_path=file_path,   # might be None
        current_user={
            "id": current_user.id,
            "email": current_user.email,
            "subscription": current_user.subscription
        },
        train_path=train_path, # might be None
        test_path=test_path,   # might be None
        target_column=target_column,
        time_column=time_column,
        drop_columns=drop_columns
    )
    response_data = await run_in_threadpool(task.get)
    return response_data

@app.post("/segment_analysis/")
async def segment_analysis(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile  = File(None),
    target_column: str     = Form(None),
    drop_columns: str      = Form(""),
    current_user: User     = Depends(get_current_active_user),
):
    # ───────── Token check ───────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ───────── Handle file uploads to Supabase ───────────
    try:
        if file:
            file_path = await handle_file_upload(user_id, file)

        elif train_file and test_file:
            train_path = await handle_file_upload(user_id, train_file)
            test_path = await handle_file_upload(user_id, test_file)
        else:
            raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")
    except Exception as e:
        raise HTTPException(500, f"File upload failed: {str(e)}")

    task = run_segment_analysis.delay(
        user_id=user_id,
        current_user={
            "id": current_user.id,
            "email": current_user.email,
            "subscription": current_user.subscription
        },
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        drop_columns=drop_columns
    )

    response_data = await run_in_threadpool(task.get)
    return response_data

@app.post("/label_clusters/")
async def label_clusters(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile = File(None),
    feature_columns: str = Form(""),
    current_user: User = Depends(get_current_active_user),
):
    # ───────── Token check ───────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ───────── Handle file uploads to Supabase ───────────
    try:
        if file:
            file_path = await handle_file_upload(user_id, file)

        elif train_file and test_file:
            train_path = await handle_file_upload(user_id, train_file)
            test_path = await handle_file_upload(user_id, test_file)
        else:
            raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")
    except Exception as e:
        raise HTTPException(500, f"File upload failed: {str(e)}")

    # Free users = Celery worker
    task = run_label_clusters.delay(
        user_id=user_id,
        current_user={
            "id": current_user.id,
            "email": current_user.email,
            "subscription": current_user.subscription
        },
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        feature_columns=feature_columns
    )
    response_data = await run_in_threadpool(task.get)
    return response_data

def process_uploaded_file(file_path: str):
    """Process a CSV file from Supabase storage and return the DataFrame"""
    try:
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
        
        try:
            # Download from Supabase to temporary file
            download_file_from_supabase(file_path, temp_path)
            
            # Process the file
            df = pd.read_csv(temp_path)
            return df
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise e

@app.post("/regression/")
async def regression(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile = File(None),
    target_column: str = Form("target"),
    drop_columns: str = Form(""),
    current_user: User = Depends(get_current_active_user),
):
    # ───────── Token check ───────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ───────── Handle file uploads to Supabase ───────────
    try:
        if file:
            file_path = await handle_file_upload(user_id, file)

        elif train_file and test_file:
            train_path = await handle_file_upload(user_id, train_file)
            test_path = await handle_file_upload(user_id, test_file)
        else:
            raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")
    except Exception as e:
        raise HTTPException(500, f"File upload failed: {str(e)}")

    task = run_regression.delay(
        user_id=user_id,
        current_user={
            "id": current_user.id,
            "email": current_user.email,
            "subscription": current_user.subscription
        },  # if `current_user` is a Pydantic model
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        drop_columns=drop_columns
    )

    response_data = await run_in_threadpool(task.get)
    return response_data
@app.post("/regression/predict/")
async def regression_predict(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile = File(None),
    drop_columns: str = Form(""),
    current_user: User = Depends(get_current_active_user),
):
    """
    Predict using previously trained regression model (stored in Supabase).
    """
    # ───────── Token check ───────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id

    # ───────── Validate model exists on Supabase ───────────
    model_supabase_path = f"{user_id}/{user_id}_best_regressor.pkl"
    try:
        _ = download_file_from_supabase(model_supabase_path)
    except Exception as e:
        raise HTTPException(404, f"No trained regression model found on Supabase. Please train a model first. Details: {str(e)}")

    # ───────── Validate at least one file is uploaded ───────────
    if not file and not (train_file and test_file):
        raise HTTPException(400, "Please upload a single prediction CSV or both train/test files.")

    # ───────── Save uploaded files to temp paths ───────────
    file_path = None
    train_path = None
    test_path = None

    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # Upload to Supabase and use that path
        supabase_path = upload_file_to_supabase(
            user_id=str(user_id),
            file_path=temp_path,
            filename=f"predict_{file.filename}"
        )

        os.remove(temp_path)  # Optional cleanup
        file_path = supabase_path  # ✅ Set this to Supabase path for task


    if train_file and test_file:
        train_contents = await train_file.read()
        test_contents = await test_file.read()

        temp_train = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp_train.write(train_contents)
        temp_train.close()
        train_path = temp_train.name

        temp_test = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp_test.write(test_contents)
        temp_test.close()
        test_path = temp_test.name

    # Free users = Celery worker
    task = run_regression_predict.delay(
        user_id=user_id,
        current_user={
            "id": current_user.id,
            "email": current_user.email,
            "subscription": current_user.subscription
        },
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        drop_columns=drop_columns
    )
    response_data = await run_in_threadpool(task.get)
    return response_data

# Response models for visualizations
class VisualizationParameters(BaseModel):
    target_column: str
    optimal_k: int

class SegmentSummary(BaseModel):
    segment: int
    count: int

class VisualizationResponse(BaseModel):
    id: str
    created_at: str
    type: str
    dataset: str
    parameters: VisualizationParameters
    thumbnailData: str
    imageData: str
    segments_summary: List[Dict[str, Any]]

class VisualizationListResponse(BaseModel):
    visualizations: List[VisualizationResponse]
    total: int



# Additional utility functions you might need

@app.delete("/visualizations/{visualization_id}")
async def delete_visualization(
    visualization_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete a specific visualization."""
    try:
        metadata = _load_metadata(current_user.id)
        
        # Find and remove the visualization
        original_count = len(metadata)
        metadata = [item for item in metadata if item.get('id') != visualization_id]
        
        if len(metadata) == original_count:
            raise HTTPException(status_code=404, detail="Visualization not found")
        
        # Save updated metadata
        _save_metadata(current_user.id, metadata)
        
        return {"message": "Visualization deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting visualization: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting visualization")

@app.get("/visualizations/{visualization_id}")
async def get_visualization(
    visualization_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_master_db_session)
):
    result = db.execute(
        """
        SELECT metadata FROM visualizations_metadata
        WHERE user_id = :user_id AND metadata->>'id' = :viz_id
        """,
        {"user_id": current_user.id, "viz_id": visualization_id}
    ).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Visualization not found")

    return json.loads(result[0])

@app.get("/visualizations")
def list_visualizations(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_master_db_session),
):
    return _load_metadata(user_id=current_user.id, db=db)
@app.get("/visualizations/{viz_id}/download")
async def download_visualization(
    viz_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_master_db_session)
):
    result = db.execute(
        """
        SELECT metadata FROM visualizations_metadata
        WHERE user_id = :user_id AND metadata->>'id' = :viz_id
        """,
        {"user_id": current_user.id, "viz_id": viz_id}
    ).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Visualization not found")

    entry = json.loads(result[0])

    img_url = (
        entry.get("imageUrl") or
        entry.get("feature_importance_detailed_url") or
        entry.get("feature_importance_detailed") or
        entry.get("imageData")
    )

    if not img_url or img_url.strip() == "":
        raise HTTPException(status_code=404, detail="No image found")

    if img_url.startswith("data:image"):
        try:
            _, b64_data = img_url.split(",", 1)
            image_bytes = base64.b64decode(b64_data)
            return StreamingResponse(
                BytesIO(image_bytes),
                media_type="image/png",
                headers={"Content-Disposition": f"attachment; filename=visualization-{viz_id}.png"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to decode base64 image: {str(e)}")

    file_path = PathL(img_url.lstrip("/"))
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    return FileResponse(str(file_path), media_type="image/png", filename=f"visualization-{viz_id}.png")

@app.delete("/visualizations/{visualization_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_visualization(
    visualization_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_master_db_session)
):
    result = db.execute(
        """
        DELETE FROM visualizations_metadata
        WHERE user_id = :user_id AND metadata->>'id' = :viz_id
        RETURNING metadata
        """,
        {"user_id": current_user.id, "viz_id": visualization_id}
    ).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Visualization not found")

    # Optional: try removing associated image files
    entry = json.loads(result[0])
    for key in ["thumbnailData", "imageData", "imageUrl", "feature_importance", "feature_importance_detailed"]:
        url = entry.get(key)
        if url and not url.startswith("data:image"):
            path = PathL(url.lstrip("/"))
            if path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete file {path}: {e}")

@app.post("/segment_analysis/")
async def segment_analysis(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile  = File(None),
    target_column: str     = Form(None),
    drop_columns: str      = Form(""),
    current_user: User     = Depends(get_current_active_user),
):
    # ───────── Token check ───────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ───────── Handle file uploads to Supabase ───────────
    try:
        if file:
            file_path = await handle_file_upload(user_id, file)

        elif train_file and test_file:
            train_path = await handle_file_upload(user_id, train_file)
            test_path = await handle_file_upload(user_id, test_file)
        else:
            raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")
    except Exception as e:
        raise HTTPException(500, f"File upload failed: {str(e)}")

    task = run_segment_analysis.delay(
        user_id=user_id,
        current_user={
            "id": current_user.id,
            "email": current_user.email,
            "subscription": current_user.subscription
        },
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        drop_columns=drop_columns
    )

    response_data = await run_in_threadpool(task.get)
    return response_data

@app.post("/visualize/")
async def visualize(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile = File(None),
    target_column: str = Form(...),
    feature_column: str = Form(...),
    current_user: User = Depends(get_current_active_user),
):
    # ───────── Token check ───────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ───────── Handle file uploads to Supabase ───────────
    try:
        if file:
            file_path = await handle_file_upload(user_id, file)

        elif train_file and test_file:
            train_path = await handle_file_upload(user_id, train_file)
            test_path = await handle_file_upload(user_id, test_file)
        else:
            raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")
    except Exception as e:
        raise HTTPException(500, f"File upload failed: {str(e)}")

    # Free users -> Celery worker
    task = run_visualization.delay(
        user_id=user_id,
        current_user={
            "id": current_user.id,
            "email": current_user.email,
            "subscription": current_user.subscription
        },
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        feature_column=feature_column
    )
    response_data = await run_in_threadpool(task.get)
    return response_data

@app.post("/counterfactual/")
async def counterfactual(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile = File(None),
    target_column: str = Form("target"),
    drop_columns: str = Form(""),
    sample_id: int = Form(None),
    sample_strategy: str = Form("random"),
    num_samples: int = Form(1),
    desired_outcome: str = Form(None),
    editable_features: str = Form(None),
    max_changes: int = Form(3),
    proximity_metric: str = Form("euclidean"),
    current_user: User = Depends(get_current_active_user),
):
    # ───────── Token check ───────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ───────── Handle file uploads to Supabase ───────────
    try:
        if file:
            file_path = await handle_file_upload(user_id, file)

        elif train_file and test_file:
            train_path = await handle_file_upload(user_id, train_file)
            test_path = await handle_file_upload(user_id, test_file)
        else:
            raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")
    except Exception as e:
        raise HTTPException(500, f"File upload failed: {str(e)}")

    # ───────── Convert input types ───────────
    # Parse desired_outcome properly
    parsed_desired = None
    if desired_outcome:
        try:
            # Try to parse as JSON first
            parsed_desired = json.loads(desired_outcome)
        except json.JSONDecodeError:
            # If not JSON, try to parse as number
            try:
                parsed_desired = float(desired_outcome)
                # Convert to int if it's a whole number
                if parsed_desired.is_integer():
                    parsed_desired = int(parsed_desired)
            except ValueError:
                # Keep as string if it's not a number
                parsed_desired = desired_outcome

    # Parse editable_features
    editable_features_list = None
    if editable_features:
        try:
            editable_features_list = json.loads(editable_features)
        except json.JSONDecodeError:
            editable_features_list = [f.strip() for f in editable_features.split(",")]

    sanitized_params = {
        "user_id": str(user_id),
        "current_user": {
            "id": str(current_user.id),
            "email": str(current_user.email),
            "subscription": str(current_user.subscription)
        },
        "file_path": file_path,
        "train_path": train_path,
        "test_path": test_path,
        "target_column": str(target_column),
        "drop_columns": str(drop_columns),
        "sample_id": int(sample_id) if sample_id is not None else None,
        "sample_strategy": str(sample_strategy),
        "num_samples": int(num_samples),
        "desired_outcome": parsed_desired,  # Already processed above
        "editable_features": editable_features_list,  # Already processed above
        "max_changes": int(max_changes),
        "proximity_metric": str(proximity_metric),
    }

    # Apply additional JSON safety check
    sanitized_params = make_json_safe(sanitized_params)
    
    task = run_counterfactual.delay(**sanitized_params)
    response_data = await run_in_threadpool(task.get)
    
    return response_data

@app.post("/survival/")
async def survival(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile = File(None),
    time_col: str = Form(...),
    event_col: str = Form(...),
    drop_cols: str = Form(""),
    current_user: User = Depends(get_current_active_user),
):
    # ───────── Token check ───────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None
    
    # ───────── Handle file uploads to Supabase ───────────
    try:
        if file:
            file_path = await handle_file_upload(user_id, file)

        elif train_file and test_file:
            train_path = await handle_file_upload(user_id, train_file)
            test_path = await handle_file_upload(user_id, test_file)
        else:
            raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")
    except Exception as e:
        raise HTTPException(500, f"File upload failed: {str(e)}")

    if file_path:
        train_path, test_path = None, None
    elif train_path and test_path:
        file_path = None
    else:
        raise HTTPException(400, "Invalid upload state: no valid dataset provided.")

    kwargs = {
        "user_id": user_id,
        "current_user": {
            "id": current_user.id,
            "email": current_user.email,
            "subscription": current_user.subscription
        },
        "time_col": time_col,
        "event_col": event_col,
        "drop_cols": drop_cols,
    }

    if file_path:
        kwargs["file_path"] = file_path
    else:
        kwargs["train_path"] = train_path
        kwargs["test_path"] = test_path

    # Free users -> Celery worker
    task = run_survival_analysis.delay(**kwargs)
    response_data = await run_in_threadpool(task.get)
    return response_data

@app.post("/what_if/")
async def what_if_analysis(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile = File(None),
    target_column: str = Form(...),
    feature_changes: str = Form(...),  # JSON string of bulk feature edits
    drop_columns: str = Form(""),
    sample_id: int = Form(None),  # Optional: specific sample ID to analyze
    current_user: User = Depends(get_current_active_user),
):
    """
    What-If Analysis endpoint for mass scenario simulation.
    
    Args:
        file: Single dataset file (CSV)
        train_file: Training dataset (if using separate train/test)
        test_file: Test dataset (if using separate train/test)
        target_column: Name of the target variable column
        feature_changes: JSON string of bulk feature changes to apply
        drop_columns: Comma-separated list of columns to drop
        sample_id: Optional specific sample ID to focus analysis on
        current_user: Authenticated user making the request
    
    Returns:
        dict: Analysis results with metrics, visualizations, and insights
    """
    user_id = current_user.id
    
    # ──────────── Token check ─────────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None
    
    # ──────────── Handle file uploads to Supabase ─────────────
    try:
        if file:
            file_path = await handle_file_upload(user_id, file)

        elif train_file and test_file:
            train_path = await handle_file_upload(user_id, train_file)
            test_path = await handle_file_upload(user_id, test_file)
        else:
            raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")
    except Exception as e:
        raise HTTPException(500, f"File upload failed: {str(e)}")

    # ──────────── Validate feature_changes JSON ─────────────
    try:
        import json
        # Validate that feature_changes is valid JSON
        json.loads(feature_changes)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON in feature_changes: {str(e)}")

    # ──────────── Run analysis via Celery worker ─────────────
    task = run_what_if.delay(
        user_id=user_id,
        current_user={
            "id": current_user.id,
            "email": current_user.email,
            "subscription": current_user.subscription
        },
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        feature_changes=feature_changes,
        drop_columns=drop_columns,
        sample_id=sample_id  # Pass sample_id to the Celery task
    )
    
    response_data = await run_in_threadpool(task.get)
    return response_data
# Add this to your database functions
async def store_counterfactual_result(user_id, sample_id, original_features, modified_features, 
                                     original_prediction, modified_prediction):
    conn = get_master_db_session()
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS counterfactual_results (
        id INTEGER PRIMARY KEY,
        user_id TEXT,
        sample_id INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        original_features TEXT,
        modified_features TEXT,
        original_prediction REAL,
        modified_prediction REAL
    )
    ''')
    
    cursor.execute('''
    INSERT INTO counterfactual_results 
    (user_id, sample_id, original_features, modified_features, original_prediction, modified_prediction)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        user_id, 
        sample_id, 
        json.dumps(original_features), 
        json.dumps(modified_features),
        float(original_prediction),
        float(modified_prediction)
    ))
    
    conn.commit()
    conn.close()
@app.post("/risk_analysis/")
async def risk_analysis(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile = File(None),
    target_column: str = Form(...),
    drop_columns: str = Form(""),
    current_user: User = Depends(get_current_active_user),
):
    user_id = current_user.id
    # ──────────── Token check ─────────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    # Handle file uploads to Supabase
    file_path = None
    train_path = None
    test_path = None

    if file:
        file_path = await handle_file_upload(user_id, file)
    elif train_file and test_file:
        train_path = await handle_file_upload(user_id, train_file)
        test_path = await handle_file_upload(user_id, test_file)
    else:
        raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")

    task = run_risk_analysis.delay(
        user_id=user_id,
        current_user={
            "id": current_user.id,
            "email": current_user.email,
            "subscription": current_user.subscription
        },
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        drop_columns=drop_columns
    )
    response_data = await run_in_threadpool(task.get)
    return response_data
@app.post("/ab_test/")
async def ab_test(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    variant_column: str = Form("variant"),
    drop_columns: str = Form(""),
    current_user: User = Depends(get_current_active_user),
):
    user_id = current_user.id
    # ──────────── Token check ─────────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    # ──────────── Upload File to Supabase ─────────────
    if not file:
        raise HTTPException(400, "A/B testing requires a single dataset file.")

    file_path = await handle_file_upload(user_id, file)

    # ──────────── Kick off Celery Task ─────────────
    task = run_ab_test.delay(
        user_id=user_id,
        file_path=file_path,
        target_column=target_column,
        variant_column=variant_column,
        drop_columns=drop_columns,
        current_user={
            "id": current_user.id,
            "email": current_user.email,
            "subscription": current_user.subscription,
        },
    )

    # Optionally block and return result (or just return task_id for async polling)
    response_data = await run_in_threadpool(task.get)
    return response_data
@app.post("/decision_paths/")
async def decision_paths(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile = File(None),
    target_column: str = Form(...),
    drop_columns: str = Form(""),
    current_user: User = Depends(get_current_active_user),
):
    # ─────────── Token check ─────────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id

    # Handle file uploads to Supabase
    file_path = None
    train_path = None
    test_path = None

    if file:
        file_path = await handle_file_upload(user_id, file)
    elif train_file and test_file:
        train_path = await handle_file_upload(user_id, train_file)
        test_path = await handle_file_upload(user_id, test_file)
    else:
        raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")

    task = run_decision_paths.delay(
        user_id=user_id,
        current_user={
            "id": current_user.id,
            "email": current_user.email,
            "subscription": current_user.subscription
        },
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        drop_columns=drop_columns
    )
    response_data = await run_in_threadpool(task.get)
    return response_data
@app.post("/forecast/")
async def forecast_time_series(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile = File(None),
    target_column: str = Form("value"),
    drop_columns: str = Form(""),
    periods: int = Form(24),
    current_user: User = Depends(get_current_active_user),
):
    try:
        # ─── Validate `periods` ───
        if periods is None or isinstance(periods, str):
            raise HTTPException(status_code=422, detail="Invalid value for forecast periods.")

        # ─── Token Check ───
        MINIMUM_TOKENS = 0.1
        if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
            raise HTTPException(403, "Insufficient token balance to begin processing.")

        user_id = current_user.id

        # ─── Handle File Uploads ───
        file_path = None
        train_path = None
        test_path = None

        if file:
            file_path = await handle_file_upload(user_id, file)
        elif train_file and test_file:
            train_path = await handle_file_upload(user_id, train_file)
            test_path = await handle_file_upload(user_id, test_file)
        else:
            raise HTTPException(400, "Upload either a single `file` or both `train_file` + `test_file`.")

        # ─── Celery Task Execution ───
        task = run_forecast.delay(
            user_id=user_id,
            file_path=file_path,
            train_path=train_path,
            test_path=test_path,
            current_user={
                "id": current_user.id,
                "email": current_user.email,
                "subscription": current_user.subscription
            },
            target_column=target_column,
            drop_columns=drop_columns,
            periods=periods
        )

        results = await run_in_threadpool(task.get)
        return results

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Time series forecast failed: {str(e)}")
class DownloadRequest(BaseModel):
    file_path: str
    file_name: str
@app.post("/api/prediction-download")
async def download_file(request: DownloadRequest):
    try:
        # Validate that the file path is within the user's directory structure
        # This assumes file_path is in format "user_id/filename"
        if not request.file_path or "/" not in request.file_path:
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Download file from Supabase
        file_bytes = download_file_from_supabase(request.file_path)
        
        # Create a temporary file for FastAPI to serve
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{request.file_name}") as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name
        
        # Return the file response
        # Note: FastAPI will handle cleanup of the temporary file
        return FileResponse(
            path=temp_path,
            filename=request.file_name,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        print(f"[❌] Download failed: {str(e)}")
        raise HTTPException(status_code=404, detail="File not found or download failed")

# Helper function for processing functions that need to work with local files
async def download_file_for_processing(file_path: str) -> str:
    """Download a file from Supabase to a temporary local file for processing"""
    try:
        file_bytes = download_file_from_supabase(file_path)
        
        # Extract filename from path
        filename = file_path.split("/")[-1]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as temp_file:
            temp_file.write(file_bytes)
            return temp_file.name
            
    except Exception as e:
        raise Exception(f"Failed to download file for processing: {str(e)}")
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

app.include_router(auth_router)
if __name__ == "__main__":
    for route in app.routes:
        print(route.path, route.methods)
    uvicorn.run(app, host="127.0.0.1", port=8000)