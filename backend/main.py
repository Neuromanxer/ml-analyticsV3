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
from jwt import ExpiredSignatureError, PyJWTError
from pydantic import BaseModel
from sqlalchemy import Table, MetaData, Column, String, Integer, Float, Boolean
from sqlalchemy.dialects.postgresql import JSONB
import json
from io import StringIO
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from .ai import generate_insights
from .worker import run_classification, run_clustering, run_segment_analysis, run_label_clusters, run_classification_predict, run_visualization, run_counterfactual
from .worker import run_regression, run_risk_analysis, run_regression_predict, run_forecast, run_survival_analysis, run_what_if, run_decision_paths
from .ecs_launcher import launch_job_on_ecs
from .worker import make_json_safe
from .auth import Base, master_engine, decode_user_from_request
from .tokens import TokenUsageLogResponse, TokenUsageLog
from .classification import ModelClassifyingTrainer, lgb_params_c, cat_params_c, xgb_params_c
from .preprocessing import preprocess_data
from .survival import calculate_business_metrics
from .anomaly_detection import train_best_anomaly_detection
from .datasets import (
    Base as DatasetBase,           # in case you want to do Base.metadata.create_all for per-user DBs
    get_user_db,

)
from .datasets import init_db as init_dataset_master_db
from .activity import router as activity_router
from .account import router as a_router
from .target import router as t_router
from .auth import _load_metadata, _save_metadata
from .account import APIStats, SubscriptionInfo, ProfileInfo, APIKeysInfo, BillingInfo, DashboardOut
# These names should match exactly what you export from auth.py
from .auth import (
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
    register_dataset,
    create_user_database,
    get_dataset_by_id,
    get_dataset_data,
    delete_dataset_crud,
    query_dataset,
    get_user_session,
    get_user_session_direct,
    get_user_engine,
    get_master_db_session,
    get_user_by_email,
    master_db_cm,
    Base,
    User

)
from .auth import get_master_db_session
from .auth import router as auth_router
from .tokens import router as token_router
from .storage import upload_file_to_supabase, download_file_from_supabase, handle_file_upload, download_file_from_supabase, list_user_files, delete_file_from_supabase, get_file_url




# from ai import generate_insights
# from worker import run_classification, run_clustering, run_segment_analysis, run_label_clusters, run_classification_predict, run_visualization, run_counterfactual
# from worker import run_regression, run_risk_analysis, run_regression_predict, run_forecast, run_survival_analysis, run_what_if, run_decision_paths
# from ecs_launcher import launch_job_on_ecs
# from worker import make_json_safe
# from auth import Base, master_engine, decode_user_from_request
# from tokens import TokenUsageLogResponse, TokenUsageLog

# from classification import ModelClassifyingTrainer, lgb_params_c, cat_params_c, xgb_params_c
# from preprocessing import preprocess_data
# from survival import calculate_business_metrics
# from anomaly_detection import train_best_anomaly_detection
# from datasets import (
#     Base as DatasetBase,           # in case you want to do Base.metadata.create_all for per-user DBs
#     get_user_db,

# )
# from datasets import init_db as init_dataset_master_db
# from activity import router as activity_router
# from account import router as a_router
# from target import router as t_router
# from auth import _load_metadata, _save_metadata
# from account import APIStats, SubscriptionInfo, ProfileInfo, APIKeysInfo, BillingInfo, DashboardOut
# # These names should match exactly what you export from auth.py
# from auth import (
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
#     register_dataset,
#     create_user_database,
#     get_dataset_by_id,
#     get_dataset_data,
#     delete_dataset_crud,
#     query_dataset,
#     get_user_session,
#     get_user_session_direct,
#     get_user_engine,
#     get_master_db_session,
#     get_user_by_email,
#     master_db_cm,
#     Base,
#     User

# )
# from auth import get_master_db_session
# from auth import router as auth_router
# from tokens import router as token_router
# from storage import upload_file_to_supabase, download_file_from_supabase, handle_file_upload, download_file_from_supabase, list_user_files, delete_file_from_supabase, get_file_url




# Create a directory for storing uploaded CSV files if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

class DatasetQuery(BaseModel):
    query: str

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

# File upload directory
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

@app.post("/dataset/profile")
async def dataset_profile(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
        profile = profile_dataset(df)
        return JSONResponse(content=profile)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to profile dataset: {str(e)}")
        import pandas as pd

def profile_dataset(df: pd.DataFrame) -> dict:
    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "data_types": df.dtypes.apply(lambda dt: str(dt)).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "outliers": {},
    }

    for col in df.select_dtypes(include="number").columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        profile["outliers"][col] = int(((df[col] < lower) | (df[col] > upper)).sum())

    return profile

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
    except PyJWTError:
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

class UsageTrackerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response: Response = await call_next(request)
        duration = time.monotonic() - start

        tracked_paths = [
            "/classification/",
            "/regression/",
            "/clustering/",
            "/forecast/",
            "/segment_analysis/",
            "/survival/",
            "/counterfactual/",
            "/visualize/",
            "/feature_impact/",
            "/risk_analysis/",
            "/decision_paths/",
            "/classification/predict/",
            "/regression/predict/",
        ]
        if request.url.path not in tracked_paths:
            return response

        # --- concave-down pricing parameters ---
        DATA_PRICE_SCALE = 17.0      # updated scale factor
        PRICE_PER_SECOND = 0.50      # compute time still billed per second

        # Estimate data volume
        content_length = request.headers.get("content-length", "0")
        bytes_estimated = int(content_length) if content_length.isdigit() else 0
        bytes_processed = getattr(request.state, "file_bytes", bytes_estimated)
        mb_used = bytes_processed / (1024 * 1024)

        # Logarithmic (concave) data cost + linear compute cost
        cost_data = DATA_PRICE_SCALE * math.log1p(mb_used)
        cost_time = PRICE_PER_SECOND * duration
        cost_for_request = round(cost_data + cost_time, 2)

        # Billing update
        OVERDRAFT_LIMIT = -1.0
        db_gen = get_master_db_session()
        try:
            db = next(db_gen)
            try:
                payload = decode_user_from_request(request)
                if not payload:
                    return response

                user_email = payload.get("sub")
                user = db.query(User).filter(User.email == user_email).first()
                if user:
                    user.tokens = (user.tokens or 0.0) - cost_for_request
                    if user.tokens < OVERDRAFT_LIMIT:
                        db.rollback()
                        raise HTTPException(
                            status_code=403,
                            detail=f"You have exceeded your usage limit. Token balance is {user.tokens:.2f}."
                        )
                    user.total_bytes_processed += bytes_processed
                    user.total_compute_seconds += duration
                    user.total_cost_dollars += cost_for_request
                    db.commit()
            except Exception as billing_err:
                print(f"Billing error: {billing_err}")
            finally:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
        except Exception as db_err:
            print(f"DB session error: {db_err}")

        return response

from fastapi import Request, HTTPException
import time
import openai
from fastapi import Request, HTTPException, Response
import time

class AITokenBillingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tracked_path = "/api/ai_insights"

        if not request.url.path.startswith(tracked_path):
            return await call_next(request)

        # ─── Run the request and time it ─────────────────────
        start_time = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start_time

        try:
            db_gen = get_master_db_session()
            db = next(db_gen)

            # Pull current user
            payload = decode_user_from_request(request)
            if not payload:
                return response

            user_email = payload.get("sub")  # Or user_id, depending on what you encoded
            user = db.query(User).filter(User.email == user_email).first()

            if not user:
                return response

            # ─── Pull token usage (if any) from OpenAI call ─────
            openai_tokens_used = getattr(request.state, "openai_tokens_used", 0)

            # OpenAI Pricing (e.g., GPT-4 Vision or GPT-4o)
            OPENAI_COST_PER_1K = 0.04  # your actual OpenAI rate
            MARKUP = 3.0               # 4x markup for profit

            ai_cost_usd = (openai_tokens_used / 1000) * OPENAI_COST_PER_1K
            ai_charge = round(ai_cost_usd * MARKUP, 3)

            # ─── Add time-based and size-based charges ─────────
            PRICE_PER_MB =  0.1875
            PRICE_PER_SECOND =  0.375

            content_length = request.headers.get("content-length")
            bytes_processed = int(content_length) if content_length and content_length.isdigit() else 0
            mb_used = bytes_processed / (1024 * 1024)

            time_charge = duration * PRICE_PER_SECOND
            size_charge = mb_used * PRICE_PER_MB

            additional_charge = round(time_charge + size_charge, 3)

            # ─── Total Charge Calculation ─────────────
            total_charge_tokens = round(ai_charge + additional_charge, 3)

            if user.tokens is None:
                user.tokens = 0.0

            user.tokens -= total_charge_tokens

            if user.tokens < -1.0:
                db.rollback()
                raise HTTPException(403, detail=f"🚫 Token balance exceeded. Required: {total_charge_tokens}, Available: {user.tokens:.2f}. Please top up.")

            # Tracking
            user.total_ai_tokens_used += openai_tokens_used
            user.total_bytes_processed += bytes_processed
            user.total_compute_seconds += duration
            user.total_cost_dollars += (ai_cost_usd + additional_charge)

            db.commit()

        except Exception as e:
            print("⚠️ Billing Middleware Error:", e)
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass

        return response

app.add_middleware(AITokenBillingMiddleware)
app.add_middleware(UsageTrackerMiddleware)
app.add_middleware(LoggingMiddleware)


# Include Routers
app.include_router(auth_router)
app.include_router(activity_router)
app.include_router(a_router)
app.include_router(token_router)
app.include_router(t_router)
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

def infer_sqlalchemy_column(name, dtype):
    """Infer SQLAlchemy column type from pandas dtype."""
    if pd.api.types.is_integer_dtype(dtype):
        return Column(name, Integer)
    elif pd.api.types.is_float_dtype(dtype):
        return Column(name, Float)
    else:
        return Column(name, String)

@app.post("/datasets/", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    current_user=Depends(get_current_active_user)
):
    logger.info(f"Current user type: {type(current_user)}, content: {current_user}")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    temp_path = None  # needed for cleanup
    try:
        # 1. Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        # 2. Upload to Supabase storage (organized by user)
        supabase_path = upload_file_to_supabase(
            user_id=str(current_user.id),
            file_path=temp_path,
            filename=file.filename
        )

        # 3. Read the CSV file BEFORE deleting it
        df = pd.read_csv(temp_path)

        # Now safe to remove temp file
        os.remove(temp_path)

        # 4. Prepare the data
        if 'id' not in df.columns:
            df.insert(0, 'id', range(1, len(df) + 1))
        row_count, column_count = df.shape
        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)  # Still valid even after reading

        table_name = name.replace(" ", "_").lower()
        dataset_info = None

        # 5. Register dataset in user database
        with get_user_db(current_user) as db:
            metadata = MetaData()
            columns = [infer_sqlalchemy_column(col, df[col].dtype) for col in df.columns]
            dataset_table = Table(table_name, metadata, *columns)

            dataset_table.create(bind=db.bind, checkfirst=True)
            df_records = df.replace({np.nan: None}).to_dict(orient="records")
            db.execute(dataset_table.insert(), df_records)

            schema = {col: str(df[col].dtype) for col in df.columns}
            dataset = register_dataset(
                db=db,
                name=name,
                description=description,
                table_name=table_name,
                file_path=supabase_path,
                row_count=row_count,
                column_count=column_count,
                schema=schema,
                overwrite_existing=True
            )

            db.commit()
            db.refresh(dataset)

            dataset_info = {
                "id": dataset.id,
                "name": dataset.name,
                "row_count": dataset.row_count,
                "column_count": dataset.column_count
            }

        # 6. Update user’s storage usage in master database
        with master_db_cm() as master_db:
            master_db.execute(text("""
                UPDATE users
                SET storage_used = storage_used + :additional
                WHERE id = :user_id
            """), {"additional": file_size_mb, "user_id": current_user.id})

        return {
            "message": "Dataset uploaded, saved, and registered successfully",
            "dataset_id": dataset_info["id"],
            "name": dataset_info["name"],
            "rows": dataset_info["row_count"],
            "columns": dataset_info["column_count"]
        }

    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Error processing dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")
@app.get("/datasets/")
def list_datasets(current_user = Depends(get_current_active_user)):
    try:
        # Using the get_user_db context manager to automatically handle session closure
        with get_user_db(current_user) as user_db: 
            # Query the datasets from the database
            datasets = user_db.query(Dataset).all()

            # Format the datasets into a dictionary
            return {
                "datasets": [
                    {
                        "id": d.id,
                        "name": d.name,
                        "created_at": d.created_at,
                        "rows": d.row_count,
                        "columns": d.column_count,
                        "description": d.description,
                    } for d in datasets
                ]
            }
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")

@app.get("/datasets/{dataset_id}")
async def get_dataset_metadata(
    dataset_id: int = Path(..., gt=0),
    current_user=Depends(get_current_active_user)
):
    with get_user_db(current_user) as db:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        return {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "created_at": dataset.created_at,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
        }

@app.get("/datasets/{dataset_id}/preview")
def get_dataset_preview(
    dataset_id: int, 
    current_user = Depends(get_current_active_user)
):
    try:
        with get_user_db(current_user) as user_db:
            dataset = user_db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                raise HTTPException(status_code=404, detail="Dataset not found")

            formatted_created_at = dataset.created_at.strftime('%Y-%m-%d') if dataset.created_at else 'Unknown date'

            # Reflect only the target table (no `bind=` needed in SQLAlchemy 2.0)
            dataset_table = Table(dataset.table_name, MetaData(), autoload_with=user_db.bind)

            stmt = dataset_table.select().limit(5)
            result = user_db.execute(stmt)
            preview_data = [dict(row._mapping) for row in result]

            return {
                "name": dataset.name,
                "created_at": formatted_created_at,
                "preview_data": preview_data
            }

    except Exception as e:
        logger.error(f"Error getting dataset preview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting dataset preview: {str(e)}")


@app.get("/datasets/{dataset_id}/data")
async def get_dataset_data(
    dataset_id: int, 
    limit: int = 100,
    current_user = Depends(get_current_active_user)
):
    """
    Get data from a dataset with pagination.
    """
    try:
        result = get_dataset_data(current_user.email, dataset_id, limit=limit)
        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Format the created_at date
        formatted_created_at = result["dataset"].created_at.strftime('%Y-%m-%d') if result["dataset"].created_at else 'Unknown date'

        return {
            "dataset": {
                "id": result["dataset"].id,
                "name": result["dataset"].name,
                "created_at": formatted_created_at  # Add created_at here
            },
            "columns": list(result["columns"]),
            "data": result["data"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving dataset data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving dataset data: {str(e)}")

@app.post("/datasets/{dataset_id}/query")
async def query_dataset(
    dataset_id: int,
    query_data: DatasetQuery,
    current_user = Depends(get_current_active_user)
):
    """
    Run a custom SQL query against a dataset.
    Only SELECT statements are allowed for security.
    """
    try:
        result = query_dataset(current_user.email, dataset_id, query_data.query)
        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        formatted_created_at = result["dataset"].created_at.strftime('%Y-%m-%d') if result["dataset"].created_at else 'Unknown date'

        return {
            "dataset": {
                "id": result["dataset"].id,
                "name": result["dataset"].name,
                "created_at": formatted_created_at
            },
            "columns": list(result["columns"]),
            "data": result["data"],
            "row_count": result["row_count"]
        }
    except ValueError as e:
        logger.warning(f"Invalid query for dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing query on dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing query: {str(e)}")

# Helper function to fetch preview data
def fetch_preview_data(file_path: str):
    """
    Fetch preview data from a CSV file
    """
    try:
        df = pd.read_csv(file_path, nrows=10)  # Read first 10 rows for preview
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        return []

API_BASE_PATH = "/datasets"

@app.get(
    f"{API_BASE_PATH}/{{dataset_id}}/download",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
)
async def download_dataset(
    dataset_id: int = Path(..., gt=0),
    current_user = Depends(get_current_active_user)
):
    """
    Return the raw CSV file for a dataset.
    """
    try:
        # 1) Open the user's DB and load the Dataset record
        with get_user_db(current_user) as db:
            ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not ds:
                raise HTTPException(status_code=404, detail="Dataset not found")

        # 2) Ensure the file exists on disk
        if not ds.file_path or not os.path.exists(ds.file_path):
            raise HTTPException(status_code=404, detail="File not found on disk")

        # 3) Stream it back
        return FileResponse(
            path=ds.file_path,
            filename=f"{ds.name}.csv",
            media_type="text/csv"
        )

    except HTTPException:
        # Re-raise 404s and auth errors
        raise
    except Exception as e:
        # Any other error
        logger.error(f"Error in download_dataset: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error downloading dataset: {e}"
        )

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
        return JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content={"message": "Invalid form data", "error": str(ve)}
        )

    except OSError as ioe:
        # Common for: I/O operation on closed file
        logger.error(f"❌ I/O error in /upload_csv: {str(ioe)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content={
                "message": "I/O error: It looks like the file stream was closed or corrupted. "
                           "Please reupload your file and try again.",
                "error": str(ioe)
            }
        )

    except Exception as e:
        logger.error(f"❌ Unexpected error in /upload_csv: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": "Unexpected error storing CSV data. Please try again.",
                "error": str(e)
            }
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
    user_id: str = Form(...),
    current_user: User = Depends(get_current_active_user),
):
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
    user_id: str = Form(...),
    current_user: User = Depends(get_current_active_user),
):
    # ──────────── Token check ─────────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    # ──────────── Upload File to Supabase ─────────────
    if not file:
        raise HTTPException(400, "A/B testing requires a single dataset file.")

    file_path = await handle_file_upload(user_id, file)

    # ──────────── Kick off Celery Task ─────────────
    task = run_ab_test_task.delay(
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