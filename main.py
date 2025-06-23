import logging
import time
import io
import joblib
import pandas as pd
import os
import jwt
import shutil
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Request, Depends, Path
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path as PathL
import json
import aiofiles
from ai import generate_insights
from worker import run_classification, run_clustering, run_segment_analysis, run_label_clusters 
from worker import run_regression, run_forecast, run_survival_analysis, run_what_if, run_decision_paths
from ecs_launcher import launch_job_on_ecs
from starlette.middleware.base import BaseHTTPMiddleware
from sklearn.model_selection import train_test_split

from tokens import TokenUsageLogResponse, TokenUsageLog
from sklearn.model_selection import train_test_split, KFold
from classification import ModelClassifyingTrainer, lgb_params_c, cat_params_c, xgb_params_c
from preprocessing import preprocess_data
from survival import calculate_business_metrics
from sqlalchemy import Column
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from fastapi import APIRouter, HTTPException, Depends, status, Query
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
from anomaly_detection import train_best_anomaly_detection
import uuid
from pydantic import EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Float, Table, MetaData, Boolean, DateTime, ForeignKey, Text, text
from datasets import (
    Base as DatasetBase,           # in case you want to do Base.metadata.create_all for per-user DBs
    Dataset,
    DatasetResponse,
    DatasetCreate,
    get_user_db,
    register_dataset,
    create_user_database,
    get_dataset_by_id,
    get_dataset_data,
    delete_dataset_crud,
    query_dataset,
    init_db as init_dataset_master_db,
    d_router
)
from starlette.concurrency import run_in_threadpool
from activity import router as activity_router
from account import router as a_router
from account import APIStats, SubscriptionInfo, ProfileInfo, APIKeysInfo, BillingInfo, DashboardOut
# These names should match exactly what you export from auth.py
from auth import (
    # Authentication & token utilities
    get_current_active_user,
    get_current_user,
    create_access_token,
    authenticate_user,

    # Pydantic schemas for auth
    UserCreate,
    UserResponse,
    Token,
    AuthTokenResponse,
    RegisterResponse,

    # Database/session helpers
    get_user_session,
    get_user_session_direct,
    get_user_engine,
    get_master_db_session,
    get_user_by_email,
    master_db_cm,
    Base,
    User

)
from auth import router as auth_router
from tokens import router as token_router
from sqlalchemy.ext.declarative import declarative_base
import uuid, os, io
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from jwt import ExpiredSignatureError, PyJWTError
from pydantic import BaseModel
from sqlalchemy import Table, MetaData, Column, String, Integer, Float, Boolean
from sqlalchemy.dialects.postgresql import JSONB
import json
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
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

print("Loaded environment:")
print("POSTGRES_USER:", os.getenv("POSTGRES_USER"))
print("POSTGRES_PASSWORD:", os.getenv("POSTGRES_PASSWORD"))
print("POSTGRES_HOST:", os.getenv("POSTGRES_HOST"))
print("POSTGRES_PORT:", os.getenv("POSTGRES_PORT"))


# Middleware for logging requests and responses
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        body = await request.body()

        logger.info(f"📥 Request: {request.method} {request.url} - Body: {body.decode('utf-8')}")
        
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(f"📤 Response: {response.status_code} - Time: {process_time:.4f}s")
        
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
class UsageTrackerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response: Response = await call_next(request)
        duration = time.monotonic() - start

        tracked_paths = ["/classification/", "/regression/", "/clustering/", "/forecast/", "/segment_analysis/", "/survival/", "/counterfactual"]
        if request.url.path not in tracked_paths:
            return response

        # Pricing
        PRICE_PER_MB = 0.02
        PRICE_PER_SECOND = 0.05
        OVERDRAFT_LIMIT = -1.0

        # Estimate or pull actual size
        content_length = request.headers.get("content-length")
        bytes_estimated = int(content_length) if content_length and content_length.isdigit() else 0
        bytes_processed = getattr(request.state, "file_bytes", bytes_estimated)
        mb_used = bytes_processed / (1024 * 1024)

        cost_for_request = round(mb_used * PRICE_PER_MB + duration * PRICE_PER_SECOND, 2)

        db_gen = get_master_db_session()
        try:
            db = next(db_gen)
            try:
                user = await get_current_active_user(request)

                if user:
                    if user.tokens is None:
                        user.tokens = 0.0

                    user.tokens -= cost_for_request

                    if user.tokens < OVERDRAFT_LIMIT:
                        db.rollback()
                        raise HTTPException(
                            status_code=403,
                            detail=f"You have exceeded your usage limit. Token balance is {user.tokens:.2f}. Please top up."
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

app.add_middleware(UsageTrackerMiddleware)
app.add_middleware(LoggingMiddleware)


# Include Routers
app.include_router(auth_router)
app.include_router(activity_router)
app.include_router(d_router)
app.include_router(a_router)
app.include_router(token_router)
# Dataset models
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

    file_path = None
    try:
        # 1. Save file
        user_upload_dir = os.path.join(UPLOAD_DIR, f"user_{current_user.id}")
        os.makedirs(user_upload_dir, exist_ok=True)

        file_path = os.path.join(user_upload_dir, file.filename)
        base_name, extension = os.path.splitext(file_path)
        counter = 1
        while os.path.exists(file_path):
            file_path = f"{base_name}_{counter}{extension}"
            counter += 1

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Load and analyze CSV
        df = pd.read_csv(file_path)
        if 'id' not in df.columns:
            df.insert(0, 'id', range(1, len(df) + 1))
        row_count, column_count = df.shape
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        table_name = name.replace(" ", "_").lower()

        # 3. Create table and register dataset
        dataset_info = None  # Initialize variable to store dataset info
        
        with get_user_db(current_user) as db:
            metadata = MetaData()
            columns = [infer_sqlalchemy_column(col, df[col].dtype) for col in df.columns]
            dataset_table = Table(
                table_name, metadata,
                *columns
            )

            dataset_table.create(bind=db.bind, checkfirst=True)

            # 4. Insert data into the new table
            df_records = df.replace({np.nan: None}).to_dict(orient="records")
            db.execute(dataset_table.insert(), df_records)

            # 5. Prepare schema as a dict
            schema = {col: str(df[col].dtype) for col in df.columns}

            # 6. Register dataset
            dataset = register_dataset(
                db=db,
                name=name,
                description=description,
                table_name=table_name,
                file_path=file_path,
                row_count=row_count,
                column_count=column_count,
                schema=schema,
                overwrite_existing=True
            )
            
            # ✅ CRITICAL FIX: Extract data while session is still active
            db.commit()  # Ensure the dataset is committed to the database
            db.refresh(dataset)  # Refresh to get the latest data including ID
            
            # Store the dataset information before session closes
            dataset_info = {
                "id": dataset.id,
                "name": dataset.name,
                "row_count": dataset.row_count,
                "column_count": dataset.column_count
            }

        # Update user's storage usage in master DB
        with master_db_cm() as master_db:
            master_db.execute(text("""
                UPDATE users
                SET storage_used = storage_used + :additional
                WHERE id = :user_id
            """), {"additional": file_size_mb, "user_id": current_user.id})

        # ✅ Use the stored dataset info instead of accessing the detached object
        return {
            "message": "Dataset uploaded, saved, and registered successfully",
            "dataset_id": dataset_info["id"],
            "name": dataset_info["name"],
            "rows": dataset_info["row_count"],
            "columns": dataset_info["column_count"]
        }

    except Exception as e:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
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



@app.delete("/datasets/{dataset_id}")
async def delete_dataset_endpoint(
    dataset_id: int,
    current_user = Depends(get_current_active_user)
):
    """
    Delete a dataset and its associated data table.
    """
    try:
        # Call the CRUD helper (not the endpoint itself)
        deleted = delete_dataset_crud(current_user.email, dataset_id)

        # If your helper returns a coroutine, await it:
        # if asyncio.iscoroutine(deleted):
        #     deleted = await deleted

        if not deleted:
            raise HTTPException(status_code=404, detail="Dataset not found")

        created_at = (
            deleted.created_at.strftime("%Y-%m-%d")
            if deleted.created_at else
            "Unknown date"
        )

        return {
            "message": "Dataset deleted successfully",
            "dataset": {
                "id": deleted.id,
                "name": deleted.name,
                "created_at": created_at
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {e}")

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

# CSV upload endpoint for compatibility
@app.post("/upload_csv/")
async def upload_csv(
    file: UploadFile = File(...),
    drop_columns: str = Form(None),
    target_column: str = Form(None),
    feature_column: str = Form(None)    
):
    logger.info("✅ /upload_csv/ endpoint called")
    try:
        # Save the uploaded file temporarily
        file_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        
        # Reset file position to beginning before reading
        await file.seek(0)
        
        # Read and save file content
        file_content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        # Process drop_columns, target_column, and feature_column if provided
        drop_cols = drop_columns.split(",") if drop_columns else []
        target_col = target_column if target_column else None
        feature_cols = feature_column.split(",") if feature_column and not isinstance(feature_column, type(Form)) else []

        # Log the received data
        logger.info(f"Drop Columns: {drop_cols}")
        logger.info(f"Target Column: {target_col}")
        logger.info(f"Feature Columns: {feature_cols}")

        return {"message": f"CSV file '{file.filename}' processed successfully"}
    except Exception as e:
        logger.error(f"❌ Error in /upload_csv: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": "Error storing CSV data", "error": str(e)})
@app.post("/classification/")
async def classification(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile  = File(None),
    target_column: str     = Form("target"),
    drop_columns: str      = Form(""),
    current_user: User     = Depends(get_current_active_user),
):
    # ───────── Token check ───────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id
    user_dir = PathL("user_uploads") / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ───────── Handle file uploads ───────────
    if file:
        file_path = os.path.join(user_dir, file.filename)
        async with aiofiles.open(file_path, "wb") as out:
            await out.write(await file.read())

    elif train_file and test_file:
        train_path = os.path.join(user_dir, train_file.filename)
        async with aiofiles.open(train_path, "wb") as out:
            await out.write(await train_file.read())

        test_path = os.path.join(user_dir, test_file.filename)
        async with aiofiles.open(test_path, "wb") as out:
            await out.write(await test_file.read())
    else:
        raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")

    # ───────── Choose execution mode ───────────
    if current_user.subscription in ["Pro", "Enterprise"]:
        # Premium = fully isolated ECS job
        response = launch_job_on_ecs(
            user_id=user_id,
            file_path=file_path,
            train_path=train_path,
            test_path=test_path,
            target_column=target_column,
            drop_columns=drop_columns
        )
        return {"status": "queued_on_ecs", "ecs_response": response}

    else:
        # Free users = Celery worker
        task = run_classification.delay(
            user_id,
            file_path,
            train_path,
            test_path,
            target_column,
            drop_columns
        )
        response_data = await run_in_threadpool(task.get)
        return response_data
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
    user_dir = PathL("user_uploads") / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ───────── Handle file uploads ───────────
    if file:
        file_path = os.path.join(user_dir, file.filename)
        async with aiofiles.open(file_path, "wb") as out:
            await out.write(await file.read())

    elif train_file and test_file:
        train_path = os.path.join(user_dir, train_file.filename)
        async with aiofiles.open(train_path, "wb") as out:
            await out.write(await train_file.read())

        test_path = os.path.join(user_dir, test_file.filename)
        async with aiofiles.open(test_path, "wb") as out:
            await out.write(await test_file.read())
    else:
        raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")

    # ───────── Choose execution mode ───────────
    if current_user.subscription in ["Pro", "Enterprise"]:
        # Premium = isolated ECS job
        response = launch_job_on_ecs(
            user_id=user_id,
            file_path=file_path,
            train_path=train_path,
            test_path=test_path,
            target_column=target_column,
            drop_columns=drop_columns,
            job_type="cluster"  # Important: ECS must know it's clustering job
        )
        return {"status": "queued_on_ecs", "ecs_response": response}

    else:
        # Free users = Celery worker
        task = run_clustering.delay(
            user_id=user_id,
            file_path=file_path,   # might be None
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
    user_dir = PathL("user_uploads") / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ───────── Handle file uploads ───────────
    if file:
        file_path = os.path.join(user_dir, file.filename)
        async with aiofiles.open(file_path, "wb") as out:
            await out.write(await file.read())

    elif train_file and test_file:
        train_path = os.path.join(user_dir, train_file.filename)
        async with aiofiles.open(train_path, "wb") as out:
            await out.write(await train_file.read())

        test_path = os.path.join(user_dir, test_file.filename)
        async with aiofiles.open(test_path, "wb") as out:
            await out.write(await test_file.read())
    else:
        raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")

    # ───────── Choose execution mode ───────────
    if current_user.subscription in ["Pro", "Enterprise"]:
        # Premium = fully isolated ECS job
        response = launch_job_on_ecs(
            user_id=user_id,
            file_path=file_path,
            train_path=train_path,
            test_path=test_path,
            target_column=target_column,
            drop_columns=drop_columns,
            job_type="segment_analysis"  # Pass job type to ECS handler
        )
        return {"status": "queued_on_ecs", "ecs_response": response}

    else:
        # Free users = Celery worker
        task = run_segment_analysis.delay(
            user_id,
            file_path,
            train_path,
            test_path,
            target_column,
            drop_columns
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
    user_dir = PathL("user_uploads") / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ───────── Handle file uploads ───────────
    if file:
        file_path = os.path.join(user_dir, file.filename)
        async with aiofiles.open(file_path, "wb") as out:
            await out.write(await file.read())

    elif train_file and test_file:
        train_path = os.path.join(user_dir, train_file.filename)
        async with aiofiles.open(train_path, "wb") as out:
            await out.write(await train_file.read())

        test_path = os.path.join(user_dir, test_file.filename)
        async with aiofiles.open(test_path, "wb") as out:
            await out.write(await test_file.read())
    else:
        raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")

    # ───────── Choose execution mode ───────────
    if current_user.subscription in ["Pro", "Enterprise"]:
        # Premium = isolated ECS job
        response = launch_job_on_ecs(
            user_id=user_id,
            file_path=file_path,
            train_path=train_path,
            test_path=test_path,
            feature_columns=feature_columns,
            job_type="label_clusters"  # Pass job type to ECS handler
        )
        return {"status": "queued_on_ecs", "ecs_response": response}

    else:
        # Free users = Celery worker
        task = run_label_clusters.delay(
            user_id,
            file_path,
            train_path,
            test_path,
            feature_columns
        )
        response_data = await run_in_threadpool(task.get)
        return response_data

# Helper function to save plot to static directory and return filename

# Helper function for consistent file processing
def process_uploaded_file(file_path):
    """Process a CSV file and return the DataFrame"""
    try:
        df = pd.read_csv(file_path)
        return df
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
    user_dir = PathL("user_uploads") / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ───────── Handle file uploads ───────────
    if file:
        file_path = os.path.join(user_dir, file.filename)
        async with aiofiles.open(file_path, "wb") as out:
            await out.write(await file.read())

    elif train_file and test_file:
        train_path = os.path.join(user_dir, train_file.filename)
        async with aiofiles.open(train_path, "wb") as out:
            await out.write(await train_file.read())

        test_path = os.path.join(user_dir, test_file.filename)
        async with aiofiles.open(test_path, "wb") as out:
            await out.write(await test_file.read())
    else:
        raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")

    # ───────── Choose execution mode ───────────
    if current_user.subscription in ["Pro", "Enterprise"]:
        # Premium users -> ECS job
        response = launch_job_on_ecs(
            user_id=user_id,
            file_path=file_path,
            train_path=train_path,
            test_path=test_path,
            target_column=target_column,
            drop_columns=drop_columns,
            job_type="regression"  # Pass job type to ECS handler
        )
        return {"status": "queued_on_ecs", "ecs_response": response}

    else:
        # Free users -> Celery worker
        task = run_regression.delay(
            user_id,
            file_path,
            train_path,
            test_path,
            target_column,
            drop_columns
        )
        response_data = await run_in_threadpool(task.get)
        return response_data

@app.post("/regression/predict/")
async def regression_predict(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile  = File(None),
    drop_columns: str      = Form(""),
    current_user: User     = Depends(get_current_active_user),
):
    # ───────── Token check ───────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id
    user_dir = PathL("user_uploads") / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ───────── Handle file uploads ───────────
    if file:
        file_path = os.path.join(user_dir, file.filename)
        async with aiofiles.open(file_path, "wb") as out:
            await out.write(await file.read())

    elif train_file and test_file:
        train_path = os.path.join(user_dir, train_file.filename)
        async with aiofiles.open(train_path, "wb") as out:
            await out.write(await train_file.read())

        test_path = os.path.join(user_dir, test_file.filename)
        async with aiofiles.open(test_path, "wb") as out:
            await out.write(await test_file.read())
    else:
        raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")

    # ───────── Choose execution mode ───────────
    if current_user.subscription in ["Pro", "Enterprise"]:
        # Premium = fully isolated ECS job (optional cloud deployment)
        response = launch_job_on_ecs(
            user_id=user_id,
            file_path=file_path,
            train_path=train_path,
            test_path=test_path,
            drop_columns=drop_columns
        )
        return {"status": "queued_on_ecs", "ecs_response": response}

    else:
        # Free users = Celery worker
        task = run_regression_predict.delay(
            user_id=user_id,
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

@app.get("/visualizations")
async def list_visualizations(
    current_user: User = Depends(get_current_active_user),
    override_id: Optional[str] = Query(None),
):
    try:
        logging.info(f"Fetching visualizations for user: {current_user.id}")
        
        user_id = current_user.id
        if override_id:
            user_id = override_id
            
        # Ensure directory exists
        #  ↓ change from Path("data", "visualizations") to either of the two options below
        meta_dir = PathL("data") / "visualizations"
        # meta_dir = Path("data/visualizations")
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize user_id for safe filename
        safe_user_id = str(user_id).replace('/', '_').replace('\\', '_')
        meta_file = meta_dir / f"{safe_user_id}.json"
        
        logging.info(f"Looking for file: {meta_file}")
        
        if not meta_file.exists():
            logging.info("No visualization file found, returning empty list")
            return []
            
        content = meta_file.read_text()
        logging.info(f"File content length: {len(content)}")
        
        return json.loads(content)
        
    except Exception as e:
        logging.error(f"Error in list_visualizations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

def _get_meta_path(user_id: str) -> Path:
    meta_dir = PathL("data") / "visualizations"
    meta_dir.mkdir(parents=True, exist_ok=True)
    safe_user_id = str(user_id).replace("/", "_").replace("\\", "_")
    return meta_dir / f"{safe_user_id}.json"

def _load_metadata(user_id: str) -> List[dict]:
    meta_path = _get_meta_path(user_id)
    if not meta_path.exists():
        return []
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Could not parse metadata.json")

def _save_metadata(user_id: str, data: List[dict]):
    meta_path = _get_meta_path(user_id)
    meta_path.write_text(json.dumps(data, indent=2))

from fastapi.responses import FileResponse, StreamingResponse
from io import BytesIO

@app.get("/visualizations/{viz_id}/download")
async def download_visualization(
    viz_id: str,
    current_user: User = Depends(get_current_active_user)
):
    user_id = current_user.id
    meta = _load_metadata(user_id)

    entry = next((e for e in meta if e["id"] == viz_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Visualization not found")

    # Try all keys
    img_url = (
        entry.get("imageUrl") or
        entry.get("feature_importance_detailed_url") or
        entry.get("feature_importance_detailed") or
        entry.get("imageData")
    )

    logging.info(f"[DOWNLOAD] img_url raw: {repr(img_url)[:100]}")

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

    # fallback for file path
    file_path = PathL(img_url.lstrip("/"))
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    return FileResponse(str(file_path), media_type="image/png", filename=f"visualization-{viz_id}.png")

@app.delete("/visualizations/{viz_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_visualization(
    viz_id: str,
    current_user: User = Depends(get_current_active_user)
):
    user_id = current_user.id
    meta = _load_metadata(user_id)

    print(f"[DELETE] Looking for viz_id: {viz_id}")
    print(f"[DELETE] Available IDs: {[e.get('id', '<missing>') for e in meta]}")

    idx = next((i for i, e in enumerate(meta) if e["id"] == viz_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Visualization not found")

    entry = meta.pop(idx)

    def _try_remove(url_key: str):
        url = entry.get(url_key)
        if not url or url.startswith("data:image"):
            return
        file_path = PathL(url.lstrip("/"))
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                print(f"Failed to delete file {file_path}: {e}")

    _try_remove("thumbnailData")
    _try_remove("imageData")
    _try_remove("imageUrl")
    _try_remove("feature_importance")
    _try_remove("feature_importance_detailed")

    _save_metadata(user_id, meta)

@app.post("/segment_analysis/")
async def segment_analysis(
    file: UploadFile = File(...),
    target_column: str = Form(None),
    current_user: User = Depends(get_current_active_user),
):
    # ───────── Token check ───────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id
    user_dir = PathL("user_uploads") / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    PathL("models").mkdir(exist_ok=True)
    PathL("data").mkdir(exist_ok=True)

    # ───────── Handle file upload ───────────
    file_path = os.path.join(user_dir, file.filename)
    async with aiofiles.open(file_path, "wb") as out:
        await out.write(await file.read())

    # ───────── Choose execution mode ───────────
    if current_user.subscription in ["Pro", "Enterprise"]:
        # Premium plan = external ECS job (optional future deployment)
        response = launch_job_on_ecs(
            user_id=user_id,
            file_path=file_path,
            target_column=target_column
        )
        return {"status": "queued_on_ecs", "ecs_response": response}

    else:
        # Free users = run on Celery worker
        task = run_segment_analysis.delay(
            user_id=user_id,
            file_path=file_path,
            target_column=target_column
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
    user_dir = PathL("user_uploads") / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ───────── Handle file uploads ───────────
    if file:
        file_path = os.path.join(user_dir, file.filename)
        async with aiofiles.open(file_path, "wb") as out:
            await out.write(await file.read())

    elif train_file and test_file:
        train_path = os.path.join(user_dir, train_file.filename)
        async with aiofiles.open(train_path, "wb") as out:
            await out.write(await train_file.read())

        test_path = os.path.join(user_dir, test_file.filename)
        async with aiofiles.open(test_path, "wb") as out:
            await out.write(await test_file.read())
    else:
        raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")

    # ───────── Choose execution mode ───────────
    if current_user.subscription in ["Pro", "Enterprise"]:
        # Premium users -> ECS job (optional cloud deployment)
        response = launch_job_on_ecs(
            user_id=user_id,
            file_path=file_path,
            train_path=train_path,
            test_path=test_path,
            target_column=target_column,
            feature_column=feature_column,
            job_type="visualization"
        )
        return {"status": "queued_on_ecs", "ecs_response": response}

    else:
        # Free users -> Celery worker
        task = run_visualization.delay(
            user_id=user_id,
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
    current_user: User = Depends(get_current_active_user),
):
    # ───────── Token check ───────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_id = current_user.id
    user_dir = PathL("user_uploads") / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ───────── Handle file uploads ───────────
    if file:
        file_path = os.path.join(user_dir, file.filename)
        async with aiofiles.open(file_path, "wb") as out:
            await out.write(await file.read())

    elif train_file and test_file:
        train_path = os.path.join(user_dir, train_file.filename)
        async with aiofiles.open(train_path, "wb") as out:
            await out.write(await train_file.read())

        test_path = os.path.join(user_dir, test_file.filename)
        async with aiofiles.open(test_path, "wb") as out:
            await out.write(await test_file.read())
    else:
        raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")

    # ───────── Choose execution mode ───────────
    if current_user.subscription in ["Pro", "Enterprise"]:
        # Premium users -> ECS job
        response = launch_job_on_ecs(
            user_id=user_id,
            file_path=file_path,
            train_path=train_path,
            test_path=test_path,
            target_column=target_column,
            drop_columns=drop_columns,
            job_type="counterfactual",
            extra_params={  # pass additional params if ECS handler supports
                "sample_id": sample_id,
                "sample_strategy": sample_strategy,
                "num_samples": num_samples
            }
        )
        return {"status": "queued_on_ecs", "ecs_response": response}

    else:
        # Free users -> Celery worker
        task = run_counterfactual.delay(
            user_id,
            file_path,
            train_path,
            test_path,
            target_column,
            drop_columns,
            sample_id,
            sample_strategy,
            num_samples
        )
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
    user_dir = PathL("user_uploads") / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None
    
    # ───────── Handle file uploads ───────────
    if file:
        file_path = os.path.join(user_dir, file.filename)
        async with aiofiles.open(file_path, "wb") as out:
            await out.write(await file.read())

    elif train_file and test_file:
        train_path = os.path.join(user_dir, train_file.filename)
        async with aiofiles.open(train_path, "wb") as out:
            await out.write(await train_file.read())

        test_path = os.path.join(user_dir, test_file.filename)
        async with aiofiles.open(test_path, "wb") as out:
            await out.write(await test_file.read())
    else:
        raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")
    if file_path:
        train_path, test_path = None, None
    elif train_path and test_path:
        file_path = None
    else:
        raise HTTPException(400, "Invalid upload state: no valid dataset provided.")
    # ───────── Choose execution mode ───────────
    if current_user.subscription in ["Pro", "Enterprise"]:
        # Premium users -> ECS job
        response = launch_job_on_ecs(
            user_id=user_id,
            file_path=file_path,
            train_path=train_path,
            test_path=test_path,
            time_col=time_col,
            event_col=event_col,
            drop_cols=drop_cols,
            job_type="survival"  # Pass job type to ECS handler
        )
        return {"status": "queued_on_ecs", "ecs_response": response}

    else:
        kwargs = {
            "user_id": user_id,
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

@app.post("/what_if_analysis/")
async def what_if_analysis(
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile = File(None),
    user_id: str = Form(...),
    sample_id: int = Form(...),
    target_column: str = Form(...),
    feature_changes: str = Form(...),
    current_user: User = Depends(get_current_active_user),
):
    # ──────────── Token check ─────────────
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, "Insufficient token balance to begin processing.")

    user_dir = PathL("user_uploads") / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ──────────── Handle file uploads ─────────────
    if file:
        file_path = os.path.join(user_dir, file.filename)
        async with aiofiles.open(file_path, "wb") as out:
            await out.write(await file.read())

    elif train_file and test_file:
        train_path = os.path.join(user_dir, train_file.filename)
        async with aiofiles.open(train_path, "wb") as out:
            await out.write(await train_file.read())

        test_path = os.path.join(user_dir, test_file.filename)
        async with aiofiles.open(test_path, "wb") as out:
            await out.write(await test_file.read())
    else:
        raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")

    # ──────────── Choose execution mode ─────────────
    if current_user.subscription in ["Pro", "Enterprise"]:
        # Premium users -> ECS job
        response = launch_job_on_ecs(
            user_id=user_id,
            file_path=file_path,
            train_path=train_path,
            test_path=test_path,
            sample_id=sample_id,
            target_column=target_column,
            feature_changes=feature_changes,
            job_type="what_if"
        )
        return {"status": "queued_on_ecs", "ecs_response": response}

    else:
        # Free users -> Celery worker
        task = run_what_if.delay(
            user_id,
            file_path,
            train_path,
            test_path,
            sample_id,
            target_column,
            feature_changes
        )
        response_data = await run_in_threadpool(task.get)
        return response_data

# Add this to your database functions
async def store_counterfactual_result(user_id, sample_id, original_features, modified_features, 
                                     original_prediction, modified_prediction):
    conn = get_db_connection()
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

    user_dir = PathL("user_uploads") / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ──────────── Handle file uploads ─────────────
    if file:
        file_path = os.path.join(user_dir, file.filename)
        async with aiofiles.open(file_path, "wb") as out:
            await out.write(await file.read())

    elif train_file and test_file:
        train_path = os.path.join(user_dir, train_file.filename)
        async with aiofiles.open(train_path, "wb") as out:
            await out.write(await train_file.read())

        test_path = os.path.join(user_dir, test_file.filename)
        async with aiofiles.open(test_path, "wb") as out:
            await out.write(await test_file.read())
    else:
        raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")

    # ──────────── Choose execution mode ─────────────
    if current_user.subscription in ["Pro", "Enterprise"]:
        response = launch_job_on_ecs(
            user_id=user_id,
            file_path=file_path,
            train_path=train_path,
            test_path=test_path,
            target_column=target_column,
            drop_columns=drop_columns,
            job_type="risk_analysis"
        )
        return {"status": "queued_on_ecs", "ecs_response": response}

    else:
        task = run_risk_analysis.delay(
            user_id,
            file_path,
            train_path,
            test_path,
            target_column,
            drop_columns
        )
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
    user_dir = PathL("user_uploads") / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    PathL("models").mkdir(exist_ok=True)

    file_path = None
    train_path = None
    test_path = None

    # ─────────── Handle file uploads ─────────────
    if file:
        file_path = os.path.join(user_dir, file.filename)
        async with aiofiles.open(file_path, "wb") as out:
            await out.write(await file.read())

    elif train_file and test_file:
        train_path = os.path.join(user_dir, train_file.filename)
        async with aiofiles.open(train_path, "wb") as out:
            await out.write(await train_file.read())

        test_path = os.path.join(user_dir, test_file.filename)
        async with aiofiles.open(test_path, "wb") as out:
            await out.write(await test_file.read())
    else:
        raise HTTPException(400, "Upload either `file` or both `train_file` + `test_file`.")

    # ─────────── ECS vs Celery execution ─────────────
    if current_user.subscription in ["Pro", "Enterprise"]:
        response = launch_job_on_ecs(
            user_id=user_id,
            file_path=file_path,
            train_path=train_path,
            test_path=test_path,
            target_column=target_column,
            drop_columns=drop_columns,
            job_type="decision_paths"
        )
        return {"status": "queued_on_ecs", "ecs_response": response}

    else:
        task = run_decision_paths.delay(
            user_id,
            file_path,
            train_path,
            test_path,
            target_column,
            drop_columns
        )
        response_data = await run_in_threadpool(task.get)
        return response_data

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
@app.post("/forecast/")
async def forecast_time_series(
    file: UploadFile = File(...),
    target_column: str = Form("value"),
    drop_columns: str = Form(""),
    periods: int = Form(24),
    current_user: User = Depends(get_current_active_user),
):
    try:
        # ─── Token check ───
        MINIMUM_TOKENS = 0.1
        if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
            raise HTTPException(403, "Insufficient token balance to begin processing.")

        user_id = current_user.id
        user_dir = f"user_uploads/{user_id}"
        os.makedirs(user_dir, exist_ok=True)

        # Save uploaded file
        await file.seek(0)
        content = await file.read()
        file_path = f"{user_dir}/{file.filename}"
        with open(file_path, "wb") as f_out:
            f_out.write(content)

        results = run_forecast(
            user_id=user_id,
            file_path=file_path,
            target_column=target_column,
            drop_columns=drop_columns,
            periods=periods
        )

        return results

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Time series forecast failed: {str(e)}")


app.include_router(auth_router)
if __name__ == "__main__":
    for route in app.routes:
        print(route.path, route.methods)
    uvicorn.run(app, host="127.0.0.1", port=8000)