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
from pathlib import Path as PathL  # Your existing alias
import json
import logging
from typing import List, Optional, Dict, Any
import tempfile
from fastapi import HTTPException, Depends, Query
import uuid
from io import BytesIO
from sqlalchemy import text
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
import os, io, csv, time, shutil, mimetypes
from typing import Optional, Tuple, Dict
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, Path, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse

from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
import shutil

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
# from .storage import download_file_from_supabase, upload_file_to_supabase
# from .biz_preprocess import PreprocessConfig, preprocess_full_dataset, build_and_save_train_test
# from .storage import _basename_from_key, _maybe_local_cache_path, supabase,  _strip_bucket_prefix, SUPABASE_BUCKET
# from .harvester import intake_and_normalize, _meta_to_json, _preview_table, save_intake_artifacts



from harvester import intake_and_normalize, _meta_to_json, _preview_table, save_intake_artifacts
from storage import _basename_from_key, supabase, _strip_bucket_prefix, SUPABASE_BUCKET, upload_file_to_supabase
from storage import download_file_from_supabase, delete_file_from_supabase
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
    Dataset,
    get_master_db_session
)
from biz_preprocess import PreprocessConfig, preprocess_full_dataset, build_and_save_train_test
from typing import List, Optional, Union
from pydantic import BaseModel, Field

class ColumnHints(BaseModel):
    id_cols: List[str] = []
    target_cls_col: Optional[str] = None
    positive_label: Optional[Union[str, int, bool]] = None
    target_reg_col: Optional[str] = None
    group_cols: List[str] = []
    date_cols: List[str] = []
    text_cols: List[str] = []
    ignore_cols: List[str] = []
    numeric_keep_continuous: List[str] = []

class PolicyKnobs(BaseModel):
    numeric_to_cat: bool = True
    numeric_bins: int = 10
    max_onehot_cardinality: int = 80
    add_date_parts: bool = True

class PreprocessRequest(BaseModel):
    # Matches your frontend call: { "clean": true, "save": true }
    clean: bool = True
    save: bool = True

    # Controls for the pipeline
    fit_encoders: bool = True                 # passed to preprocess_full_dataset(...)
    filename: Optional[str] = None            # override output name (e.g., "data.csv")
    sample_rows: Optional[int] = Field(5000, ge=100, le=500_000)
    return_preview: bool = False              # if you want to return a small preview

    # Optional overrides for autodetection + policy knobs
    overrides: Optional[ColumnHints] = None
    policy: Optional[PolicyKnobs] = None
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
router = APIRouter(prefix="/datasets", tags=["datasets"])

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
    import os
import io
import glob
from typing import Optional
from urllib.parse import urlparse

from fastapi import HTTPException

# Optional cloud deps are imported lazily inside helpers.

# If you already defined this earlier, reuse it:
def user_dataset_root(user_id: int, dataset_id: int) -> str:
    return os.path.abspath(os.path.join(".", "data", "users", str(user_id), "datasets", str(dataset_id)))


def _path_ok(p: Optional[str]) -> bool:
    return bool(p) and os.path.isfile(p) and os.path.getsize(p) > 0



def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _download_to(path: str, content: bytes):
    _ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(content)
    return path


def _download_http(uri: str, dest_path: str) -> str:
    try:
        import requests  # type: ignore
    except ImportError as e:
        raise HTTPException(500, detail="requests not installed for HTTP download") from e
    resp = requests.get(uri, stream=True, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(502, detail=f"HTTP fetch failed: {resp.status_code}")
    _ensure_dir(os.path.dirname(dest_path))
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return dest_path


def _download_s3(uri: str, dest_path: str) -> str:
    try:
        import boto3  # type: ignore
    except ImportError as e:
        raise HTTPException(500, detail="boto3 not installed for S3 access") from e
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = boto3.client("s3")
    _ensure_dir(os.path.dirname(dest_path))
    s3.download_file(bucket, key, dest_path)
    return dest_path


def _download_gcs(uri: str, dest_path: str) -> str:
    try:
        from google.cloud import storage  # type: ignore
    except ImportError as e:
        raise HTTPException(500, detail="google-cloud-storage not installed for GCS access") from e
    parsed = urlparse(uri)
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")
    client = storage.Client()  # auth via env (GOOGLE_APPLICATION_CREDENTIALS)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    _ensure_dir(os.path.dirname(dest_path))
    blob.download_to_filename(dest_path)
    return dest_path


def _download_supabase(uri: str, dest_path: str) -> str:
    """
    supabase://<bucket>/<path/to/file.csv>
    Requires env SUPABASE_URL, SUPABASE_ANON_KEY (or service key) and 'supabase' package.
    """
    try:
        from supabase import create_client  # type: ignore
    except ImportError as e:
        raise HTTPException(500, detail="supabase-py not installed for Supabase access") from e

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(500, detail="Supabase env vars not set (SUPABASE_URL, SUPABASE_ANON_KEY/SERVICE_ROLE_KEY)")

    parsed = urlparse(uri)
    bucket = parsed.netloc
    path = parsed.path.lstrip("/")
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Download as bytes
    res = client.storage.from_(bucket).download(path)
    if not isinstance(res, (bytes, bytearray)):
        raise HTTPException(502, detail="Supabase download returned no bytes")
    return _download_to(dest_path, bytes(res))



def _strip_bucket_prefix(key: str) -> str:
    """
    Accepts things like:
      - '1/orders.csv'
      - 'user-uploads/1/orders.csv'
      - '/user-uploads/1/orders.csv'
    Returns inside-bucket key: '1/orders.csv'
    """
    k = key.replace("\\", "/").lstrip("/")
    if k.startswith("user-uploads/"):
        k = k[len("user-uploads/"):]
    return k
def resolve_and_cache_dataset_csv(*, db: Session, dataset_id: int, user_id: int) -> str:
    """
    Find the dataset by id only, read its stored Supabase key, download bytes,
    and write to ./data/users/<user_id>/datasets/<dataset_id>/cache/original.csv
    """


    ds = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    raw_key = (
        getattr(ds, "file_path", None)
        or getattr(ds, "supabase_path", None)
        or getattr(ds, "storage_uri", None)
        or getattr(ds, "uri", None)
    )
    if not raw_key:
        raise HTTPException(status_code=404, detail="Dataset has no stored file path")

    key = _strip_bucket_prefix(raw_key)  # -> "1/<filename>.csv"

    # Download from Supabase
    blob = download_file_from_supabase(key)  # must return bytes
    if not isinstance(blob, (bytes, bytearray)):
        raise HTTPException(status_code=502, detail="Supabase download did not return bytes")

    # Cache locally
    root = user_dataset_root(user_id, dataset_id)
    cache_dir = os.path.join(root, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cached_csv = os.path.join(cache_dir, "original.csv")
    with open(cached_csv, "wb") as f:
        f.write(blob)
    return os.path.abspath(cached_csv)
def get_dataset_csv_path(db: Session, dataset_id: int, user_id: int) -> str:
    """
    Use ONLY the DB to find the original CSV:
      - reads Dataset.file_path (or supabase_path/uri/storage_uri if you use those)
      - downloads the file via Supabase helper
      - caches it at: ./data/users/<uid>/datasets/<did>/cache/original.csv
    """

    ds = (
        db.query(Dataset)
          .filter(Dataset.id == int(dataset_id), Dataset.user_id == int(user_id))
          .first()
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    raw_key = (
        getattr(ds, "file_path", None)
        or getattr(ds, "supabase_path", None)
        or getattr(ds, "storage_uri", None)
        or getattr(ds, "uri", None)
    )
    if not raw_key:
        raise HTTPException(status_code=404, detail="Dataset has no stored file path")

    key = _strip_bucket_prefix(str(raw_key))  # -> '1/<filename>.csv'

    # Download & cache locally
    root = user_dataset_root(user_id, dataset_id)
    cache_dir = os.path.join(root, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cached_csv = os.path.join(cache_dir, "original.csv")

    try:
        blob = download_file_from_supabase(key)  # bytes
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to download '{key}': {e}")

    with open(cached_csv, "wb") as f:
        f.write(blob)

    return os.path.abspath(cached_csv)
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
# ---- Supabase init ----
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_UPLOADS_BUCKET", "user-uploads")

SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY)
if SUPABASE_ENABLED:
    from supabase import create_client
    _sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Path/Key helpers
# -----------------------------
def _safe_filename(name: str) -> str:
    name = os.path.basename(name or "").replace("\x00", "")
    # keep alnum and a few safe symbols
    name = "".join(c for c in name if c.isalnum() or c in (" ", ".", "_", "-", "(", ")")).strip()
    return name or f"upload_{int(time.time())}.csv"

def _strip_bucket_prefix(key: str) -> str:
    """
    Turn 'user-uploads/123/My File.csv' or '/user-uploads/123/My File.csv'
    into '123/My File.csv' so UI can keep basename stable.
    """
    if not key:
        return key
    key = key.lstrip("/")
    if key.startswith(SUPABASE_BUCKET + "/"):
        return key[len(SUPABASE_BUCKET) + 1 :]
    return key

def _supakey_original(user_id: int, filename: str) -> str:
    # Flat per-user namespace as requested
    return f"{user_id}/{filename}"

def _supakey_encoder(user_id: int, dataset_id: int, name: str) -> str:
    return f"{user_id}/enc_{dataset_id}_{name}"

def _supakey_processed(user_id: int, dataset_id: int, name: str) -> str:
    return f"{user_id}/proc_{dataset_id}_{name}"

def _quick_csv_shape(fp: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        with open(fp, "r", encoding="utf-8", newline="") as f:
            sample = f.read(64_000)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
            except Exception:
                dialect = csv.excel
            reader = csv.reader(f, dialect)
            header = next(reader, [])
            cols = len(header)
            rows = sum(1 for _ in reader)
            return rows, cols
    except Exception:
        return None, None

def _write_upload_to_disk(upload: UploadFile, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp = dest_path + ".tmp"
    with open(tmp, "wb") as out:
        shutil.copyfileobj(upload.file, out, length=1024 * 1024)
    os.replace(tmp, dest_path)

def _content_type_for_ext(filename: str, fallback: str = "text/csv") -> str:
    ctype = mimetypes.guess_type(filename)[0]
    return ctype or fallback

# -----------------------------
# Supabase Storage helpers
# -----------------------------
def sb_upload_from_path(bucket: str, key: str, local_path: str, content_type: str = "application/octet-stream", upsert: bool = True):
    if not SUPABASE_ENABLED:
        raise RuntimeError("Supabase is not configured")
    with open(local_path, "rb") as fh:
        _sb.storage.from_(bucket).upload(
            path=key,
            file=fh,
            file_options={"content-type": content_type, "upsert": str(upsert).lower()},
        )

def sb_download_to_path(bucket: str, key: str, dest_path: str):
    if not SUPABASE_ENABLED:
        raise RuntimeError("Supabase is not configured")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    data = _sb.storage.from_(bucket).download(key)
    tmp = dest_path + ".tmp"
    with open(tmp, "wb") as out:
        out.write(data)
    os.replace(tmp, dest_path)

def sb_signed_url(bucket: str, key: str, seconds: int = 300) -> Optional[str]:
    if not SUPABASE_ENABLED:
        return None
    try:
        resp = _sb.storage.from_(bucket).create_signed_url(key, seconds)
        # v2 returns dict with 'signedURL' or 'signed_url'
        if isinstance(resp, dict):
            return resp.get("signedURL") or resp.get("signed_url")
        return None
    except Exception as e:
        logger.warning(f"Signed URL failed for {key}: {e}")
        return None

@router.get(
    "{dataset_id}/download",
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


import os
import io
import re
import math
import time
from typing import Dict, Any, List, Optional, Iterable, Tuple

import pandas as pd

# ---- Tunables ---------------------------------------------------------------

DEFAULT_DATE_CANDIDATES = [
    "date", "created", "created_at", "updated", "updated_at",
    "timestamp", "order_date", "signup_date"
]

EMAIL_RE = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$"
)

# Limit how many distinct values we keep per column for value_counts preview
TOP_VALUES_LIMIT = 10
UNIQUE_TRACK_CAP = 10_000      # cap memory for approximate "unique <= ?" tracking
RESERVOIR_CAP = 3_000          # per-column reservoir for approx quantiles (numeric)


# ---- Utilities --------------------------------------------------------------

def _try_read_csv_sample(
    path: str,
    nrows: int = 2000,
    encodings: Iterable[str] = ("utf-8", "utf-8-sig", "latin-1")
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read a small sample to infer delimiter/encoding quickly and robustly.
    Returns (df, meta) where meta includes chosen encoding & sep.
    """
    last_err = None
    for enc in encodings:
        try:
            # sep=None lets pandas sniff the delimiter using 'python' engine
            df = pd.read_csv(path, nrows=nrows, sep=None, engine="python", encoding=enc)
            return df, {"encoding": enc, "sep": None}
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("Failed to read CSV sample")

def _count_invalid_emails(series: pd.Series) -> int:
    s = series.astype("string", errors="ignore")
    mask = s.notna() & ~s.str.match(EMAIL_RE)
    return int(mask.sum())

def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)

def _series_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _reservoir_update(reservoir: List[float], values: Iterable[float], cap: int = RESERVOIR_CAP, state: Dict[str, int] = None):
    """
    Simple reservoir sampling for approx quantiles. 'state' keeps total seen count.
    """
    if state is None:
        state = {"seen": 0}
    seen = state["seen"]
    import random

    for v in values:
        if pd.isna(v):
            continue
        if len(reservoir) < cap:
            reservoir.append(float(v))
        else:
            j = random.randint(0, seen)
            if j < cap:
                reservoir[j] = float(v)
        seen += 1
    state["seen"] = seen

def _quantiles_from_reservoir(reservoir: List[float]) -> Dict[str, float]:
    if not reservoir:
        return {}
    s = sorted(reservoir)
    def q(p: float) -> float:
        if not s: return math.nan
        idx = int(p * (len(s) - 1))
        return float(s[idx])
    return {
        "p05": q(0.05),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
    }


# ---- Main: compute_dataset_health ------------------------------------------

def compute_dataset_health(
    csv_path: str,
    *,
    primary_key: Optional[str] = None,
    required_fields: Optional[List[str]] = None,
    date_fields: Optional[List[str]] = None,
    chunk_size: int = 100_000,
    preview_limit: int = 20,
    infer_dates: bool = True,
) -> Dict[str, Any]:
    """
    Stream a CSV in chunks to compute dataset health without loading it all in memory.

    Returns:
      {
        status, row_count, duplicate_keys, missing_required, null_rate,
        date_range: {min, max} | null,
        issues: [{severity, field, message, count}],
        preview: [ {col: val, ...}, ... ],
        columns: [
          {
            name, dtype, non_nulls, nulls, null_rate,
            unique_approx, top_values: [{value, count}, ...],
            numeric: {min, max, mean, p05, p25, p50, p75, p95}?,
            text: {min_len, max_len, avg_len}?
          },
          ...
        ],
        runtime_ms
      }
    """
    t0 = time.time()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    # ---- Phase 0: sample for encoding & delimiter and early schema cues ----
    sample_df, meta = _try_read_csv_sample(csv_path)
    encoding = meta.get("encoding", "utf-8")
    sep = meta.get("sep", None)   # None => pandas will sniff again

    # infer candidate date columns if requested
    required_fields = list(required_fields or [])
    candidate_dates = list(date_fields or [])
    if infer_dates and not candidate_dates:
        # use defaults that actually exist or sample columns with parsable date ratio > 0.7
        for c in DEFAULT_DATE_CANDIDATES:
            if c in sample_df.columns:
                candidate_dates.append(c)
        if not candidate_dates:
            for c in sample_df.columns:
                s = _to_datetime(sample_df[c])
                non_null = s.notna().sum()
                total = len(s)
                if total and (non_null / total) >= 0.7:
                    candidate_dates.append(c)

    # classify column "kinds" from sample
    numeric_cols: List[str] = []
    string_cols: List[str] = []
    for c in sample_df.columns:
        s_num = pd.to_numeric(sample_df[c], errors="coerce")
        numeric_ratio = (s_num.notna().sum() / max(1, len(s_num)))
        if numeric_ratio >= 0.8:
            numeric_cols.append(c)
        else:
            string_cols.append(c)

    # ---- Accumulators (global + per-column) --------------------------------
    total_rows = 0
    total_cells = 0
    total_nulls = 0

    # per-required
    missing_required_rows = 0
    per_field_missing: Dict[str, int] = {f: 0 for f in required_fields}

    # duplicates (primary key)
    duplicate_keys = 0
    seen_keys = set() if primary_key else None

    # date range (use first available candidate in data)
    chosen_date_col = None
    min_date = None
    max_date = None

    # preview rows (first N)
    preview_rows: List[Dict[str, Any]] = []

    # issues
    issues: List[Dict[str, Any]] = []

    # per-column stats
    per_col_nulls: Dict[str, int] = {}
    per_col_non_nulls: Dict[str, int] = {}
    per_col_unique_cap: Dict[str, set] = {c: set() for c in sample_df.columns}
    per_col_unique_gt_cap: Dict[str, bool] = {c: False for c in sample_df.columns}
    per_col_top_counts: Dict[str, Dict[Any, int]] = {c: {} for c in sample_df.columns}

    # numeric trackers
    per_num_min: Dict[str, float] = {c: math.inf for c in numeric_cols}
    per_num_max: Dict[str, float] = {c: -math.inf for c in numeric_cols}
    per_num_sum: Dict[str, float] = {c: 0.0 for c in numeric_cols}
    per_num_count: Dict[str, int] = {c: 0 for c in numeric_cols}
    per_num_reservoir: Dict[str, List[float]] = {c: [] for c in numeric_cols}
    per_num_state: Dict[str, Dict[str, int]] = {c: {"seen": 0} for c in numeric_cols}

    # text trackers
    per_txt_minlen: Dict[str, int] = {c: math.inf for c in string_cols}
    per_txt_maxlen: Dict[str, int] = {c: -math.inf for c in string_cols}
    per_txt_sumlen: Dict[str, int] = {c: 0 for c in string_cols}
    per_txt_count: Dict[str, int] = {c: 0 for c in string_cols}

    # ---- Phase 1: Full streaming pass --------------------------------------
    reader = pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        encoding=encoding,
        sep=sep,
        engine="python",           # robust with sep=None
        low_memory=True,
        on_bad_lines="skip",       # don't abort on bad records
    )

    for chunk_idx, chunk in enumerate(reader):
        cols = list(chunk.columns)

        # preview: take head of the first non-empty chunk
        if not preview_rows and len(chunk) > 0:
            preview_rows = chunk.head(preview_limit).to_dict(orient="records")

        # totals
        total_rows += len(chunk)
        total_cells += int(chunk.shape[0] * chunk.shape[1])
        nulls_here = int(chunk.isna().sum().sum())
        total_nulls += nulls_here

        # required fields accounting
        if required_fields:
            masks = []
            for f in required_fields:
                if f in cols:
                    miss_f = int(chunk[f].isna().sum())
                    per_field_missing[f] += miss_f
                    masks.append(chunk[f].isna())
                else:
                    # field missing entirely in this chunk
                    per_field_missing[f] += len(chunk)
                    masks.append(pd.Series([True] * len(chunk), index=chunk.index))

            any_missing = masks[0]
            for m in masks[1:]:
                any_missing = any_missing | m
            missing_required_rows += int(any_missing.sum())

        # duplicate primary key
        if primary_key and primary_key in cols and seen_keys is not None:
            keys = chunk[primary_key].astype("object").tolist()
            for k in keys:
                if k in seen_keys:
                    duplicate_keys += 1
                else:
                    seen_keys.add(k)

        # choose date column (first hit among candidates)
        if chosen_date_col is None and candidate_dates:
            for c in candidate_dates:
                if c in cols:
                    chosen_date_col = c
                    break

        # update date range
        if chosen_date_col and chosen_date_col in cols:
            sdt = _to_datetime(chunk[chosen_date_col])
            cmin = sdt.min()
            cmax = sdt.max()
            if pd.notna(cmin):
                min_date = cmin if (min_date is None or cmin < min_date) else min_date
            if pd.notna(cmax):
                max_date = cmax if (max_date is None or cmax > max_date) else max_date

            # future-date issue (simple check)
            future_count = int((sdt > pd.Timestamp.now()).sum())
            if future_count:
                issues.append({
                    "severity": "warn",
                    "field": chosen_date_col,
                    "message": "Dates in the future detected",
                    "count": future_count
                })
        
        # per-column nulls/non-nulls and uniques (capped)
        isna = chunk.isna()
        for c in cols:
            nn = int((~isna[c]).sum())
            n = int(isna[c].sum())
            per_col_non_nulls[c] = per_col_non_nulls.get(c, 0) + nn
            per_col_nulls[c] = per_col_nulls.get(c, 0) + n

            # track small-cardinality value counts (for "top values")
            if nn:
                vc_map = per_col_top_counts[c]
                # Use value_counts on non-nulls only (small chunk)
                vc = chunk.loc[~isna[c], c].value_counts(dropna=True)
                # Increment counts (limit map size to avoid explosion)
                for val, cnt in vc.items():
                    if len(vc_map) < UNIQUE_TRACK_CAP or val in vc_map:
                        vc_map[val] = vc_map.get(val, 0) + int(cnt)

                # approximate uniqueness by storing up to UNIQUE_TRACK_CAP exemplars
                s = per_col_unique_cap[c]
                if len(s) < UNIQUE_TRACK_CAP:
                    # add as many as possible (convert to python hashables)
                    for v in chunk.loc[~isna[c], c].head(UNIQUE_TRACK_CAP - len(s)).tolist():
                        try:
                            s.add(v)
                        except TypeError:
                            # unhashable (e.g., list), fallback to str
                            s.add(str(v))
                else:
                    per_col_unique_gt_cap[c] = True

        # numeric aggregations
        for c in numeric_cols:
            if c not in cols:
                continue
            sn = _series_numeric(chunk[c])
            non_null = sn.notna()
            vals = sn[non_null].values

            per_num_count[c] += int(non_null.sum())
            per_num_sum[c] += float(pd.Series(vals).sum()) if len(vals) else 0.0
            if len(vals):
                vmin = float(pd.Series(vals).min())
                vmax = float(pd.Series(vals).max())
                per_num_min[c] = min(per_num_min[c], vmin)
                per_num_max[c] = max(per_num_max[c], vmax)
                _reservoir_update(per_num_reservoir[c], vals, cap=RESERVOIR_CAP, state=per_num_state[c])

        # text stats
        for c in string_cols:
            if c not in cols:
                continue
            s = chunk[c].astype("string", errors="ignore")
            lens = s.dropna().str.len()
            if not lens.empty:
                per_txt_count[c] += int(lens.count())
                per_txt_sumlen[c] += int(lens.sum())
                per_txt_minlen[c] = min(per_txt_minlen[c], int(lens.min()))
                per_txt_maxlen[c] = max(per_txt_maxlen[c], int(lens.max()))

        # custom validations (examples)
        if "email" in cols:
            bad = _count_invalid_emails(chunk["email"])
            if bad:
                issues.append({
                    "severity": "error",
                    "field": "email",
                    "message": "Invalid email format",
                    "count": int(bad)
                })
        if "amount" in cols:
            a = _series_numeric(chunk["amount"])
            neg = int((a < 0).sum())
            if neg:
                issues.append({
                    "severity": "error",
                    "field": "amount",
                    "message": "Negative amounts found",
                    "count": int(neg)
                })

    # ---- Finalize -----------------------------------------------------------
    overall_null_rate = float(total_nulls / total_cells) if total_cells else 0.0

    # Consolidate issues for required fields
    for f, miss in per_field_missing.items():
        if miss:
            issues.append({
                "severity": "error",
                "field": f,
                "message": "Missing required values",
                "count": int(miss)
            })

    # Per-column report
    columns_report: List[Dict[str, Any]] = []
    all_columns = list(sample_df.columns)  # keep original order from sample
    for c in all_columns:
        non_nulls = per_col_non_nulls.get(c, 0)
        nulls = per_col_nulls.get(c, 0)
        col_null_rate = float(nulls / (nulls + non_nulls)) if (nulls + non_nulls) else 0.0

        top_map = per_col_top_counts.get(c, {})
        # take top N values
        top_items = sorted(top_map.items(), key=lambda kv: kv[1], reverse=True)[:TOP_VALUES_LIMIT]
        top_values = [{"value": ("" if (isinstance(k, float) and math.isnan(k)) else k), "count": int(v)} for k, v in top_items]

        col_entry: Dict[str, Any] = {
            "name": c,
            "dtype": "numeric" if c in numeric_cols else "text",
            "non_nulls": int(non_nulls),
            "nulls": int(nulls),
            "null_rate": col_null_rate,
            "unique_approx": len(per_col_unique_cap[c]) if not per_col_unique_gt_cap[c] else f">{UNIQUE_TRACK_CAP}",
            "top_values": top_values,
        }

        if c in numeric_cols:
            q = _quantiles_from_reservoir(per_num_reservoir[c])
            mean = (per_num_sum[c] / per_num_count[c]) if per_num_count[c] else None
            col_entry["numeric"] = {
                "min": None if per_num_min[c] is math.inf else per_num_min[c],
                "max": None if per_num_max[c] == -math.inf else per_num_max[c],
                "mean": mean,
                **q
            }
        else:
            # text stats
            avg_len = (per_txt_sumlen[c] / per_txt_count[c]) if per_txt_count[c] else None
            min_len = None if per_txt_minlen[c] is math.inf else per_txt_minlen[c]
            max_len = None if per_txt_maxlen[c] == -math.inf else per_txt_maxlen[c]
            col_entry["text"] = {
                "min_len": min_len,
                "max_len": max_len,
                "avg_len": avg_len
            }

        columns_report.append(col_entry)
    preview_rows = json.loads(sample_df.to_json(orient="records", date_format="iso"))
    payload = {
        "status": "ready",
        "row_count": int(total_rows),
        "duplicate_keys": int(duplicate_keys),
        "missing_required": int(missing_required_rows),
        "null_rate": overall_null_rate,
        "date_range": {
            "min": min_date.strftime("%Y-%m-%d") if min_date is not None else None,
            "max": max_date.strftime("%Y-%m-%d") if max_date is not None else None,
        } if chosen_date_col else None,
        "issues": _merge_similar_issues(issues),
        "preview": preview_rows[:preview_limit],
        "columns": columns_report,
        "runtime_ms": int((time.time() - t0) * 1000),
    }
    return payload
def _merge_similar_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge issues with same (severity, field, message) to keep the list tidy.
    """
    acc: Dict[Tuple[str, str, str], int] = {}
    for it in issues:
        key = (it.get("severity","info"), it.get("field",""), it.get("message",""))
        acc[key] = acc.get(key, 0) + int(it.get("count", 0) or 0)
    out = []
    for (sev, field, msg), cnt in acc.items():
        out.append({"severity": sev, "field": field, "message": msg, "count": int(cnt)})
    # stable order: errors first, then warns, then info
    order = {"error": 0, "warn": 1, "info": 2}
    out.sort(key=lambda x: (order.get(x["severity"], 99), -x["count"], x["field"]))
    return out
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
def json_sanitize(obj):
    """
    Recursively convert NaN/Inf -> None, numpy scalars -> Python types,
    pandas Timestamps -> ISO strings, NaT -> None, ndarrays -> lists.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if obj is pd.NaT:
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [json_sanitize(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(v) for v in obj]
    return obj

# --- import your preprocessing APIs (already in your codebase) ---
from biz_preprocess import PreprocessConfig, preprocess_full_dataset, build_and_save_train_test

# -----------------------------
# Request/response models


# -----------------------------
# JSON sanitization (avoid: "Out of range float values are not JSON compliant")
# -----------------------------
def _is_non_finite(x: Any) -> bool:
    try:
        return isinstance(x, float) and (math.isnan(x) or math.isinf(x))
    except Exception:
        return False

def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NaN/Inf to None and numpy scalars to Python types."""
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return None if _is_non_finite(obj) else obj
    if isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        return None if _is_non_finite(val) else val
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return sanitize_for_json(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return sanitize_for_json(obj.to_dict())
    return obj  # best effort

import re
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Smart coercers & small utils
# -----------------------------
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
)
_HEXLIKE_RE = re.compile(r"^[0-9a-fA-F]{16,}$")
_ONLY_WORDCHARS_RE = re.compile(r"^\w+$")


def _is_integer_like(x: pd.Series) -> float:
    """Fraction of non-null values that are (close to) integers."""
    y = x.dropna().astype(float)
    if y.empty:
        return 0.0
    return float((np.isclose(y, np.round(y))).mean())

def _fractional_part_ratio(x: pd.Series) -> float:
    y = x.dropna().astype(float)
    if y.empty:
        return 0.0
    frac = np.abs(y - np.round(y))
    return float((frac > 1e-9).mean())

def _unique_ratio(x: pd.Series) -> float:
    x_n = x.dropna()
    if x_n.empty:
        return 0.0
    return float(x_n.nunique() / len(x_n))

def _maybe_id_numeric(x: pd.Series) -> bool:
    """Numeric id: high uniqueness, integer-like, wide range or long digit-length."""
    if x.dropna().empty:
        return False
    ur = _unique_ratio(x)
    if ur < 0.9:
        return False
    int_ratio = _is_integer_like(x)
    if int_ratio < 0.95:
        return False
    # wide digit-length or large range indicate identifiers
    y = x.dropna().astype(float)
    rng = float(y.max() - y.min())
    max_abs = float(np.max(np.abs(y)))
    # many IDs are large ints; also monotonic sequences are common
    monotonic = y.is_monotonic_increasing or y.is_monotonic_decreasing
    long_digits = np.log10(max_abs + 1e-9) > 6  # ~7+ digits
    return bool(monotonic or long_digits or rng > 1e6)


def _minority_label(series: pd.Series) -> Optional[Any]:
    vc = series.dropna().value_counts()
    if vc.empty:
        return None
    # prefer binary [0,1]/[False,True]/[no,yes]/[negative,positive] style
    # else choose minority class
    if len(vc) == 2:
        return vc.idxmin()
    return vc.idxmin()

# -----------------------------
# Numeric profiling & typing
# -----------------------------
class NumProfile(Tuple):
    pass

def _profile_numeric(x: pd.Series) -> Dict[str, float]:
    y = x.dropna().astype(float)
    if y.empty:
        return {
            "unique_ratio": 0.0,
            "is_integer_like": 0.0,
            "frac_nonzero_ratio": 0.0,
            "n_unique": 0.0,
            "n": 0.0,
            "range": 0.0,
        }
    return {
        "unique_ratio": _unique_ratio(y),
        "is_integer_like": _is_integer_like(y),
        "frac_nonzero_ratio": _fractional_part_ratio(y),
        "n_unique": float(y.nunique()),
        "n": float(len(y)),
        "range": float(y.max() - y.min()),
    }

def _classify_numeric(x: pd.Series) -> str:
    """
    Returns: 'continuous' | 'discrete' | 'id_like'
    Logic:
      - id_like: very high uniqueness & integer-like (or long digits/monotonic)
      - discrete: small unique set (<=20 or <=5% of rows) AND mostly integer-like
      - continuous: otherwise (esp. many unique or many decimals)
    """
    if x.dropna().empty:
        return "discrete"  # harmless default
    if _maybe_id_numeric(x):
        return "id_like"
    prof = _profile_numeric(x)
    n = prof["n"]
    n_unique = prof["n_unique"]
    ur = prof["unique_ratio"]
    is_int = prof["is_integer_like"] >= 0.95
    has_decimals = prof["frac_nonzero_ratio"] > 0.2

    small_card = (n_unique <= 20) or (n_unique <= max(5.0, 0.05 * n))
    if is_int and small_card:
        return "discrete"
    if ur > 0.5 or has_decimals or prof["range"] > 100 and n_unique > 30:
        return "continuous"
    # tie-breaker
    return "continuous" if not is_int else "discrete"

# -----------------------------
# Master detection
# -----------------------------
def _looks_like_date(series: pd.Series, sample_n: int = 200) -> bool:
    s = series.dropna().astype(str).head(sample_n)
    if s.empty: return False
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    return parsed.notna().mean() >= 0.5

def detect_columns(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    low = {c: str(c).lower() for c in cols}
    def has_any(c, keys): 
        name = low[c]
        return any(k in name for k in keys)

    id_cols = [c for c in cols if has_any(c, ["id", "uuid", "guid"])]
    id_cols = id_cols[:1] or (["ID"] if "ID" in cols else [])

    # target_cls: small-unique or obvious names
    cand_cls = [c for c in cols if has_any(c, ["target","label","class","churn","default","is_","flag"])]
    target_cls_col = None
    for c in cand_cls + cols:
        s = df[c]
        uniq = s.nunique(dropna=True)
        if (uniq <= 10) or (s.dropna().dtype == bool):
            target_cls_col = c; break

    # target_reg: numeric + name hint
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cand_reg = [c for c in num_cols if has_any(c, ["target","amount","price","revenue","score","value"])]
    target_reg_col = cand_reg[0] if cand_reg else None

    group_cols = [c for c in cols if has_any(c, ["store","shop","segment","category","group","channel"])][:3]
    date_cols = [c for c in cols if (_looks_like_date(df[c]) or has_any(c, ["date","time","created","updated","timestamp","ordered","processed"]))]
    obj_cols = [c for c in cols if df[c].dtype == "O"]
    # text-ish if high uniqueness ratio
    text_cols = [c for c in obj_cols if (df[c].nunique(dropna=True) / max(1, len(df))) > 0.2]
    ignore_cols = []

    # numeric_keep_continuous heuristic
    keep = []
    for c in num_cols:
        uniq = df[c].nunique(dropna=True)
        uniq_ratio = uniq / max(1, len(df))
        if (uniq >= 30) or (uniq_ratio >= 0.05) or any(k in low[c] for k in ["age","tenure","amount","price","cost","revenue","qty","quantity","score","margin","days","duration","time","rate"]):
            keep.append(c)

    return {
        "id_cols": id_cols,
        "target_cls_col": target_cls_col,
        "target_reg_col": target_reg_col,
        "group_cols": group_cols,
        "date_cols": date_cols,
        "text_cols": text_cols,
        "ignore_cols": ignore_cols,
        "numeric_keep_continuous": keep,
    }

# -----------------------------
# Positive label inference
# -----------------------------
def guess_positive_label(df: pd.DataFrame, target_cls_col: Optional[str]) -> Optional[Any]:
    if not target_cls_col:
        return None
    s = df[target_cls_col]
    # Normalize common truthy strings
    s_norm = (
        s.astype(str)
         .str.strip().str.lower()
         .replace({"true": "1", "false": "0", "yes": "1", "no": "0"})
    )
    # If it's binary 0/1-ish, choose '1' as positive if present
    vals = set(s_norm.unique())
    if {"0", "1"} <= vals or "1" in vals:
        return "1"
    # Otherwise pick the minority class
    return _minority_label(s)
# ---------- Helpers for null handling & safety ----------

_STANDARD_NULL_MARKERS = ["", "na", "n/a", "none", "null", "nan", "nil", "-", "--"]
def _std_null_marker_list():
    out = set()
    for s in _STANDARD_NULL_MARKERS:
        out.update([s, s.upper(), s.title()])
    return list(out)

def standardize_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols):
        df[obj_cols] = df[obj_cols].apply(lambda s: s.astype(str).str.strip())
        df[obj_cols] = df[obj_cols].replace(_std_null_marker_list(), np.nan)
    return df

def compute_null_rate(df: pd.DataFrame) -> float:
    if df.empty: return 0.0
    total = df.shape[0] * df.shape[1]
    return 0.0 if total == 0 else float(df.isna().sum().sum()) / float(total)

def _safe_list(x, default=None):
    if x is None: return (default or [])
    return list(x) if isinstance(x, (list, tuple)) else [x]
# Working directory helpers
# -----------------------------
def user_dataset_root(user_id: int, dataset_id: int) -> str:
    return os.path.abspath(os.path.join(".", "data", "users", str(user_id), "datasets", str(dataset_id)))
def ensure_dirs(root: str) -> Dict[str, str]:
    """
    Local staging dirs only. Supabase storage is flat per-user, so we do NOT
    mirror remote structure locally.
    """
    data_dir = os.path.join(root, "raw")
    # temp-ish locals for compatibility if anything writes here:
    enc_dir  = os.path.join(root, ".tmp", "encoders")
    proc_dir = os.path.join(root, ".tmp", "processed")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(enc_dir,  exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    return {
        "data_dir": data_dir,
        "encoder_info_dir": enc_dir,
        "data_process_dir": proc_dir,
    }
def make_processed_filename(dataset_id: int, base: str = "data.csv") -> str:
    return f"{dataset_id}_{base}"

def make_encoder_filename(dataset_id: int, base: str = "one_hot_encoder.pkl") -> str:
    return f"{dataset_id}_{base}"
# GET all stages for this dataset
@router.get("/{dataset_id}/stages")
def get_stages(dataset_id: int, current_user=Depends(get_current_active_user)):
    with get_user_db(current_user) as db:
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {
            "dataset_id": ds.id,
            "stage": getattr(ds, "stage", None),
            "stages": getattr(ds, "stages_json", []) or []
        }

# PATCH stages (replace or merge)
from pydantic import BaseModel
class StagesPatch(BaseModel):
    stage: Optional[str] = None
    stages: Optional[List[str]] = None      # full replace
    add: Optional[List[str]] = None         # append/merge

@router.patch("/{dataset_id}/stages")
def patch_stages(
    dataset_id: int,
    payload: StagesPatch,
    current_user=Depends(get_current_active_user)
):
    with get_user_db(current_user) as db:
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")

        stages = (getattr(ds, "stages_json", []) or [])
        if payload.stages is not None:
            stages = list(dict.fromkeys(payload.stages))  # replace, de-dup
        if payload.add:
            stages = list(dict.fromkeys([*stages, *payload.add]))
        if payload.stage is not None:
            ds.stage = payload.stage
        ds.stages_json = stages

        db.add(ds); db.commit(); db.refresh(ds)
        return {"dataset_id": ds.id, "stage": ds.stage, "stages": ds.stages_json}

@router.get("/{dataset_id}/health", tags=["datasets"]) 
def get_dataset_health(dataset_id: int, current_user=Depends(get_current_active_user)):
    # fetch dataset row
    with get_user_db(current_user) as db:
        ds = db.query(Dataset).get(dataset_id)
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")
        key = (ds.file_path or "").replace("\\", "/").lstrip("/")  # normalize key

    # Case 2: file_path is a Supabase key -> download, write temp, inspect
    try:
        blob: bytes = download_file_from_supabase(key)  # raises on error
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Unable to download file: {e}")

    # preserve extension so pandas chooses the right parser if you extend later
    suffix = PathL(key).suffix or ".csv"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(blob)
            tmp_path = tmp.name
        payload = compute_dataset_health(
            csv_path=tmp_path,
            primary_key=None,
            required_fields=None,
            date_fields=None,
            chunk_size=100_000,
            preview_limit=20,
            infer_dates=True,
        )
        safe = json_sanitize(payload)
        return JSONResponse(content=safe)  # standards-compliant JSON
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
import re
from pathlib import Path as PathL

def make_processed_name(original_name: str, dataset_id: int) -> str:
    p = PathL(original_name)
    stem = p.stem
    # strip any trailing __processed__<digits>
    stem = re.sub(r"__processed__\d+$", "", stem)
    return f"{stem}__processed__{dataset_id}{p.suffix or '.csv'}"
def _basename_from_key(key: str) -> str:
    """Extract filename from a storage key."""
    if not key:
        return ""
    return os.path.basename(key.rstrip("/"))

def _key_only(object_key: str, bucket: str) -> str:
    """
    Return the key relative to the bucket, e.g. '1/file.csv'
    from any of:
      'user-uploads/1/file.csv', '/user-uploads/1/file.csv', '1/file.csv', '/1/file.csv'
    """
    if not object_key:
        return ""
    k = str(object_key).lstrip("/")
    if k.startswith(bucket + "/"):
        k = k[len(bucket) + 1:]
    return k.lstrip("/")

def _bytes_from_supabase_download(resp) -> bytes:
    """
    Normalize download response into raw bytes for different client shapes.
    """
    # raw bytes
    if isinstance(resp, (bytes, bytearray)):
        return bytes(resp)
    # httpx.Response-like
    if hasattr(resp, "content"):
        return resp.content
    # dict shape
    if isinstance(resp, dict):
        if resp.get("error"):
            raise RuntimeError(f"Supabase download error: {resp['error']}")
        if "data" in resp and resp["data"] is not None:
            return resp["data"]
    raise RuntimeError(f"Unexpected Supabase download type: {type(resp)}")

def _build_supabase_key(user_id: int, filename: str) -> str:
    """
    Build the correct Supabase storage key in format: {user_id}/{filename}
    """
    if not filename:
        raise ValueError("Filename cannot be empty")
    
    # Clean the filename of any leading path separators
    clean_filename = filename.lstrip("/")
    
    # If filename already has user_id prefix, extract just the filename
    if "/" in clean_filename:
        parts = clean_filename.split("/")
        if len(parts) >= 2 and parts[0].isdigit():
            # Format is "user_id/filename" - extract just the filename
            clean_filename = "/".join(parts[1:])
    
    return f"{user_id}/{clean_filename}"

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


import re
from pathlib import Path as PathL

def make_processed_name(source_name: str, dataset_id: int) -> str:
    """
    Given a source filename, return an idempotent processed filename:
    Foo.csv -> Foo__processed__<id>.csv
    Foo__processed__<id>.csv -> Foo__processed__<id>.csv  (unchanged)
    Foo__processed__123.csv  -> Foo__processed__<id>.csv  (normalized to this run's id)
    """
    p = PathL(source_name)
    # strip any existing __processed__<digits> suffix on the *stem*
    clean_stem = re.sub(r"__processed__\d+$", "", p.stem)
    return f"{clean_stem}__processed__{dataset_id}{p.suffix or '.csv'}"

@router.post("/{dataset_id}/preprocess")
def preprocess_dataset_endpoint(
    dataset_id: int,
    body: PreprocessRequest,
    current_user = Depends(get_current_active_user),
):
    user_id = getattr(current_user, "id", None) or getattr(current_user, "user_id", None)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    # 🔑 Resolve original → local cache
    # 🔑 Resolve ORIGINAL → local cache (never chain-process the processed file)
    with get_user_db(current_user) as db:
        ds = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # prefer the true original if we have it; otherwise use current pointer
        src_key = getattr(ds, "original_file_path", None) or getattr(ds, "file_path", None)
        if not src_key:
            raise HTTPException(status_code=400, detail="Dataset has no file path")

        try:
            # download/copy the exact object we want to preprocess
            src_csv_path = _download_object_to_temp(src_key, user_id=user_id, dataset_id=dataset_id)
        except Exception:
            # final fallback to your existing resolver (if it uses ds.file_path internally)
            src_csv_path = resolve_and_cache_dataset_csv(db=db, dataset_id=dataset_id, user_id=user_id)

        old_key = getattr(ds, "file_path", None)
        original_name = os.path.basename(_strip_bucket_prefix(src_key)) if src_key else "data.csv"

    # Workspace
    root = user_dataset_root(user_id, dataset_id)
    dirs = ensure_dirs(root)
    data_dir = dirs["data_dir"]; enc_dir = dirs["encoder_info_dir"]; proc_dir = dirs["data_process_dir"]

    # Materialize train/test placeholders
    train_csv_path = os.path.join(data_dir, "train.csv")
    shutil.copy2(src_csv_path, train_csv_path)

    test_csv_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_csv_path):
        PathL(test_csv_path).write_text("", encoding="utf-8")

    # Read sample
    try:
        sample_df = pd.read_csv(
            train_csv_path,
            nrows=5000,
            na_values=_std_null_marker_list(),
            keep_default_na=True,
            low_memory=False,
        )
        sample_df = standardize_missing_markers(sample_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    null_rate_before_sample = compute_null_rate(sample_df)

    # Hints & config
    try:
        hints = detect_columns(sample_df)
    except Exception:
        hints = {}

    positive_label = guess_positive_label(sample_df, (hints or {}).get("target_cls_col"))

    cfg = PreprocessConfig(
        data_dir=data_dir + os.sep,
        encoder_info_dir=enc_dir + os.sep,
        data_process_dir=proc_dir + os.sep,
        id_cols=(hints.get("id_cols") or ["ID"]),
        target_cls_col=hints.get("target_cls_col"),
        positive_label=positive_label,
        target_reg_col=hints.get("target_reg_col"),
        group_cols=_safe_list(hints.get("group_cols", []))[:3],
        date_cols=_safe_list(hints.get("date_cols", [])),
        text_cols=_safe_list(hints.get("text_cols", [])),
        ignore_cols=_safe_list(hints.get("ignore_cols", [])),
        numeric_to_cat=True,
        numeric_bins=10,
        numeric_keep_continuous=_safe_list(hints.get("numeric_keep_continuous", [])),
        max_onehot_cardinality=80,
        add_date_parts=True,
        verbose=True,
    )

    try:
        build_and_save_train_test(cfg, train_csv="train.csv", test_csv="test.csv")
    except Exception:
        pass

    # CHANGE: write processed to a DIFFERENT object key (don't overwrite original)
    p = PathL(original_name)
    # Idempotent processed name (no double __processed__X)
    processed_fname = make_processed_name(original_name, dataset_id)


    # Full preprocessing → upload processed file under processed_fname
    try:
        artifacts = preprocess_full_dataset(
            cfg,
            csv_path=train_csv_path,
            user_id=str(user_id),      # stored at user-uploads/{user_id}/...
            filename=processed_fname,  # CHANGE: separate processed name
            fit_encoders=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocess failed: {e}")

    new_key = artifacts.get("data_csv")  # processed object key
    if not new_key:
        raise HTTPException(status_code=500, detail="Pipeline did not return processed data key")

    # Persist: store BOTH paths
    with get_user_db(current_user) as db:
        ds = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found for update")

        prior_key = getattr(ds, "file_path", None)

        # CHANGE: set original_file_path once (if not already set)
        if not getattr(ds, "original_file_path", None):
            setattr(ds, "original_file_path", prior_key)

        # CHANGE: if you have a dedicated processed_file_path column, populate it
        if hasattr(ds, "processed_file_path"):
            ds.processed_file_path = new_key

        # Optionally keep file_path as "current" pointer → processed
        ds.file_path  = new_key
        ds.stage      = "processed"
        ds.processed_at = datetime.utcnow()

        if artifacts.get("encoder_pkl"):
            setattr(ds, "encoder_path", artifacts["encoder_pkl"])

        db.add(ds)
        db.commit()
        db.refresh(ds)

    # CHANGE: do NOT delete the original — we are preserving it
    # (remove the deletion block entirely)

    # Metrics
    null_rate_after_full = artifacts.get("null_rate_after_full")
    metrics = {
        "null_rate_before_sample": round(null_rate_before_sample, 6),
        "null_rate_after_full": (round(float(null_rate_after_full), 6)
                                 if null_rate_after_full is not None else None),
        "rows_processed": artifacts.get("rows_processed"),
        "cols_processed": artifacts.get("cols_processed"),
    }

    # CHANGE: return both keys (and optional signed URLs if you have a helper)
    def _signed_or_none(key: str):
        try:
            bucket = os.environ.get("SUPABASE_BUCKET", "user-uploads")
            return sb_signed_url(bucket, _strip_bucket_prefix(key), seconds=300)  # if you have this helper
        except Exception:
            return None

    original_key = getattr(ds, "original_file_path", None) or old_key
    processed_key = new_key

    payload = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "config_used": {
            "id_cols": cfg.id_cols,
            "target_cls_col": cfg.target_cls_col,
            "positive_label": cfg.positive_label,
            "target_reg_col": cfg.target_reg_col,
            "group_cols": cfg.group_cols,
            "date_cols": cfg.date_cols,
            "text_cols": cfg.text_cols,
            "ignore_cols": cfg.ignore_cols,
            "numeric_keep_continuous": cfg.numeric_keep_continuous,
            "numeric_bins": cfg.numeric_bins,
            "numeric_to_cat": cfg.numeric_to_cat,
            "max_onehot_cardinality": cfg.max_onehot_cardinality,
            "add_date_parts": cfg.add_date_parts,
        },
        "dirs": {
            "data_dir": cfg.data_dir,
            "encoder_info_dir": cfg.encoder_info_dir,
            "data_process_dir": cfg.data_process_dir,
        },
        "artifacts": artifacts,
        "metrics": metrics,
        "message": "Preprocess & Clean completed",
        # CHANGE: clearly report both files
        "files": {
            "original": {
                "key": original_key,
                "signedUrl": _signed_or_none(original_key) if original_key else None,
                "filename": original_name,
            },
            "processed": {
                "key": processed_key,
                "signedUrl": _signed_or_none(processed_key),
                "filename": processed_fname,
            },
        },
        # legacy fields for backward-compat:
        "replaced_original": False,           # CHANGE: we no longer replace
        "old_file_path": original_key,        # original
        "new_file_path": processed_key,       # processed
    }

    return JSONResponse(content=sanitize_for_json(payload), status_code=200)
@router.get("/{dataset_id}/original")
async def get_original_file(
    dataset_id: int = Path(..., gt=0),
    download: int = 0,
    current_user=Depends(get_current_active_user),
):
    """
    Default: returns {"signedUrl": "..."} so the frontend can download directly.
    If `?download=1`: ensure local cached copy exists, then stream it as FileResponse.
    """
    user_id = getattr(current_user, "id", None) or getattr(current_user, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    with get_user_db(current_user) as db:
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # ds.file_path holds object key 'userId/filename.csv'
        object_key = ds.file_path
        if not object_key:
            raise HTTPException(status_code=404, detail="No file associated with this dataset")

        filename = getattr(ds, "original_filename", os.path.basename(object_key))

        if not download:
            # Return a short-lived signed URL (front-end will do fetch → blob)
            signed = sb_signed_url(SUPABASE_BUCKET, object_key, seconds=300)
            if not signed:
                # Fall back: make sure we can at least stream from local cache
                download = 1
            else:
                return {"signedUrl": signed, "filename": filename}

        # Ensure local cached copy exists
        root = user_dataset_root(user_id, dataset_id)
        dirs = ensure_dirs(root)
        cache_path = os.path.join(dirs["data_dir"], filename)
        if not os.path.exists(cache_path):
            try:
                sb_download_to_path(SUPABASE_BUCKET, object_key, cache_path)
            except Exception as e:
                logger.error(f"Failed to download original to cache: {e}")
                raise HTTPException(status_code=500, detail="Could not fetch file from remote storage")

        return FileResponse(path=cache_path, filename=filename, media_type="text/csv")
@router.put("/{dataset_id}/file")
async def replace_dataset_file(
    dataset_id: int = Path(..., gt=0),
    file: UploadFile = File(...),
    current_user=Depends(get_current_active_user),
):
    """
    Replace the original file for a dataset.
    - Saves a local cached copy to your staging root (./data/users/{uid}/datasets/{id}/raw/<filename>)
    - Uploads canonical original to Supabase: user-uploads/{user_id}/{filename}
    - Updates Dataset.file_path to the *Supabase object key* (without bucket)
    - Recomputes quick row/col counts
    """
    user_id = getattr(current_user, "id", None) or getattr(current_user, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    # 1) Load Dataset
    with get_user_db(current_user) as db:
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # 2) Sanitize and persist locally
        filename = _safe_filename(file.filename or f"{ds.name}.csv")
        root = user_dataset_root(user_id, dataset_id)
        dirs = ensure_dirs(root)
        local_path = os.path.join(dirs["data_dir"], filename)
        _write_upload_to_disk(file, local_path)

        # 3) Upload to Supabase (canonical)
        supa_key = _supakey_original(user_id, filename)
        try:
            if not SUPABASE_ENABLED:
                raise RuntimeError("Supabase not configured")
            sb_upload_from_path(
                SUPABASE_BUCKET,
                supa_key,
                local_path,
                content_type=_content_type_for_ext(filename, "text/csv"),
                upsert=True,
            )
        except Exception as e:
            logger.error(f"Supabase upload failed: {e}")
            # You still have local cache; but original should be remote canonical.
            # Decide policy: fail hard so UI knows the replace didn't complete.
            raise HTTPException(status_code=500, detail="Failed to store file in remote storage")

        # 4) Update dataset metadata
        rows, cols = _quick_csv_shape(local_path)
        ds.file_path = f"{supa_key}"  # store object key (no bucket prefix)
        if rows is not None:
            ds.row_count = rows
        if cols is not None:
            ds.column_count = cols
        if hasattr(ds, "original_filename"):
            ds.original_filename = filename
        if hasattr(ds, "updated_at"):
            ds.updated_at = datetime.now(timezone.utc)
        # reset stages if you track them
        if hasattr(ds, "stage"):
            ds.stage = "uploaded"
        if hasattr(ds, "stages_json"):
            ds.stages_json = ["uploaded", "parsed"]

        db.add(ds)
        db.commit()
        db.refresh(ds)

    # 5) Hand a short-lived URL back for convenience (UI can cache for Download button)
    signed = sb_signed_url(SUPABASE_BUCKET, supa_key, seconds=300)

    return {
        "dataset": {
            "id": ds.id,
            "name": ds.name,
            "filename": filename,
            "row_count": ds.row_count,
            "column_count": ds.column_count,
            "file_key": ds.file_path,      # '123/filename.csv'
            "bucket": SUPABASE_BUCKET,
            "signedUrl": signed,
        }
    }

@router.post("/{dataset_id}/intake")
def run_intake_and_normalization(
    dataset_id: int,
    base_currency: str = Query("USD"),
    country_hint: str = Query("US"),
    preview_rows: int = Query(50, ge=1, le=500),
    current_user = Depends(get_current_active_user),
):
    """
    Reads the dataset CSV for `dataset_id`, runs Intake & Normalization,
    stores artifacts, and returns a JSON-friendly summary + table previews.
    """
    user_id = getattr(current_user, "id", getattr(current_user, "user_id", None))
    if user_id is None:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    # 1) Resolve the *local* CSV path for this dataset
    # Prefer your existing helper if available:
    src_csv_path = None
    try:
        with get_user_db(current_user) as db:
            src_csv_path = resolve_and_cache_dataset_csv(db=db, dataset_id=dataset_id, user_id=user_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to resolve dataset CSV: {e}")

    # Fallback example (if you don't have resolve_and_cache_dataset_csv here):
    # with get_user_db(current_user) as db:
    #     ds = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
    #     if not ds:
    #         raise HTTPException(status_code=404, detail="Dataset not found")
    #     # Download from Supabase to a temp file
    #     src_csv_path = _download_dataset_csv_to_temp(ds.file_path, user_id=user_id, dataset_id=dataset_id)

    if not src_csv_path or not os.path.exists(src_csv_path):
        raise HTTPException(status_code=404, detail="Local CSV for dataset not found")

    # 2) Run Intake & Normalization (from prior code)
    try:
        result = intake_and_normalize(
            src_csv_path,
            base_currency=base_currency,
            country_hint=country_hint,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intake & normalization failed: {e}")

    # 3) Persist artifacts (normalized CSV + meta JSON) to Supabase
    artifacts: Dict[str, Any] = {}
    temp_dir = tempfile.mkdtemp(prefix=f"d{dataset_id}_intake_")
    try:
        # Save normalized CSV (full)
        norm_csv = os.path.join(temp_dir, f"dataset_{dataset_id}_normalized.csv")
        result.df_normalized.to_csv(norm_csv, index=False)

        # Save meta JSON (JSON-friendly)
        meta_json_path = os.path.join(temp_dir, f"dataset_{dataset_id}_intake_meta.json")
        meta_dict = _meta_to_json(result.meta)
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, ensure_ascii=False, indent=2)

        # Upload artifacts to Supabase
        supa_norm = upload_file_to_supabase(
            user_id=str(user_id),
            file_path=norm_csv,
            filename=f"{dataset_id}/intake/normalized.csv",
        )
        supa_meta = upload_file_to_supabase(
            user_id=str(user_id),
            file_path=meta_json_path,
            filename=f"{dataset_id}/intake/intake_meta.json",
        )
        artifacts["normalized_csv"] = supa_norm
        artifacts["intake_meta_json"] = supa_meta

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist artifacts: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 4) Build lightweight previews (do NOT return full DataFrames)
    preview_raw = _preview_table(result.df_raw, n=preview_rows)
    preview_norm = _preview_table(result.df_normalized, n=preview_rows)
    # 4b) Compute stats
    stats_dict = {
        "raw_rows": int(result.df_raw.shape[0]),
        "raw_cols": int(result.df_raw.shape[1]),
        "normalized_rows": int(result.df_normalized.shape[0]),
        "normalized_cols": int(result.df_normalized.shape[1]),
    }

    # After you compute: meta_dict, preview_raw, preview_norm, artifacts, stats_dict
    save_intake_artifacts(
        dataset_id=dataset_id,
        meta=meta_dict,
        stats=stats_dict,  # {"raw_rows":..., "raw_cols":..., "normalized_rows":..., "normalized_cols":...}
        preview={"raw": preview_raw, "normalized": preview_norm},
        artifacts=artifacts,  # optional
    )

    # 5) Response
    return {
        "ok": True,
        "dataset_id": dataset_id,
        "artifacts": artifacts,
        "meta": meta_dict,
        "preview": {
            "raw": preview_raw,
            "normalized": preview_norm,
        },
        "stats": {
            "raw_rows": int(result.df_raw.shape[0]),
            "raw_cols": int(result.df_raw.shape[1]),
            "normalized_rows": int(result.df_normalized.shape[0]),
            "normalized_cols": int(result.df_normalized.shape[1]),
        },
    }

if __name__ == "__main__":
    init_db()