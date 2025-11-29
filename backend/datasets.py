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



from harvester import intake_and_normalize, _meta_to_json, _preview_table
from storage import _basename_from_key, supabase, _strip_bucket_prefix, SUPABASE_BUCKET, upload_file_to_supabase
from storage import download_file_from_supabase, delete_file_from_supabase, save_intake_artifacts
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

class DatasetQuery(BaseModel):
    query: str

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
    target_cls_col: Optional[str] = None
    positive_label: Optional[Union[str, int]] = None
# Base configuration
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "ethanhong")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "printing")
MASTER_DB_NAME = os.environ.get("MASTER_DB_NAME", "master_ml_insights")
REFRESH_TOKEN_EXPIRE_DAYS = 7  # Example: Refresh token expires after 7 days
# Master database for user management
MASTER_DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{MASTER_DB_NAME}"
master_engine = create_engine(
    MASTER_DB_URL,                 # your existing URL
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True,            # <- tests connections before use; swaps dead ones
    pool_recycle=300,              # <- recycle before many servers/NATs idle-drop (~5 min)
    pool_use_lifo=True,            # (2.0+) faster reuse of warm conns
    # echo="debug",                # optional: useful while diagnosing
)
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
    """
    Robust datetime parsing:
    - try default
    - if too many NaT, try dayfirst
    - if still weak and column looks numeric, try epoch units
    Picks the best (fewest NaT).
    """
    def score(s: pd.Series) -> int:
        return int(s.notna().sum())

    # 1) default
    best = pd.to_datetime(series, errors="coerce", utc=False)
    best_score = score(best)

    # 2) dayfirst
    cand = pd.to_datetime(series, errors="coerce", utc=False, dayfirst=True)
    s2 = score(cand)
    if s2 > best_score:
        best, best_score = cand, s2

    # 3) epoch numbers (if mostly numeric-like)
    # e.g., 1694640000, 1694640000000, etc.
    if best_score < max(5, int(0.6 * len(series))):  # only if default was weak
        ser_num = pd.to_numeric(series, errors="coerce")
        if ser_num.notna().mean() > 0.6:
            for unit in ("s", "ms", "us", "ns"):
                cand = pd.to_datetime(ser_num, unit=unit, errors="coerce", utc=False)
                s3 = score(cand)
                if s3 > best_score:
                    best, best_score = cand, s3

    return best

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
    parsed = pd.to_datetime(s, errors="coerce")
    return parsed.notna().mean() >= 0.5
def detect_columns(
    df: pd.DataFrame,
    *,
    meta: Optional[Dict] = None,     # intake meta JSON (already sanitized)
    context: Optional[str] = None,   # e.g., "orders", "customers", ...
    max_cls_cardinality: int = 10,   # treat <=10 unique as classification-ish
) -> Dict[str, Any]:
    cols: List[str] = [c for c in df.columns if isinstance(c, str)]
    low: Dict[str, str] = {c: c.casefold() for c in cols}

    def has_any(name: str, keys: List[str]) -> bool:
        n = low.get(name, "")
        return any(k in n for k in keys)

    canon_by_norm: Dict[str, str] = {}
    if isinstance(meta, dict):
        canon_by_norm = meta.get("canonical_by_normalized") or meta.get("canonical_by_normalized".lower(), {}) or {}

    def _norms_for(*canonicals: str) -> List[str]:
        return [n for n, c in canon_by_norm.items() if c in canonicals]

    # ---------- IDs ----------
    id_cols = _norms_for("order_id") or _norms_for("customer_id")
    if not id_cols:
        # heuristics if meta absent
        id_aliases = ["order_id","order","orderno","oid","customer_id","cust_id","client_id","user_id","uid","shopper_id","id"]
        id_cols = [c for c in cols if has_any(c, id_aliases)]
        if not id_cols and "id" in cols:
            id_cols = ["id"]

    # ---------- Dates ----------
    date_cols = _norms_for("created_at", "updated_at")
    if not date_cols:
        date_aliases = ["date","time","created","updated","timestamp","ordered","processed","processed_at","order_date"]
        date_cols = [c for c in cols if has_any(c, date_aliases) or _looks_like_date(df[c])]

    # ---------- Classification Target ----------
    # Context-sensitive name hints
    base_aliases = [
        "target","label","class","y","outcome",
        "churn","is_churn","fraud","is_fraud","refund","refunded","is_refund",
        "returned","is_returned","subscribed","is_subscribed",
        "active","is_active","cancelled","is_cancelled",
        "purchased","conversion","converted","success","won","lost",
        "is_new_customer","new_customer"
    ]
    if (context or (meta or {}).get("context_inferred")) == "orders":
        # prefer purchase/return/refund outcomes in orders datasets
        base_aliases = [
            "purchased","conversion","converted","returned","is_returned",
            "refund","refunded","is_refund","cancelled","is_cancelled",
            "success","won","lost","target","label","class","y","outcome","churn","is_churn"
        ] + base_aliases

    named_hits = [c for c in cols if any(a in low[c] for a in base_aliases)]
    boolish = []
    lowcard = []
    for c in cols:
        s = df[c].dropna()
        # skip obvious identifiers as targets
        if c in id_cols:
            continue
        # booleanish
        uniq_norm = set(s.astype(str).str.strip().str.casefold().unique())
        if uniq_norm <= {"0","1"} or uniq_norm <= {"true","false"}:
            boolish.append(c)
            continue
        # low-cardinality non-numeric (likely categorical)
        try:
            uniq = int(s.nunique(dropna=True))
        except Exception:
            uniq = 0
        if uniq and uniq <= max_cls_cardinality and not pd.api.types.is_float_dtype(s.dtype):
            lowcard.append(c)

    # candidate ranking with scores
    cand_ranked: List[Tuple[str, float, str]] = []
    for c in named_hits:
        cand_ranked.append((c, 1.0, "name_match"))
    for c in boolish:
        if c not in [x[0] for x in cand_ranked]:
            cand_ranked.append((c, 0.9, "booleanish"))
    for c in lowcard:
        if c not in [x[0] for x in cand_ranked]:
            # score by inverse cardinality
            card = max(1, int(df[c].nunique(dropna=True)))
            cand_ranked.append((c, 0.8 - min(0.6, 0.02 * (card - 2)), f"low_cardinality:{card}"))

    cand_ranked.sort(key=lambda t: (-t[1], t[0]))
    target_cls_col = cand_ranked[0][0] if cand_ranked else None

    # ---------- Regression Target ----------
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    reg_aliases = ["target","amount","price","revenue","score","value","aov","ltv","gmv","margin"]
    cand_reg = [c for c in num_cols if any(a in low[c] for a in reg_aliases)]
    target_reg_col = cand_reg[0] if cand_reg else None

    # ---------- Groups / Text ----------
    group_cols = [c for c in cols if any(k in low[c] for k in ["store","shop","segment","category","group","channel","country","state","region"])][:3]

    obj_cols = [c for c in cols if df[c].dtype == "object"]
    text_cols = []
    for c in obj_cols:
        s = df[c].dropna()
        # high uniqueness ratio among strings → treat as text-ish
        if len(s) == 0:
            continue
        uniq_ratio = s.nunique(dropna=True) / max(1, len(s))
        if uniq_ratio > 0.20:
            text_cols.append(c)

    # ---------- Ignore ----------
    ignore_cols: List[str] = []
    # If intake PII identified columns to ignore, honor them
    pii = (meta or {}).get("pii") or {}
    if isinstance(pii, dict):
        actions = pii.get("actions") or {}
        ignore_cols.extend([c for c, act in actions.items() if act == "ignore"])

    # ---------- numeric_keep_continuous ----------
    numeric_keep_continuous = []
    for c in num_cols:
        s = df[c].dropna()
        uniq = int(s.nunique(dropna=True)) if len(s) else 0
        uniq_ratio = (uniq / max(1, len(s))) if len(s) else 0.0
        if (
            uniq >= 30
            or uniq_ratio >= 0.05
            or any(k in low[c] for k in ["age","tenure","amount","price","cost","revenue","qty","quantity","score","margin","days","duration","time","rate"])
        ):
            numeric_keep_continuous.append(c)

    return {
        "id_cols": id_cols,
        "target_cls_col": target_cls_col,
        "target_cls_candidates": [{"col": c, "score": sc, "reason": r} for c, sc, r in cand_ranked],
        "target_reg_col": target_reg_col,
        "group_cols": group_cols[:3],
        "date_cols": date_cols,
        "text_cols": text_cols,
        "ignore_cols": ignore_cols,
        "numeric_keep_continuous": numeric_keep_continuous,
        "why": {
            "context": context or (meta or {}).get("context_inferred"),
            "used_meta_canon": bool(canon_by_norm),
        },
    }


from typing import Optional, Any, Dict, List, Tuple
import pandas as pd
import numpy as np

def guess_positive_label(
    df: pd.DataFrame,
    target_cls_col: Optional[str],
) -> Optional[Any]:
    """
    Choose a 'positive' label for classification metrics.
    - Prefer '1' for truly binary 0/1 or booleanish columns.
    - Else map common truthy/positive strings.
    - Else pick the minority class (stable tiebreak via sorted order).
    Returns the label **as it appears in the column** (original dtype if possible).
    """
    if not target_cls_col or target_cls_col not in df.columns:
        return None

    s = df[target_cls_col].dropna()
    if s.empty:
        return None

    # normalize strings for detection, but keep original for return
    s_norm = s.astype(str).str.strip().str.casefold()
    # common mappings
    truthy = {"1", "true", "yes", "y", "t", "purchase", "purchased", "won", "success", "churn", "fraud", "refund", "returned"}
    falsy  = {"0", "false", "no", "n", "f", "lost", "nonchurn", "clean", "no_refund", "not_returned"}

    uniq_norm = set(s_norm.unique())

    # Strict binary (0/1 or bool-ish)
    if uniq_norm <= {"0","1"} or uniq_norm <= {"false","true"}:
        # return the original representation of the positive if possible
        # prefer whatever literal maps to "1"/"true"
        # try exact "1" match first
        mask_pos = s_norm == "1"
        if mask_pos.any():
            return s.loc[mask_pos].iloc[0]
        mask_pos = s_norm == "true"
        if mask_pos.any():
            return s.loc[mask_pos].iloc[0]
        # fallback
        return s.iloc[0]

    # If any known truthy token appears, return that token's original
    for tok in truthy:
        mask = s_norm == tok
        if mask.any():
            return s.loc[mask].iloc[0]

    # Minority class fallback (keeps original labels)
    return _minority_label(s)


def _minority_label(series: pd.Series) -> Optional[Any]:
    if series.empty:
        return None
    counts = series.value_counts(dropna=True)
    # choose smallest count; stable tiebreak on stringified label
    min_count = counts.min()
    cands = sorted(counts[counts == min_count].index, key=lambda x: str(x))
    return cands[0] if cands else None

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
# =========================== datasets.preprocess.py ===========================
# Paste-ready endpoint + helper to resolve/load intake meta
# ============================================================================

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import os, shutil, json, tempfile
import pandas as pd

# --- project imports you already have somewhere ---
# from .auth import get_current_active_user, get_user_db, Dataset
# from .paths import user_dataset_root, ensure_dirs
# from .storage import _strip_bucket_prefix, sb_signed_url, sb_download_to_path
# from .datasets import resolve_and_cache_dataset_csv, _download_object_to_temp
# from .preprocess import (
#     _std_null_marker_list, standardize_missing_markers, compute_null_rate,
#     detect_columns, guess_positive_label, _safe_list,
#     PreprocessConfig, build_and_save_train_test, make_processed_name,
#     preprocess_full_dataset, save_intake_artifacts, sanitize_for_json
# )
# from .fs import PathL

# If you're importing directly (not dotted), keep as-is:
from auth import get_current_active_user, get_user_db, Dataset
from storage import sb_download_to_path, sb_signed_url, _strip_bucket_prefix, save_intake_artifacts

from datasets import resolve_and_cache_dataset_csv, _download_object_to_temp
# ---------------------------------------------------------------------------
# Intake meta loader (DB hints -> sibling files -> canonical fallback)
# ---------------------------------------------------------------------------

MIN_META_KEYS = {"data_gaps", "assumptions", "priors", "prior_provenance"}

def _read_json_from_supabase(bucket: str, key: str) -> dict | None:
    """Download an object to a temp file and parse JSON. Return None if not found/invalid."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            sb_download_to_path(bucket, key, tmp.name)  # must raise on 404/permissions
            with open(tmp.name, "r", encoding="utf-8") as f:
                data = json.load(f)
        if isinstance(data, str):
            data = json.loads(data)
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def _meta_is_minimal(meta: dict) -> bool:
    try:
        return all(k in meta for k in MIN_META_KEYS)
    except Exception:
        return False
import json, os, tempfile
from contextlib import contextmanager

# If you already have these helpers, import them instead:
# - sb_download_to_path(bucket, key, dest_path)
# - _strip_bucket_prefix
# - SUPABASE_BUCKET
# - get_user_db, Dataset

MIN_META_KEYS = {"data_gaps", "assumptions", "priors", "prior_provenance"}

def _read_json_from_supabase(bucket: str, key: str) -> dict | None:
    """Download an object to a temp file and parse JSON. Return None if not found/invalid."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            sb_download_to_path(bucket, key, tmp.name)  # must raise on 404/permissions
            with open(tmp.name, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, str):
                # Some older writers might have double-serialized JSON
                data = json.loads(data)
            if not isinstance(data, dict):
                return None
            return data
    except Exception:
        return None

def _meta_is_minimal(meta: dict) -> bool:
    try:
        return all(k in meta for k in MIN_META_KEYS)
    except Exception:
        return False
import posixpath

def key_join(*parts) -> str:
    return posixpath.join(*(str(p).strip("/\\") for p in parts if p not in (None, "")))

def key_norm(key: str) -> str:
    # turn Windows backslashes into forward slashes and strip leading slash
    return "/".join(str(key).split("\\")).lstrip("/")

def load_intake_meta(*, user_id: int | str, dataset_id: int | str, current_user=None) -> dict | None:
    bucket = os.environ.get("SUPABASE_BUCKET", "user-uploads")
    uid, did = str(user_id), str(dataset_id)

    def _push(lst, k):
        k = key_norm(k)
        if k and k not in lst:
            lst.append(k)

    candidates: list[str] = []
    # 1–3) DB-driven and sibling locations
    with get_user_db(current_user) as db:
        ds = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
        if not ds:
            return None

        explicit = getattr(ds, "intake_meta_path", None) or getattr(ds, "meta_path", None)
        if explicit:
            _push(candidates, explicit)

        orig = getattr(ds, "original_file_path", None) or getattr(ds, "file_path", None)
        if orig:
            prefix = posixpath.dirname(key_norm(orig))
            _push(candidates, f"{prefix}/meta.json")

        cur = getattr(ds, "file_path", None)
        if cur:
            prefix = posixpath.dirname(key_norm(cur))
            _push(candidates, f"{prefix}/meta.json")

    # 4) Intake variants (what your intake route writes)
    _push(candidates, key_join(uid, did, "intake", "intake_meta.json"))
    _push(candidates, key_join(uid, did, "intake", "meta.json"))

    # 5) Canonical per-dataset + 6) user-level fallback
    _push(candidates, key_join(uid, did, "meta.json"))
    _push(candidates, key_join(uid, "meta.json"))

    first_nonminimal: dict | None = None
    for k in candidates:
        meta = _read_json_from_supabase(bucket, k)
        if isinstance(meta, dict) and meta:
            if _meta_is_minimal(meta):
                return meta
            if first_nonminimal is None:
                first_nonminimal = meta

    if first_nonminimal:
        return _normalize_intake_meta(first_nonminimal)

    return None


def _meta_is_minimal(meta: dict) -> bool:
    # Be permissive: accept either inferred types or header map (your compiler needs just one)
    return isinstance(meta, dict) and (
        ("inferred_types" in meta and isinstance(meta["inferred_types"], dict))
        or ("header_map" in meta and isinstance(meta["header_map"], dict))
    )

def _normalize_intake_meta(meta: dict) -> dict:
    out = dict(meta)
    out.setdefault("data_gaps", {})
    out.setdefault("assumptions", {})
    out.setdefault("priors", {})
    out.setdefault("prior_provenance", {})
    # Optional: relocate/rename keys if older schema variants exist
    return out


def load_or_backfill_intake_meta(
    *,
    user_id: int | str,
    dataset_id: int | str,
    current_user,
    base_currency: str = "USD",
    country_hint: str = "US",
) -> dict:
    """
    Load previously computed intake meta for a dataset; if missing, run intake once,
    persist artifacts (meta.json), and return the resulting meta dict.

    Requires:
      - load_intake_meta (loader that searches DB hints, sibling files, and canonical path)
      - get_user_db, Dataset (ORM/session helpers)
      - _download_object_to_temp / resolve_and_cache_dataset_csv (to fetch the CSV locally)
      - intake_and_normalize, _meta_to_json (intake pipeline + JSON-safe serializer)
      - save_intake_artifacts (persists meta.json under user-uploads/{user_id}/{dataset_id}/meta.json)

    Returns:
      dict: JSON-safe meta with at least keys:
            {"data_gaps","assumptions","priors","prior_provenance", ...}
    """
    # 1) Try to load if it already exists
    meta = load_intake_meta(user_id=user_id, dataset_id=dataset_id, current_user=current_user)
    if isinstance(meta, dict) and meta:
        return meta

    # 2) Resolve original CSV path
    with get_user_db(current_user) as db:
        ds = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
        if not ds:
            raise ValueError("Dataset not found")

        src_key = getattr(ds, "original_file_path", None) or getattr(ds, "file_path", None)
        if not src_key:
            raise ValueError("Dataset has no file path")

        try:
            local_csv = _download_object_to_temp(src_key, user_id=user_id, dataset_id=dataset_id)
        except Exception:
            # Fallback: your existing resolver (may read via ds.file_path internally)
            local_csv = resolve_and_cache_dataset_csv(db=db, dataset_id=dataset_id, user_id=user_id)

    # 3) Run intake once (lightweight compared to full FE/encoding)
    intake_res = intake_and_normalize(local_csv, base_currency=base_currency, country_hint=country_hint)
    meta_json = _meta_to_json(intake_res.meta)

    # 4) Persist meta.json in a canonical per-dataset location
    written_key = f"{user_id}/{dataset_id}/meta.json"  # expected canonical key
    try:
        res = save_intake_artifacts(
            dataset_id=dataset_id,
            user_id=user_id,
            meta=meta_json,    # primary payload to persist
            stats={},          # optional
            preview=None,      # optional
            artifacts=None,    # optional
        )
        # If your saver returns the actual key/path, honor it
        if isinstance(res, dict):
            written_key = (
                res.get("meta_key")
                or res.get("paths", {}).get("meta")
                or written_key
            )
    except Exception:
        # Persistence failure should surface, but if you prefer soft-fail, remove this raise
        raise

    # 5) Optionally record the meta path on the Dataset row for quicker future lookups
    try:
        with get_user_db(current_user) as db:
            ds = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
            if ds is not None:
                if hasattr(ds, "intake_meta_path"):
                    setattr(ds, "intake_meta_path", written_key)
                elif hasattr(ds, "meta_path"):
                    setattr(ds, "meta_path", written_key)
                db.add(ds)
                db.commit()
    except Exception:
        # Non-fatal: the meta is still persisted in storage
        pass

    return meta_json
def _read_sample_robust(path: str, *, nrows: int = 5000) -> pd.DataFrame | None:
    """
    Try several encodings & delimiters; normalize nulls afterward.
    Returns None if we can't read anything usable.
    """
    # fast path
    try:
        df = pd.read_csv(
            path, nrows=nrows, low_memory=False, na_values=_std_null_marker_list(),
            keep_default_na=True, encoding="utf-8-sig"
        )
        return standardize_missing_markers(df)
    except Exception:
        pass

    encodings = ["utf-8", "latin-1", "utf-16", "cp1252"]
    seps = [",", ";", "\t", "|"]
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    path,
                    nrows=nrows,
                    sep=sep,
                    engine="python",    # more forgiving
                    na_values=_std_null_marker_list(),
                    keep_default_na=True,
                    encoding=enc,
                    low_memory=False,
                    on_bad_lines="skip",
                )
                return standardize_missing_markers(df)
            except Exception:
                continue

    # final hail mary: sniff dialect with csv
    try:
        with open(path, "rb") as fh:
            raw = fh.read(128 * 1024)  # 128KB is plenty to sniff
        try:
            text = raw.decode("utf-8-sig")
        except Exception:
            text = raw.decode("latin-1", errors="ignore")
        try:
            dialect = csv.Sniffer().sniff(text.splitlines()[0])
            sep = dialect.delimiter
        except Exception:
            sep = ","
        df = pd.read_csv(
            io.BytesIO(raw), nrows=nrows, sep=sep, engine="python",
            na_values=_std_null_marker_list(), keep_default_na=True, low_memory=False
        )
        return standardize_missing_markers(df)
    except Exception:
        return None

def _remote_csv_has_rows(processed_key: str | None) -> bool:
    """
    Best-effort: download the processed object to a temp file and check head(1).
    Return False on any error.
    """
    if not processed_key:
        return False
    try:
        bucket = os.environ.get("SUPABASE_BUCKET", "user-uploads")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name
        try:
            # you already have this downloader in your codebase
            sb_download_to_path(bucket, _strip_bucket_prefix(processed_key), tmp_path)
            df = pd.read_csv(tmp_path, nrows=1, low_memory=False)
            return (df is not None) and (df.shape[0] > 0)
        finally:
            try: os.remove(tmp_path)
            except Exception: pass
    except Exception:
        return False

def _upload_dataframe_as_processed(df: pd.DataFrame, *, user_id: str | int, processed_fname: str) -> str | None:
    """
    Write df → CSV → upload via your existing helper.
    Returns storage path (object key) or None on failure.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
        try:
            # your helper signature: upload_file_to_supabase(user_id, local_path, filename)
            storage_path = upload_file_to_supabase(str(user_id), tmp_path, processed_fname)
            return storage_path
        finally:
            try: os.remove(tmp_path)
            except Exception: pass
    except Exception:
        return None
@router.post("/{dataset_id}/preprocess")
def preprocess_dataset_endpoint(
    dataset_id: int,
    body: "PreprocessRequest",  # type: ignore[name-defined]
    current_user = Depends(get_current_active_user),
):
    try:
        user_id = getattr(current_user, "id", None) or getattr(current_user, "user_id", None)
        if user_id is None:
            raise HTTPException(status_code=401, detail="Unauthenticated")

        # Resolve ORIGINAL → local cache (never chain-process a processed file)
        with get_user_db(current_user) as db:
            ds = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
            if not ds:
                raise HTTPException(status_code=404, detail="Dataset not found")

            src_key = getattr(ds, "original_file_path", None) or getattr(ds, "file_path", None)
            if not src_key:
                raise HTTPException(status_code=400, detail="Dataset has no file path")

            try:
                src_csv_path = _download_object_to_temp(src_key, user_id=user_id, dataset_id=dataset_id)
            except Exception:
                src_csv_path = resolve_and_cache_dataset_csv(db=db, dataset_id=dataset_id, user_id=user_id)

            prior_file_path = getattr(ds, "file_path", None)
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

        # Read sample (robust)
        sample_df = _read_sample_robust(train_csv_path, nrows=5000)

        # Quick empty check: headers-only or all-null rows?
        if sample_df is None or sample_df.shape[0] == 0 or sample_df.dropna(how="all").shape[0] == 0:
            raise HTTPException(
                status_code=422,
                detail="Uploaded CSV appears to have no data rows (headers-only or fully empty after cleaning).",
            )

        # Try to load previously saved meta.json (from intake)
        meta = None
        try:
            meta = load_intake_meta(user_id=user_id, dataset_id=dataset_id, current_user=current_user)
        except Exception:
            meta = None

        # Backfill meta if needed
        if meta is None:
            intake_res = intake_and_normalize(src_csv_path, base_currency="USD", country_hint="US")
            meta = _meta_to_json(intake_res.meta)
            save_intake_artifacts(
                dataset_id=dataset_id,
                user_id=user_id,
                meta=meta,
                stats={},
                preview=None,
                artifacts=None,
            )

        # Re-map sample_df columns to normalized names using intake meta (if available)
        try:
            header_map = (meta or {}).get("header_map")
            if header_map:
                # meta.header_map was original_name -> ColumnMeta, rebuild mapping:
                norm_map = {orig: info["normalized_name"] for orig, info in header_map.items()}
                sample_df_norm = sample_df.rename(columns=norm_map)
            else:
                sample_df_norm = sample_df
        except Exception:
            sample_df_norm = sample_df
        meta_for_detect = meta if isinstance(meta, dict) else _meta_to_json(meta) if meta is not None else {}

        # Source of truth for classification target:
        # 1) user override in request
        # 2) intake suggestions from meta
        # 3) fallback: detect on sample_df_norm here (only for target resolution)
        user_tgt = getattr(body, "target_cls_col", None)
        user_pos = getattr(body, "positive_label", None)
        suggested = (meta or {}).get("suggested") or {}
        meta_tgt = suggested.get("target_cls_col")
        meta_pos = suggested.get("positive_label")
        target_source = "detect_now"
        if user_tgt:
            target_cls_norm = user_tgt
            positive_label  = user_pos if user_pos is not None else guess_positive_label(sample_df_norm, user_tgt)
            target_source = "user_override"
        elif meta_tgt:
            target_cls_norm = meta_tgt
            positive_label  = meta_pos if meta_pos is not None else guess_positive_label(sample_df_norm, meta_tgt)
            target_source = "intake_suggested"
        else:
            try:
                det_now_for_target = detect_columns(
                    sample_df_norm,
                    meta=meta_for_detect,
                    context=(meta_for_detect or {}).get("context_inferred"),
                )
            except Exception:
                det_now_for_target = {}
            target_cls_norm = det_now_for_target.get("target_cls_col")
            positive_label  = guess_positive_label(sample_df_norm, target_cls_norm)

        def _coerce_label_dtype(df, col, label):
            if label is None or col not in df.columns:
                return label
            try:
                ser = df[col]
                # try using the pandas dtype to cast back
                return ser.dtype.type(label) if hasattr(ser.dtype, "type") else label
            except Exception:
                return label

        positive_label = _coerce_label_dtype(sample_df_norm, target_cls_norm, positive_label)

        try:
            hints = detect_columns(sample_df_norm, meta=meta_for_detect, context=(meta_for_detect or {}).get("context_inferred"))
        except Exception:
            hints = {}

        canon_by_norm = (meta_for_detect or {}).get("canonical_by_normalized") or {}

        def _norms_for(*canonicals):
            return [n for n, c in canon_by_norm.items() if c in canonicals]

        default_ids = ["id"] if "id" in sample_df_norm.columns else []
        id_cols_norm      = _norms_for("order_id") or _norms_for("customer_id") or (hints.get("id_cols") or default_ids)
        date_cols_norm    = _norms_for("created_at", "updated_at") or (hints.get("date_cols") or [])
        target_reg_norm   = hints.get("target_reg_col") or None
        group_cols_norm   = (hints.get("group_cols") or [])[:3]
        text_cols_norm    = hints.get("text_cols") or []
        ignore_cols_norm  = hints.get("ignore_cols") or []
        keep_continuous   = hints.get("numeric_keep_continuous") or []

        cfg = PreprocessConfig(
            data_dir=data_dir,
            encoder_info_dir=enc_dir,
            data_process_dir=proc_dir,
            id_cols=id_cols_norm,
            target_cls_col=target_cls_norm,
            positive_label=positive_label,
            target_reg_col=target_reg_norm,
            group_cols=_safe_list(group_cols_norm)[:3],
            date_cols=_safe_list(date_cols_norm),
            text_cols=_safe_list(text_cols_norm),
            ignore_cols=_safe_list(ignore_cols_norm),
            numeric_to_cat=True,
            numeric_bins=10,
            numeric_keep_continuous=_safe_list(keep_continuous),
            max_onehot_cardinality=80,
            add_date_parts=True,
            verbose=True,
        )

        # Optional split build (not fatal)
        try:
            build_and_save_train_test(cfg, train_csv="train.csv", test_csv="test.csv")
        except Exception:
            pass

        processed_fname = make_processed_name(original_name, dataset_id)

        # ------------------ FULL PREPROCESS CALL ------------------
        try:
            artifacts = preprocess_full_dataset(
                cfg,
                csv_path=train_csv_path,
                user_id=str(user_id),
                dataset_id=dataset_id,
                filename=processed_fname,
                fit_encoders=True,
            )
        except HTTPException as e:
            logger.exception("preprocess_full_dataset raised HTTPException: %s", e)
            raise
        except Exception as e:
            logger.exception("preprocess_full_dataset crashed: %s", e)
            raise HTTPException(status_code=500, detail=f"Preprocess failed: {e}")

        logger.info(
            "preprocess_endpoint: preprocess_full_dataset returned | keys=%s",
            list(artifacts.keys()),
        )

        # ------------------ AFTER preprocess_full_dataset RETURNS ------------------

        # Extract storage info (added by preprocess_full_dataset)
        storage_mode = artifacts.get("storage_mode", "full")
        storage_reason = artifacts.get("storage_reason")

        processed_key = (
            artifacts.get("processed_key")       # sb:... or fs:...
            or artifacts.get("data_csv")
            or artifacts.get("processed_csv")
            or artifacts.get("normalized_csv")
        )

        rows_processed = artifacts.get("rows_processed")
        cols_processed = artifacts.get("cols_processed")

        logger.info(
            "preprocess_endpoint: initial artifacts snapshot | "
            "storage_mode=%s storage_reason=%s rows=%s cols=%s "
            "processed_key=%s data_csv=%s null_rate_after_full=%s",
            storage_mode,
            storage_reason,
            rows_processed,
            cols_processed,
            processed_key,
            artifacts.get("data_csv"),
            artifacts.get("null_rate_after_full"),
        )

        def _is_supabase_key(key: str | None) -> bool:
            # Only bare Supabase-style paths should be treated as remote objects.
            # Anything starting with fs: is local-only.
            if not key:
                return False
            return not str(key).startswith("fs:")

        # Only call _remote_csv_has_rows for Supabase keys
        if _is_supabase_key(processed_key):
            try:
                logger.info("preprocess_endpoint: probing remote csv for rows, key=%s", processed_key)
                is_nonempty_remote = _remote_csv_has_rows(processed_key)
                logger.info(
                    "preprocess_endpoint: remote csv probe complete, key=%s, is_nonempty_remote=%s",
                    processed_key,
                    is_nonempty_remote,
                )
            except Exception as e:
                logger.warning(
                    "preprocess_endpoint: remote csv probe failed for %s: %s",
                    processed_key,
                    e,
                    exc_info=True,
                )
                is_nonempty_remote = False
        else:
            is_nonempty_remote = False
            logger.info(
                "preprocess_endpoint: processed_key is local or missing; skipping remote probe. key=%s",
                processed_key,
            )

        # ------------------ Guardrail: replace EMPTY output ------------------
        try:
            rp = int(rows_processed) if rows_processed is not None else None
        except Exception as e:
            logger.warning("preprocess_endpoint: failed to cast rows_processed=%r to int: %s", rows_processed, e)
            rp = None

        try:
            cp = int(cols_processed) if cols_processed is not None else None
        except Exception as e:
            logger.warning("preprocess_endpoint: failed to cast cols_processed=%r to int: %s", cols_processed, e)
            cp = None

        produced_empty = (
            (rp is not None and rp == 0)
            or (cp is not None and cp == 0)
            or (rp in (None, 0) and not is_nonempty_remote)
        )

        logger.info(
            "preprocess_endpoint: empty-check snapshot | rp=%s cp=%s is_nonempty_remote=%s produced_empty=%s",
            rp,
            cp,
            is_nonempty_remote,
            produced_empty,
        )

        fallback_used = False
        if produced_empty:
            logger.warning("preprocess_endpoint: produced_empty=True, entering fallback_light_normalize")
            # Light-normalize the original and upload as processed
            try:
                light = intake_and_normalize(src_csv_path, base_currency="USD", country_hint="US")
                df_light = light.df_normalized
                logger.info(
                    "preprocess_endpoint: fallback intake_and_normalize produced df_light shape=%s",
                    df_light.shape if df_light is not None else None,
                )
            except Exception as e:
                logger.warning(
                    "preprocess_endpoint: fallback intake_and_normalize failed, using sample_df. err=%s",
                    e,
                    exc_info=True,
                )
                df_light = sample_df  # at least pass through sample if intake fails

            if df_light is not None and not df_light.empty:
                storage_path = _upload_dataframe_as_processed(
                    df_light,
                    user_id=user_id,
                    processed_fname=processed_fname,
                )
                logger.info(
                    "preprocess_endpoint: fallback _upload_dataframe_as_processed returned storage_path=%s",
                    storage_path,
                )
                if not storage_path:
                    logger.error(
                        "preprocess_endpoint: fallback upload failed – storage_path is None/empty"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="Processed pipeline yielded empty output and fallback upload failed.",
                    )
                processed_key = storage_path
                # refresh artifacts & metrics to reflect fallback
                artifacts["data_csv"] = storage_path
                artifacts["rows_processed"] = int(df_light.shape[0])
                artifacts["cols_processed"] = int(df_light.shape[1])
                try:
                    artifacts["null_rate_after_full"] = float(compute_null_rate(df_light))
                except Exception as e:
                    logger.warning(
                        "preprocess_endpoint: compute_null_rate(df_light) failed in fallback: %s",
                        e,
                        exc_info=True,
                    )
                    artifacts["null_rate_after_full"] = None
                fallback_used = True
                storage_mode = "full"
                storage_reason = "fallback_light_normalize"
                logger.info(
                    "preprocess_endpoint: fallback complete | new_processed_key=%s rows=%s cols=%s",
                    processed_key,
                    artifacts.get("rows_processed"),
                    artifacts.get("cols_processed"),
                )
            else:
                logger.error(
                    "preprocess_endpoint: produced_empty=True but df_light is None/empty – aborting with 422"
                )
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Processing produced an empty dataset and fallback also had 0 rows. "
                        "Please upload a file with at least one non-empty data row."
                    ),
                )

        # If we *expected* to store but ended with no key, treat as error
        if not processed_key and storage_mode in ("full", "supabase"):
            logger.error(
                "preprocess_endpoint: expected processed_key for storage_mode=%s, but got processed_key=%r",
                storage_mode,
                processed_key,
            )
            raise HTTPException(status_code=500, detail="Pipeline did not return processed data key")

        # ---- SAFER null-rate metric on sample_df ----
        try:
            null_rate_before_sample = compute_null_rate(sample_df)
            logger.info(
                "preprocess_endpoint: compute_null_rate(sample_df)=%s",
                null_rate_before_sample,
            )
        except Exception as e:
            logger.warning(
                "preprocess_endpoint: compute_null_rate(sample_df) failed: %s",
                e,
                exc_info=True,
            )
            null_rate_before_sample = None

        # ------------------ DB PERSIST ------------------
        logger.info(
            "preprocess_endpoint: entering DB persist | processed_key=%s storage_mode=%s storage_reason=%s",
            processed_key,
            storage_mode,
            storage_reason,
        )
        with get_user_db(current_user) as db:
            ds = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
            if not ds:
                logger.error("preprocess_endpoint: dataset %s not found during persist", dataset_id)
                raise HTTPException(status_code=404, detail="Dataset not found for update")

            if not getattr(ds, "original_file_path", None):
                setattr(ds, "original_file_path", getattr(ds, "file_path", None))

            if processed_key:
                if hasattr(ds, "processed_file_path"):
                    ds.processed_file_path = processed_key

                ds.file_path    = processed_key
                ds.stage        = "processed"
                ds.processed_at = datetime.utcnow()

            if artifacts.get("encoder_pkl"):
                setattr(ds, "encoder_path", artifacts["encoder_pkl"])

            db.add(ds)
            db.commit()
            db.refresh(ds)

            original_key = getattr(ds, "original_file_path", None) or prior_file_path

            logger.info(
                "preprocess_endpoint: DB persist complete | original_key=%s file_path=%s stage=%s",
                original_key,
                ds.file_path,
                ds.stage,
            )

        # ------------------ METRICS BUILD ------------------
        null_rate_after_full = artifacts.get("null_rate_after_full")
        logger.info(
            "preprocess_endpoint: metrics raw values | "
            "null_rate_before_sample=%s null_rate_after_full=%s",
            null_rate_before_sample,
            null_rate_after_full,
        )

        try:
            metrics = {
                "null_rate_before_sample": (
                    round(float(null_rate_before_sample), 6)
                    if null_rate_before_sample is not None
                    else None
                ),
                "null_rate_after_full": (
                    round(float(null_rate_after_full), 6)
                    if null_rate_after_full is not None
                    else None
                ),
                "rows_processed": rows_processed,
                "cols_processed": cols_processed,
            }
            logger.info("preprocess_endpoint: metrics build OK | metrics=%s", metrics)
        except Exception as e:
            logger.exception("preprocess_endpoint: metrics build failed: %s", e)
            raise HTTPException(status_code=500, detail=f"metrics build failed: {e}")

                # ------------------ SIGNED URLS ------------------
        def _is_supabase_key(key: str | None) -> bool:
            """
            Only bare Supabase-style object paths should be signed.
            Local 'fs:' keys or empty values are never sent to Supabase.
            """
            if not key:
                return False
            return not str(key).startswith("fs:")

        def _signed_or_none(key: str | None):
            if not _is_supabase_key(key):
                logger.info(
                    "preprocess_endpoint: skipping signed URL for key=%s (local-or-empty)",
                    key,
                )
                return None
            try:
                bucket = os.environ.get("SUPABASE_BUCKET", "user-uploads")
                url = sb_signed_url(bucket, _strip_bucket_prefix(key), seconds=300)
                logger.info("preprocess_endpoint: signed URL generated for key=%s", key)
                return url
            except Exception as e:
                logger.warning(
                    "preprocess_endpoint: sb_signed_url failed for %s: %s",
                    key,
                    e,
                    exc_info=True,
                )
                return None

        # Prefer the explicit Supabase key from artifacts, but fall back to processed_key
        supabase_processed_key = artifacts.get("data_csv")
        if not supabase_processed_key and storage_mode == "supabase":
            # Backward/defensive: if we're in Supabase mode and processed_key
            # looks like a Supabase object, use that.
            if _is_supabase_key(processed_key):
                supabase_processed_key = processed_key

        processed_signed_url = _signed_or_none(supabase_processed_key)
        original_signed_url  = _signed_or_none(original_key)

        msg = "Preprocess & Clean completed"
        if fallback_used:
            msg += " (fallback used: light normalize)"
        if storage_mode not in ("full", "supabase"):
            if storage_reason:
                msg += f" (storage_mode={storage_mode}, reason={storage_reason})"
            else:
                msg += f" (storage_mode={storage_mode})"

        # ------------------ PAYLOAD BUILD ------------------
        logger.info("preprocess_endpoint: building payload object")
        payload = {
            "dataset_id": dataset_id,
            "user_id": user_id,
            "config_used": {
                "id_cols": cfg.id_cols,
                "target_cls_col": cfg.target_cls_col,
                "positive_label": cfg.positive_label,
                "target_source": target_source,
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
            "message": msg,
            "files": {
                "original": {
                    "key": original_key,
                    "signedUrl": original_signed_url,
                    "filename": original_name,
                },
                "processed": {
                    "key": processed_key,
                    "signedUrl": processed_signed_url,
                    "filename": processed_fname if processed_key else None,
                },
            },
            "replaced_original": False,
            "old_file_path": original_key,
            "new_file_path": processed_key,
            "meta": {
                "data_gaps": (meta or {}).get("data_gaps"),
                "assumptions": (meta or {}).get("assumptions"),
                "priors": (meta or {}).get("priors"),
                "prior_provenance": (meta or {}).get("prior_provenance"),
                "suggested": (meta or {}).get("suggested"),
                "hints": {
                    "target_cls_candidates": (hints or {}).get("target_cls_candidates"),
                    "why": (hints or {}).get("why"),
                },
            },
            "storage": {
                "mode": storage_mode,
                "reason": storage_reason,
            },
            "processed_key": processed_key,
        }

        logger.info(
            "preprocess_endpoint: payload built | "
            "processed_key=%s storage_mode=%s message=%s",
            processed_key,
            storage_mode,
            msg,
        )

        # Final safety: log and catch JSON serialization issues
        try:
            logger.info("preprocess_endpoint: about to serialize + return JSONResponse")
            resp_content = sanitize_for_json(payload)
            logger.info("preprocess_endpoint: sanitize_for_json completed successfully")
            return JSONResponse(content=resp_content, status_code=200)
        except Exception as e:
            logger.exception("sanitize_for_json or JSONResponse failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Serialization failed: {type(e).__name__}: {e}")

    # --------------- TOP-LEVEL CATCH-ALL FOR THE ENDPOINT ---------------
    except HTTPException:
        # Already structured for the client; we just want the traceback in logs.
        logger.exception("HTTPException in preprocess_dataset_endpoint")
        raise
    except Exception as e:
        logger.exception("Unhandled exception in preprocess_dataset_endpoint: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}: {e}")

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
import math
import numpy as np

def clean_json(obj):
    """
    Recursively walk dict/list/tuple and:
      - cast numpy scalars to native Python
      - turn NaN/Inf/-Inf into None
    """
    # numpy scalar -> python
    if isinstance(obj, (np.generic,)):
        obj = obj.item()

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj

    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [clean_json(v) for v in obj]

    # pandas NA types
    try:
        import pandas as pd
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj
import os
ALLOWED_EXTS = {'.csv', '.tsv', '.xlsx', '.xls', '.parquet', '.json', '.jsonl'}

def _profile_light(path: str, fmt: str) -> Dict[str, Any]:
    """
    Lightweight schema + row_count estimate without fully reading huge files.
    Uses your readers where possible, but prefers sampling.
    """
    fmt = fmt.lower()

    # CSV / TSV: sample + cheap line count
    if fmt in ("csv", "tsv"):
        sep = "\t" if fmt == "tsv" else None
        try:
            df_sample = pd.read_csv(path, nrows=SAMPLE_ROWS, sep=sep, engine="python")
        except Exception:
            df_sample = pd.read_csv(path, nrows=SAMPLE_ROWS, sep=sep)

        try:
            with open(path, "rb") as f:
                total_lines = sum(1 for _ in f)
            # header line present if pandas inferred header
            est_rows = max(total_lines - 1, len(df_sample))
        except Exception:
            est_rows = len(df_sample)

        schema = {c: str(df_sample[c].dtype) for c in df_sample.columns}
        return {"schema": schema, "row_estimate": est_rows, "sample": df_sample}

    # XLSX/XLS: sample first sheet (pandas supports nrows)
    if fmt in ("xlsx", "xls"):
        df_sample = pd.read_excel(path, nrows=SAMPLE_ROWS)
        schema = {c: str(df_sample[c].dtype) for c in df_sample.columns}
        return {"schema": schema, "row_estimate": None, "sample": df_sample}

    # Parquet: try pyarrow metadata (no full read)
    if fmt == "parquet":
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            row_count = pf.metadata.num_rows if pf.metadata else None
            # schema via arrow
            arrow_schema = pf.schema_arrow if hasattr(pf, "schema_arrow") else None
            schema = {f.name: str(f.type) for f in (arrow_schema or [])}
            # sample: read first row group, but bounded
            sample_df = None
            if pf.metadata and pf.metadata.num_row_groups > 0:
                tb = pf.read_row_group(0)
                # cap to SAMPLE_ROWS if needed
                if tb.num_rows > SAMPLE_ROWS:
                    tb = tb.slice(0, SAMPLE_ROWS)
                sample_df = tb.to_pandas()
            return {"schema": schema, "row_estimate": row_count, "sample": sample_df}
        except Exception:
            # fallback: minimal info
            return {"schema": {}, "row_estimate": None, "sample": None}

    # JSON/JSONL: sample without loading all
    if fmt in ("jsonl", "json"):
        # JSONL: read first N lines
        if fmt == "jsonl":
            rows = []
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= SAMPLE_ROWS: break
                        line = line.strip()
                        if not line: continue
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            continue
                df = pd.DataFrame(rows) if rows else pd.DataFrame()
                schema = {c: str(df[c].dtype) for c in df.columns}
                # estimate by counting lines
                try:
                    with open(path, "rb") as f:
                        row_estimate = sum(1 for _ in f)
                except Exception:
                    row_estimate = len(rows) or None
                return {"schema": schema, "row_estimate": row_estimate, "sample": df}
            except Exception:
                return {"schema": {}, "row_estimate": None, "sample": None}
        else:
            # plain JSON: try to read and normalize a small slice
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    part = data[:SAMPLE_ROWS]
                    df = pd.json_normalize(part) if part else pd.DataFrame()
                    row_estimate = len(data)
                else:
                    df = pd.json_normalize(data)
                    row_estimate = 1
                schema = {c: str(df[c].dtype) for c in df.columns}
                return {"schema": schema, "row_estimate": row_estimate, "sample": df}
            except Exception:
                return {"schema": {}, "row_estimate": None, "sample": None}

    raise HTTPException(status_code=400, detail=f"Unsupported format '{fmt}'")
def _detect_format_from_name(filename: str) -> str:
    return (PathL(filename).suffix.lower().lstrip('.') or '').strip()
# Tunables to avoid heavy ingestion inline
MAX_INLINE_MB   = 25           # if file > this, skip DB ingest
MAX_INLINE_ROWS = 200_000      # if rows > this, skip DB ingest
SAMPLE_ROWS     = 5_000        # sample for profiling

def safe_table_name(name: str) -> str:
    return name.strip().replace(" ", "_").lower()

# Postgres 32-bit integer range
INT32_MIN = -2_147_483_648
INT32_MAX =  2_147_483_647
def infer_sqlalchemy_column(name: str, series: pd.Series):
    """
    SAFER inference for SQLAlchemy column types.
    Avoids integer overflow on large identifiers (EAN/UPC/GTIN/skus/order ids).
    """

    # Clean nulls for checks
    s = series.dropna()

    # ---- 1. Identify "ID-like" columns (store as TEXT) ----
    id_keywords = ["ean", "upc", "gtin", "sku", "barcode", "code"]
    if any(tok in name.lower() for tok in id_keywords):
        return Column(name, Text)

    # ---- 2. Check if integer dtype ----
    if pd.api.types.is_integer_dtype(series):
        # If column contains extremely large integers → use BigInteger
        if len(s) > 0:
            try:
                min_val = int(s.min())
                max_val = int(s.max())
                if min_val >= INT32_MIN and max_val <= INT32_MAX:
                    return Column(name, Integer)
                else:
                    return Column(name, BigInteger)
            except Exception:
                # mixed weird values → store as TEXT
                return Column(name, Text)
        else:
            return Column(name, Integer)

    # ---- 3. Floats ----
    if pd.api.types.is_float_dtype(series):
        return Column(name, Float)

    # ---- 4. Fallback: store as TEXT ----
    return Column(name, Text)
import logging, re
from pathlib import Path as PathL
from fastapi import HTTPException, Depends, Query
from sqlalchemy import MetaData, Table, text
async def _process_single_upload(
    file: UploadFile,
    name: str,
    description: Optional[str],
    delimiter: Optional[str],
    fmt_hint: Optional[str],
    current_user,
):
    logger.info(f"Upload by user={getattr(current_user,'id',None)}: {file.filename}")

    ext = PathL(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTS))}"
        )

    fmt = (fmt_hint or _detect_format_from_name(file.filename)).lower()  # csv/tsv/xlsx/xls/parquet/json/jsonl
    if fmt == "":  # no suffix somehow
        fmt = "csv"

    temp_path = None
    try:
        # 1) Persist to disk
        await file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".bin") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)

        # 2) Upload to Supabase as-is
        supabase_path = upload_file_to_supabase(
            user_id=str(current_user.id),
            file_path=temp_path,
            filename=file.filename,
        )

        # 3) Lightweight profiling (no full read)
        prof = _profile_light(temp_path, fmt)
        schema = prof.get("schema") or {}
        row_estimate = prof.get("row_estimate")
        sample_df: Optional[pd.DataFrame] = prof.get("sample")

        table_name = safe_table_name(name)

        # 4) Decide whether to inline-ingest
        do_ingest = (
            file_size_mb <= MAX_INLINE_MB and
            (row_estimate is None or row_estimate <= MAX_INLINE_ROWS) and
            sample_df is not None and not sample_df.empty
        )

        # 5) Register + optional ingest
        with get_user_db(current_user) as db:
            dataset_meta = {}

            if do_ingest:
                metadata = MetaData()
                cols = [infer_sqlalchemy_column(col, sample_df[col].dtype) for col in sample_df.columns]
                dataset_table = Table(table_name, metadata, *cols)
                dataset_table.create(bind=db.bind, checkfirst=True)

                # For CSV/TSV small files you can attempt a full read; otherwise just insert the sample
                df_to_insert = sample_df
                if fmt in ("csv", "tsv") and row_estimate and row_estimate <= MAX_INLINE_ROWS:
                    try:
                        # cautious “maybe full read”
                        df_full = pd.read_csv(temp_path, sep=("\t" if fmt == "tsv" else None), engine="python")
                        df_to_insert = df_full
                    except Exception:
                        pass

                if "id" not in df_to_insert.columns:
                    df_to_insert.insert(0, "id", range(1, len(df_to_insert) + 1))

                records = df_to_insert.replace({np.nan: None}).to_dict(orient="records")
                if records:
                    db.execute(dataset_table.insert(), records)

                dataset_meta.update({
                    "row_count": len(df_to_insert),
                    "column_count": df_to_insert.shape[1],
                    "external_only": False,
                })
            else:
                dataset_meta.update({
                    "row_count": row_estimate,
                    "column_count": len(schema) if schema else None,
                    "external_only": True,
                })

            dataset = register_dataset(
                db=db,
                name=name,
                description=description,
                table_name=table_name if not dataset_meta["external_only"] else None,
                file_path=supabase_path,
                row_count=dataset_meta["row_count"],
                column_count=dataset_meta["column_count"],
                schema=schema,
                overwrite_existing=True,
            )
            db.commit()
            db.refresh(dataset)

        # 6) Update storage used
        with master_db_cm() as master_db:
            master_db.execute(
                text("""
                    UPDATE users
                    SET storage_used = storage_used + :additional
                    WHERE id = :user_id
                """),
                {"additional": file_size_mb, "user_id": current_user.id},
            )

        return {
            "message": "Dataset uploaded successfully",
            "dataset_id": dataset.id,
            "name": dataset.name,
            "rows": row_estimate,
            "columns": len(schema) if schema else None,
            "external_only": dataset_meta["external_only"],
            "size_mb": round(file_size_mb, 2),
            "path": supabase_path,
            "ingested_inline": not dataset_meta["external_only"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing dataset")
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {e}")
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

@router.post("/", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    delimiter: Optional[str] = Form(None),
    fmt_hint: Optional[str] = Form(None),
    current_user = Depends(get_current_active_user),
):
    return await _process_single_upload(file, name, description, delimiter, fmt_hint, current_user)


# 3) Optional: multi-file route
@router.post("/batch", status_code=status.HTTP_201_CREATED)
async def upload_dataset_batch(
    files: List[UploadFile] = File(...),
    meta_json: Optional[str] = Form(None),   # JSON mapping by index or filename
    current_user = Depends(get_current_active_user),
):
    metas = {}
    if meta_json:
        try:
            metas = json.loads(meta_json)
        except Exception:
            raise HTTPException(400, "meta_json must be valid JSON")

    results = []
    for idx, f in enumerate(files):
        m = metas.get(str(idx)) or metas.get(f.filename) or {}
        name       = m.get("name") or f.filename
        desc       = m.get("description")
        delimiter  = m.get("delimiter")
        fmt_hint   = m.get("fmt_hint")
        res = await _process_single_upload(f, name, desc, delimiter, fmt_hint, current_user)
        results.append(res)
    return {"items": results}

@router.get("/")
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
                        "created_at": d.created_at.date().isoformat() if d.created_at else None,
                        "rows": d.row_count,
                        "columns": d.column_count,
                        "description": d.description,
                    } for d in datasets
                ]
            }
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")

@router.get("/{dataset_id}")
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
from fastapi import Query, HTTPException, Depends
from sqlalchemy import MetaData, Table, select, inspect
from sqlalchemy.exc import NoSuchTableError, ProgrammingError, OperationalError
from sqlalchemy.sql.schema import quoted_name
# Updated dataset preview endpoint
def _try_preview_from_keys(keys, limit, *, user_id: int, dataset_id: int):
    for key in [k for k in keys if k]:
        tmp = None
        try:
            logger.info(f"Attempting to preview key: {key}")
            tmp = _download_object_to_temp(key, user_id=user_id, dataset_id=dataset_id)
            
            # Add file size check
            if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
                logger.warning(f"Preview: downloaded file is empty for key {key}")
                continue
                
            rows = fetch_preview_data(tmp)
            logger.info(f"Preview: fetch_preview_data returned type {type(rows)} for key {key}")
            
            if rows is None:
                logger.warning(f"Preview: fetch_preview_data returned None for key {key}")
                continue
            if not isinstance(rows, list):
                logger.warning(f"Preview: unexpected rows type {type(rows)} for key {key}")
                continue
            if len(rows) == 0:
                logger.warning(f"Preview: no rows found for key {key}")
                continue
                
            return rows[:limit], key
        except Exception as e:
            logger.error(f"Preview: failed reading {key}: {e}", exc_info=True)
        finally:
            if tmp and os.path.exists(tmp):
                try: 
                    os.remove(tmp)
                except Exception as cleanup_e: 
                    logger.warning(f"Failed to cleanup temp file {tmp}: {cleanup_e}")
    return None, None

def fetch_preview_data(file_path: str, max_rows: int = 100) -> list:
    """
    Safely read preview data from a CSV file.
    Returns a list of dictionaries, or empty list if file cannot be read.
    """
    if not file_path or not os.path.exists(file_path):
        logger.warning(f"Preview file does not exist: {file_path}")
        return []
    
    try:
        # Check file size first
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.warning(f"Preview file is empty: {file_path}")
            return []
        
        logger.info(f"Reading preview from {file_path} (size: {file_size} bytes)")
        
        # Try pandas first (more robust for various CSV formats)
        try:
            df = pd.read_csv(file_path, nrows=max_rows, encoding='utf-8')
            if df.empty:
                logger.warning(f"CSV file contains no data: {file_path}")
                return []
            logger.info(f"Successfully read {len(df)} rows with pandas")
            return df.fillna('').to_dict('records')
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(file_path, nrows=max_rows, encoding=encoding)
                    if df.empty:
                        continue
                    logger.info(f"Successfully read {len(df)} rows with pandas using {encoding} encoding")
                    return df.fillna('').to_dict('records')
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Pandas CSV read failed: {e}")
        
        # Fallback to standard csv module
        rows = []
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding, newline='') as f:
                    # Try to detect if file has headers
                    sample = f.read(1024)
                    f.seek(0)
                    
                    sniffer = csv.Sniffer()
                    has_header = sniffer.has_header(sample)
                    
                    reader = csv.DictReader(f) if has_header else csv.reader(f)
                    
                    for i, row in enumerate(reader):
                        if i >= max_rows:
                            break
                            
                        if isinstance(row, dict):
                            # DictReader case
                            rows.append(row)
                        else:
                            # Regular reader case - convert to dict
                            if i == 0 and not has_header:
                                # Create generic column names
                                headers = [f'col_{j}' for j in range(len(row))]
                                rows.append(dict(zip(headers, row)))
                            else:
                                # This shouldn't happen with proper logic above
                                continue
                
                if rows:
                    logger.info(f"Successfully read {len(rows)} rows with csv module using {encoding} encoding")
                    break
                    
            except Exception as e:
                logger.warning(f"CSV read failed with {encoding}: {e}")
                continue
        
        return rows[:max_rows]
        
    except Exception as e:
        logger.error(f"Failed to read CSV preview from {file_path}: {e}", exc_info=True)
        return []

@router.get("/{dataset_id}/preview")
def get_dataset_preview(
    dataset_id: int,
    limit: int = Query(20, gt=1, le=200),
    current_user = Depends(get_current_active_user),
):
    try:
        # Who am I (for storage keys)
        uid = getattr(current_user, "id", None) or getattr(current_user, "user_id", None)
        if not uid:
            raise HTTPException(status_code=401, detail="Unauthenticated")

        # Open the tenant DB and load the dataset row
        with get_user_db(current_user) as db:
            ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not ds:
                raise HTTPException(status_code=404, detail="Dataset not found")

            created_at_str = ds.created_at.strftime("%Y-%m-%d") if ds.created_at else "Unknown date"
            engine = db.bind  # keep a handle for reflection as a last resort

        # 1) PROCESSED FIRST (use either explicit processed path or current pointer)
        processed_keys = []
        if getattr(ds, "processed_file_path", None):
            processed_keys.append(ds.processed_file_path)
        if getattr(ds, "file_path", None):
            processed_keys.append(ds.file_path)
        # de-dup while preserving order
        seen = set()
        processed_keys = [k for k in processed_keys if k and not (k in seen or seen.add(k))]

        rows, _used_key = _try_preview_from_keys(processed_keys, limit, user_id=uid, dataset_id=dataset_id)
        if rows:
            return {"name": ds.name, "created_at": created_at_str, "preview_data": rows}

        # 2) ORIGINAL FALLBACK
        original_keys = []
        if getattr(ds, "original_file_path", None):
            original_keys.append(ds.original_file_path)
        original_keys = [k for k in original_keys if k and k not in processed_keys]

        rows, _used_key = _try_preview_from_keys(original_keys, limit, user_id=uid, dataset_id=dataset_id)
        if rows:
            return {"name": ds.name, "created_at": created_at_str, "preview_data": rows}

        # 3) DB REFLECTION (last resort)
        table_name = getattr(ds, "table_name", None)
        table_schema = getattr(ds, "table_schema", None) or getattr(ds, "schema", None)
        if table_name and engine is not None:
            md = MetaData()
            t = None
            try:
                t = Table(
                    quoted_name(table_name, True),
                    md,
                    schema=quoted_name(table_schema, True) if table_schema else None,
                    autoload_with=engine,
                )
            except (NoSuchTableError, ProgrammingError, OperationalError):
                t = None
            except Exception:
                t = None

            if t is None:
                # Probe other schemas if needed
                try:
                    insp = inspect(engine)
                    for sch in insp.get_schema_names():
                        if sch in ("pg_catalog", "information_schema"):
                            continue
                        if table_name in set(insp.get_table_names(schema=sch)):
                            t = Table(
                                quoted_name(table_name, True), md,
                                schema=quoted_name(sch, True),
                                autoload_with=engine,
                            )
                            break
                except Exception as e:
                    logger.warning(f"Schema probe failed in preview: {e}")

            if t is not None:
                with engine.connect() as conn:
                    result = conn.execute(select(t).limit(limit))
                    rows = [dict(r._mapping) for r in result]
                return {"name": ds.name, "created_at": created_at_str, "preview_data": rows}

        # Nothing available
        raise HTTPException(status_code=404, detail="File not available for preview")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset preview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting dataset preview: {e}")
@router.get("/{dataset_id}/data")
async def fetch_dataset_data(
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

@router.post("/{dataset_id}/query")
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

@router.get(
    "/{dataset_id}//download",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
)
async def download_dataset(
    dataset_id: int,
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

@router.post("/profile")
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
from typing import Literal, Dict, Any, List
import hashlib, json, os, shutil, tempfile, time
from uuid import uuid4
from fastapi import Query, HTTPException, Depends
def _grade_mapping(alias_hits, required_hits, ts_mean, num_mean, gaps_dict):
    score = 0
    score += 40 * min(alias_hits / max(1, required_hits), 1.0)
    score += 20 * float(ts_mean or 0.0)
    score += 20 * float(num_mean or 0.0)
    penalty = 0
    for r in (gaps_dict or {}).values():
        st = (r.get("status") or "").lower()
        crit = (r.get("criticality") or "").lower()
        if st == "missing" and crit == "high":
            penalty += 20
        elif st in ("missing","partial") and crit == "high":
            penalty += 10
    score = max(0, min(100, score - penalty))
    return ("A" if score>=93 else "A-" if score>=90 else "B+" if score>=87 else
            "B" if score>=83 else "B-" if score>=80 else "C+" if score>=77 else
            "C" if score>=73 else "C-" if score>=70 else "D")

def _build_intake_proof(result, base_currency: str) -> dict:
    meta = result.meta
    df_raw = result.df_raw
    df_norm = result.df_normalized

    # header metrics
    alias_hits = sum(1 for m in meta.header_map.values() if getattr(m, "canonical_hint", None))
    unit_tags = sorted({t for m in meta.header_map.values() for t in getattr(m, "tags", [])})
    # duplicates resolved
    uniq_norm_names = {m.normalized_name for m in meta.header_map.values()}
    duplicates_resolved = max(0, len(meta.header_map) - len(uniq_norm_names))

    # parse success means by semantic
    ts_rates, num_rates, bool_rates = [], [], []
    id_like_cols = []
    for col, ti in (meta.inferred_types or {}).items():
        vr = float(getattr(ti, "valid_rate", 0.0) or 0.0)
        sem = getattr(ti, "semantic", "")
        if sem == "timestamp": ts_rates.append(vr)
        if sem in ("numeric","money","percent"): num_rates.append(vr)
        if sem == "boolean": bool_rates.append(vr)
        if sem == "id": id_like_cols.append(col)

    ts_mean  = round(sum(ts_rates)/len(ts_rates), 4) if ts_rates else 0.0
    num_mean = round(sum(num_rates)/len(num_rates), 4) if num_rates else 0.0
    bool_mean= round(sum(bool_rates)/len(bool_rates),4) if bool_rates else 0.0

    # json expansion (your meta.json_columns is a dict of {col:[...]} or a list)
    if isinstance(meta.json_columns, dict):
        json_cols = list(meta.json_columns.keys())
    else:
        json_cols = list(meta.json_columns or [])
    new_cols_added = max(0, int(df_norm.shape[1]) - int(df_raw.shape[1]))

    # winsorization impact
    capped_counts = {}
    for c in df_norm.columns:
        if not c.endswith("_capped"): 
            continue
        base = c[:-7]  # remove "_capped"
        if base in df_norm.columns:
            try:
                lhs = pd.to_numeric(df_norm[c], errors="coerce")
                rhs = pd.to_numeric(df_norm[base], errors="coerce")
                capped_counts[c] = int((lhs.notna() & rhs.notna() & (lhs != rhs)).sum())
            except Exception:
                capped_counts[c] = None

    # derived numeric cols
    base_ccy = (base_currency or "USD").lower()
    derived_numeric = [c for c in df_norm.columns if c.endswith(("_num","_rate", f"_{base_ccy}"))]

    # gaps & assumptions (already computed in meta)
    gaps = meta.data_gaps or {}
    gap_summary = {
        "missing_high":   [k for k,v in gaps.items() if (v.get("status")=="missing" and v.get("criticality")=="high")],
        "missing_medium": [k for k,v in gaps.items() if (v.get("status")=="missing" and v.get("criticality")=="medium")],
        "present_but_sparse": [k for k,v in gaps.items() if v.get("status")=="partial"],
    }
    from harvester import EXPECTED_SIGNALS
    # grade
    required_hits = len(EXPECTED_SIGNALS)
    letter = _grade_mapping(alias_hits, required_hits, ts_mean, num_mean, gaps)

    return {
        "dialect": {
            "delimiter": meta.dialect.delimiter,
            "encoding": meta.dialect.encoding,
            "has_header": bool(meta.dialect.has_header),
            "number_format": {"decimal": meta.dialect.decimal, "thousands": meta.dialect.thousands},
            "delimiter_confidence": float(meta.dialect.confidence or 0.0),
        },
        "parsing": {
            "timestamp_parse_mean": ts_mean,
            "numeric_parse_mean":   num_mean,
            "boolean_parse_mean":   bool_mean,
        },
        "headers": {
            "alias_hits": int(alias_hits),
            "duplicates_resolved": int(duplicates_resolved),
            "unit_tags_detected": unit_tags,
        },
        "json_expansion": {
            "json_expanded_columns": json_cols,
            "new_columns_added": int(new_cols_added),
        },
        "standardization": {
            "base_currency": base_currency,
            "derived_numeric_columns": derived_numeric,
        },
        "winsorization": {
            "capped_columns": list(capped_counts.keys()),
            "rows_capped_per_column": capped_counts,
            "quantiles": {"lower_q": 0.01, "upper_q": 0.99},
        },
        "coverage": {
            "row_coverage": round(float(df_norm.shape[0])/max(1.0, float(df_raw.shape[0])), 4),
            "col_coverage": round(float(df_norm.shape[1])/max(1.0, float(df_raw.shape[1])), 4),
            "id_like_columns": id_like_cols,
        },
        "currency_markers": meta.currency_symbols or {},
        "gap_summary": gap_summary,
        "assumptions_applied": meta.assumptions or {},
        "anomaly_count": len(meta.anomalies or []),
        "anomalies": meta.anomalies or [],
        "mapping_quality": letter,
    }

@router.post("/{dataset_id}/intake")
def run_intake_and_normalization(
    dataset_id: int,
    base_currency: str = Query("USD"),
    country_hint: str = Query("US"),
    # ⬇️ new controls (defaults keep responses small)
    include_preview: bool = Query(False, description="Include tiny table samples"),
    preview_rows: int = Query(10, ge=1, le=100, description="If include_preview=true"),
    schema_cols: int = Query(12, ge=1, le=200, description="How many columns to sample in schema"),
    meta_level: Literal["none","summary","full"] = Query("summary"),
    current_user = Depends(get_current_active_user),
):
    """
    Runs Intake & Normalization and returns a concise receipt.
    No table data is returned unless include_preview=true.
    """
    t0 = time.perf_counter()
    corr_id = f"corr-{uuid4().hex[:8]}"

    user_id = getattr(current_user, "id", getattr(current_user, "user_id", None))
    if user_id is None:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    # 1) Resolve CSV
    try:
        hardcoded = os.getenv("HARDCODE_CSV")
        if hardcoded:
            src_csv_path = os.path.abspath(hardcoded)
        else:
            with get_user_db(current_user) as db:
                src_csv_path = resolve_and_cache_dataset_csv(
                    db=db, dataset_id=dataset_id, user_id=user_id
                )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to resolve dataset CSV: {e}")

    if not src_csv_path or not os.path.exists(src_csv_path):
        raise HTTPException(status_code=404, detail=f"Local CSV for dataset not found: {src_csv_path}")

    # 2) Intake
    try:
        result = intake_and_normalize(
            src_csv_path,
            base_currency=base_currency,
            country_hint=country_hint,
        )
    except Exception as e:
        logger.exception("Intake failed for dataset %s", dataset_id)
        raise HTTPException(status_code=500, detail=f"Intake & normalization failed: {type(e).__name__}: {e}")

    # 3) Persist canonical artifacts
    temp_dir = tempfile.mkdtemp(prefix=f"d{dataset_id}_intake_")
    meta_dict: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {}
    try:
        # normalized CSV
        norm_csv = os.path.join(temp_dir, f"dataset_{dataset_id}_normalized.csv")
        result.df_normalized.to_csv(norm_csv, index=False)
        norm_bytes = open(norm_csv, "rb").read()
        norm_sha256 = hashlib.sha256(norm_bytes).hexdigest()
        norm_size = len(norm_bytes)

        # meta json (full)
        meta_json_path = os.path.join(temp_dir, f"dataset_{dataset_id}_intake_meta.json")
        meta_dict = _meta_to_json(result.meta)
        meta_bytes = json.dumps(meta_dict, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        with open(meta_json_path, "wb") as f:
            f.write(meta_bytes)
        meta_sha256 = hashlib.sha256(meta_bytes).hexdigest()
        meta_size = len(meta_bytes)

        canon_norm_key = f"{dataset_id}/intake/normalized.csv"
        canon_meta_key = f"{dataset_id}/intake/intake_meta.json"
        legacy_dataset_meta_key = f"{dataset_id}/meta.json"

        supa_norm = upload_file_to_supabase(user_id=str(user_id), file_path=norm_csv,      filename=canon_norm_key)
        supa_meta = upload_file_to_supabase(user_id=str(user_id), file_path=meta_json_path, filename=canon_meta_key)
        try:
            upload_file_to_supabase(user_id=str(user_id), file_path=meta_json_path, filename=legacy_dataset_meta_key)
        except Exception:
            logger.exception("Legacy dataset meta.json copy failed for dataset %s", dataset_id)

        artifacts.update({
            "normalized_csv_key": f"{user_id}/{canon_norm_key}",
            "normalized_csv_sha256": norm_sha256,
            "normalized_csv_bytes": norm_size,
            "intake_meta_key": f"{user_id}/{canon_meta_key}",
            "intake_meta_sha256": meta_sha256,
            "intake_meta_bytes": meta_size,
            "legacy_meta_key": f"{user_id}/{legacy_dataset_meta_key}",
        })

        # Persist pointers on dataset row
        try:
            with get_user_db(current_user) as db:
                ds = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
                if ds:
                    ds.intake_meta_path = f"{user_id}/{canon_meta_key}"
                    ds.meta_path = ds.meta_path or f"{user_id}/{legacy_dataset_meta_key}"
                    db.add(ds); db.commit()
        except Exception:
            logger.exception("Failed to persist intake_meta_path for dataset %s", dataset_id)

    except Exception as e:
        logger.exception("Artifact upload failed for dataset %s", dataset_id)
        raise HTTPException(status_code=500, detail=f"Failed to persist artifacts: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 4) Build compact receipt payload
    stats_dict = {
        "raw_rows": int(result.df_raw.shape[0]),
        "raw_cols": int(result.df_raw.shape[1]),
        "normalized_rows": int(result.df_normalized.shape[0]),
        "normalized_cols": int(result.df_normalized.shape[1]),
    }
    proof = _build_intake_proof(result, base_currency=base_currency)
    # tiny schema sample (names + dtypes only)
    cols: List[str] = list(result.df_normalized.columns)[:schema_cols]
    schema_sample = [{"name": c, "dtype": str(result.df_normalized[c].dtype)} for c in cols]
    # type distribution
    type_dist: Dict[str, int] = {}
    for dtype, count in result.df_normalized.dtypes.value_counts().items():
        type_dist[str(dtype)] = int(count)

    # optional minimal preview (first few rows) — OFF by default
    preview_payload: Dict[str, Any] | None = None
    if include_preview:
        preview_payload = {
            "normalized": clean_json(_preview_table(result.df_normalized, n=preview_rows)),
            "raw": clean_json(_preview_table(result.df_raw, n=min(preview_rows, 10))),  # keep raw very small
        }

    # optional meta levels
    if meta_level == "none":
        meta_payload = None
    elif meta_level == "full":
        meta_payload = clean_json(meta_dict)
    else:  # "summary"
        meta_payload = {
            "number_format": meta_dict.get("number_format"),
            "timezone_inference": meta_dict.get("timezone_inference"),
            "anomaly_count": len(meta_dict.get("anomalies") or []),
            "json_columns": {k: len(v or []) for k, v in (meta_dict.get("json_columns") or {}).items()},
        }

    duration_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "ok": True,
        "message": "Intake & normalization succeeded",
        "correlation_id": corr_id,
        "dataset_id": dataset_id,
        "duration_ms": duration_ms,
        "stats": stats_dict,
        "schema": {
            "total_columns": stats_dict["normalized_cols"],
            "sample": schema_sample,
            "type_distribution": type_dist,
        },
        "artifacts": artifacts,
        "meta": meta_payload,            # unchanged
        "preview": preview_payload,      # unchanged
        "proof": (
            {k:v for k,v in proof.items() if k in [
                "dialect","parsing","headers","gap_summary","anomaly_count","mapping_quality"
            ]}
            if meta_level == "summary" else
            ({} if meta_level == "none" else proof)
        ),
    }


if __name__ == "__main__":
    init_db()
