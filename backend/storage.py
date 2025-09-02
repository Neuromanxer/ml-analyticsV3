import os
import tempfile
from supabase import create_client, Client
from fastapi import File, Form, UploadFile, HTTPException, Depends
from pathlib import Path as PathL
import aiofiles
import logging
from dotenv import load_dotenv
from typing import Optional, List, Tuple
from typing import Optional, Any, Dict, List
import logging
import shutil
logger = logging.getLogger(__name__)
load_dotenv()
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "user-uploads")

SUPABASE_PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF", "cosnptezznpqmzotozpt")
FULL_BUCKET_PREFIX   = f"{SUPABASE_PROJECT_REF}/storage/buckets/{SUPABASE_BUCKET}"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
import mimetypes
import os
import os
import mimetypes
from typing import Optional

# assumes you've created: supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "user-uploads")

import os
import mimetypes
from typing import Optional

SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "user-uploads")

def upload_file_to_supabase(
    user_id: str,
    file_path: str,
    filename: Optional[str] = None,
    dest_path: Optional[str] = None,
    *,
    upsert: bool = True,
    cache_control: str | int = "3600",
    content_type: Optional[str] = None,
) -> str:
    if not filename and dest_path:
        filename = dest_path
    if not filename:
        raise ValueError("upload_file_to_supabase: filename (or dest_path) is required")

    upload_path = f"{str(user_id).strip('/')}/{str(filename).lstrip('/')}"

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Local file not found: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"Refusing to upload zero-byte file: {file_path}")

    if content_type is None:
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # IMPORTANT: all values must be strings
    opts = {
        "content-type": str(content_type),
        "cache-control": str(cache_control),
        # Some supabase-py versions treat options like headers -> strings only
        "upsert": "true" if upsert else "false",
    }

    resp = supabase.storage.from_(SUPABASE_BUCKET).upload(upload_path, file_bytes, opts)

    # Normalize error extraction
    err = None
    if isinstance(resp, dict):
        err = resp.get("error")
    else:
        err = getattr(resp, "error", None) or getattr(resp, "message", None)

    if err:
        # Optional fallback: delete & re-upload if "already exists"
        emsg = str(err).lower()
        if upsert and ("already exists" in emsg or "resource already exists" in emsg or "409" in emsg):
            del_resp = supabase.storage.from_(SUPABASE_BUCKET).remove([upload_path])
            if isinstance(del_resp, dict) and del_resp.get("error"):
                raise Exception(f"Supabase remove failed for {upload_path}: {del_resp['error']}")
            resp2 = supabase.storage.from_(SUPABASE_BUCKET).upload(
                upload_path,
                file_bytes,
                {"content-type": str(content_type), "cache-control": str(cache_control)}
            )
            if isinstance(resp2, dict) and resp2.get("error"):
                raise Exception(f"Upload failed after delete for {upload_path}: {resp2['error']}")
            return upload_path
        raise Exception(f"Upload failed for {upload_path}: {err}")

    return upload_path


async def handle_file_upload(user_id: str, file: UploadFile) -> str:
    """Handle file upload to Supabase and return the upload path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:
        upload_path = upload_file_to_supabase(user_id, temp_path, file.filename)
        return upload_path
    finally:
        os.unlink(temp_path)


# def download_file_from_supabase(file_path: str) -> bytes:
#     response = supabase.storage.from_(SUPABASE_BUCKET).download(file_path)
#     if getattr(response, "error", None):
#         raise Exception(f"Download failed: {response.error.message}")
#     return response  # already bytes
def download_file_from_supabase(file_path: str) -> bytes:
    # Normalize Windows backslashes to forward slashes for Supabase
    clean_path = file_path.replace("\\", "/")

    # Perform the download
    response = supabase.storage.from_(SUPABASE_BUCKET).download(clean_path)

    # Raise error if download failed
    if getattr(response, "error", None):
        raise Exception(f"Download failed: {response.error.message}")

    return response  # already bytes


def list_user_files(user_id: str):
    response = supabase.storage.from_(SUPABASE_BUCKET).list(user_id)
    if getattr(response, "error", None):
        raise Exception(f"Failed to list files: {response.error.message}")


def delete_file_from_supabase(file_path: str):
    response = supabase.storage.from_(SUPABASE_BUCKET).remove([file_path])
    if isinstance(response, dict) and "error" in response and response["error"]:
        raise Exception(f"Failed to delete file: {response['error']['message']}")
    return response  # or just return True


def get_file_url(file_path: str, expires_in: int = 3600) -> str:
    response = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(file_path, expires_in)

    # If an error exists in the response (depends on client), handle it
    if isinstance(response, dict) and "error" in response and response["error"]:
        raise Exception(f"Failed to create signed URL: {response['error']['message']}")

    # ✅ Safely return the signed URL from the dict
    return response.get("signedURL", "")
def list_user_files(user_id: str):
    response = supabase.storage.from_(SUPABASE_BUCKET).list(user_id)
    if getattr(response, "error", None):
        raise Exception(f"Failed to list files: {response.error.message}")
    return response  # ✅ return the list of files
def delete_file_from_supabase(file_path: str):
    """Deletes a file from Supabase storage."""
    try:
        response = supabase.storage.from_(SUPABASE_BUCKET).remove([file_path])
        
        # Check if there was an error in the response
        if getattr(response, "error", None):
            raise Exception(f"Failed to delete file: {response.error.message}")

        # If no error, return a success message or a status
        return {"message": f"✅ File '{file_path}' deleted successfully."}

    except Exception as e:
        # Log the error and raise it
        logging.error(f"❌ Error deleting file '{file_path}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")
import os, tempfile, shutil
from typing import Optional, List, Tuple
from sqlalchemy import inspect
from sqlalchemy.exc import NoSuchTableError, ProgrammingError, OperationalError
from sqlalchemy.sql.schema import quoted_name


def ensure_dirs(p: str):
    os.makedirs(p, exist_ok=True)
# If you already defined this earlier, reuse it:
def user_dataset_root(user_id: int, dataset_id: int) -> str:
    return os.path.abspath(os.path.join(".", "data", "users", str(user_id), "datasets", str(dataset_id)))



def _strip_bucket_prefix(key: Optional[str]) -> Optional[str]:
    if not key:
        return key
    k = key.lstrip("/")
    if k.startswith(SUPABASE_BUCKET + "/"):
        return k[len(SUPABASE_BUCKET) + 1 :]
    return k

def _basename_from_key(key: str) -> str:
    return os.path.basename(_strip_bucket_prefix(key) or key or "")
def _intake_paths(dataset_id: int) -> dict:
    base = f"datasets/{dataset_id}/intake"
    return {
        "meta": f"{base}/meta.json",
        "stats": f"{base}/stats.json",
        "preview_norm": f"{base}/preview.normalized.json",
        "preview_raw": f"{base}/preview.raw.json",            # optional
        "artifacts": f"{base}/artifacts.json",                # optional
    }
import json
import numpy as np
import pandas as pd
from typing import Any

def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        # avoid NaN in JSON
        return None if (v != v) else v
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (pd.Series, pd.Index)):
        return o.tolist()
    if isinstance(o, (pd.DataFrame,)):
        return o.to_dict(orient="records")
    return str(o)

def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, ensure_ascii=False)

import tempfile, os

def save_intake_artifacts(
    *,
    dataset_id: int,
    meta: dict,
    stats: dict,
    preview: dict,          # {"raw": [...], "normalized": [...]}
    artifacts: dict = None  # optional; whatever you want to reference
) -> dict:
    """
    Writes JSON blobs to temp files and uploads them to Supabase under
    datasets/{id}/intake/*.json. Returns a dict of uploaded paths.
    """
    paths = _intake_paths(dataset_id)
    uploaded = {}

    # (1) meta.json
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(_dumps(meta).encode("utf-8"))
        tmp.flush()
        uploaded["meta"] = upload_file_to_supabase(file_path=tmp.name,
                                                   filename=os.path.basename(paths["meta"]),
                                                   user_id="system",  # or current_user.id
                                                   dest_path=paths["meta"])
    # (2) stats.json
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(_dumps(stats).encode("utf-8"))
        tmp.flush()
        uploaded["stats"] = upload_file_to_supabase(file_path=tmp.name,
                                                    filename=os.path.basename(paths["stats"]),
                                                    user_id="system",
                                                    dest_path=paths["stats"])
    # (3) preview.normalized.json
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        norm = preview.get("normalized") or []
        tmp.write(_dumps(norm).encode("utf-8"))
        tmp.flush()
        uploaded["preview_norm"] = upload_file_to_supabase(file_path=tmp.name,
                                                           filename=os.path.basename(paths["preview_norm"]),
                                                           user_id="system",
                                                           dest_path=paths["preview_norm"])

    # (4) preview.raw.json (optional)
    raw = preview.get("raw")
    if raw is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(_dumps(raw).encode("utf-8"))
            tmp.flush()
            uploaded["preview_raw"] = upload_file_to_supabase(file_path=tmp.name,
                                                              filename=os.path.basename(paths["preview_raw"]),
                                                              user_id="system",
                                                              dest_path=paths["preview_raw"])

    # (5) artifacts.json (optional)
    if artifacts is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(_dumps(artifacts).encode("utf-8"))
            tmp.flush()
            uploaded["artifacts"] = upload_file_to_supabase(file_path=tmp.name,
                                                            filename=os.path.basename(paths["artifacts"]),
                                                            user_id="system",
                                                            dest_path=paths["artifacts"])

    return uploaded
def _download_json(path: str) -> dict | list:
    # Your helper returns bytes
    blob = download_file_from_supabase(path)
    return json.loads(blob.decode("utf-8"))

def load_intake_artifacts(dataset_id: int) -> dict:
    """
    Returns:
    {
      "meta": {...},
      "stats": {...},
      "preview": {"normalized": [...], "raw": [...]},
      "artifacts": {...}     # optional
    }
    """
    paths = _intake_paths(dataset_id)

    # Required:
    meta = _download_json(paths["meta"])
    stats = _download_json(paths["stats"])
    preview_norm = _download_json(paths["preview_norm"])

    # Optional:
    preview_raw = []
    artifacts = {}

    try:
        preview_raw = _download_json(paths["preview_raw"])
    except Exception:
        pass

    try:
        artifacts = _download_json(paths["artifacts"])
    except Exception:
        pass

    return {
        "meta": meta,
        "stats": stats,
        "preview": {"normalized": preview_norm, "raw": preview_raw},
        "artifacts": artifacts,
    }
def delete_supabase_prefix(prefix: str) -> int:
    """Delete all objects under prefix in SUPABASE_BUCKET. Return count deleted."""
    sb = supabase
    if not sb:
        return 0
    try:
        storage = sb.storage.from_(SUPABASE_BUCKET)
        # List paginated; delete by chunks
        page = 0
        deleted = 0
        while True:
            res = storage.list(path=prefix, search="", limit=1000, offset=page * 1000)
            files = res or []
            if not files:
                break
            keys = [f"{prefix.rstrip('/')}/{obj['name']}" for obj in files if obj.get("name")]
            if keys:
                storage.remove(keys)
                deleted += len(keys)
            page += 1
        return deleted
    except Exception as e:
        logger.warning(f"[Supabase] Delete prefix '{prefix}' failed: {e}")
        return 0
# 5) Local artifacts
ARTIFACTS_ROOT = PathL("artifacts")  # matches your training code

def delete_local_artifacts_for_user(user_id: int) -> None:
    p = ARTIFACTS_ROOT / str(user_id)
    try:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    except Exception as e:
        logger.warning(f"[Artifacts] Failed removing {p}: {e}")