import os
import tempfile
from supabase import create_client, Client
from fastapi import File, Form, UploadFile, HTTPException, Depends
from pathlib import Path as PathL
import aiofiles
import logging
from dotenv import load_dotenv
from typing import Optional, List, Tuple
load_dotenv()
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "user-uploads")

SUPABASE_PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF", "cosnptezznpqmzotozpt")
FULL_BUCKET_PREFIX   = f"{SUPABASE_PROJECT_REF}/storage/buckets/{SUPABASE_BUCKET}"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
import mimetypes
import os
def upload_file_to_supabase(user_id: str, file_path: str, filename: str) -> str:
    upload_path = f"{user_id}/{filename}"
    print(f"[upload_file_to_supabase] user_id={user_id!r}, filename={filename!r}, key={upload_path!r}")
    # Step 1: Check if file exists
    list_response = supabase.storage.from_(SUPABASE_BUCKET).list(user_id)
    existing_file_names = [f["name"] for f in list_response or []]

    if filename in existing_file_names:
        delete_response = supabase.storage.from_(SUPABASE_BUCKET).remove([upload_path])
        if isinstance(delete_response, dict) and delete_response.get("error"):
            raise Exception(f"Failed to delete existing file: {delete_response['error']['message']}")

    # Step 2: Upload
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    upload_response = supabase.storage.from_(SUPABASE_BUCKET).upload(
        upload_path,
        file_bytes,
        {"content-type": content_type}
    )

    # ✅ FIX: Safely check for error in dict response
    if isinstance(upload_response, dict) and upload_response.get("error"):
        raise Exception(f"Upload failed: {upload_response['error']['message']}")

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
