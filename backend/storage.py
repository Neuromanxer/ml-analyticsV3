import os
import tempfile
from supabase import create_client, Client
from fastapi import File, Form, UploadFile, HTTPException, Depends
from pathlib import Path as PathL
import aiofiles

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "user-uploads")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
def upload_file_to_supabase(user_id: str, file_path: str, filename: str):
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    upload_path = f"{user_id}/{filename}"
    response = supabase.storage.from_(SUPABASE_BUCKET).upload(
        upload_path,
        file_bytes,
        {"content-type": "text/csv"}  # Only headers here
    )

    if getattr(response, "error", None):
        raise Exception(f"Upload failed: {response.error.message}")

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


def download_file_from_supabase(file_path: str) -> bytes:
    response = supabase.storage.from_(SUPABASE_BUCKET).download(file_path)
    if getattr(response, "error", None):
        raise Exception(f"Download failed: {response.error.message}")
    return response  # response is already bytes


def list_user_files(user_id: str):
    response = supabase.storage.from_(SUPABASE_BUCKET).list(user_id)
    if getattr(response, "error", None):
        raise Exception(f"Failed to list files: {response.error.message}")
    return response.data


def delete_file_from_supabase(file_path: str):
    response = supabase.storage.from_(SUPABASE_BUCKET).remove([file_path])
    if getattr(response, "error", None):
        raise Exception(f"Failed to delete file: {response.error.message}")
    return response.data


def get_file_url(file_path: str, expires_in: int = 3600):
    response = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(file_path, expires_in)
    if getattr(response, "error", None):
        raise Exception(f"Failed to create signed URL: {response.error.message}")
    
    # This is the fix
    return response.data.signed_url

