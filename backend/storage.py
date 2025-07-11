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
        {"content-type": "text/csv", "upsert": "true"} 
    )

    
    if response.get("error"):
        raise Exception(f"Upload failed: {response['error']['message']}")
    
    return upload_path
async def handle_file_upload(user_id: str, file: UploadFile) -> str:
    """Handle file upload to Supabase and return the upload path"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name
    
    try:
        # Upload to Supabase
        upload_path = upload_file_to_supabase(user_id, temp_path, file.filename)
        return upload_path
    finally:
        # Clean up temporary file
        os.unlink(temp_path)
def download_file_from_supabase(file_path: str) -> bytes:
    """Download a file from Supabase storage"""
    response = supabase.storage.from_(SUPABASE_BUCKET).download(file_path)
    if isinstance(response, dict) and response.get("error"):
        raise Exception(f"Download failed: {response['error']['message']}")
    return response

# Additional utility functions you might need:

def list_user_files(user_id: str):
    """List all files for a user in Supabase storage"""
    response = supabase.storage.from_(SUPABASE_BUCKET).list(user_id)
    if isinstance(response, dict) and response.get("error"):
        raise Exception(f"Failed to list files: {response['error']['message']}")
    return response

def delete_file_from_supabase(file_path: str):
    """Delete a file from Supabase storage"""
    response = supabase.storage.from_(SUPABASE_BUCKET).remove([file_path])
    if isinstance(response, dict) and response.get("error"):
        raise Exception(f"Failed to delete file: {response['error']['message']}")
    return response

def get_file_url(file_path: str, expires_in: int = 3600):
    """Get a signed URL for a file in Supabase storage"""
    response = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(file_path, expires_in)
    if isinstance(response, dict) and response.get("error"):
        raise Exception(f"Failed to create signed URL: {response['error']['message']}")
    return response.get("signedURL")
