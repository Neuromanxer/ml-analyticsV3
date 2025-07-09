from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import os
import shutil
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from database_models import get_db, register_dataset, get_all_datasets, get_dataset_by_id, get_dataset_data, delete_dataset

# Create a router
router = APIRouter()

# Pydantic models for request/response
class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    
class DatasetResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    table_name: str
    row_count: int
    column_count: int
    created_at: datetime
    
    class Config:
        orm_mode = True

# Directory for storing uploaded CSV files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload a CSV file and register it as a dataset
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are allowed"
        )
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Register the dataset in the database
    try:
        dataset = register_dataset(
            db=db,
            name=name,
            description=description,
            file_path=file_path,
            # For now, assume owner_id=1 or None until authentication is implemented
            owner_id=None
        )
        return dataset
    except Exception as e:
        # Clean up the file if registration fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register dataset: {str(e)}"
        )

@router.get("/datasets", response_model=List[DatasetResponse])
def list_datasets(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get a list of all datasets
    """
    datasets = get_all_datasets(db, skip=skip, limit=limit)
    return datasets

@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Get a dataset by ID including preview data
    """
    dataset_info = get_dataset_data(db, dataset_id)
    if not dataset_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    return dataset_info

@router.delete("/datasets/{dataset_id}")
def remove_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Delete a dataset and its associated data
    """
    success = delete_dataset(db, dataset_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    return {"message": "Dataset deleted successfully"}

@router.get("/datasets-page", response_class=HTMLResponse)
async def get_datasets_page(db: Session = Depends(get_db)):
    """
    Render the datasets HTML page
    """
    datasets = get_all_datasets(db)
    
    # Generate HTML for the page (this could be replaced with a template engine in a real app)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Insights - Datasets</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .upload-form {{
                border: 1px solid #ddd;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .dataset-card {{
                margin-bottom: 15px;
            }}
        </style>
    </head>
    <body>
        <div class="container mt-5">
            <h1 class="mb-4">Datasets Management</h1>
            
            <div class="upload-form">
                <h3>Upload New Dataset</h3>
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label for="name" class="form-label">Dataset Name</label>
                        <input type="text" class="form-control" id="name" name="name" required>
                    </div>
                    <div class="mb-3">
                        <label for="description" class="form-label">Description</label>
                        <textarea class="form-control" id="description" name="description" rows="3"></textarea>
                    </div>
                    <div class="mb-3">
                        <label for="file" class="form-label">CSV File</label>
                        <input class="form-control" type="file" id="file" name="file" accept=".csv" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Upload Dataset</button>
                </form>
                <div id="uploadStatus" class="alert mt-3" style="display: none;"></div>
            </div>
            
            <h3>Available Datasets</h3>
            <div id="datasetsList" class="row">
                <!-- Datasets will be loaded here -->
            </div>
        </div>
        
        <!-- Modal for Dataset Preview -->
        <div class="modal fade" id="datasetPreviewModal" tabindex="-1" aria-hidden="true">
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Dataset Preview</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div id="datasetInfo"></div>
                        <div class="table-responsive">
                            <table class="table table-striped" id="previewTable">
                                <thead id="previewTableHead"></thead>
                                <tbody id="previewTableBody"></tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Load datasets when page loads
            document.addEventListener('DOMContentLoaded', loadDatasets);
            
            // Handle form submission
            document.getElementById('uploadForm').addEventListener('submit', function(e) {
                e.preventDefault();
                uploadDataset();
            });
            
            function loadDatasets() {
                fetch('/datasets')
                    .then(response => response.json())
                    .then(datasets => {
                        const datasetsListEl = document.getElementById('datasetsList');
                        datasetsListEl.innerHTML = '';
                        
                        if (datasets.length === 0) {
                            datasetsListEl.innerHTML = '<div class="col-12"><p>No datasets available. Upload your first dataset using the form above.</p></div>';
                            return;
                        }
                        
                        datasets.forEach(dataset => {
                            const datasetCard = document.createElement('div');
                            datasetCard.className = 'col-md-4';
                            datasetCard.innerHTML = `
                                <div class="card dataset-card">
                                    <div class="card-body">
                                        <h5 class="card-title">${dataset.name}</h5>
                                        <h6 class="card-subtitle mb-2 text-muted">${dataset.row_count} rows, ${dataset.column_count} columns</h6>
                                        <p class="card-text">${dataset.description || 'No description provided'}</p>
                                        <p class="card-text"><small class="text-muted">Created: ${new Date(dataset.created_at).toLocaleString()}</small></p>
                                        <button class="btn btn-sm btn-primary preview-btn" data-id="${dataset.id}">Preview</button>
                                        <button class="btn btn-sm btn-danger delete-btn" data-id="${dataset.id}">Delete</button>
                                    </div>
                                </div>
                            `;
                            datasetsListEl.appendChild(datasetCard);
                        });
                        
                        // Add event listeners for preview buttons
                        document.querySelectorAll('.preview-btn').forEach(btn => {
                            btn.addEventListener('click', function() {
                                previewDataset(this.getAttribute('data-id'));
                            });
                        });
                        
                        // Add event listeners for delete buttons
                        document.querySelectorAll('.delete-btn').forEach(btn => {
                            btn.addEventListener('click', function() {
                                if (confirm('Are you sure you want to delete this dataset? This action cannot be undone.')) {
                                    deleteDataset(this.getAttribute('data-id'));
                                }
                            });
                        });
                    })
                    .catch(error => {
                        console.error('Error loading datasets:', error);
                    });
            }
            
            function uploadDataset() {
                const formData = new FormData(document.getElementById('uploadForm'));
                const statusEl = document.getElementById('uploadStatus');
                
                statusEl.style.display = 'block';
                statusEl.className = 'alert alert-info mt-3';
                statusEl.textContent = 'Uploading dataset... This may take a while for large files.';
                
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(data => {
                            throw new Error(data.detail || 'Upload failed');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    statusEl.className = 'alert alert-success mt-3';
                    statusEl.textContent = 'Dataset uploaded successfully!';
                    document.getElementById('uploadForm').reset();
                    loadDatasets();
                    setTimeout(() => {
                        statusEl.style.display = 'none';
                    }, 5000);
                })
                .catch(error => {
                    statusEl.className = 'alert alert-danger mt-3';
                    statusEl.textContent = 'Error: ' + error.message;
                });
            }
            
            function previewDataset(datasetId) {
                fetch(`/datasets/${datasetId}`)
                    .then(response => response.json())
                    .then(data => {
                        const modal = new bootstrap.Modal(document.getElementById('datasetPreviewModal'));
                        
                        // Populate dataset info
                        const datasetInfoEl = document.getElementById('datasetInfo');
                        datasetInfoEl.innerHTML = `
                            <h4>${data.dataset.name}</h4>
                            <p>${data.dataset.description || 'No description provided'}</p>
                            <p><strong>Rows:</strong> ${data.dataset.row_count} | <strong>Columns:</strong> ${data.dataset.column_count}</p>
                        `;
                        
                        // Populate table headers
                        const previewTableHeadEl = document.getElementById('previewTableHead');
                        previewTableHeadEl.innerHTML = '<tr>' + 
                            data.columns.map(col => `<th>${col}</th>`).join('') + 
                            '</tr>';
                        
                        // Populate table data
                        const previewTableBodyEl = document.getElementById('previewTableBody');
                        previewTableBodyEl.innerHTML = '';
                        
                        data.data.forEach(row => {
                            const tr = document.createElement('tr');
                            data.columns.forEach(col => {
                                const td = document.createElement('td');
                                td.textContent = row[col] !== null ? row[col] : 'null';
                                tr.appendChild(td);
                            });
                            previewTableBodyEl.appendChild(tr);
                        });
                        
                        modal.show();
                    })
                    .catch(error => {
                        console.error('Error previewing dataset:', error);
                        alert('Failed to load dataset preview');
                    });
            }
            
            function deleteDataset(datasetId) {
                fetch(`/datasets/${datasetId}`, {
                    method: 'DELETE'
                })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(data => {
                            throw new Error(data.detail || 'Delete failed');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    alert('Dataset deleted successfully');
                    loadDatasets();
                })
                .catch(error => {
                    console.error('Error deleting dataset:', error);
                    alert('Error: ' + error.message);
                });
            }
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)