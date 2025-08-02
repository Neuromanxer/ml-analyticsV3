

from datetime import datetime, date
from typing import Optional, List, Literal

# FastAPI Router Implementation
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from io import StringIO
import logging
import secrets
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path as FPath,
    status,
)
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy import create_engine, Column, Integer, String, Float, Table, MetaData, Boolean, DateTime, ForeignKey, Text, text
  # <-- assumes your User is defined in auth.py
import os
import stripe
from fastapi import FastAPI, UploadFile, File, Form, Request, Depends, Path
from pydantic import BaseModel
from datetime import datetime

from .auth import get_current_active_user, get_master_db_session, User, Base
from .preprocessing import preprocess_data

# from auth import get_current_active_user, get_master_db_session, User, Base
# from preprocessing import preprocess_data


router = APIRouter(prefix="/api", tags=["api"])
class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    description = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="activity_logs")


# Add to User model
activity_logs = relationship("ActivityLog", back_populates="user", cascade="all, delete-orphan")
class ActivityLogResponse(BaseModel):
    title: str
    description: str
    timestamp: datetime

    class Config:
        orm_mode = True


@router.get("/activity", response_model=List[ActivityLogResponse])
def get_activity(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_master_db_session)):
    return (
        db.query(ActivityLog)
        .filter(ActivityLog.user_id == current_user.id)
        .order_by(ActivityLog.timestamp.desc())
        .limit(10)
        .all()
    )

def suggest_model_from_df(df, target_column: str):
    import numpy as np

    suggestions = []
    added_model_types = set()

    # Check target column type
    if target_column and target_column in df.columns:
        target_series = df[target_column].dropna()
        unique_vals = target_series.nunique()
        is_numeric = np.issubdtype(target_series.dtype, np.number)

        if is_numeric:
            if unique_vals == 2:
                model_type = "Classification Model"
                priority = "high"
            elif unique_vals < 10:
                model_type = "Classification Model"
                priority = "medium"
            else:
                model_type = "Regression Model"
                priority = "high"
        else:
            model_type = "Classification Model"
            priority = "medium"

        suggestions.append({
            "model_type": model_type,
            "icon": "🎯" if "Classification" in model_type else "📈",
            "description": "Automatically selected based on your target column",
            "priority": priority,
            "endpoint": f"/api/{model_type.lower().replace(' ', '_')}",
            "requirements": {
                "suggested_target": target_column,
                "target_column": target_column
            },
            "use_cases": ["Auto-selected", "Smart Detection"]
        })

        added_model_types.add(model_type)

    # Detect time column candidates
    time_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower() or "time" in col.lower():
            time_col = col
            break

    if time_col:
        model_type = "Time Series Analysis"
        suggestions.append({
            "model_type": model_type,
            "icon": "⏳",
            "description": f"Detected time-related column: '{time_col}'",
            "priority": "high" if target_column else "medium",
            "endpoint": "/api/forecast",
            "requirements": {
                "suggested_target": target_column or "value",
                "target_column": target_column or "value"
            },
            "use_cases": ["Demand Forecasting", "Trend Prediction"]
        })
        added_model_types.add(model_type)

    # Always include these baseline/general models if not added
    general_models = [
        {
            "model_type": "KMeans Clustering",
            "icon": "🔗",
            "description": "No target column needed. Clusters your dataset by similarity.",
            "endpoint": "/api/clustering",
            "requirements": {
                "suggested_target": "auto_detect",
                "target_column": "none"
            },
            "use_cases": ["Customer Segmentation", "Pattern Discovery"]
        },
        {
            "model_type": "Survival Analysis",
            "icon": "⚕️",
            "description": "Handles time-to-event data (e.g., churn, failure, dropout).",
            "endpoint": "/api/survival",
            "requirements": {
                "suggested_target": target_column or "event",
                "target_column": target_column or "event"
            },
            "use_cases": ["Churn Modeling", "Time-to-Failure"]
        },
        {
            "model_type": "Counterfactual Analysis",
            "icon": "🔄",
            "description": "Explore what-if scenarios and causal relationships.",
            "endpoint": "/api/counterfactual",
            "requirements": {
                "suggested_target": target_column or "outcome",
                "target_column": target_column or "outcome"
            },
            "use_cases": ["Policy Testing", "A/B Analysis"]
        },
        {
            "model_type": "Customer Segments",
            "icon": "📊",
            "description": "Uncover behavioral clusters and optimize marketing.",
            "endpoint": "/api/segment_analysis",
            "requirements": {
                "suggested_target": "auto_detect",
                "target_column": "n/a"
            },
            "use_cases": ["Retention", "Persona Discovery"]
        },
        {
            "model_type": "Deep Learning Model",
            "icon": "🧠",
            "description": "Complex nonlinear pattern detection using neural networks.",
            "endpoint": "/api/deep_learning",
            "requirements": {
                "suggested_target": target_column or "auto_detect",
                "target_column": target_column or "auto_detect"
            },
            "use_cases": ["Image/NLP", "High-Dimensional Modeling"]
        }
    ]

    # Add remaining general models with default low priority
    for model in general_models:
        if model["model_type"] not in added_model_types:
            model["priority"] = "low"
            suggestions.append(model)

    return {
        "all_model_options": suggestions,
        "data_shape": df.shape,
        "missing_values": int(df.isnull().sum().sum()),
        "numeric_columns": int(len(df.select_dtypes(include=['number']).columns)),
        "categorical_columns": int(len(df.select_dtypes(include=['object', 'category']).columns))
    }


def make_json_serializable(obj):
    """Convert numpy/pandas types to Python native types for JSON serialization."""
    import numpy as np

    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    else:
        return obj

@router.post("/suggest-model")
async def suggest_model(
    file: UploadFile = File(...),
    target_column: str = Form(default="")
):
    """
    Analyze uploaded dataset and suggest multiple model options with configurations.
    
    Args:
        file: CSV file to analyze
        target_column: Optional target column name for supervised learning
        
    Returns:
        JSON with primary suggestion, all model options, and data characteristics
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read and parse CSV
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
        
        # Basic data validation
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        if len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="Dataset must have at least 2 columns")
        
        # Get model suggestions
        df = pd.read_csv("your_data.csv")
        df_cleaned, CATS, NUMS = preprocess_data(df, RMV=[])


        result = suggest_model_from_df(df_cleaned, target_column)
        
        # Add filename
        result["filename"] = file.filename
        
        # Make sure all values are JSON serializable
        result = make_json_serializable(result)
        
        return JSONResponse(content=result)
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or corrupted")
    
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8 encoded CSV")
    
    except Exception as e:
        logging.error(f"Model suggestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/ai_recommendations")
async def get_ai_recommendations(
    file: UploadFile = File(...),
    goal: str = Form(...),
    dataType: str = Form(...),
    target: Optional[str] = Form(None),
    success: Optional[str] = Form(None),
    timeColumn: Optional[str] = Form(None),
    user_plan: str = Form("Free"),
    current_user: User = Depends(get_current_active_user)
):
    try:
        import io, json, openai, numpy as np
        from fastapi import HTTPException
        import pandas as pd
        MINIMUM_TOKENS = 0.1
        if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
            raise HTTPException(403, detail="Insufficient token balance to begin processing.")

        # Read the uploaded file
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        # ───────────── Free Plan Fallback ─────────────
        if user_plan.lower() == "free":
            model_suggestion = suggest_model_from_df(df, target or "")
            return {
                "suggested_model_type": model_suggestion["suggested_model_type"],
                "notes": model_suggestion["notes"],
                "target_dtype": model_suggestion.get("target_dtype"),
                "unique_values": model_suggestion.get("unique_values")
            }

        # ───────────── Pro Plan AI Recommendation ─────────────
        data_shape = df.shape
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        datetime_columns = []

        for col in df.columns:
            try:
                pd.to_datetime(df[col].dropna().head(100))
                datetime_columns.append(col)
            except:
                continue

        target_info = ""
        if target and target in columns:
            target_series = df[target]
            if target_series.dtype in ['object', 'category']:
                unique_values = target_series.nunique()
                target_info = f"Target '{target}' is categorical with {unique_values} unique values"
            else:
                target_info = f"Target '{target}' is numeric with range [{target_series.min():.2f}, {target_series.max():.2f}]"

        prompt = f"""
        As a data science advisor, analyze this dataset and recommend the best analysis approach:

        DATASET INFORMATION:
        - Shape: {data_shape[0]} rows, {data_shape[1]} columns
        - Columns: {', '.join(columns)}
        - Numeric columns: {', '.join(numeric_columns)}
        - Categorical columns: {', '.join(categorical_columns)}
        - Datetime columns: {', '.join(datetime_columns)}
        - Target info: {target_info}

        USER REQUIREMENTS:
        - Goal: {goal}
        - Data Type: {dataType}
        - Target Column: {target or 'none specified'}
        - Success Criteria: {success or 'none specified'}
        - Time Column: {timeColumn or 'none specified'}

        AVAILABLE ANALYSIS PATHS:
        - /classification/
        - /regression/
        - /clustering/
        - /forecast/
        - /segment_analysis/
        - /survival/
        - /counterfactual/

        Please respond in JSON with keys:
        {{
            "recommended_path": "...",
            "target_variable": "...",
            "drop_columns": [...],
            "time_column": "...",
            "forecast_periods": ...,
            "impactful_features": [...],
            "reasoning": "...",
            "confidence": "...",
            "alternative_approaches": [...]
        }}
        """

        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data science advisor. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        token_usage = completion.usage.total_tokens  # Total = prompt + completion
                # Estimate OpenAI cost (adjust rate if you know your real cost)
        openai_cost_per_1k_tokens = 0.04  # Approximate for GPT-4-turbo
        actual_cost_usd = (token_usage / 1000) * openai_cost_per_1k_tokens

        # Apply your profit margin
        markup_multiplier = 4.0
        user_token_cost = round(actual_cost_usd * markup_multiplier, 3)  # e.g., $0.05 * 4 = 0.20

        ai_response = completion.choices[0].message.content.strip()
        if ai_response.startswith("```json"):
            ai_response = ai_response.split("```json")[1].split("```")[0].strip()
        elif ai_response.startswith("```"):
            ai_response = ai_response.split("```")[1].split("```")[0].strip()

        recommendation = json.loads(ai_response)
        for key in ["recommended_path", "target_variable", "drop_columns", "time_column",
                    "forecast_periods", "impactful_features", "reasoning", "confidence", "alternative_approaches"]:
            recommendation.setdefault(key, None)
        if not isinstance(recommendation.get("drop_columns"), list):
            recommendation["drop_columns"] = []
        if not isinstance(recommendation.get("impactful_features"), list):
            recommendation["impactful_features"] = []
        if not isinstance(recommendation.get("alternative_approaches"), list):
            recommendation["alternative_approaches"] = []

        return {
            "ai_recommendation": recommendation,
            "dataset_summary": {
                "shape": data_shape,
                "columns": columns,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "datetime_columns": datetime_columns,
                "missing_values": {k: int(v) for k, v in df.isnull().sum().items() if v > 0}
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Any, Dict
from openai import OpenAI

client = OpenAI(api_key="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")  # This auto-uses OPENAI_API_KEY from env

from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class AIInsightsRequest(BaseModel):
    id: str
    type: str  # e.g. "classification", "regression", etc.
    dataset: str
    created_at: str

    # Optional analysis-specific keys
    target_column: Optional[str] = None
    target_col: Optional[str] = None  # Some entries use this instead
    time_column: Optional[str] = None
    event_column: Optional[str] = None

    parameters: Optional[Dict[str, Any]] = {}
    test_scores: Optional[Dict[str, Any]] = {}
    cv_scores: Optional[Dict[str, Any]] = {}
    model_info: Optional[Dict[str, Any]] = {}

    top_features: Optional[List[Dict[str, Any]]] = []
    business_metrics: Optional[Dict[str, float]] = {}
    impact_metrics: Optional[Dict[str, Any]] = {}

    thumbnailData: Optional[str] = None
    imageData: Optional[str] = None
    visualizations: Optional[Dict[str, str]] = {}

    # Prediction-related
    metrics: Optional[Dict[str, Any]] = {}
    conversion_rate: Optional[float] = None
    output_file: Optional[str] = None
    output_file_name: Optional[str] = None
    total_rows: Optional[int] = None
    columns_added: Optional[List[str]] = []

    # Segmentation-related
    segments_summary: Optional[List[Dict[str, Any]]] = []
    cluster_stats: Optional[Dict[str, Any]] = {}
    cluster_counts: Optional[Dict[str, int]] = {}
    cluster_feature_differences: Optional[Dict[str, Any]] = {}

    # Feature alignment info
    feature_alignment: Optional[Dict[str, Any]] = {}
from fastapi import APIRouter, Depends, HTTPException, Request

@router.post("/ai_insights")
async def generate_ai_insights(
    data: AIInsightsRequest,
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    MINIMUM_TOKENS = 0.1
    if current_user.tokens is None or current_user.tokens < MINIMUM_TOKENS:
        raise HTTPException(403, detail="Insufficient token balance to begin processing.")

    try:
        messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"""
You're an expert business strategist. Analyze this machine learning result and provide a concise summary of **what actions the business owner should take** to maximize profits, improve conversions, or reduce losses.

Avoid technical jargon unless needed. Focus on what the data implies for the business, and make your suggestions practical and immediately actionable.

Here are the inputs:

• Type: {data.type}
• Dataset: {data.dataset}
• Target Column: {data.target_column or data.time_column}
• Key Parameters: {data.parameters}
• Test Scores: {data.test_scores}
• Cross-Validation Scores: {data.cv_scores}
• Top Influential Features: {data.top_features}
• Business Metrics:
    - Revenue per Customer: ${data.business_metrics.get('revenue_per_customer')}
    - Cost per Lost Customer: ${data.business_metrics.get('cost_per_lost_customer')}
    - Revenue per Lead: ${data.business_metrics.get('revenue_per_lead')}
    - Average Order Value: ${data.business_metrics.get('average_order_value')}

Start your response with a short executive summary. Then give 3-5 **actionable recommendations** based on the data. Include expected impact (e.g., “could improve revenue by X” or “reduce churn by Y%”) if possible.
"""
            }
        ]
    }
]


        # Optional: include image if available
        if data.imageData and data.imageData.startswith("data:image"):
            base64_content = data.imageData.split(",")[-1]
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_content}"
                }
            })

        # Generate response
        response = client.chat.completions.create(
            model= "gpt-4.1",
            messages=messages,
            temperature=0.7,
            max_tokens=700
        )

        insights = response.choices[0].message.content.strip()

        # Optionally track token usage
        if hasattr(response, "usage"):
            request.state.openai_tokens_used = response.usage.total_tokens

        return {
            "entry_id": data.id,
            "insights": insights
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {str(e)}")