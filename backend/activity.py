

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

# from .auth import get_current_active_user, get_master_db_session, User, Base
# from .preprocessing import preprocess_data

from auth import get_current_active_user, get_master_db_session, User, Base
from preprocessing import preprocess_data


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
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
import logging
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

class WhatIfInputRequest(BaseModel):
    """
    Request model for what-if analysis input processing.
    
    This model is used by the /what_if/inputs/ endpoint to extract and analyze 
    features suitable for what-if analysis from column statistics.
    """
    column_stats: Dict[str, List[Any]]
    sample_row: Optional[Dict[str, Any]] = None

class WhatIfPredictionRequest(BaseModel):
    """
    Request model for what-if prediction analysis with mass column adjustments.
    
    This model handles bulk changes to entire columns rather than individual rows.
    """
    target_column: str
    drop_columns: Optional[List[str]] = []
    feature_changes: Dict[str, Dict[str, Any]]  # Mass column adjustments
    original_data: List[Dict[str, Any]]  # The original dataset to modify

class FeatureType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical" 
    BOOLEAN = "boolean"

class AdjustmentType(Enum):
    PERCENT = "percent"
    DISCRETE = "discrete"
    BOOST_CATEGORY = "boost_category"
    TOGGLE = "toggle"
    RANGE = "range"

@dataclass
class FeatureInfo:
    name: str
    label: str
    type: FeatureType
    adjustment_type: AdjustmentType
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    options: Optional[List[Any]] = None
    default: Optional[Any] = None
    adjustment_values: Optional[List[Union[int, float]]] = None
    recommendation_note: Optional[str] = None
    null_percentage: Optional[float] = None

class WhatIfProcessor:
    """Handles mass column adjustments for what-if analysis."""
    
    def __init__(self):
        self.random_state = np.random.RandomState(42)  # For reproducible results
    
    def apply_mass_changes(self, df: pd.DataFrame, feature_changes: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply mass changes to entire columns based on feature_changes configuration.
        
        Args:
            df: Original DataFrame
            feature_changes: Dictionary of column names to change configurations
            
        Returns:
            Modified DataFrame with changes applied
        """
        modified_df = df.copy()
        
        for column_name, change_config in feature_changes.items():
            if column_name not in modified_df.columns:
                logger.warning(f"Column '{column_name}' not found in dataset")
                continue
                
            try:
                modified_df[column_name] = self._apply_column_change(
                    modified_df[column_name], 
                    change_config
                )
                logger.info(f"Applied {change_config['type']} change to column '{column_name}'")
                
            except Exception as e:
                logger.error(f"Failed to apply change to column '{column_name}': {str(e)}")
                # Continue with other columns even if one fails
                continue
        
        return modified_df
    
    def _apply_column_change(self, series: pd.Series, change_config: Dict[str, Any]) -> pd.Series:
        """Apply a specific change configuration to a pandas Series."""
        
        change_type = change_config.get('type')
        
        if change_type == 'percentage':
            return self._apply_percentage_change(series, change_config)
        elif change_type == 'fixed':
            return self._apply_fixed_change(series, change_config)
        elif change_type == 'boost_category':
            return self._apply_category_boost(series, change_config)
        elif change_type == 'force_boolean':
            return self._apply_boolean_force(series, change_config)
        elif change_type == 'range_scale':
            return self._apply_range_scale(series, change_config)
        else:
            logger.warning(f"Unknown change type: {change_type}")
            return series
    
    def _apply_percentage_change(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Apply percentage change to numeric column."""
        percentage = config.get('value', 0)
        if percentage == 0:
            return series
        
        # Only apply to numeric, non-null values
        mask = pd.to_numeric(series, errors='coerce').notna()
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        # Apply percentage change
        multiplier = 1 + (percentage / 100)
        modified_series = series.copy()
        modified_series[mask] = numeric_series[mask] * multiplier
        
        return modified_series
    
    def _apply_fixed_change(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Apply fixed value change to numeric column."""
        fixed_value = config.get('value', 0)
        if fixed_value == 0:
            return series
        
        # Only apply to numeric, non-null values
        mask = pd.to_numeric(series, errors='coerce').notna()
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        # Apply fixed change
        modified_series = series.copy()
        modified_series[mask] = numeric_series[mask] + fixed_value
        
        return modified_series
    
    def _apply_range_scale(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Scale values to fit within a new range."""
        new_min = config.get('new_min')
        new_max = config.get('new_max')
        
        if new_min is None or new_max is None:
            return series
        
        # Only apply to numeric, non-null values
        mask = pd.to_numeric(series, errors='coerce').notna()
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        if not mask.any():
            return series
        
        # Get current min/max of non-null values
        current_min = numeric_series[mask].min()
        current_max = numeric_series[mask].max()
        
        if current_min == current_max:
            # All values are the same, set to middle of new range
            modified_series = series.copy()
            modified_series[mask] = (new_min + new_max) / 2
            return modified_series
        
        # Scale to new range
        modified_series = series.copy()
        scaled_values = (numeric_series[mask] - current_min) / (current_max - current_min)
        modified_series[mask] = scaled_values * (new_max - new_min) + new_min
        
        return modified_series
    
    def _apply_category_boost(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Boost frequency of a specific category in categorical column."""
        target_category = config.get('category')
        boost_percentage = config.get('boost_percentage', 0)
        
        if not target_category or boost_percentage <= 0:
            return series
        
        modified_series = series.copy()
        non_null_mask = series.notna()
        
        if not non_null_mask.any():
            return series
        
        # Calculate how many additional instances we need
        current_count = (series == target_category).sum()
        total_non_null = non_null_mask.sum()
        
        # Calculate target count based on boost percentage
        additional_needed = int((boost_percentage / 100) * total_non_null)
        
        if additional_needed <= 0:
            return series
        
        # Find non-target category instances to convert
        non_target_mask = (series != target_category) & non_null_mask
        non_target_indices = series[non_target_mask].index.tolist()
        
        if len(non_target_indices) == 0:
            return series
        
        # Randomly select indices to convert
        convert_count = min(additional_needed, len(non_target_indices))
        indices_to_convert = self.random_state.choice(
            non_target_indices, 
            size=convert_count, 
            replace=False
        )
        
        # Apply the conversion
        modified_series.loc[indices_to_convert] = target_category
        
        return modified_series
    
    def _apply_boolean_force(self, series: pd.Series, config: Dict[str, Any]) -> pd.Series:
        """Force all values in boolean column to specific value."""
        target_value = config.get('value')
        
        if target_value is None:
            return series
        
        # Convert to boolean and apply to non-null values
        modified_series = series.copy()
        non_null_mask = series.notna()
        modified_series[non_null_mask] = target_value
        
        return modified_series
    
    def generate_change_summary(self, original_df: pd.DataFrame, 
                               modified_df: pd.DataFrame, 
                               feature_changes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of what changes were applied."""
        summary = {
            'total_changes': len(feature_changes),
            'rows_affected': len(modified_df),
            'changes_applied': {}
        }
        
        for column_name, change_config in feature_changes.items():
            if column_name not in original_df.columns:
                continue
                
            change_summary = {
                'type': change_config.get('type'),
                'configuration': change_config
            }
            
            # Add before/after statistics
            if column_name in modified_df.columns:
                orig_col = original_df[column_name]
                mod_col = modified_df[column_name]
                
                if pd.api.types.is_numeric_dtype(orig_col):
                    change_summary['before'] = {
                        'mean': float(orig_col.mean()) if orig_col.notna().any() else None,
                        'median': float(orig_col.median()) if orig_col.notna().any() else None,
                        'std': float(orig_col.std()) if orig_col.notna().any() else None
                    }
                    change_summary['after'] = {
                        'mean': float(mod_col.mean()) if mod_col.notna().any() else None,
                        'median': float(mod_col.median()) if mod_col.notna().any() else None,
                        'std': float(mod_col.std()) if mod_col.notna().any() else None
                    }
                else:
                    # For categorical data, show value counts
                    change_summary['before'] = orig_col.value_counts().head(5).to_dict()
                    change_summary['after'] = mod_col.value_counts().head(5).to_dict()
            
            summary['changes_applied'][column_name] = change_summary
        
        return summary

class FeatureAnalyzer:
    """Handles analysis and classification of features for what-if scenarios."""
    
    # Configurable thresholds
    MIN_NON_NULL_RATIO = 0.2
    MIN_NON_NULL_COUNT = 3
    MAX_CATEGORICAL_UNIQUE = 50
    NUMERIC_UNIQUE_THRESHOLD = 20
    OUTLIER_QUANTILES = (0.05, 0.95)
    
    # Reserved column names to exclude
    RESERVED_COLUMNS = {"id", "target", "label", "prediction", "score"}
    
    def __init__(self, min_non_null_ratio: float = 0.2, 
                 max_categorical_unique: int = 50):
        self.MIN_NON_NULL_RATIO = min_non_null_ratio
        self.MAX_CATEGORICAL_UNIQUE = max_categorical_unique
    
    def analyze_features(self, column_stats: Dict[str, List[Any]]) -> List[FeatureInfo]:
        """Analyze all features and return list of editable features."""
        features = []
        
        for name, values in column_stats.items():
            feature_info = self.analyze_feature(name, values)
            if feature_info:
                features.append(feature_info)
        
        return features
    
    def analyze_feature(self, name: str, values: List[Any]) -> Optional[FeatureInfo]:
        """Analyze a single feature and return FeatureInfo if suitable for mass editing."""
        try:
            if not self._is_editable_feature(name):
                return None
            
            non_nulls = self._get_non_null_values(values)
            if not self._has_sufficient_data(non_nulls, len(values)):
                return None
            
            null_percentage = (len(values) - len(non_nulls)) / len(values) * 100
            feature_type = self._determine_feature_type(non_nulls)
            
            if feature_type == FeatureType.NUMERIC:
                return self._analyze_numeric_feature(name, non_nulls, null_percentage)
            elif feature_type == FeatureType.CATEGORICAL:
                return self._analyze_categorical_feature(name, non_nulls, null_percentage)
            elif feature_type == FeatureType.BOOLEAN:
                return self._analyze_boolean_feature(name, non_nulls, null_percentage)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to analyze feature '{name}': {str(e)}")
            return None
    
    def _is_editable_feature(self, name: str) -> bool:
        """Check if feature name suggests it should be editable."""
        return name.lower() not in self.RESERVED_COLUMNS
    
    def _get_non_null_values(self, values: List[Any]) -> List[Any]:
        """Extract non-null values, handling various null representations."""
        return [v for v in values if pd.notnull(v) and v is not None 
                and str(v).lower() not in ['', 'nan', 'null', 'none']]
    
    def _has_sufficient_data(self, non_nulls: List[Any], total_count: int) -> bool:
        """Check if feature has sufficient non-null data for analysis."""
        if len(non_nulls) < self.MIN_NON_NULL_COUNT:
            return False
        
        non_null_ratio = len(non_nulls) / total_count if total_count > 0 else 0
        return non_null_ratio >= self.MIN_NON_NULL_RATIO
    
    def _determine_feature_type(self, non_nulls: List[Any]) -> FeatureType:
        """Determine the type of feature based on its values."""
        # Try to convert to numeric
        numeric_values = []
        for val in non_nulls:
            try:
                numeric_val = pd.to_numeric(val)
                if not pd.isna(numeric_val):
                    numeric_values.append(numeric_val)
            except (ValueError, TypeError):
                pass
        
        # If most values are numeric, treat as numeric
        if len(numeric_values) / len(non_nulls) > 0.8:
            return FeatureType.NUMERIC
        
        # Check for boolean
        unique_vals = set(str(v).lower() for v in non_nulls)
        boolean_indicators = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}
        if unique_vals.issubset(boolean_indicators):
            return FeatureType.BOOLEAN
        
        # Check if suitable for categorical
        unique_count = len(set(non_nulls))
        if unique_count <= self.MAX_CATEGORICAL_UNIQUE:
            return FeatureType.CATEGORICAL
        
        # Default to categorical for non-numeric with reasonable unique count
        return FeatureType.CATEGORICAL if unique_count <= 100 else FeatureType.NUMERIC
    
    def _analyze_numeric_feature(self, name: str, non_nulls: List[Any], 
                                null_percentage: float) -> FeatureInfo:
        """Analyze numeric feature and create FeatureInfo."""
        numeric_vals = [pd.to_numeric(v) for v in non_nulls if pd.notnull(pd.to_numeric(v, errors='coerce'))]
        
        if not numeric_vals:
            return None
        
        stats = {
            'mean': float(np.mean(numeric_vals)),
            'median': float(np.median(numeric_vals)),
            'std': float(np.std(numeric_vals)),
            'min': float(np.min(numeric_vals)),
            'max': float(np.max(numeric_vals))
        }
        
        # Determine appropriate adjustment values for percentage changes
        adjustment_values = [-50, -25, -10, -5, 0, 5, 10, 25, 50, 100]
        
        # Create recommendation note
        recommendation_note = f"Current mean: {stats['mean']:.2f}, Range: {stats['min']:.2f} to {stats['max']:.2f}"
        
        return FeatureInfo(
            name=name,
            label=self._create_readable_label(name),
            type=FeatureType.NUMERIC,
            adjustment_type=AdjustmentType.PERCENT,
            min_val=stats['min'],
            max_val=stats['max'],
            mean=stats['mean'],
            median=stats['median'],
            std=stats['std'],
            adjustment_values=adjustment_values,
            recommendation_note=recommendation_note,
            null_percentage=null_percentage
        )
    
    def _analyze_categorical_feature(self, name: str, non_nulls: List[Any], 
                                   null_percentage: float) -> FeatureInfo:
        """Analyze categorical feature and create FeatureInfo."""
        unique_vals = list(set(non_nulls))
        value_counts = pd.Series(non_nulls).value_counts()
        
        # Get most common category as default
        most_common = value_counts.index[0] if len(value_counts) > 0 else unique_vals[0]
        
        recommendation_note = f"Most common: {most_common} ({value_counts.iloc[0]} occurrences)"
        
        return FeatureInfo(
            name=name,
            label=self._create_readable_label(name),
            type=FeatureType.CATEGORICAL,
            adjustment_type=AdjustmentType.BOOST_CATEGORY,
            options=unique_vals,
            default=most_common,
            recommendation_note=recommendation_note,
            null_percentage=null_percentage
        )
    
    def _analyze_boolean_feature(self, name: str, non_nulls: List[Any], 
                               null_percentage: float) -> FeatureInfo:
        """Analyze boolean feature and create FeatureInfo."""
        # Normalize boolean values
        true_indicators = {'true', '1', 'yes', 'y'}
        false_indicators = {'false', '0', 'no', 'n'}
        
        normalized_vals = []
        for val in non_nulls:
            str_val = str(val).lower()
            if str_val in true_indicators:
                normalized_vals.append(True)
            elif str_val in false_indicators:
                normalized_vals.append(False)
        
        true_count = sum(normalized_vals)
        false_count = len(normalized_vals) - true_count
        
        recommendation_note = f"Current: {true_count} True, {false_count} False"
        
        return FeatureInfo(
            name=name,
            label=self._create_readable_label(name),
            type=FeatureType.BOOLEAN,
            adjustment_type=AdjustmentType.TOGGLE,
            options=[True, False],
            default=true_count > false_count,
            recommendation_note=recommendation_note,
            null_percentage=null_percentage
        )
    
    def _create_readable_label(self, name: str) -> str:
        """Convert column name to readable label."""
        # Replace underscores and capitalize
        label = name.replace('_', ' ').replace('-', ' ')
        return ' '.join(word.capitalize() for word in label.split())

def _generate_scenario_description(feature_changes: Dict[str, Dict[str, Any]]) -> str:
    """Generate a human-readable description of the scenario."""
    descriptions = []
    
    for column_name, change_config in feature_changes.items():
        change_type = change_config.get('type')
        
        if change_type == 'percentage':
            percentage = change_config.get('value', 0)
            if percentage > 0:
                descriptions.append(f"{column_name} increased by {percentage}%")
            elif percentage < 0:
                descriptions.append(f"{column_name} decreased by {abs(percentage)}%")
        
        elif change_type == 'fixed':
            value = change_config.get('value', 0)
            if value > 0:
                descriptions.append(f"{column_name} increased by {value}")
            elif value < 0:
                descriptions.append(f"{column_name} decreased by {abs(value)}")
        
        elif change_type == 'boost_category':
            category = change_config.get('category')
            boost = change_config.get('boost_percentage', 0)
            descriptions.append(f"{column_name} boosted '{category}' by {boost}%")
        
        elif change_type == 'force_boolean':
            value = change_config.get('value')
            descriptions.append(f"{column_name} set to {value}")
    
    if not descriptions:
        return "No changes applied"
    
    return "Scenario: " + ", ".join(descriptions)

# Router endpoints

@router.post("/what_if/inputs/")
async def get_bulk_editable_features(req: WhatIfInputRequest):
    """
    Extract and analyze features suitable for what-if analysis.
    
    Returns a list of features with their metadata and suggested adjustment strategies.
    """
    try:
        stats = req.column_stats
        
        if not stats:
            return {"features": [], "message": "No column statistics provided"}
        
        analyzer = FeatureAnalyzer()
        editable_features = []
        
        for name, values in stats.items():
            if not values:  # Skip empty columns
                continue
                
            feature_info = analyzer.analyze_feature(name, values)
            if feature_info:
                # Convert to dict for JSON serialization
                feature_dict = {
                    "name": feature_info.name,
                    "label": feature_info.label,
                    "type": feature_info.type.value,
                    "adjustment_type": feature_info.adjustment_type.value,
                    "null_percentage": round(feature_info.null_percentage, 2)
                }
                
                # Add type-specific fields
                if feature_info.type == FeatureType.NUMERIC:
                    feature_dict.update({
                        "min": feature_info.min_val,
                        "max": feature_info.max_val,
                        "step": feature_info.step,
                        "mean": round(feature_info.mean, 4),
                        "median": round(feature_info.median, 4),
                        "std": round(feature_info.std, 4),
                        "adjustment_values": feature_info.adjustment_values
                    })
                elif feature_info.type == FeatureType.CATEGORICAL:
                    feature_dict.update({
                        "options": feature_info.options,
                        "recommendation_note": feature_info.recommendation_note
                    })
                elif feature_info.type == FeatureType.BOOLEAN:
                    feature_dict["default"] = feature_info.default
                
                editable_features.append(feature_dict)
        
        # Sort features for better UX (numeric first, then categorical, then boolean)
        type_order = {FeatureType.NUMERIC.value: 0, FeatureType.CATEGORICAL.value: 1, FeatureType.BOOLEAN.value: 2}
        editable_features.sort(key=lambda x: (type_order.get(x["type"], 3), x["name"]))
        
        return {
            "features": editable_features,
            "summary": {
                "total_features": len(editable_features),
                "numeric_features": len([f for f in editable_features if f["type"] == "numeric"]),
                "categorical_features": len([f for f in editable_features if f["type"] == "categorical"]),
                "boolean_features": len([f for f in editable_features if f["type"] == "boolean"])
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing what-if inputs: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": "Failed to process features", "details": str(e)})


@router.get("/what_if/health")
async def health_check():
    """Health check endpoint for what-if analysis router."""
    return {"status": "healthy", "service": "what-if-analysis"}