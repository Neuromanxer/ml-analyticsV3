from auth import Base

from datetime import datetime, date
from typing import Optional, List, Literal

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
from auth import get_current_active_user, get_master_db_session, User, Base  # <-- assumes your User is defined in auth.py
import os
import stripe
from fastapi import FastAPI, UploadFile, File, Form, Request, Depends, Path
from pydantic import BaseModel
from datetime import datetime
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

@router.post("/ai_recommendations")
async def get_ai_recommendations(
    file: UploadFile = File(...),
    goal: str = Form(...),
    dataType: str = Form(...),
    target: Optional[str] = Form(None),
    success: Optional[str] = Form(None),
    timeColumn: Optional[str] = Form(None)
):
    """
    Get AI-powered recommendations for data analysis approach
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel file.")
        
        # Get basic data information
        data_shape = df.shape
        columns = df.columns.tolist()
        data_types = df.dtypes.to_dict()
        missing_values = df.isnull().sum().to_dict()
        
        # Analyze column types
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        datetime_columns = []
        
        # Try to identify datetime columns
        for col in df.columns:
            try:
                pd.to_datetime(df[col].dropna().head(100))
                datetime_columns.append(col)
            except:
                continue
        
        # Analyze target variable if specified
        target_info = ""
        if target and target in columns:
            target_series = df[target]
            if target_series.dtype in ['object', 'category']:
                unique_values = target_series.nunique()
                target_info = f"Target '{target}' is categorical with {unique_values} unique values"
            else:
                target_info = f"Target '{target}' is numeric with range [{target_series.min():.2f}, {target_series.max():.2f}]"
        
        # Create comprehensive prompt for AI recommendation
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
        - /classification/ (for categorical prediction)
        - /regression/ (for numeric prediction)
        - /clustering/ (for pattern discovery)
        - /forecast/ (for time series prediction)
        - /segment_analysis/ (for customer/group segmentation)
        - /survival/ (for time-to-event analysis)
        - /counterfactual/ (for causal analysis)

        Please provide a JSON response with these exact keys:
        {{
            "recommended_path": "one of the available paths",
            "target_variable": "recommended target column or null",
            "drop_columns": ["list", "of", "columns", "to", "drop"],
            "time_column": "recommended time column or null",
            "forecast_periods": "number for forecasting or null",
            "impactful_features": ["list", "of", "most", "relevant", "features"],
            "reasoning": "brief explanation of recommendation",
            "confidence": "high/medium/low",
            "alternative_approaches": ["alternative", "paths", "to", "consider"]
        }}

        Consider:
        1. Match the analysis type to the user's goal
        2. Identify the most suitable target variable
        3. Suggest columns to drop (IDs, duplicates, irrelevant)
        4. For time series, identify appropriate time column
        5. For forecasting, suggest reasonable forecast periods
        6. Identify the most impactful features for the analysis
        """

        # Get AI recommendation
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data science advisor. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        
        ai_response = completion.choices[0].message.content.strip()
        
        # Try to parse JSON response
        try:
            # Clean the response if it has markdown formatting
            if ai_response.startswith('```json'):
                ai_response = ai_response.split('```json')[1].split('```')[0].strip()
            elif ai_response.startswith('```'):
                ai_response = ai_response.split('```')[1].split('```')[0].strip()
            
            recommendation = json.loads(ai_response)
            
            # Validate the recommendation structure
            required_keys = ["recommended_path", "target_variable", "drop_columns", 
                           "time_column", "forecast_periods", "impactful_features", 
                           "reasoning", "confidence"]
            
            for key in required_keys:
                if key not in recommendation:
                    recommendation[key] = None
            
            # Ensure lists are actually lists
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
                    "missing_values": {k: int(v) for k, v in missing_values.items() if v > 0}
                }
            }
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "ai_text": ai_response,
                "dataset_summary": {
                    "shape": data_shape,
                    "columns": columns,
                    "numeric_columns": numeric_columns,
                    "categorical_columns": categorical_columns,
                    "datetime_columns": datetime_columns
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

