# filename: eda_endpoint.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from io import BytesIO

# Perform general EDA
def perform_eda(df: pd.DataFrame):
    eda_summary = {}
    
    # Basic info
    eda_summary['shape'] = df.shape
    eda_summary['columns'] = list(df.columns)
    eda_summary['dtypes'] = df.dtypes.astype(str).to_dict()
    
    # Nulls
    eda_summary['null_values'] = df.isna().sum().to_dict()
    eda_summary['null_percentage'] = (df.isna().sum()/len(df)*100).to_dict()
    
    # Descriptive stats
    eda_summary['describe'] = df.describe(include='all').to_dict()
    
    # Unique values for categorical columns
    cat_features = df.select_dtypes(include='object').columns.tolist()
    unique_vals = {col: df[col].unique().tolist() for col in cat_features}
    eda_summary['unique_values'] = unique_vals
    
    return eda_summary

# Determine target type
def detect_target_type(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        return "Target column not found."
    
    unique_vals = df[target_column].nunique()
    dtype = df[target_column].dtype
    
    if dtype == 'object' or unique_vals <= 20:
        return "classification"
    elif np.issubdtype(dtype, np.number):
        return "regression"
    else:
        return "unknown"

# Suggest best model based on EDA
def suggest_models(df: pd.DataFrame, target_column: str):
    task_type = detect_target_type(df, target_column)
    n_samples, n_features = df.shape
    
    model_suggestions = []
    
    if task_type == "classification":
        if n_samples < 1000:
            model_suggestions = ["Logistic Regression", "Decision Tree", "Random Forest"]
        elif n_samples < 10000:
            model_suggestions = ["Random Forest", "Gradient Boosting (XGBoost/LightGBM)", "SVM"]
        else:
            model_suggestions = ["Gradient Boosting", "Deep Learning / Neural Network"]
            
    elif task_type == "regression":
        if n_samples < 1000:
            model_suggestions = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
        elif n_samples < 10000:
            model_suggestions = ["Random Forest Regressor", "Gradient Boosting Regressor"]
        else:
            model_suggestions = ["Gradient Boosting Regressor", "Neural Network Regressor"]
    else:
        model_suggestions = ["Manual inspection required"]
    
    return {"task_type": task_type, "suggested_models": model_suggestions}
