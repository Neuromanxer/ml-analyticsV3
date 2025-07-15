from fastapi import APIRouter, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import pandas as pd
import json

router = APIRouter()

@router.post("/analyze/define-target")
async def define_target_variable(
    file: UploadFile,
    target_type: str = Form(...),
    custom_formula: Optional[str] = Form(""),
    column_map: str = Form(...)
):
    try:
        # Load CSV and normalize column names
        df = pd.read_csv(file.file)
        df.columns = df.columns.str.lower().str.strip()

        # Parse user-provided column mapping
        column_mapping = json.loads(column_map)

        # Rename user columns to expected internal names
        for internal_name, user_column in column_mapping.items():
            user_column_lower = user_column.lower()
            if user_column_lower not in df.columns:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Column '{user_column}' not found in uploaded file."}
                )
            df.rename(columns={user_column_lower: internal_name}, inplace=True)

        # Parse dates if present
        if "last_activity" in df.columns:
            df["last_activity"] = pd.to_datetime(df["last_activity"], errors="coerce")
        if "first_activity" in df.columns:
            df["first_activity"] = pd.to_datetime(df["first_activity"], errors="coerce")

        # ──────────────── Generate Target Variable ──────────────── #
        if target_type == "clv":
            df["aov"] = df["total_spent"] / df["total_visits"]
            df["days_active"] = (df["last_activity"] - df["first_activity"]).dt.days.clip(lower=1)
            df["frequency"] = df["total_visits"] / df["days_active"]
            df["target"] = df["aov"] * df["frequency"] * (df["days_active"] / 30)

        elif target_type == "revenue_per_customer":
            df["target"] = df["total_spent"] / df["total_visits"]

        elif target_type == "churn":
            df["target"] = ((pd.Timestamp.now() - df["last_activity"]).dt.days > 60).astype(int)

        elif target_type == "aov":
            df["target"] = df["total_spent"] / df["total_visits"]

        elif target_type == "purchase_frequency":
            df["days_active"] = (df["last_activity"] - df["first_activity"]).dt.days.clip(lower=1)
            df["target"] = df["total_visits"] / df["days_active"]

        elif target_type == "customer_lifetime":
            df["target"] = (df["last_activity"] - df["first_activity"]).dt.days

        elif target_type == "repeat_purchase_rate":
            df["target"] = (df["total_visits"] > 1).astype(int)

        elif target_type == "recency":
            df["target"] = (pd.Timestamp.now() - df["last_activity"]).dt.days

        elif target_type == "revenue_per_visit":
            df["target"] = df["total_spent"] / df["total_visits"]

        elif target_type == "loyalty_score":
            df["days_active"] = (df["last_activity"] - df["first_activity"]).dt.days.clip(lower=1)
            df["frequency_score"] = df["total_visits"] / df["days_active"]
            df["recency_score"] = 1 / (pd.Timestamp.now() - df["last_activity"]).dt.days.clip(lower=1)
            df["target"] = (df["frequency_score"] * 0.6) + (df["recency_score"] * 0.4)

        elif target_type == "engagement_score":
            df["days_active"] = (df["last_activity"] - df["first_activity"]).dt.days.clip(lower=1)
            df["target"] = df["total_visits"] / df["days_active"]

        elif target_type == "profit_per_customer":
            df["target"] = df["total_spent"] * df["margin"]

        elif target_type == "inactivity_risk_score":
            df["days_since_last"] = (pd.Timestamp.now() - df["last_activity"]).dt.days
            df["target"] = (df["days_since_last"] > 90).astype(int)

        elif target_type == "vip_flag":
            df["target"] = (df["total_spent"] > 500).astype(int)

        elif target_type == "referral_score":
            df["target"] = df["referrals"] / (df["total_visits"] + 1)

        elif target_type == "custom" and custom_formula:
            df["target"] = df.eval(custom_formula)

        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Target type '{target_type}' is not supported."}
            )

        # Return full dataset with 'target' column first
        columns_ordered = ["target"] + [col for col in df.columns if col != "target"]
        return df[columns_ordered].to_dict(orient="records")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing: {str(e)}"}
        )
