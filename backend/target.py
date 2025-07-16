from fastapi import APIRouter, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import pandas as pd
import json

router = APIRouter()
from fastapi import APIRouter, UploadFile, Form
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
    column_map: str = Form(...),
    vip_threshold: float = Form(500.0),  # New parameter for VIP threshold
    at_risk_days: int = Form(90),        # New parameter for at-risk threshold
    upsell_threshold: float = Form(0.7)  # New parameter for high upsell threshold
):
    try:
        # Load CSV and normalize column names
        df = pd.read_csv(file.file)
        df.columns = df.columns.str.lower().str.strip()

        # Parse user-provided column mapping
        column_mapping = json.loads(column_map)

        # Rename user columns to internal names
        for internal_name, user_column in column_mapping.items():
            user_column_lower = user_column.lower()
            if user_column_lower not in df.columns:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Column '{user_column}' not found in uploaded file."}
                )
            df.rename(columns={user_column_lower: internal_name}, inplace=True)

        # Convert visit date
        if "visit_date" in df.columns:
            df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")

        # Group to customer-level
        agg = df.groupby("customer_id").agg({
            "amount_paid": "sum",
            "visit_date": ["min", "max", "count"]
        })
        agg.columns = ["total_spent", "first_activity", "last_activity", "total_visits"]
        agg = agg.reset_index()
        agg["days_active"] = (agg["last_activity"] - agg["first_activity"]).dt.days.clip(lower=1)

        # Optional: referrals
        if "referral_code" in df.columns:
            referral_counts = df.groupby("referral_code")["customer_id"].count()
            agg["referrals"] = agg["customer_id"].map(referral_counts).fillna(0)
        else:
            agg["referrals"] = 0

        # Optional: margin
        if "margin" in df.columns:
            margin_df = df.groupby("customer_id")["margin"].mean().reset_index()
            agg = agg.merge(margin_df, on="customer_id", how="left")
        else:
            agg["margin"] = 0

        # ──────────────── Compute Target ──────────────── #
        if target_type == "clv":
            agg["aov"] = agg["total_spent"] / agg["total_visits"]
            agg["frequency"] = agg["total_visits"] / agg["days_active"]
            agg["target"] = agg["aov"] * agg["frequency"] * (agg["days_active"] / 30)
            agg["clv"] = agg["target"]  # Add clv column for summary stats

        elif target_type == "revenue_per_customer":
            agg["target"] = agg["total_spent"] / agg["total_visits"]

        elif target_type == "churn":
            agg["target"] = ((pd.Timestamp.now() - agg["last_activity"]).dt.days > 60).astype(int)

        elif target_type == "aov":
            agg["target"] = agg["total_spent"] / agg["total_visits"]

        elif target_type == "purchase_frequency":
            agg["target"] = agg["total_visits"] / agg["days_active"]

        elif target_type == "customer_lifetime":
            agg["target"] = (agg["last_activity"] - agg["first_activity"]).dt.days

        elif target_type == "repeat_purchase_rate":
            agg["target"] = (agg["total_visits"] > 1).astype(int)

        elif target_type == "recency":
            agg["target"] = (pd.Timestamp.now() - agg["last_activity"]).dt.days
            agg["recency"] = agg["target"]  # Add recency column for summary stats

        elif target_type == "revenue_per_visit":
            agg["target"] = agg["total_spent"] / agg["total_visits"]

        elif target_type == "loyalty_score":
            agg["frequency_score"] = agg["total_visits"] / agg["days_active"]
            agg["recency_score"] = 1 / (pd.Timestamp.now() - agg["last_activity"]).dt.days.clip(lower=1)
            agg["target"] = (agg["frequency_score"] * 0.6) + (agg["recency_score"] * 0.4)

        elif target_type == "engagement_score":
            agg["target"] = agg["total_visits"] / agg["days_active"]

        elif target_type == "profit_per_customer":
            agg["target"] = agg["total_spent"] * agg["margin"]

        elif target_type == "inactivity_risk_score":
            agg["days_since_last"] = (pd.Timestamp.now() - agg["last_activity"]).dt.days
            agg["target"] = (agg["days_since_last"] > 90).astype(int)

        elif target_type == "vip_flag":
            agg["target"] = (agg["total_spent"] > vip_threshold).astype(int)

        elif target_type == "referral_score":
            agg["target"] = agg["referrals"] / (agg["total_visits"] + 1)

        elif target_type == "custom" and custom_formula:
            agg["target"] = agg.eval(custom_formula)

        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Target type '{target_type}' is not supported."}
            )

        # ──────────────── Calculate Derived Columns for Summary Stats ──────────────── #
        
        # Calculate CLV if not already calculated
        if "clv" not in agg.columns:
            agg["aov"] = agg["total_spent"] / agg["total_visits"]
            agg["frequency"] = agg["total_visits"] / agg["days_active"]
            agg["clv"] = agg["aov"] * agg["frequency"] * (agg["days_active"] / 30)
        
        # Calculate recency if not already calculated
        if "recency" not in agg.columns:
            agg["recency"] = (pd.Timestamp.now() - agg["last_activity"]).dt.days
        
        # Calculate VIP flag
        agg["is_vip"] = (agg["total_spent"] > vip_threshold).astype(int)
        
        # Calculate at-risk flag
        agg["is_at_risk"] = (agg["recency"] > at_risk_days).astype(int)
        
        # Calculate upsell score (example: based on CLV and recency)
        # Normalize CLV and recency to 0-1 scale, then combine
        clv_normalized = (agg["clv"] - agg["clv"].min()) / (agg["clv"].max() - agg["clv"].min())
        recency_normalized = 1 - ((agg["recency"] - agg["recency"].min()) / (agg["recency"].max() - agg["recency"].min()))
        agg["upsell_score"] = (clv_normalized * 0.7 + recency_normalized * 0.3).fillna(0)

        # ──────────────── Calculate Summary Statistics ──────────────── #
        summary = {
            "total_customers": len(agg),
            "vip_count": int(agg["is_vip"].sum()),
            "at_risk_customers": int(agg["is_at_risk"].sum()),
            "high_upsell_score": int((agg["upsell_score"] > upsell_threshold).sum()),
            "average_clv": round(agg["clv"].mean(), 2),
            "average_recency": round(agg["recency"].mean(), 2),
        }

        # Reorder and return result
        columns_ordered = ["target"] + [col for col in agg.columns if col != "target"]
        agg = agg.replace([float('inf'), float('-inf')], pd.NA).fillna("")
        
        return {
            "data": agg[columns_ordered].to_dict(orient="records"),
            "summary": summary
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing: {str(e)}"}
        )
def generate_customer_summary(agg_df: pd.DataFrame, vip_threshold: float = 500.0, at_risk_days: int = 90, upsell_threshold: float = 0.7) -> Dict:
    try:
        summary = {
            "total_customers": len(agg_df),
            "vip_count": int((agg_df["total_spent"] > vip_threshold).sum()),
            "at_risk_customers": int((agg_df["recency"] > at_risk_days).sum()),
            "high_upsell_score": int((agg_df["upsell_score"] > upsell_threshold).sum()),
            "average_clv": round(agg_df["clv"].mean(), 2),
            "average_recency": round(agg_df["recency"].mean(), 2),
        }
        return summary
    except Exception as e:
        raise ValueError(f"Failed to generate summary stats: {e}")
