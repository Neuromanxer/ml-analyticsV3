# account.py

from datetime import datetime, date
from typing import Optional, List, Literal
from .activity import ActivityLog
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

from .auth import get_current_active_user, get_master_db_session, User, Base  # <-- assumes your User is defined in auth.py
import os
import stripe
from fastapi import FastAPI, UploadFile, File, Form, Request, Depends, Path

stripe.api_key = os.environ.get("STRIPE_API_KEY", "")

DOMAIN = "http://localhost:8000"  # ✅ for local development


class BillingTable(Base):
    """
    Stores billing-related information for each user.
    - user_id: FK to users.id
    - payment_method: e.g. "Visa ****1234"
    - address: e.g. "123 Main St, City, State, ZIP"
    - history: a JSON array of invoice objects (date, amount, invoice_id, etc.)
    """
    __tablename__ = "billing"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False
    )

    payment_method = Column(String(255), nullable=True)
    address = Column(String(512), nullable=True)

    # Store invoice history as a JSON array; defaults to an empty list if not provided
    history = Column(JSON, nullable=False, default=list)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    # Relationship back to the User model (one‐to‐one)
    user = relationship("User", back_populates="billing", uselist=False)

class APIStats(BaseModel):
    mbUsed: float
    computeSeconds: float
    totalCost: float

class SubscriptionInfo(BaseModel):
    planName: str
    status: str
    renewalDate: date

class AccountStatsOut(BaseModel):
    mbUsed: float
    computeSeconds: float
    tokensRemaining: int
    plan_status: str     # keep this if you’re using snake_case for plan

    class Config:
        # allow both snake_case and camelCase from DB if needed
        allow_population_by_field_name = True

class ProfileInfo(BaseModel):
    firstName: str
    lastName: str
    email: str
    phone: Optional[str]
    company: Optional[str]
    role: Optional[str]
    bio: Optional[str]
    notifications: str
    timezone: str


class APIKeysInfo(BaseModel):
    production: str
    development: str


class BillingInfo(BaseModel):
    paymentMethod: str
    address: str
    history: List[dict]  # e.g. [{"date": "2025-01-01", "amount": "$29.00"}]

class ActivityLogResponse(BaseModel):
    title:       str
    description: str
    timestamp:   datetime

    class Config:
        orm_mode = True
class DashboardOut(BaseModel):
    apiStats:     APIStats
    subscription: SubscriptionInfo
    profile:      ProfileInfo
    apiKeys:      APIKeysInfo
    billing:      BillingInfo

    # ← new field here:
    activity:     List[ActivityLogResponse]


class ChangePlanRequest(BaseModel):
    planName: Literal["Free", "Starter", "Pro", "Enterprise"]  # match whatever tiers you support


class ChangePlanResponse(BaseModel):
    success: bool
    message: str
    newPlan: str
    checkoutUrl: Optional[str]  # ✅ THIS IS REQUIRED


class UpdateProfileRequest(BaseModel):
    firstName: str = Field(..., max_length=50)
    lastName: str = Field(..., max_length=50)
    email: EmailStr
    phone: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    bio: Optional[str] = None
    notifications: str = Field(..., pattern="^(all|important|none)$")
    timezone: str = Field(..., pattern="^(pst|est|utc|gmt)$")
    password: Optional[str] = Field(None, min_length=6)



# ─────────────────────────────────────────────────────────────────────────────
# 3) Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def get_user_subscription(
    user_id: int,
    db: Session
) -> SubscriptionInfo:
    """
    Fetch the subscription record for a given user_id.
    Currently returns a default “Free” subscription that renews on the
    first of next month. Replace this stub with a real DB lookup later.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No user found with ID {user_id}"
        )

    plan_name = user.subscription or "Free"
    status_str = "Active"  # or derive from a real billing status

    today = date.today()
    if today.month == 12:
        first_of_next = date(year=today.year + 1, month=1, day=1)
    else:
        first_of_next = date(year=today.year, month=today.month + 1, day=1)

    return SubscriptionInfo(
        planName=plan_name,
        status=status_str,
        renewalDate=first_of_next
    )


def get_user_api_keys(
    user_id: int,
    db: Session
) -> APIKeysInfo:
    """
    Look up the given user by ID and return an APIKeysInfo containing
    prod_api_key and dev_api_key. Raises 404 if the user does not exist.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No user found with ID {user_id}"
        )

    prod_key = user.prod_api_key or ""
    dev_key = user.dev_api_key or ""
    return APIKeysInfo(production=prod_key, development=dev_key)


# ─────────────────────────────────────────────────────────────────────────────
# 4) Plan → call limit mapping
# ─────────────────────────────────────────────────────────────────────────────

PLAN_CALL_LIMITS: dict[str, int] = {
    "Free":        1000,
    "Pro":         10000,
    "Enterprise":  100000,
}


# ─────────────────────────────────────────────────────────────────────────────
# 5) Router and endpoints
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/account", tags=["account"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# plan ↔ Stripe Price ID mapping
PRICE_IDS: dict[str,str] = {
    "Starter":     "price_1RXvKQRu3MBe4WUoBFTgEKQU", 
    "Pro":"price_1RXvIBRu3MBe4WUoVH8oIyvh",
    "Enterprise":  "",
}
@router.post("/change-plan", response_model=ChangePlanResponse)
async def change_plan(
    payload: ChangePlanRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_master_db_session),
):
    new_plan = payload.planName

    # 0) Dev/Admin shortcut: bypass Stripe and apply directly
    if current_user.role in ("admin", "dev"):
        print(f"🛠 Bypassing Stripe for {current_user.email} (role: {current_user.role})")
        
        if current_user.subscription == new_plan:
            return ChangePlanResponse(
                success=False,
                message=f"You are already on the {new_plan} plan.",
                newPlan=new_plan,
                checkoutUrl=None
            )

        current_user.subscription = new_plan
        db.add(current_user)
        db.commit()

        return ChangePlanResponse(
            success=True,
            message=f"Plan changed to {new_plan} (dev override).",
            newPlan=new_plan,
            checkoutUrl=None
        )

    # 1) Regular user: already subscribed?
    if current_user.subscription == new_plan:
        return ChangePlanResponse(
            success=False,
            message=f"You are already on the {new_plan} plan.",
            newPlan=new_plan,
            checkoutUrl=None
        )

    # 2) Get Stripe Price ID
    price_id = PRICE_IDS.get(new_plan)
    if not price_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown plan: {new_plan}"
        )

    # 3) Create Stripe Checkout Session
    try:
        checkout_session = stripe.checkout.Session.create(
            mode="subscription",
            customer=current_user.stripe_customer_id,
            line_items=[{
                "price": price_id,
                "quantity": 1,
            }],
            success_url=f"{DOMAIN}/account?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{DOMAIN}/account?canceled=true",

            # ✅ Add this metadata block
            metadata={
                "user_id": str(current_user.id),
                "plan": new_plan.lower()
            }
        )

        print("✅ Stripe Checkout URL:", checkout_session.url)
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Payment provider error: {e.user_message or str(e)}"
        )

    # 4) Send user to Stripe
    return ChangePlanResponse(
        success=True,
        message="Redirect to payment gateway to complete upgrade.",
        newPlan=new_plan,
        checkoutUrl=checkout_session.url,
    )

TOKEN_AMOUNTS_BY_PLAN = {
    "starter": 35,
    "professional": 80,
    "enterprise": 150
}
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
@router.post("/webhooks/stripe")
async def stripe_webhook(
    request: Request,
    db: Session = Depends(get_master_db_session)
):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]

        # ✅ Preferred: use metadata (most reliable)
        plan_name = session.get("metadata", {}).get("plan")
        cust_id = session.get("customer")

        if not plan_name or not cust_id:
            raise HTTPException(status_code=400, detail="Missing customer or plan info in metadata")

        plan_name = plan_name.lower()
        tokens_to_add = TOKEN_AMOUNTS_BY_PLAN.get(plan_name)

        if tokens_to_add is None:
            raise HTTPException(status_code=400, detail=f"No token amount mapped for plan: {plan_name}")

        # Lookup user by stripe_customer_id
        user = db.query(User).filter(User.stripe_customer_id == cust_id).first()

        if user:
            user.subscription = plan_name
            user.tokens = (user.tokens or 0) + tokens_to_add
            db.add(user)
            db.commit()

            print(f"✅ Added {tokens_to_add} tokens to user {user.email} for plan '{plan_name}'")

    return {"received": True}


@router.get("/dashboard", response_model=DashboardOut)
async def get_account_dashboard(
    current_user: User = Depends(get_current_active_user),
    db: Session      = Depends(get_master_db_session),
) -> DashboardOut:
    # — existing steps A–G unchanged —
    sub_info = get_user_subscription(current_user.id, db)
    api_stats = APIStats(
        mbUsed=round((current_user.total_bytes_processed or 0) / (1024 * 1024), 2),
        computeSeconds=round(current_user.total_compute_seconds or 0, 2),
        totalCost=round(current_user.total_cost_dollars or 0.0, 2),
    )
    subscription = sub_info
    profile = ProfileInfo(
        firstName=current_user.first_name or "",
        lastName=current_user.last_name  or "",
        email=current_user.email,
        phone=current_user.phone or "",
        company=current_user.company or "",
        role=current_user.role or "",
        bio=current_user.bio or "",
        notifications=current_user.notifications_pref or "all",
        timezone=current_user.timezone or "utc",
    )
    api_keys = get_user_api_keys(current_user.id, db)
    billing_row = db.query(BillingTable).filter(BillingTable.user_id == current_user.id).first()
    if not billing_row:
        billing_info = BillingInfo(paymentMethod="", address="", history=[])
    else:
        billing_info = BillingInfo(
            paymentMethod=billing_row.payment_method or "",
            address=billing_row.address or "",
            history=billing_row.history or [],
        )

    # — NEW: load last 10 activity logs —
    activity = (
        db.query(ActivityLog)
          .filter(ActivityLog.user_id == current_user.id)
          .order_by(ActivityLog.timestamp.desc())
          .limit(10)
          .all()
    )

    return DashboardOut(
        apiStats=api_stats,
        subscription=subscription,
        profile=profile,
        apiKeys=api_keys,
        billing=billing_info,
        activity=activity,               # ← pass it in here
    )
@router.get("/stats", response_model=AccountStatsOut)
async def get_account_stats(
    current_user: User = Depends(get_current_active_user)
):
    """
    Returns JSON for the Account Stats page:
     - MB used
     - Compute seconds
     - Tokens remaining (instead of total cost)
     - Plan status
    """
    mb_used = round((current_user.total_bytes_processed or 0) / (1024 * 1024), 2)
    compute_secs = round(current_user.total_compute_seconds or 0.0, 2)
    tokens_left = current_user.tokens or 0
    plan = current_user.subscription or "Free"

    return AccountStatsOut(
        mbUsed=mb_used,
        computeSeconds=compute_secs,
        tokensRemaining=tokens_left,
        plan_status=plan
    )

@router.get("/keys", response_model=APIKeysInfo)
async def read_api_keys(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_master_db_session),
):
    """
    Returns the production and development API keys for the authenticated user.
    """
    return get_user_api_keys(current_user.id, db)




@router.get("/profile", response_model=ProfileInfo)
async def get_profile(
    current_user: User = Depends(get_current_active_user)
):
    """
    Return basic profile information for the logged‐in user.
    """
    return ProfileInfo(
        firstName=current_user.first_name or "",
        lastName=current_user.last_name or "",
        email=current_user.email,
        phone=current_user.phone or "",
        company=current_user.company or "",
        role=current_user.role or "",
        bio=current_user.bio or "",
        notifications=current_user.notifications_pref or "all",
        timezone=current_user.timezone or "utc"
    )


@router.put("/update-profile")
async def update_profile(
    payload: UpdateProfileRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_master_db_session)
):
    """
    Update the current user's profile fields and (optionally) password.
    """
    # 1) If email is changing, ensure uniqueness
    if payload.email != current_user.email:
        existing = db.query(User).filter(User.email == payload.email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is already in use by another account."
            )

    # 2) Apply profile changes
    current_user.first_name = payload.firstName
    current_user.last_name = payload.lastName
    current_user.email = payload.email
    current_user.phone = payload.phone or ""
    current_user.company = payload.company or ""
    current_user.role = payload.role or ""
    current_user.bio = payload.bio or ""
    current_user.notifications_pref = payload.notifications
    current_user.timezone = payload.timezone

    # 3) If password is provided, hash and set it
    if payload.password:
        current_user.hashed_password = pwd_context.hash(payload.password)

    try:
        db.add(current_user)
        db.commit()
        return {"success": True, "message": "Profile updated successfully."}
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile."
        )

@router.get(
    "/keys",
    response_model=APIKeysInfo,
    summary="Get production + development API keys"
)
async def read_api_keys(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_master_db_session),
):
    """
    Returns the production and development API keys for the authenticated user.
    """
    return get_user_api_keys(current_user.id, db)

class UpdateProfileRequest(BaseModel):
    firstName: str = Field(..., max_length=50)
    lastName: str = Field(..., max_length=50)
    email: EmailStr
    phone: str | None = None
    company: str | None = None
    role: str | None = None
    bio: str | None = None
    notifications: str = Field(..., pattern="^(all|important|none)$")
    timezone: str = Field(..., pattern="^(pst|est|utc|gmt)$")
    password: str | None = Field(None, min_length=6)

class PasswordChangeRequest(BaseModel):
    new_password: str

@router.put("/user/change-password")
def change_password(
    payload: PasswordChangeRequest,
    db: Session = Depends(get_master_db_session),
    current_user: User = Depends(get_current_active_user),
):
    hashed_pw = pwd_context.hash(payload.new_password)
    current_user.hashed_password = hashed_pw
    db.commit()
    return {"message": "Password changed successfully."}
@router.get("/profile")
def get_profile(current_user: User = Depends(get_current_active_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.first_name,
        "subscription": current_user.subscription,
        "api_call_count": current_user.api_call_count,
        "storage_used": current_user.storage_used
    }
@router.put("/update-profile")
async def update_profile(
    payload: UpdateProfileRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_master_db_session)
):
    """
    Update the current user's profile. Fields:
      - firstName, lastName, email, phone, company, role, bio, notifications, timezone
      - Optional: password (min length 6)
    """

    # 1) If email is changing, ensure no other user has that email
    if payload.email != current_user.email:
        existing = db.query(User).filter(User.email == payload.email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is already in use by another account."
            )

    # 2) Apply changes
    current_user.first_name = payload.firstName
    current_user.last_name = payload.lastName
    current_user.email = payload.email
    current_user.phone = payload.phone or ""
    current_user.company = payload.company or ""
    current_user.role = payload.role or ""
    current_user.bio = payload.bio or ""
    current_user.notifications_pref = payload.notifications
    current_user.timezone = payload.timezone

    # 3) If password provided, hash and set
    if payload.password:
        hashed = pwd_context.hash(payload.password)
        current_user.hashed_password = hashed

    try:
        db.add(current_user)
        db.commit()
        return {"success": True, "message": "Profile updated successfully."}
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile."
        )
from .storage import list_user_files, get_file_url
import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from io import BytesIO

@router.post("/request-export")
def request_data_export(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_master_db_session)
):
    try:
        user_id = str(current_user.id)

        # --- 1. Profile Info ---
        profile = {
            "user_id": user_id,
            "email": current_user.email,
            "first_name": current_user.first_name,
            "last_name": current_user.last_name,
            "company": current_user.company,
            "role": current_user.role,
            "bio": current_user.bio,
            "timezone": current_user.timezone,
            "agreed_to_terms": current_user.agreed_to_terms,
            "agreed_at": str(current_user.agreed_at),
            "policy_version": current_user.policy_version,
            "tokens_remaining": current_user.tokens,
            "total_cost_dollars": current_user.total_cost_dollars,
        }

        # --- 2. Files ---
        file_info = []
        files = list_user_files(user_id)
        for f in files or []:
            file_info.append({
                "file_name": f["name"],
                "size": f["metadata"]["size"],
                "created_at": f["created_at"],
                "download_url": get_file_url(f"{user_id}/{f['name']}", expires_in=3600)
            })

        # --- 3. Visualizations ---
        vis_result = db.execute(text("""
            SELECT type, created_at, metadata FROM visualizations_metadata
            WHERE user_id = :user_id
            ORDER BY created_at DESC
        """), {"user_id": user_id})

        visualizations = [
            {
                "type": row[0],
                "created_at": row[1],
                "metadata": row[2] if isinstance(row[2], dict) else json.loads(row[2])
            } for row in vis_result.fetchall()
        ]

        # --- 4. Package JSON ---
        export_data = {
            "profile": profile,
            "uploaded_files": file_info,
            "ai_insights": visualizations,
        }

        return export_data

    except Exception as e:
        logging.error(f"❌ Failed to generate export for {current_user.id}: {str(e)}")
        raise HTTPException(500, "Failed to generate export")
from .storage import delete_file_from_supabase
@router.delete("/delete-all-data")
def delete_all_user_data(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_master_db_session)):
    try:
        user_id = str(current_user.id)

        # Step 1: Delete files associated with the user
        files = list_user_files(user_id)
        if not files:
            logging.info(f"No files found for user {user_id}")
        
        for f in files or []:
            file_path = f"{user_id}/{f['name']}"
            logging.info(f"Deleting file: {file_path}")
            delete_file_from_supabase(file_path)

        # Step 2: Delete visualizations metadata
        logging.info(f"Deleting visualizations metadata for user {user_id}")
        delete_vis_response = db.execute(text("""
            DELETE FROM visualizations_metadata WHERE user_id = :user_id
        """), {"user_id": user_id})

        # Step 3: Delete user profile info (if stored in a user-specific table, e.g., 'user_profiles')
        logging.info(f"Deleting user profile info for user {user_id}")
        delete_profile_response = db.execute(text("""
            DELETE FROM user_profiles WHERE user_id = :user_id
        """), {"user_id": user_id})

        # Step 4: Delete any other user-specific data (e.g., audit logs, activities)
        logging.info(f"Deleting user activity logs for user {user_id}")
        delete_activity_response = db.execute(text("""
            DELETE FROM user_activity WHERE user_id = :user_id
        """), {"user_id": user_id})

        # Step 5: Delete user from the main user table
        logging.info(f"Deleting user record for user {user_id}")
        delete_user_response = db.execute(text("""
            DELETE FROM users WHERE id = :user_id
        """), {"user_id": user_id})

        # Commit the changes
        db.commit()

        return {"message": f"✅ Successfully deleted all data for user {current_user.id}."}

    except Exception as e:
        logging.exception(f"❌ Failed to delete all data for user {current_user.id}: {str(e)}")
        db.rollback()  # Rollback any partial changes
        raise HTTPException(status_code=500, detail="Failed to delete all user data.")