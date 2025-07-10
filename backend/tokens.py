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
from .auth import get_current_active_user, get_master_db_session, User, Base  # <-- assumes your User is defined in auth.py
import os
import stripe
from fastapi import FastAPI, UploadFile, File, Form, Request, Depends, Path
from pydantic import BaseModel
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TokenUsageLogResponse(BaseModel):
    endpoint: str
    tokens_used: float
    timestamp: datetime

    class Config:
        orm_mode = True


class TopUpRequest(BaseModel):
    amount: int



class TokenUsageLog(Base):
    __tablename__ = "token_usage_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    endpoint = Column(String)
    tokens_used = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="token_usage_logs")

router = APIRouter(prefix="/tokens", tags=["auth"])
TOKEN_PRICE_CENTS = 100

FRONTEND_BASE_URL = "https://ml-insights-frontend.onrender.com"
from pydantic import BaseModel, AnyHttpUrl

class CheckoutSessionResponse(BaseModel):
    checkoutUrl: AnyHttpUrl
class TopUpRequest(BaseModel):
    amount: int
def log_current_db_user(db: Session):
    result = db.execute(text("SELECT current_user")).fetchone()
    logger.info(f"PostgreSQL current_user: {result[0]}")

@router.post("/topup-tokens", response_model=CheckoutSessionResponse)
def top_up_tokens(
    req: TopUpRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session     = Depends(get_master_db_session),
):
    log_current_db_user(db)
    # —————— Bypass Stripe for admin/dev roles ——————
    if current_user.role in ("admin", "dev"):
        # credit immediately
        current_user.tokens = (current_user.tokens or 0) + req.amount
        db.commit()
        return CheckoutSessionResponse(
            checkoutUrl=f"{FRONTEND_BASE_URL}/success.html?added={req.amount}"
        )

    # —————— Everyone else goes through Stripe ——————
    try:
        session = stripe.checkout.Session.create(
            customer=current_user.stripe_customer_id,
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {"name": f"{req.amount} Tokens"},
                    "unit_amount": req.amount * TOKEN_PRICE_CENTS,
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url=f"{FRONTEND_BASE_URL}/success.html?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_BASE_URL}/cancel.html",
            metadata={"user_id": str(current_user.id), "amount": str(req.amount)},
        )
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=402, detail=f"Stripe error: {e.user_message}")

    return CheckoutSessionResponse(checkoutUrl=session.url)
@router.get("/token-usage", response_model=List[TokenUsageLogResponse])
def get_token_usage(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_master_db_session)):
    return db.query(TokenUsageLog).filter(TokenUsageLog.user_id == current_user.id).order_by(TokenUsageLog.timestamp.desc()).limit(20).all()

class VerifyPaymentResponse(BaseModel):
    session_id: str
    amount: int               # tokens
    payment_intent: str       # the Stripe PaymentIntent ID
    amount_charged: float     # in dollars

@router.get("/verify-payment", response_model=VerifyPaymentResponse)
def verify_payment(
    session_id: str,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_master_db_session),
):
    # Pull the session from Stripe
    try:
        session = stripe.checkout.Session.retrieve(session_id)
    except stripe.error.StripeError as e:
        raise HTTPException(400, detail=f"Invalid session: {e.user_message}")

    # Ensure it belongs to this user
    meta_user = session.metadata.get("user_id")
    if meta_user != str(current_user.id):
        raise HTTPException(403, detail="Not your session")

    # Only accept successful payments
    if session.payment_status != "paid":
        raise HTTPException(400, detail="Payment not completed")

    # The number of tokens requested is in metadata
    amount = int(session.metadata.get("amount", 0))
    pi_id = session.payment_intent
    cents = session.amount_total or 0

    return VerifyPaymentResponse(
        session_id    = session_id,
        amount        = int(session.metadata.get("amount", 0)),
        payment_intent= pi_id,
        amount_charged= cents / 100.0,
    )