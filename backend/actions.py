# routers/actions.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Date,
    UniqueConstraint, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.exc import IntegrityError

# Reuse your auth and DB helpers
from auth import get_current_active_user, get_user_db

router = APIRouter(prefix="/actions", tags=["actions"])

UserBase = declarative_base()

# -----------------------------
# DB Models
# -----------------------------
class Plan(UserBase):
    __tablename__ = "plans"
    id = Column(Integer, primary_key=True, index=True)
    goal_amount = Column(Float, nullable=False)
    period = Column(String(32), nullable=False)               # this_month | next_30d | custom
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    risk = Column(String(16), nullable=False)                 # conservative | balanced | aggressive
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    actions = relationship("Action", back_populates="plan", cascade="all, delete-orphan")


class Action(UserBase):
    __tablename__ = "actions"
    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(Integer, ForeignKey("plans.id", ondelete="CASCADE"), index=True, nullable=False)

    name = Column(String(128), nullable=False)
    channel = Column(String(64), nullable=False)
    cost = Column(Float, nullable=False, default=0.0)         # per action/send
    cooldown = Column(Integer, nullable=True)                 # days
    daily_cap = Column(Integer, nullable=True)
    provider = Column(String(128), nullable=True)
    active = Column(Boolean, default=True, nullable=False)
    sort_order = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    plan = relationship("Plan", back_populates="actions")

    __table_args__ = (
        UniqueConstraint("plan_id", "name", "channel", name="uq_actions_plan_name_channel"),
    )


# -----------------------------
# Schemas
# -----------------------------
class ActionIn(BaseModel):
    name: str = Field(..., max_length=128)
    channel: str = Field(..., max_length=64)
    cost: float = 0.0
    cooldown: Optional[int] = None
    daily_cap: Optional[int] = None
    provider: Optional[str] = None
    active: bool = True
    sort_order: Optional[int] = None

class ActionOut(ActionIn):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class SaveStep1Request(BaseModel):
    goal_amount: float = Field(..., ge=0)
    period: Literal["this_month", "next_30d", "custom"]
    start_date: date
    end_date: date
    risk: Literal["conservative", "balanced", "aggressive"]
    actions: List[ActionIn] = Field(default_factory=list)
    plan_id: Optional[int] = None  # send to update existing plan

class PlanOut(BaseModel):
    id: int
    goal_amount: float
    period: str
    start_date: date
    end_date: date
    risk: str
    created_at: datetime
    updated_at: datetime
    actions: List[ActionOut]

    class Config:
        from_attributes = True


# -----------------------------
# Helpers
# -----------------------------
def ensure_tables(db):
    UserBase.metadata.create_all(bind=db.bind, tables=[Plan.__table__, Action.__table__])

def get_plan_or_404(db, plan_id: int) -> Plan:
    obj = db.query(Plan).get(plan_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Plan not found")
    return obj


# -----------------------------
# Endpoints
# -----------------------------

@router.post("/save_step1", response_model=PlanOut, status_code=status.HTTP_201_CREATED)
def save_step1(body: SaveStep1Request, current_user=Depends(get_current_active_user)):
    """
    Atomically upsert a Plan (Step 1 inputs) and replace its Actions.
    """
    if body.start_date > body.end_date:
        raise HTTPException(status_code=400, detail="start_date must be <= end_date")

    with get_user_db(current_user) as db:
        ensure_tables(db)

        # Create or update plan
        if body.plan_id:
            plan = get_plan_or_404(db, body.plan_id)
            plan.goal_amount = body.goal_amount
            plan.period = body.period
            plan.start_date = body.start_date
            plan.end_date = body.end_date
            plan.risk = body.risk
        else:
            plan = Plan(
                goal_amount=body.goal_amount,
                period=body.period,
                start_date=body.start_date,
                end_date=body.end_date,
                risk=body.risk,
            )
            db.add(plan)
            db.flush()  # get plan.id before inserting actions

        # Replace actions for this plan
        db.query(Action).filter(Action.plan_id == plan.id).delete()

        for idx, a in enumerate(body.actions):
            row = Action(
                plan_id=plan.id,
                name=a.name,
                channel=a.channel,
                cost=a.cost or 0.0,
                cooldown=a.cooldown,
                daily_cap=a.daily_cap,
                provider=a.provider,
                active=a.active if a.active is not None else True,
                sort_order=a.sort_order if a.sort_order is not None else idx,
            )
            db.add(row)

        try:
            db.commit()
        except IntegrityError as ie:
            db.rollback()
            # Likely (plan_id, name, channel) uniqueness violation
            raise HTTPException(status_code=409, detail="Duplicate (name, channel) within this plan") from ie

        db.refresh(plan)
        # eager load actions
        actions = db.query(Action).filter(Action.plan_id == plan.id).order_by(Action.sort_order.asc(), Action.id.asc()).all()
        plan.actions = actions
        return plan


@router.get("/plan/{plan_id}", response_model=PlanOut)
def get_plan(plan_id: int, current_user=Depends(get_current_active_user)):
    with get_user_db(current_user) as db:
        ensure_tables(db)
        plan = get_plan_or_404(db, plan_id)
        actions = db.query(Action).filter(Action.plan_id == plan.id).order_by(Action.sort_order.asc(), Action.id.asc()).all()
        plan.actions = actions
        return plan


@router.get("/plan/latest", response_model=PlanOut)
def get_latest_plan(current_user=Depends(get_current_active_user)):
    with get_user_db(current_user) as db:
        ensure_tables(db)
        plan = db.query(Plan).order_by(Plan.created_at.desc()).first()
        if not plan:
            raise HTTPException(status_code=404, detail="No plans yet")
        actions = db.query(Action).filter(Action.plan_id == plan.id).order_by(Action.sort_order.asc(), Action.id.asc()).all()
        plan.actions = actions
        return plan
