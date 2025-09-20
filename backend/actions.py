# routers/actions.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Date,
    UniqueConstraint, ForeignKey
)
from sqlalchemy import Column, Integer, BigInteger, Text, Boolean, Numeric, JSON, DateTime
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.orm import Session
from decimal import Decimal
# Reuse your auth and DB helpers
from auth import get_current_active_user, get_user_db

router = APIRouter(prefix="/actions", tags=["actions"])

Base = declarative_base()

# -----------------------------
# DB Models
# -----------------------------
# --- Plan model: add user_id (+ optional dataset_id) and timezone-aware timestamps ---
class Plan(Base):
    __tablename__ = "plans"
    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(BigInteger, nullable=False, index=True)
    dataset_id = Column(BigInteger, nullable=True, index=True)

    goal_amount = Column(Numeric(12,2), nullable=False)
    period = Column(String(32), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    risk = Column(String(16), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    actions = relationship("Action", back_populates="plan", cascade="all, delete-orphan")

class Action(Base):
    __tablename__ = "actions"  # matches the table you just migrated
    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(Integer, ForeignKey("plans.id", ondelete="CASCADE"), index=True, nullable=True)  # keep nullable while legacy rows exist

    name = Column(String(128), nullable=False)
    channel = Column(String(64), nullable=False)
    cost = Column(Numeric(10,4), nullable=False, default=0)
    cooldown = Column(Integer)
    daily_cap = Column(Integer)
    provider = Column(String(128))
    active = Column(Boolean, nullable=False, default=True)
    sort_order = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    plan = relationship("Plan", back_populates="actions")

    __table_args__ = (
        UniqueConstraint("plan_id", "name", "channel", name="uq_actions_plan_name_channel"),
    )
# ---------- Schemas ----------
class ActionIn(BaseModel):
    name: str
    channel: str
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
        from_attributes = True  # Pydantic v2

class SaveStep1Request(BaseModel):
    plan_id: Optional[int] = None
    goal_amount: float = Field(..., ge=0)
    period: Literal["this_month", "next_30d", "custom"]
    start_date: date
    end_date: date
    risk: Literal["conservative","balanced","aggressive"]
    actions: List[ActionIn] = Field(default_factory=list)

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
def ensure_schema(db: Session, default_user_id: int | None = None) -> None:
    # 1) create tables if missing (includes user_id in the definition)
    db.execute(text("""
    CREATE TABLE IF NOT EXISTS plans (
      id           BIGSERIAL PRIMARY KEY,
      user_id      BIGINT,              -- nullable during backfill
      dataset_id   BIGINT,
      goal_amount  NUMERIC(12,2) NOT NULL,
      period       VARCHAR(32) NOT NULL,
      start_date   DATE NOT NULL,
      end_date     DATE NOT NULL,
      risk         VARCHAR(16) NOT NULL,
      created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
      updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """))

    db.execute(text("""
    CREATE TABLE IF NOT EXISTS actions (
      id          BIGSERIAL PRIMARY KEY,
      plan_id     BIGINT REFERENCES plans(id) ON DELETE CASCADE,
      name        VARCHAR(128) NOT NULL,
      channel     VARCHAR(64) NOT NULL,
      cost        NUMERIC(10,4) NOT NULL DEFAULT 0,
      cooldown    INT,
      daily_cap   INT,
      provider    VARCHAR(128),
      active      BOOLEAN NOT NULL DEFAULT TRUE,
      sort_order  INT NOT NULL DEFAULT 0,
      created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
      updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """))

    # 2) upgrade existing schemas in-place (idempotent)
    db.execute(text("ALTER TABLE plans   ADD COLUMN IF NOT EXISTS user_id BIGINT"))
    db.execute(text("ALTER TABLE plans   ADD COLUMN IF NOT EXISTS dataset_id BIGINT"))
    db.execute(text("ALTER TABLE actions ADD COLUMN IF NOT EXISTS plan_id BIGINT"))
    db.execute(text("ALTER TABLE actions ADD COLUMN IF NOT EXISTS sort_order INT NOT NULL DEFAULT 0"))
    db.execute(text("ALTER TABLE actions ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE"))

    # make sure FK exists
    db.execute(text("""
    DO $$
    BEGIN
      IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'actions_plan_fk') THEN
        ALTER TABLE actions
          ADD CONSTRAINT actions_plan_fk
          FOREIGN KEY (plan_id) REFERENCES plans(id) ON DELETE CASCADE;
      END IF;
    END $$;
    """))

    # indexes
    db.execute(text("CREATE INDEX IF NOT EXISTS ix_plans_user_id     ON plans(user_id)"))
    db.execute(text("CREATE INDEX IF NOT EXISTS ix_plans_dataset_id  ON plans(dataset_id)"))
    db.execute(text("CREATE INDEX IF NOT EXISTS ix_actions_plan_id   ON actions(plan_id)"))

    # 3) backfill user_id for existing rows (so your filters work)
    if default_user_id is not None:
        db.execute(text("UPDATE plans SET user_id = :uid WHERE user_id IS NULL"), {"uid": default_user_id})

    db.commit()
def get_plan_or_404(db: Session, plan_id: int, user_id: int) -> Plan:
    obj = (db.query(Plan)
             .filter(Plan.id == plan_id, Plan.user_id == user_id)
             .first())
    if not obj:
        raise HTTPException(status_code=404, detail="Plan not found")
    return obj
# ---------- Endpoints ----------
@router.post("/save_step1", response_model=PlanOut, status_code=status.HTTP_201_CREATED)
def save_step1(body: SaveStep1Request, current_user=Depends(get_current_active_user)):
    if body.start_date > body.end_date:
        raise HTTPException(status_code=400, detail="start_date must be <= end_date")

    with get_user_db(current_user) as db:
        ensure_schema(db, default_user_id=current_user.id)

        # Create or update Plan
        if body.plan_id:
            plan = get_plan_or_404(db, body.plan_id, current_user.id)
            plan.goal_amount = Decimal(str(body.goal_amount))
            plan.period = body.period
            plan.start_date = body.start_date
            plan.end_date = body.end_date
            plan.risk = body.risk
        else:
            plan = Plan(
                user_id=current_user.id,
                goal_amount=Decimal(str(body.goal_amount)),
                period=body.period,
                start_date=body.start_date,
                end_date=body.end_date,
                risk=body.risk,
            )
            db.add(plan)
            db.flush()  # get plan.id

        # Replace actions for this plan
        db.query(Action).filter(Action.plan_id == plan.id).delete()
        for idx, a in enumerate(body.actions):
            db.add(Action(
                plan_id=plan.id,
                name=a.name,
                channel=a.channel,
                cost=Decimal(str(a.cost or 0)),
                cooldown=a.cooldown,
                daily_cap=a.daily_cap,
                provider=a.provider,
                active=True if a.active is None else a.active,
                sort_order=idx if a.sort_order is None else a.sort_order,
            ))

        try:
            db.commit()
        except IntegrityError as ie:
            db.rollback()
            raise HTTPException(status_code=409, detail="Duplicate (name, channel) within this plan") from ie

        db.refresh(plan)
        plan.actions = (db.query(Action)
                          .filter(Action.plan_id == plan.id)
                          .order_by(Action.sort_order.asc(), Action.id.asc())
                          .all())
        return plan

@router.get("/plan/{plan_id}", response_model=PlanOut)
def get_plan(plan_id: int, current_user=Depends(get_current_active_user)):
    with get_user_db(current_user) as db:
        ensure_tables(db)
        plan = get_plan_or_404(db, plan_id, current_user.id)
        plan.actions = (db.query(Action)
                          .filter(Action.plan_id == plan.id)
                          .order_by(Action.sort_order.asc(), Action.id.asc())
                          .all())
        return plan

@router.get("/plan/latest", response_model=PlanOut)
def get_latest_plan(current_user=Depends(get_current_active_user)):
    with get_user_db(current_user) as db:
        ensure_tables(db)
        plan = (db.query(Plan)
                  .filter(Plan.user_id == current_user.id)
                  .order_by(Plan.created_at.desc())
                  .first())
        if not plan:
            raise HTTPException(status_code=404, detail="No plans yet")
        plan.actions = (db.query(Action)
                          .filter(Action.plan_id == plan.id)
                          .order_by(Action.sort_order.asc(), Action.id.asc())
                          .all())
        return plan