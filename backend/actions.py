# routers/actions.py
from __future__ import annotations

from datetime import datetime, date
from decimal import Decimal
from typing import List, Optional, Literal

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column, Integer, BigInteger, String, Boolean, DateTime, Date,
    Numeric, ForeignKey, UniqueConstraint, text
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, relationship, Session
from sqlalchemy.sql import func

# Auth / DB helpers (your existing utilities)
from auth import get_current_active_user, get_user_db

router = APIRouter(prefix="/actions", tags=["actions"])
Base = declarative_base()

# -----------------------------
# DB Models
# -----------------------------
class Plan(Base):
    __tablename__ = "plans"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    dataset_id = Column(BigInteger, nullable=True, index=True)

    goal_amount = Column(Numeric(12, 2), nullable=False)
    period = Column(String(32), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    risk = Column(String(16), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    actions = relationship("Action", back_populates="plan", cascade="all, delete-orphan")


class Action(Base):
    __tablename__ = "actions"

    id = Column(Integer, primary_key=True, index=True)
    # kept nullable=True while legacy rows exist; see ensure_schema to tighten later
    plan_id = Column(Integer, ForeignKey("plans.id", ondelete="CASCADE"), index=True, nullable=True)

    name = Column(String(128), nullable=False)
    channel = Column(String(64), nullable=False)
    cost = Column(Numeric(10, 4), nullable=False, default=0)
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

# -----------------------------
# Pydantic Schemas
# -----------------------------
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
    risk: Literal["conservative", "balanced", "aggressive"]
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

# -----------------------------
# Schema bootstrap (idempotent)
# -----------------------------
def ensure_schema(db: Session, default_user_id: int | None = None) -> None:
    # 1) Create base tables if missing
    db.execute(text("""
    CREATE TABLE IF NOT EXISTS plans (
      id           BIGSERIAL PRIMARY KEY,
      user_id      BIGINT,
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

    # 2) In-place upgrades (idempotent columns)
    db.execute(text("ALTER TABLE plans   ADD COLUMN IF NOT EXISTS user_id BIGINT"))
    db.execute(text("ALTER TABLE plans   ADD COLUMN IF NOT EXISTS dataset_id BIGINT"))
    db.execute(text("ALTER TABLE actions ADD COLUMN IF NOT EXISTS plan_id BIGINT"))
    db.execute(text("ALTER TABLE actions ADD COLUMN IF NOT EXISTS sort_order INT NOT NULL DEFAULT 0"))
    db.execute(text("ALTER TABLE actions ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE"))

    # 3) Ensure FK
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

    # 4) Indexes
    db.execute(text("CREATE INDEX IF NOT EXISTS ix_plans_user_id    ON plans(user_id)"))
    db.execute(text("CREATE INDEX IF NOT EXISTS ix_plans_dataset_id ON plans(dataset_id)"))
    db.execute(text("CREATE INDEX IF NOT EXISTS ix_actions_plan_id  ON actions(plan_id)"))

    # 5) Drop ANY legacy unique on (user_id,name,channel) by inspecting columns; then add the correct unique
    # 6b) Ensure created_at / updated_at have defaults + backfill any NULLs
    db.execute(text("""
    DO $$
    BEGIN
    -- created_at: ensure DEFAULT now()
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name='actions' AND column_name='created_at'
        AND column_default IS NOT NULL
    ) THEN
        ALTER TABLE actions ALTER COLUMN created_at SET DEFAULT now();
    END IF;

    -- updated_at: ensure DEFAULT now()
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name='actions' AND column_name='updated_at'
        AND column_default IS NOT NULL
    ) THEN
        ALTER TABLE actions ALTER COLUMN updated_at SET DEFAULT now();
    END IF;

    -- backfill any NULLs that were inserted before defaults existed
    UPDATE actions SET created_at = now()  WHERE created_at IS NULL;
    UPDATE actions SET updated_at = now()  WHERE updated_at IS NULL;
    END $$;
    """))
    # --- Ensure timestamp defaults exist + backfill nulls (idempotent) ---
    # PLANS
    db.execute(text("UPDATE plans   SET created_at = NOW() WHERE created_at IS NULL"))
    db.execute(text("UPDATE plans   SET updated_at = NOW() WHERE updated_at IS NULL"))
    db.execute(text("ALTER TABLE plans   ALTER COLUMN created_at SET DEFAULT now()"))
    db.execute(text("ALTER TABLE plans   ALTER COLUMN updated_at SET DEFAULT now()"))
    db.execute(text("ALTER TABLE plans   ALTER COLUMN created_at SET NOT NULL"))
    db.execute(text("ALTER TABLE plans   ALTER COLUMN updated_at SET NOT NULL"))

    # ACTIONS
    db.execute(text("UPDATE actions SET created_at = NOW() WHERE created_at IS NULL"))
    db.execute(text("UPDATE actions SET updated_at = NOW() WHERE updated_at IS NULL"))
    db.execute(text("ALTER TABLE actions ALTER COLUMN created_at SET DEFAULT now()"))
    db.execute(text("ALTER TABLE actions ALTER COLUMN updated_at SET DEFAULT now()"))
    db.execute(text("ALTER TABLE actions ALTER COLUMN created_at SET NOT NULL"))
    db.execute(text("ALTER TABLE actions ALTER COLUMN updated_at SET NOT NULL"))


    # 6) Optionally tighten plan_id to NOT NULL after backfill
    db.execute(text("""
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='actions' AND column_name='plan_id' AND is_nullable='NO'
      ) THEN
        IF NOT EXISTS (SELECT 1 FROM actions WHERE plan_id IS NULL) THEN
          ALTER TABLE actions ALTER COLUMN plan_id SET NOT NULL;
        END IF;
      END IF;
    END $$;
    """))

    # 7) Backfill user_id for plans so filters work
    if default_user_id is not None:
        db.execute(text("UPDATE plans SET user_id = :uid WHERE user_id IS NULL"), {"uid": default_user_id})

    db.commit()

# -----------------------------
# Helpers
# -----------------------------
def get_plan_or_404(db: Session, plan_id: int, user_id: int) -> Plan:
    obj = (db.query(Plan)
             .filter(Plan.id == plan_id, Plan.user_id == user_id)
             .first())
    if not obj:
        raise HTTPException(status_code=404, detail="Plan not found")
    return obj

# -----------------------------
# Endpoints
# -----------------------------
@router.post("/save_step1", response_model=PlanOut, status_code=status.HTTP_201_CREATED)
def save_step1(body: SaveStep1Request, current_user=Depends(get_current_active_user)):
    if body.start_date > body.end_date:
        raise HTTPException(status_code=400, detail="start_date must be <= end_date")

    def _canon(s: str | None) -> str:
        return (s or "").strip()

    def _key(name: str, channel: str) -> tuple[str, str]:
        return (_canon(name).lower(), _canon(channel).lower())

    with get_user_db(current_user) as db:
        ensure_schema(db, default_user_id=current_user.id)

        # Upsert plan
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
            db.flush()  # ensure plan.id exists

        # Optional: serialize per-plan updates to avoid races
        db.query(Plan).filter(Plan.id == plan.id).with_for_update().one()

        # Normalize & de-duplicate incoming actions
        seen: set[tuple[str, str]] = set()
        cleaned: list[Action] = []
        for idx, a in enumerate(body.actions or []):
            name = _canon(a.name)
            chan = _canon(a.channel)
            if not name or not chan:
                continue
            k = _key(name, chan)
            if k in seen:
                continue
            seen.add(k)
            cleaned.append(Action(
                plan_id=plan.id,
                name=name,
                channel=chan,
                cost=Decimal(str(a.cost or 0)),
                cooldown=a.cooldown,
                daily_cap=a.daily_cap,
                provider=_canon(a.provider),
                active=True if a.active is None else bool(a.active),
                sort_order=idx if a.sort_order is None else a.sort_order,
            ))

        # Hard-replace actions: delete then insert (flush after delete!)
        db.query(Action).filter(Action.plan_id == plan.id).delete(synchronize_session=False)
        db.flush()  # make the delete visible before inserts

        for act in cleaned:
            db.add(act)

        # Commit with diagnostic on constraint name if it fails
        try:
            db.commit()
        except IntegrityError as ie:
            db.rollback()
            cname = None
            try:
                cname = getattr(getattr(ie, "orig", None), "diag", None).constraint_name
            except Exception:
                pass
            logging.exception("IntegrityError on save_step1; constraint=%s", cname)
            raise HTTPException(status_code=409, detail=f"Duplicate constraint hit: {cname or 'unknown'}") from ie

        # Reload and return
        db.refresh(plan)
        plan.actions = (db.query(Action)
                          .filter(Action.plan_id == plan.id)
                          .order_by(Action.sort_order.asc(), Action.id.asc())
                          .all())
        return plan


@router.get("/plan/{plan_id}", response_model=PlanOut)
def get_plan(plan_id: int, current_user=Depends(get_current_active_user)):
    with get_user_db(current_user) as db:
        ensure_schema(db)
        plan = get_plan_or_404(db, plan_id, current_user.id)
        plan.actions = (db.query(Action)
                          .filter(Action.plan_id == plan.id)
                          .order_by(Action.sort_order.asc(), Action.id.asc())
                          .all())
        return plan


@router.get("/plan/latest", response_model=PlanOut)
def get_latest_plan(current_user=Depends(get_current_active_user)):
    with get_user_db(current_user) as db:
        ensure_schema(db)
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
