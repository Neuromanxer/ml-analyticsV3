# routers/decision_cards.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from sqlalchemy import String, Integer, Float, Boolean
from sqlalchemy.orm import Mapped, mapped_column, Session, DeclarativeBase
from sqlalchemy.types import JSON
from typing import Dict, List, Any, Optional
import asyncio, uuid, time

from auth import master_engine as engine  # <- good: use your app DB

router = APIRouter(prefix="/cards", tags=["cards"])

class Base(DeclarativeBase): pass

class Card(Base):
    __tablename__ = "cards"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    goal: Mapped[str] = mapped_column(String)
    timeframeDays: Mapped[int] = mapped_column(Integer)
    budgetCap: Mapped[float] = mapped_column(Float)
    risk: Mapped[int] = mapped_column(Integer)
    constraints: Mapped[dict] = mapped_column(JSON)
    datasetIds: Mapped[list] = mapped_column(JSON)
    pipeline: Mapped[list] = mapped_column(JSON)
    plan: Mapped[list] = mapped_column(JSON)
    logs: Mapped[list] = mapped_column(JSON)
    launched: Mapped[bool] = mapped_column(Boolean, default=False)
    totals: Mapped[dict | None] = mapped_column(JSON, nullable=True)

Base.metadata.create_all(engine)

def get_session():
    return Session(engine)

def new_card(body) -> Card:
    return Card(
        id=str(uuid.uuid4()),
        goal=body.goal,
        timeframeDays=body.timeframeDays,
        budgetCap=body.budgetCap,
        risk=body.risk,
        constraints=body.constraints,
        datasetIds=body.datasetIds or [],
        pipeline=[
            {"id": "goal", "label": "1 Goal", "status": "PENDING"},
            {"id": "intake", "label": "2 Intake", "status": "PENDING"},
            {"id": "preprocess", "label": "3 Preprocess", "status": "PENDING"},
            {"id": "derive", "label": "4 Derive", "status": "PENDING"},
            {"id": "planner", "label": "5 Planner", "status": "PENDING"},
            {"id": "synthesis", "label": "6 AI Synthesis", "status": "PENDING"},
        ],
        plan=[],
        logs=[],
        launched=False,
        totals=None,
    )

_ws_conns: Dict[str, List[WebSocket]] = {}

def _ts() -> str:
    return time.strftime("%H:%M:%S")

async def _broadcast(card_id: str, payload: dict):
    for ws in list(_ws_conns.get(card_id, [])):
        try:
            await ws.send_json(payload)
        except Exception:
            pass

def _set_status_db(card_id: str, step_id: str, status: str, *, meta=None, anomalies=None, duration_ms: Optional[int]=None):
    with get_session() as s:
        card = s.get(Card, card_id)
        if not card:
            return
        for st in card.pipeline:
            if st["id"] == step_id:
                st["status"] = status
                if duration_ms is not None:
                    st["durationMs"] = duration_ms
                if meta is not None:
                    st["meta"] = meta
                if anomalies is not None:
                    st["anomalies"] = anomalies
        s.add(card)
        s.commit()

def _make_plan(card: Card):
    # <-- FIX: use attributes, not dict indexing
    risk = card.risk
    budget = card.budgetCap
    rm = 1 + (risk/100) * 0.5
    bf = min(budget / 5000, 1)

    def s(est, cost, lo, hi, _id, title, summary):
        return {
            "id": _id, "title": title, "summary": summary,
            "estNet": round(est * rm * bf),
            "cost": round(cost * bf),
            "confLow": round(lo * rm * bf),
            "confHigh": round(hi * rm * bf),
        }

    steps = [
        s(2400, 200, 1800, 3200, "step1", "Dynamic Pricing Adjustment", "Optimize pricing on high-margin items"),
        s(1800, 300, 1200, 2600, "step2", "Cross-sell Bundle Campaign", "Promote complementary products"),
        s(1200, 150,  800, 1800, "step3", "Inventory Velocity Push", "Progressive discounts on slow movers"),
    ]
    total = sum(x["estNet"] for x in steps)
    var = int(total * 0.2)
    totals = {"estNet": total, "confLow": total - var, "confHigh": total + var}
    return steps, totals

async def _simulate_pipeline(card_id: str):
    steps = ["goal", "intake", "preprocess", "derive", "planner", "synthesis"]
    delays = [0.4, 0.6, 0.4, 0.3, 0.5, 0.3]

    for sid, d in zip(steps, delays):
        await _broadcast(card_id, {"type": "step_started", "step_id": sid, "ts": _ts()})
        _set_status_db(card_id, sid, "RUNNING")
        await asyncio.sleep(d)

        status = "SUCCESS"
        meta, anomalies = {}, []
        if sid == "intake":
            meta = {"rows": 3892, "features": 12}
        elif sid == "preprocess":
            meta = {"clean_rows": 3801, "quality": 97.3}
            anomalies = ["high_variance_sku_42"] if (time.time() % 2 > 1) else []
        elif sid == "derive":
            meta = {"features": 127}
        elif sid == "planner":
            meta = {"plans_generated": 3, "confidence": 0.89}

        duration_ms = int(d * 1000)
        _set_status_db(card_id, sid, status, meta=meta, anomalies=anomalies, duration_ms=duration_ms)
        await _broadcast(card_id, {
            "type": "step_completed", "step_id": sid, "status": status,
            "durationMs": duration_ms, "meta": meta, "anomalies": anomalies
        })
        await _broadcast(card_id, {"type": "log", "ts": _ts(), "msg": f"{sid} completed {status}"})

    # finalize plan/totals in DB
    with get_session() as s:
        card = s.get(Card, card_id)
        if card:
            plan, totals = _make_plan(card)
            card.plan = plan
            card.totals = totals
            s.add(card)
            s.commit()
            await _broadcast(card_id, {"type": "plan_ready", "plan": plan, "totals": totals})

# ===== Schemas & routes =====
class CreateCardIn(BaseModel):
    goal: str = "Increase AOV"
    timeframeDays: int = 30
    budgetCap: float = 5000
    risk: int = 50
    constraints: Dict[str, bool] = {"excludeLowStock": False, "capDiscount12": True}
    datasetIds: Optional[List[str]] = []

class PatchCardIn(BaseModel):
    goal: Optional[str] = None
    timeframeDays: Optional[int] = None
    budgetCap: Optional[float] = None
    risk: Optional[int] = None
    constraints: Optional[Dict[str, bool]] = None
    datasetIds: Optional[List[str]] = None

@router.post("/")
@router.post("")
def create_card(body: CreateCardIn):
    with get_session() as s:
        card = new_card(body)
        s.add(card)
        s.commit()
        s.refresh(card)
        return card.__dict__

@router.get("/{card_id}")
def get_card(card_id: str):
    with get_session() as s:
        card = s.get(Card, card_id)
        if not card:
            raise HTTPException(404, "Card not found")
        return card.__dict__

@router.patch("/{card_id}")
def patch_card(card_id: str, body: PatchCardIn):
    with get_session() as s:
        card = s.get(Card, card_id)
        if not card:
            raise HTTPException(404, "Card not found")
        data = body.model_dump(exclude_unset=True)
        for k, v in data.items():
            setattr(card, k, v)
        s.add(card)
        s.commit()
        s.refresh(card)
        return card.__dict__

@router.post("/{card_id}/launch", status_code=202)
async def launch(card_id: str):
    with get_session() as s:
        card = s.get(Card, card_id)
        if not card:
            raise HTTPException(404, "Card not found")
        card.launched = True
        s.add(card)
        s.commit()
    await _broadcast(card_id, {"type": "log", "ts": _ts(), "msg": "Decision card launched"})
    return {}

@router.post("/{card_id}/pipeline/run")
async def run_pipeline(card_id: str):
    with get_session() as s:
        if not s.get(Card, card_id):
            raise HTTPException(404, detail=f"Card not found: {card_id}")
    job_id = str(uuid.uuid4())
    asyncio.create_task(_simulate_pipeline(card_id))
    return {"job_id": job_id}

@router.post("/{card_id}/pipeline/steps/{step_id}/retry", status_code=202)
async def retry_step(card_id: str, step_id: str):
    with get_session() as s:
        card = s.get(Card, card_id)
        if not card:
            raise HTTPException(404, "Card not found")
    await _broadcast(card_id, {"type": "log", "ts": _ts(), "msg": f"Retrying {step_id}"})
    _set_status_db(card_id, step_id, "RUNNING", anomalies=[])
    await asyncio.sleep(0.8)
    _set_status_db(card_id, step_id, "SUCCESS", duration_ms=420)
    await _broadcast(card_id, {"type": "step_completed", "step_id": step_id, "status": "SUCCESS", "durationMs": 420})
    await _broadcast(card_id, {"type": "log", "ts": _ts(), "msg": f"{step_id} retry completed successfully"})
    return {}

@router.post("/{card_id}/pipeline/steps/{step_id}/rollback", status_code=202)
async def rollback_step(card_id: str, step_id: str):
    with get_session() as s:
        card = s.get(Card, card_id)
        if not card:
            raise HTTPException(404, "Card not found")
        idx = next((i for i, sp in enumerate(card.pipeline) if sp["id"] == step_id), None)
        if idx is None:
            raise HTTPException(404, "Step not found")
        for j in range(idx, len(card.pipeline)):
            sp = card.pipeline[j]
            sp.update({"status": "PENDING", "durationMs": None, "meta": {}, "anomalies": []})
        if idx <= 4:
            card.plan = []
            card.totals = None
        s.add(card)
        s.commit()
    await _broadcast(card_id, {"type": "log", "ts": _ts(), "msg": f"Rolled back from {step_id}"})
    return {}

@router.websocket("/{card_id}/events")
async def ws_events(websocket: WebSocket, card_id: str):
    await websocket.accept()
    _ws_conns.setdefault(card_id, []).append(websocket)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        if card_id in _ws_conns:
            try:
                _ws_conns[card_id].remove(websocket)
            except ValueError:
                pass
