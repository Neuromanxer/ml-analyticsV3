from pydantic import BaseModel
from planner import compile_plan
from planner_router import CompileRequest, IntentIn, SignalsIn, ArtifactsIn

payload = CompileRequest(
    intent=IntentIn(goal="predict", mode="train", risk_preset="balanced"),
    signals=SignalsIn(hasLabel=True, labelType="binary", hasTime=True, horizonDays=14, rows=1000, cols=20),
    artifacts=ArtifactsIn(),
    top_k=1,
)

plan = compile_plan(
    intent=payload.intent.model_dump(),
    signals=payload.signals.model_dump(),
    artifacts=payload.artifacts.model_dump(),
    top_k=payload.top_k,
)

print(plan.summary)
for step in plan.steps:
    print(step)
