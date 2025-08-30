# planner.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable, Set
import itertools
import math
import time

StepDict = Dict[str, Any]
DEFAULT_MAX_LATENCY_MS = 1000_000  # 10 minutes hard ceiling
# ----------------------------
# Inputs to the planner
# ----------------------------
@dataclass
class Intent:
    goal: str = "predict"    # "predict" | "uplift" | "forecast" | "segment" | "survival" | "what_if" | "ab_test"
    mode: str = "train"      # "train" | "predict" | "analyze"
    risk_preset: str = "balanced"  # "conservative" | "balanced" | "aggressive"
    constraints: Dict[str, Any] = field(default_factory=dict)  # budgetCap, latencyMs, required, forbidden, forceRetrain

@dataclass
class Signals:
    hasLabel: Optional[bool] = None
    labelType: Optional[str] = None             # "binary" | "multiclass" | "numeric"
    hasTime: Optional[bool] = None
    isCensored: Optional[bool] = None
    horizonDays: Optional[int] = None
    eventRate: Optional[float] = None
    rows: Optional[int] = None
    cols: Optional[int] = None
    tsColumn: Optional[str] = None
    idColumn: Optional[str] = None

@dataclass
class Artifacts:
    classification: bool = False
    regression: bool = False
    forecast: bool = False
    survival: bool = False
    clustering: bool = False
    # You can extend with versions, timestamps, drift flags, etc.

# ----------------------------
# Step specification & registry
# ----------------------------
@dataclass(frozen=True)
class StepSpec:
    name: str
    endpoint: str
    depends_on: Tuple[str, ...] = ()
    produces: Tuple[str, ...] = ()
    consumes: Tuple[str, ...] = ()
    # Estimates (tunable / could be learned)
    base_cost: float = 1.0            # abstract cost units
    base_latency_ms: int = 30000      # default 30s
    base_risk: float = 0.02           # failure/instability prior
    idempotent: bool = True
    once_per_dataset: bool = False
    optional: bool = False
    family: str = "aux"               # "core" | "analysis" | "inference" | "aux"
    # Eligibility function
    
    when: Optional[Callable[[Intent, Signals, Artifacts], bool]] = None

    def eligible(self, intent: Intent, sig: Signals, art: Artifacts) -> bool:
        return True if self.when is None else bool(self.when(intent, sig, art))

# ----------------------------
# Planner weights & scoring
# ----------------------------
@dataclass
class Weights:
    w_value: float = 1.0
    w_speed: float = 0.25     # penalize latency
    w_cost: float = 0.25      # penalize cost
    w_risk: float = 0.5       # penalize risk
    w_coverage: float = 0.25  # reward broadness/utility
    w_cache_bonus: float = 0.5  # reward using existing artifacts

def weights_for_preset(risk_preset: str) -> Weights:
    if risk_preset == "conservative":
        return Weights(w_value=0.9, w_speed=0.2, w_cost=0.3, w_risk=0.9, w_coverage=0.2, w_cache_bonus=0.8)
    if risk_preset == "aggressive":
        return Weights(w_value=1.2, w_speed=0.5, w_cost=0.15, w_risk=0.2, w_coverage=0.3, w_cache_bonus=0.3)
    return Weights()  # balanced

# Utility/“value” priors for core families given intent
FAMILY_VALUE = {
    # goal -> family -> value
    "predict":     {"classification": 1.0, "regression": 1.0, "forecast": 0.4, "survival": 0.4, "clustering": 0.3},
    "uplift":      {"classification": 0.9, "regression": 0.6, "forecast": 0.3, "survival": 0.3, "clustering": 0.4},
    "forecast":    {"classification": 0.2, "regression": 0.5, "forecast": 1.1, "survival": 0.5, "clustering": 0.3},
    "segment":     {"classification": 0.4, "regression": 0.4, "forecast": 0.3, "survival": 0.3, "clustering": 1.0},
    "survival":    {"classification": 0.4, "regression": 0.4, "forecast": 0.3, "survival": 1.1, "clustering": 0.3},
    "what_if":     {"classification": 1.0, "regression": 1.0, "forecast": 0.6, "survival": 0.6, "clustering": 0.3},
    "ab_test":     {"classification": 1.0, "regression": 0.7, "forecast": 0.5, "survival": 0.5, "clustering": 0.3},
}

# ----------------------------
# Step registry (edit safely)
# ----------------------------
def get_registry() -> Dict[str, StepSpec]:
    R = {}

    def reg(s: StepSpec):
        R[s.name] = s
        return s

    # Always begin with derive
    reg(StepSpec(
        name="derive",
        endpoint="/api/plan/derive",
        base_cost=2.0,
        base_latency_ms=20000,
        family="aux",
        produces=("mappings","options","actions","target_horizon","capacity","budget"),
        when=lambda i,s,a: True
    ))

    # Core families
    reg(StepSpec("classification", "/classification/",
        depends_on=("derive",),
        produces=("model.classification","metrics","conformal","decision_curve"),
        base_cost=6.0, base_latency_ms=60000, family="core",
        when=lambda i,s,a: i.mode=="train" and bool(s.hasLabel) and s.labelType in ("binary","multiclass")))
    reg(StepSpec("regression", "/regression/",
        depends_on=("derive",),
        produces=("model.regression","metrics"),
        base_cost=6.0, base_latency_ms=60000, family="core",
        when=lambda i,s,a: i.mode=="train" and bool(s.hasLabel) and s.labelType=="numeric"))
    reg(StepSpec("forecast", "/forecast/",
        depends_on=("derive",),
        produces=("model.forecast","metrics"),
        base_cost=7.0, base_latency_ms=70000, family="core",
        when=lambda i,s,a: i.mode=="train" and bool(s.hasTime) and bool(s.horizonDays) and not s.hasLabel))
    reg(StepSpec("survival", "/survival/",
        depends_on=("derive",),
        produces=("model.survival","metrics"),
        base_cost=7.0, base_latency_ms=75000, family="core",
        when=lambda i,s,a: i.mode=="train" and bool(s.isCensored)))
    reg(StepSpec("clustering", "/clustering/",
        depends_on=("derive",),
        produces=("clusters","centroids"),
        base_cost=4.0, base_latency_ms=40000, family="core",
        when=lambda i,s,a: i.mode!="predict" and not s.hasLabel and not s.isCensored and not s.horizonDays))

    # Analysis / explainers
    reg(StepSpec("feature_impact", "/feature_impact/",
        depends_on=("classification","regression","forecast","survival"),
        produces=("shap_summary","drivers"),
        base_cost=3.5, base_latency_ms=30000, family="analysis",
        when=lambda i,s,a: i.mode!="predict" and (a.classification or a.regression or a.forecast or a.survival)))
    reg(StepSpec("risk_analysis", "/risk_analysis/",
        depends_on=("classification",),
        produces=("ece","coverage","abstention","stops"),
        base_cost=2.0, base_latency_ms=15000, family="analysis",
        when=lambda i,s,a: a.classification))
    reg(StepSpec("decision_paths", "/decision_paths/",
        depends_on=("classification",),
        produces=("tree_paths","surrogate_rules"),
        base_cost=2.5, base_latency_ms=20000, family="analysis",
        when=lambda i,s,a: a.classification))
    reg(StepSpec("segment_analysis", "/segment_analysis/",
        depends_on=("derive",),
        produces=("segments","lifts"),
        base_cost=2.0, base_latency_ms=12000, family="analysis",
        when=lambda i,s,a: i.mode!="predict"))

    # Inference / what-if / counterfactuals / predict
    reg(StepSpec("classification_predict", "/classification/predict/",
        depends_on(("classification",) if not Artifacts.classification else tuple(),),
        produces=("scores","predictions"),
        base_cost=1.5, base_latency_ms=8000, family="inference",
        when=lambda i,s,a: i.mode in ("predict","analyze") and (a.classification or s.hasLabel and s.labelType in ("binary","multiclass"))))
    reg(StepSpec("regression_predict", "/regression/predict/",
        depends_on(("regression",) if not Artifacts.regression else tuple(),),
        produces=("scores","predictions"),
        base_cost=1.5, base_latency_ms=8000, family="inference",
        when=lambda i,s,a: i.mode in ("predict","analyze") and (a.regression or s.labelType=="numeric")))
    reg(StepSpec("what_if", "/what_if/",
        depends_on=("classification","regression"),
        produces=("counterfactual_eval","elasticities"),
        base_cost=2.0, base_latency_ms=12000, family="inference",
        when=lambda i,s,a: i.goal in ("what_if","predict","uplift") and (a.classification or a.regression)))
    reg(StepSpec("counterfactual", "/counterfactual/",
        depends_on=("classification","regression"),
        produces=("dice_examples","recourse"),
        base_cost=3.0, base_latency_ms=20000, family="inference",
        when=lambda i,s,a: i.goal in ("what_if","predict","uplift") and (a.classification or a.regression)))
    reg(StepSpec("ab_test", "/ab_test/",
        depends_on=("classification","segment_analysis"),
        produces=("lift_estimates","power","sample_allocation"),
        base_cost=2.5, base_latency_ms=18000, family="inference",
        when=lambda i,s,a: i.goal in ("ab_test","uplift","predict") and a.classification))
    reg(StepSpec("visualize", "/visualize/",
        depends_on=("derive",),
        produces=("dash_assets",),
        base_cost=1.0, base_latency_ms=8000, family="aux",
        when=lambda i,s,a: True))

    return R

# ----------------------------
# Planning core
# ----------------------------
@dataclass
class PlanSummary:
    score: float
    value: float
    total_cost: float
    total_latency_ms: int
    total_risk: float
    families: Set[str]
    chosen_core: Optional[str]

@dataclass
class CompiledPlan:
    steps: List[StepDict]
    summary: PlanSummary

def _family_value(intent: Intent, family: str) -> float:
    table = FAMILY_VALUE.get(intent.goal, FAMILY_VALUE["predict"])
    return table.get(family, 0.3)

def _estimate_step_value(step: StepSpec, intent: Intent) -> float:
    base = _family_value(intent, step.family if step.family != "aux" else "classification")  # aux inherits some value
    # bonus for key analysis supporting Decision Card v2
    if step.name in ("risk_analysis","decision_paths","feature_impact","segment_analysis"):
        base += 0.2
    if step.name in ("what_if","counterfactual","ab_test"):
        base += 0.15
    if step.name.endswith("_predict"):
        base += 0.25 if intent.mode in ("predict","analyze") else 0.0
    return max(0.0, base)

def _compute_plan_score(steps: List[StepSpec], weights: Weights, intent: Intent, art: Artifacts) -> Tuple[float, PlanSummary]:
    total_cost = sum(s.base_cost for s in steps)
    total_latency = sum(s.base_latency_ms for s in steps)
    # aggregate risk as 1 - Π(1 - r)
    agg_risk = 1.0 - math.prod([1.0 - s.base_risk for s in steps])
    families = {s.family for s in steps if s.family != "aux"}
    chosen_core = next((s.name for s in steps if s.family=="core"), None)
    # value = sum per-step value with small diminishing returns
    values = [_estimate_step_value(s, intent) for s in steps]
    diminishing = sum(v / (1.0 + 0.15*i) for i, v in enumerate(sorted(values, reverse=True)))
    # cache bonus: if artifacts let us skip training, we’ll add later in compile based on skipped steps
    value = diminishing
    # final score (higher is better)
    score = (
        weights.w_value * value
        - weights.w_speed * (total_latency / 60000.0)      # normalize to minutes
        - weights.w_cost * total_cost
        - weights.w_risk * (agg_risk * 10.0)               # scale risk to ~[0,10]
        + weights.w_coverage * len(families) * 0.1
    )
    summary = PlanSummary(score=score, value=value, total_cost=total_cost, total_latency_ms=total_latency,
                          total_risk=agg_risk, families=families, chosen_core=chosen_core)
    return score, summary

def _toposort(reg: Dict[str, StepSpec], selected: List[StepSpec]) -> List[StepSpec]:
    names = {s.name for s in selected}
    deps = {s.name: set(s.depends_on) & names for s in selected}
    out: List[StepSpec] = []
    resolved: Set[str] = set()
    remaining = {s.name: s for s in selected}
    while remaining:
        ready = [n for n, d in deps.items() if d <= resolved]
        if not ready:
            # cycle; break deterministically by name
            ready = [sorted(remaining.keys())[0]]
        for n in ready:
            out.append(remaining[n])
            resolved.add(n)
            remaining.pop(n, None)
            deps.pop(n, None)
    # ensure derive first if present
    out_sorted = sorted(out, key=lambda s: (0 if s.name=="derive" else 1))
    return out_sorted

def _prune_by_constraints(steps: List[StepSpec], intent: Intent):
    cons = intent.constraints or {}
    req: Set[str] = set(cons.get("required", []))
    forb: Set[str] = set(cons.get("forbidden", []))
    budget_cap: Optional[float] = None if cons.get("ignoreBudgetCaps", False) else cons.get("budgetCap")

    # If we're ignoring latency, null the cap entirely.
    if cons.get("ignoreLatencyCaps", True):
        latency_cap: Optional[int] = None
    else:
        latency_raw: Optional[int] = cons.get("latencyMs")
        latency_cap = (
            min(int(latency_raw), DEFAULT_MAX_LATENCY_MS)
            if isinstance(latency_raw, (int, float)) else DEFAULT_MAX_LATENCY_MS
        )

    steps = [s for s in steps if s.name not in forb]
    missing_req = [r for r in req if r not in [s.name for s in steps]]
    return steps, missing_req, budget_cap, latency_cap

def _apply_caps(steps: List[StepSpec], budget_cap: Optional[float], latency_cap: Optional[int]) -> List[StepSpec]:
    if budget_cap is None and latency_cap is None:
        return steps
    out: List[StepSpec] = []
    cost_acc = 0.0
    lat_acc = 0
    for s in steps:
        next_cost = cost_acc + s.base_cost
        next_lat = lat_acc + s.base_latency_ms
        if (budget_cap is not None and next_cost > budget_cap) or (latency_cap is not None and next_lat > latency_cap):
            # skip optional/analysis first
            if s.family in ("analysis","aux") and s.optional:
                continue
            # otherwise stop adding more optional analysis
            if s.family in ("analysis","aux"):
                continue
            # if it's core/inference and cap is exceeded, we break
            break
        out.append(s)
        cost_acc, lat_acc = next_cost, next_lat
    return out

# ----------------------------
# Public API
# ----------------------------
def compile_plan(intent: Dict[str, Any], signals: Dict[str, Any], artifacts: Dict[str, bool], top_k: int = 1) -> CompiledPlan:
    i = Intent(**intent)
    s = Signals(**signals)
    a = Artifacts(**artifacts)
    reg = get_registry()
    W = weights_for_preset(i.risk_preset)
    cons = i.constraints or {}
    if cons.get("ignoreLatencyScore", True):
        W.w_speed = 0.0
    # 1) Candidate core families based on signals
    families: List[List[str]] = []
    if i.mode == "train":
        if s.hasLabel and s.labelType in ("binary","multiclass"):
            families.append(["classification"])
        if s.hasLabel and s.labelType == "numeric":
            families.append(["regression"])
        if s.hasTime and s.horizonDays and not s.hasLabel:
            families.append(["forecast"])
        if s.isCensored:
            families.append(["survival"])
        if not s.hasLabel and not s.isCensored and not s.horizonDays:
            families.append(["clustering"])
    else:
        # predict/analyze: prefer existing artifacts; else fall back to train
        arts = []
        if a.classification: arts.append(["classification"])
        if a.regression: arts.append(["regression"])
        if a.forecast: arts.append(["forecast"])
        if a.survival: arts.append(["survival"])
        if not arts:
            # fallback like train
            return compile_plan({**intent, "mode":"train"}, signals, artifacts, top_k)
        families = arts

    if not families:
        # default fallback
        families = [["clustering"]]

    # 2) Build branch candidates
    branches: List[List[StepSpec]] = []
    for fam in families:
        # core step(s) + derive
        selected = [reg["derive"]]
        core = reg[fam[0]]
        if core.eligible(i, s, a):
            selected.append(core)

        # analysis add-ons (only if will be satisfied)
        if fam[0] in ("classification","regression","forecast","survival"):
            # feature impact
            if reg["feature_impact"].eligible(i, s, a) or core.name in reg["feature_impact"].depends_on:
                selected.append(reg["feature_impact"])
        if fam[0] == "classification":
            if reg["risk_analysis"].eligible(i, s, a) or core.name in reg["risk_analysis"].depends_on:
                selected.append(reg["risk_analysis"])
            if reg["decision_paths"].eligible(i, s, a) or core.name in reg["decision_paths"].depends_on:
                selected.append(reg["decision_paths"])

        # segment analysis is broadly useful
        if reg["segment_analysis"].eligible(i, s, a):
            selected.append(reg["segment_analysis"])

        # inference (mode-aware)
        if i.mode in ("predict","analyze"):
            if fam[0] == "classification":
                selected.append(reg["classification_predict"])
            elif fam[0] == "regression":
                selected.append(reg["regression_predict"])

            # what-if / counterfactuals / ab_test if classification or regression exists
            if fam[0] in ("classification","regression"):
                if reg["what_if"].eligible(i, s, a):
                    selected.append(reg["what_if"])
                if reg["counterfactual"].eligible(i, s, a):
                    selected.append(reg["counterfactual"])
                if reg["ab_test"].eligible(i, s, a):
                    selected.append(reg["ab_test"])

        # visualization always last
        selected.append(reg["visualize"])

        # prune by constraints
        selected, missing_req, budget_cap, latency_cap = _prune_by_constraints(selected, i)
        # append required if eligible
        for r in missing_req:
            if r in reg and reg[r].eligible(i, s, a) and reg[r] not in selected:
                selected.append(reg[r])

        # honor caps (drop optional analysis first)
        selected = _apply_caps(selected, budget_cap, latency_cap)

        # topologically sort
        selected = _toposort(reg, selected)
        branches.append(selected)

    # 3) Score branches & pick best
    scored: List[Tuple[float, PlanSummary, List[StepSpec]]] = []
    for sel in branches:
        score, summary = _compute_plan_score(sel, W, i, a)
        # Cache bonus: if artifact exists for core step, boost
        core = summary.chosen_core
        if core and getattr(a, core, False):
            score += W.w_cache_bonus
            summary.score = score
        scored.append((score, summary, sel))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_summary, best_steps = scored[0]

    # 4) Materialize plan steps for execution (attach idempotency keys & metadata)
    run_id = f"plan_{int(time.time())}"
    out_steps: List[StepDict] = []
    for idx, st in enumerate(best_steps):
        out_steps.append({
            "idx": idx,
            "name": st.name,
            "endpoint": st.endpoint,
            "depends_on": list(st.depends_on),
            "produces": list(st.produces),
            "consumes": list(st.consumes),
            "est_cost": st.base_cost,
            "est_latency_ms": st.base_latency_ms,
            "est_risk": st.base_risk,
            "idempotent": st.idempotent,
            "once_per_dataset": st.once_per_dataset,
            "family": st.family,
            "headers": {
                "Idempotency-Key": f"{run_id}:{idx}:{st.name}"
            }
        })

    return CompiledPlan(steps=out_steps, summary=best_summary)

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    intent = {
        "goal": "predict",
        "mode": "train",
        "risk_preset": "balanced",
        "constraints": {
            "budgetCap": 15.0,
            "latencyMs": 160000,
            "required": [],
            "forbidden": []
        }
    }
    signals = {
        "hasLabel": True,
        "labelType": "binary",
        "hasTime": True,
        "horizonDays": 14,
        "isCensored": False,
        "rows": 120000,
        "cols": 45
    }
    artifacts = {
        "classification": False,
        "regression": False,
        "forecast": False,
        "survival": False,
        "clustering": False
    }

    plan = compile_plan(intent, signals, artifacts)
    from pprint import pprint
    pprint(plan.summary)
    pprint(plan.steps)
