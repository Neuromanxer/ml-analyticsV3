# planner.py

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union, Protocol
from pydantic import BaseModel, Field, conint, confloat
import pandas as pd
class SignalsIn(BaseModel):
    hasLabel: Optional[bool] = None
    labelType: Optional[str] = Field(None, pattern="^(binary|multiclass|numeric)$")
    hasTime: Optional[bool] = None
    isCensored: Optional[bool] = None
    horizonDays: Optional[conint(ge=1, le=3650)] = None  
    eventRate: Optional[confloat(ge=0.0, le=1.0)] = None  
    rows: Optional[int] = None
    cols: Optional[int] = None
    tsColumn: Optional[str] = None
    idColumn: Optional[str] = None

# ---------------------------------------------------------------------
# Constants & small types
# ---------------------------------------------------------------------

DEFAULT_MAX_LATENCY_MS = 600_000  # 10 minutes hard ceiling

# Semantic helper sets
_BOOL = {"boolean", "bool"}
_CAT  = {"categorical", "category"}
_NUM  = {"numeric", "number", "float", "integer", "int", "double"}
_TIME = {"timestamp", "datetime", "date", "time"}

# Step dict type used by compiled plan
StepDict = Dict[str, Any]


# Optional typing protocol to describe "gap" records if callers supply them
class GapRecord(Protocol):
    status: str            # e.g. "missing" | "ok" | ...
    criticality: str       # e.g. "high" | "medium" | "low"
    coverage: Optional[float]


# ---------------------------------------------------------------------
# Planner inputs
# ---------------------------------------------------------------------

@dataclass
class Intent:
    goal: str = "predict"    # "predict" | "uplift" | "forecast" | "segment" | "survival" | "what_if" | "ab_test"
    mode: str = "train"      # "train" | "predict" | "analyze"
    risk_preset: str = "balanced"  # "conservative" | "balanced" | "aggressive"
    constraints: Dict[str, Any] = field(default_factory=dict)  # budgetCap, latencyMs, required, forbidden, forceRetrain, ...


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
    # extend as needed (versions, timestamps, drift flags, etc.)


# ---------------------------------------------------------------------
# Step specification & registry
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class StepSpec:
    name: str
    endpoint: str
    depends_on: Tuple[str, ...] = ()
    produces: Tuple[str, ...] = ()
    consumes: Tuple[str, ...] = ()

    # Estimates (tunable / could be learned)
    base_cost: float = 1.0            # abstract cost units
    base_latency_ms: int = 30_000     # default 30s
    base_risk: float = 0.02           # failure/instability prior

    idempotent: bool = True
    once_per_dataset: bool = False
    optional: bool = False            # <-- used by pruning/caps
    family: str = "aux"               # "core" | "analysis" | "inference" | "aux"

    # Eligibility function
    when: Optional[Callable[[Intent, Signals, Artifacts], bool]] = None

    def eligible(self, intent: Intent, sig: Signals, art: Artifacts) -> bool:
        return True if self.when is None else bool(self.when(intent, sig, art))


class PlanHint(BaseModel):
    goal: Optional[str] = None                  # "predict" | "forecast" | "segment" | ...
    preferred_families: List[str] = []          # e.g., ["classification"]
    id_column: Optional[str] = None
    time_column: Optional[str] = None
    label_candidates: List[str] = []            # e.g., ["converted", "purchase", "churned"]
    label_type: Optional[str] = None            # "binary" | "numeric" | "multiclass"
    horizon_days: Optional[int] = None
    notes: List[str] = []


# ---------------------------------------------------------------------
# Weights & value priors
# ---------------------------------------------------------------------

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
FAMILY_VALUE: Dict[str, Dict[str, float]] = {
    # goal -> family -> value
    "predict":     {"classification": 1.0, "regression": 1.0, "forecast": 0.4, "survival": 0.4, "clustering": 0.3},
    "uplift":      {"classification": 0.9, "regression": 0.6, "forecast": 0.3, "survival": 0.3, "clustering": 0.4},
    "forecast":    {"classification": 0.2, "regression": 0.5, "forecast": 1.1, "survival": 0.5, "clustering": 0.3},
    "segment":     {"classification": 0.4, "regression": 0.4, "forecast": 0.3, "survival": 0.3, "clustering": 1.0},
    "survival":    {"classification": 0.4, "regression": 0.4, "forecast": 0.3, "survival": 1.1, "clustering": 0.3},
    "what_if":     {"classification": 1.0, "regression": 1.0, "forecast": 0.6, "survival": 0.6, "clustering": 0.3},
    "ab_test":     {"classification": 1.0, "regression": 0.7, "forecast": 0.5, "survival": 0.5, "clustering": 0.3},
}


def _family_value(intent: Intent, family: str) -> float:
    table = FAMILY_VALUE.get(intent.goal, FAMILY_VALUE["predict"])
    return table.get(family, 0.3)


def _estimate_step_value(step: StepSpec, intent: Intent) -> float:
    base = _family_value(intent, step.family if step.family != "aux" else "classification")  # aux ≈ support value
    # Bonus for analysis that supports Decision Card v2
    if step.name in ("risk_analysis", "decision_paths", "feature_impact", "segment_analysis"):
        base += 0.2
    if step.name in ("what_if", "counterfactual", "ab_test"):
        base += 0.15
    if step.name.endswith("_predict"):
        base += 0.25 if intent.mode in ("predict", "analyze") else 0.0
    return max(0.0, base)


# ---------------------------------------------------------------------
# Step registry (safe to edit)
# ---------------------------------------------------------------------

def get_registry() -> Dict[str, StepSpec]:
    R: Dict[str, StepSpec] = {}

    def reg(s: StepSpec):
        R[s.name] = s
        return s

    # Always begin with derive
    reg(StepSpec(
        name="derive",
        endpoint="/api/plan/derive",
        base_cost=2.0,
        base_latency_ms=20_000,
        family="aux",
        once_per_dataset=True,
        optional=True,   # can be dropped if absolutely necessary (though we try to keep it)
        produces=("mappings", "options", "actions", "target_horizon", "capacity", "budget"),
        when=lambda i, s, a: True,
    ))

    # Core families
    reg(StepSpec(
        "classification", "/classification/",
        depends_on=("derive",),
        produces=("model.classification", "metrics", "conformal", "decision_curve"),
        base_cost=6.0, base_latency_ms=60_000, family="core",
        when=lambda i, s, a: i.mode == "train" and bool(s.hasLabel) and s.labelType in ("binary", "multiclass"),
    ))
    reg(StepSpec(
        "regression", "/regression/",
        depends_on=("derive",),
        produces=("model.regression", "metrics"),
        base_cost=6.0, base_latency_ms=60_000, family="core",
        when=lambda i, s, a: i.mode == "train" and bool(s.hasLabel) and s.labelType == "numeric",
    ))
    reg(StepSpec(
        "forecast", "/forecast/",
        depends_on=("derive",),
        produces=("model.forecast", "metrics"),
        base_cost=7.0, base_latency_ms=70_000, family="core",
        when=lambda i, s, a: i.mode == "train" and bool(s.hasTime) and bool(s.horizonDays) and not s.hasLabel,
    ))
    reg(StepSpec(
        "survival", "/survival/",
        depends_on=("derive",),
        produces=("model.survival", "metrics"),
        base_cost=7.0, base_latency_ms=75_000, family="core",
        when=lambda i, s, a: i.mode == "train" and bool(s.isCensored),
    ))
    reg(StepSpec(
        "clustering", "/clustering/",
        depends_on=("derive",),
        produces=("clusters", "centroids"),
        base_cost=4.0, base_latency_ms=40_000, family="core",
        when=lambda i, s, a: i.mode != "predict" and not s.hasLabel and not s.isCensored and not s.horizonDays,
    ))

    # Analysis / explainers (mark optional so caps can prune)
    reg(StepSpec(
        "feature_impact", "/feature_impact/",
        depends_on=("classification", "regression", "forecast", "survival"),
        produces=("shap_summary", "drivers"),
        base_cost=3.5, base_latency_ms=30_000, family="analysis", optional=True,
        when=lambda i, s, a: i.mode != "predict" and (
            (s.hasLabel and s.labelType in ("binary", "multiclass", "numeric"))
            or a.classification or a.regression or a.forecast or a.survival
        ),
    ))
    reg(StepSpec(
        "risk_analysis", "/risk_analysis/",
        depends_on=("classification",),
        produces=("ece", "coverage", "abstention", "stops"),
        base_cost=2.0, base_latency_ms=15_000, family="analysis", optional=True,
        when=lambda i, s, a: (s.hasLabel and s.labelType in ("binary", "multiclass")) or a.classification,
    ))
    reg(StepSpec(
        "decision_paths", "/decision_paths/",
        depends_on=("classification",),
        produces=("tree_paths", "surrogate_rules"),
        base_cost=2.5, base_latency_ms=20_000, family="analysis", optional=True,
        when=lambda i, s, a: (s.hasLabel and s.labelType in ("binary", "multiclass")) or a.classification,
    ))
    reg(StepSpec(
        "segment_analysis", "/segment_analysis/",
        depends_on=("derive",),
        produces=("segments", "lifts"),
        base_cost=2.0, base_latency_ms=12_000, family="analysis", optional=True,
        when=lambda i, s, a: i.mode != "predict",
    ))

    # Inference / what-if / counterfactuals / predict
    reg(StepSpec(
        name="classification_predict",
        endpoint="/classification/predict/",
        depends_on=("classification",),
        produces=("scores", "predictions"),
        base_cost=1.5,
        base_latency_ms=8_000,
        family="inference",
        optional=False,  # keep for predict/analyze
        when=lambda i, s, a: i.mode in ("predict", "analyze") and (
            a.classification or (s.hasLabel and s.labelType in ("binary", "multiclass"))
        ),
    ))
    reg(StepSpec(
        name="regression_predict",
        endpoint="/regression/predict/",
        depends_on=("regression",),
        produces=("scores", "predictions"),
        base_cost=1.5,
        base_latency_ms=8_000,
        family="inference",
        optional=False,  # keep for predict/analyze
        when=lambda i, s, a: i.mode in ("predict", "analyze") and (
            a.regression or (s.hasLabel and s.labelType == "numeric")
        ),
    ))
    reg(StepSpec(
        "what_if", "/what_if/",
        depends_on=("classification", "regression"),
        produces=("counterfactual_eval", "elasticities"),
        base_cost=2.0, base_latency_ms=12_000, family="inference", optional=True,
        when=lambda i, s, a: i.goal in ("what_if", "predict", "uplift") and (
            (s.hasLabel and s.labelType in ("binary", "multiclass", "numeric")) or
            a.classification or a.regression
        ),
    ))
    reg(StepSpec(
        "counterfactual", "/counterfactual/",
        depends_on=("classification", "regression"),
        produces=("dice_examples", "recourse"),
        base_cost=3.0, base_latency_ms=20_000, family="inference", optional=True,
        when=lambda i, s, a: i.goal in ("what_if", "predict", "uplift") and (a.classification or a.regression),
    ))
    reg(StepSpec(
        "ab_test", "/ab_test/",
        depends_on=("classification", "segment_analysis"),
        produces=("lift_estimates", "power", "sample_allocation"),
        base_cost=2.5, base_latency_ms=18_000, family="inference", optional=True,
        when=lambda i, s, a: i.goal in ("ab_test", "uplift", "predict") and a.classification,
    ))
    reg(StepSpec(
        "visualize", "/visualize/",
        depends_on=("derive",),
        produces=("dash_assets",),
        base_cost=1.0, base_latency_ms=8_000, family="aux", optional=True,
        when=lambda i, s, a: True,
    ))

    return R


# ---------------------------------------------------------------------
# Planning core
# ---------------------------------------------------------------------

@dataclass
class PlanSummary:
    score: float
    value: float
    total_cost: float
    total_latency_ms: int
    total_risk: float
    families: Set[str]
    chosen_core: Optional[str]


class TargetSpecModel(BaseModel):
    task: Optional[str] = "auto"
    target_column: Optional[str] = None
    ts_column: Optional[str] = None
    id_column: Optional[str] = None
    horizon_days: Optional[int] = None
    positive_class: Optional[Any] = None


@dataclass
class CompiledPlan:
    steps: List[StepDict]
    summary: PlanSummary


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _sem_val(inferred: Dict[str, Any], col: str) -> Optional[str]:
    v = inferred.get(col)
    if isinstance(v, dict):
        s = (v.get("semantic") or "").strip().lower()
        return s or None
    if isinstance(v, str):
        return v.strip().lower()
    return None


def _infer_label_type_from_preview(rows: List[Dict[str, Any]], target_col: str) -> Optional[str]:
    if not rows or not target_col:
        return None
    try:
        s = pd.Series([r.get(target_col) for r in rows])
        # numeric?
        as_num = pd.to_numeric(s, errors="coerce")
        if as_num.notna().mean() >= 0.70:
            # numeric but maybe low-cardinality -> classification
            nunique = as_num.nunique(dropna=True)
            n = len(as_num)
            if nunique <= 10 or (n > 0 and nunique / max(1, n) <= 0.05):
                return "binary" if nunique <= 2 else "multiclass"
            return "numeric"
        # categorical cases
        uniq = pd.Series(s.dropna()).unique().tolist()
        return "binary" if len(uniq) <= 2 else "multiclass"
    except Exception:
        # fallback heuristic without pandas
        values = [r.get(target_col) for r in rows if target_col in r]
        vs = [v for v in values if v is not None]
        uniq = {str(v) for v in vs}
        if len(uniq) == 0:
            return None
        if len(uniq) <= 2:
            return "binary"
        if len(uniq) <= 10:
            return "multiclass"
        return "numeric"


def _toposort(registry: Dict[str, StepSpec], selected: List[StepSpec]) -> List[StepSpec]:
    """
    Topologically sort selected steps using their 'depends_on'.
    Missing dependencies are ignored (treated as already satisfied).
    """
    name_map = {s.name: s for s in selected}
    in_deg: Dict[str, int] = {s.name: 0 for s in selected}
    children: Dict[str, Set[str]] = {s.name: set() for s in selected}

    for s in selected:
        for dep in s.depends_on:
            if dep in name_map:
                in_deg[s.name] += 1
                children[dep].add(s.name)

    queue = [n for n, d in in_deg.items() if d == 0]
    ordered: List[str] = []
    while queue:
        n = queue.pop(0)
        ordered.append(n)
        for c in children.get(n, ()):
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)

    if len(ordered) != len(selected):  # cycle fallback
        return selected
    return [name_map[n] for n in ordered]


def _gap_risk_multiplier(gaps: Mapping[str, Union[GapRecord, Dict[str, Any]]]) -> float:
    """
    Returns an additive risk in [0, 0.20] based on missing critical fields.
    Accepts GapRecord objects or JSON/dicts (e.g., from persisted meta).
    """
    def _get(v, key, default=None):
        if hasattr(v, key):        # dataclass/obj
            return getattr(v, key)
        if isinstance(v, dict):    # json/dict
            return v.get(key, default)
        return default

    add = 0.0
    for g in gaps.values():
        status = (_get(g, "status", "") or "").lower()
        if status != "missing":
            continue

        crit = (_get(g, "criticality", "") or "").lower()
        coverage = _get(g, "coverage", None)
        cov_factor = 1.0 - float(coverage) if isinstance(coverage, (int, float)) else 1.0

        if crit == "high":
            add += 0.08 * cov_factor
        elif crit == "medium":
            add += 0.04 * cov_factor
        else:
            add += 0.02 * cov_factor

    return min(add, 0.20)


def _compute_plan_score(
    steps: List[StepSpec],
    weights: Weights,
    intent: Intent,
    art: Artifacts,
    gaps: Optional[Mapping[str, Union[GapRecord, Dict[str, Any]]]] = None,
) -> Tuple[float, PlanSummary]:
    total_cost = sum(s.base_cost for s in steps)
    total_latency = sum(s.base_latency_ms for s in steps)

    step_risk = 1.0 - math.prod([1.0 - s.base_risk for s in steps])

    # Add gap-derived risk as another independent failure channel
    gap_add = _gap_risk_multiplier(gaps or {})
    agg_risk = 1.0 - (1.0 - step_risk) * (1.0 - gap_add)   # in [0,1]

    families = {s.family for s in steps if s.family != "aux"}
    chosen_core = next((s.name for s in steps if s.family == "core"), None)

    values = [_estimate_step_value(s, intent) for s in steps]
    diminishing = sum(v / (1.0 + 0.15 * i) for i, v in enumerate(sorted(values, reverse=True)))
    value = diminishing

    score = (
        weights.w_value * value
        - weights.w_speed * (total_latency / 60_000.0)
        - weights.w_cost * total_cost
        - weights.w_risk * (agg_risk * 10.0)
        + weights.w_coverage * len(families) * 0.1
    )

    summary = PlanSummary(
        score=score,
        value=value,
        total_cost=total_cost,
        total_latency_ms=total_latency,
        total_risk=agg_risk,
        families=families,
        chosen_core=chosen_core,
    )
    return score, summary


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
    """
    Greedily include essential steps (derive/core/predict) and add optional ones
    only if both caps (when present) won't be exceeded. Dependencies must still be
    satisfied upstream; we assume toposort happens after this call.
    """
    if budget_cap is None and latency_cap is None:
        return steps

    out: List[StepSpec] = []
    cost_acc = 0.0
    lat_acc = 0

    def is_essential(st: StepSpec) -> bool:
        if st.name == "derive":
            return True
        if st.family == "core":
            return True
        if st.family == "inference" and st.name.endswith("_predict"):
            return True
        return False

    for st in steps:
        next_cost = cost_acc + st.base_cost
        next_lat = lat_acc + st.base_latency_ms

        # Always keep essentials, even if caps would be exceeded
        if is_essential(st):
            out.append(st)
            cost_acc = next_cost
            lat_acc = next_lat
            continue

        # For optional/non-essential steps, include only if under caps
        over_budget = (budget_cap is not None and next_cost > budget_cap)
        over_latency = (latency_cap is not None and next_lat > latency_cap)

        if over_budget or over_latency:
            # skip this step
            continue

        out.append(st)
        cost_acc = next_cost
        lat_acc = next_lat

    return out

def _compute_event_rate_binary(values: List[Any], positive_class: Optional[Union[str, int, bool]] = None) -> Optional[float]:
    if not values:
        return None
    if positive_class is None:
        def _is_pos(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)) and pd.notna(v):
                return v != 0
            if isinstance(v, str):
                t = v.strip().lower()
                return t in {"true", "yes", "y", "1"}
            return bool(v)
        trues = sum(1 for v in values if _is_pos(v))
    else:
        trues = sum(1 for v in values if v == positive_class)
    denom = max(1, len(values))
    return trues / denom
# ---------------------------------------------------------------------
# Signal derivation from intake meta + target
# ---------------------------------------------------------------------

def signals_from_meta(meta: Dict[str, Any], target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Derive Signals-like dict from intake meta.
    Works whether meta came from dataclass->dict or a plain json dict.
    """
    out: Dict[str, Any] = {
        "hasLabel": None, "labelType": None, "hasTime": None, "isCensored": None,
        "horizonDays": None, "rows": None, "cols": None, "tsColumn": None, "idColumn": None, "eventRate": None,
    }

    meta = meta or {}
    stats = meta.get("stats") or {}
    hm    = meta.get("header_map") or {}
    inferred = meta.get("inferred_types") or {}
    schema = meta.get("schema") or {}
    schema_cols = schema.get("columns") or []

    # rows / cols
    out["rows"] = stats.get("normalized_rows") or stats.get("raw_rows") or None
    out["cols"] = (
        stats.get("normalized_cols")
        or stats.get("raw_cols")
        or (len(hm) if hm else (len(schema_cols) or None))
    )

    # schema role map if present
    role_map = {}
    if schema_cols:
        role_map = {c.get("name"): (c.get("role") or c.get("type")) for c in schema_cols if isinstance(c, dict)}

    # ---- idColumn
    for name, role in role_map.items():
        if isinstance(role, str) and role.lower() in {"id", "identifier", "primary_key", "customer_id", "order_id", "user_id"}:
            out["idColumn"] = name
            break
    if out["idColumn"] is None and hm:
        ID_CANON = {"id", "customer_id", "user_id", "account_id", "order_id"}
        for _, meta_c in hm.items():
            canon = (meta_c.get("canonical_hint") if isinstance(meta_c, dict) else getattr(meta_c, "canonical_hint", None))
            norm  = (meta_c.get("normalized_name") if isinstance(meta_c, dict) else getattr(meta_c, "normalized_name", None))
            if isinstance(canon, str) and isinstance(norm, str) and canon.strip().lower() in ID_CANON:
                out["idColumn"] = norm
                break
    if out["idColumn"] is None:
        for name in inferred.keys():
            if _sem_val(inferred, name) == "id":
                out["idColumn"] = name
                break

    # ---- tsColumn / hasTime
    for name, role in role_map.items():
        if isinstance(role, str) and role.lower() in {"timestamp", "datetime", "date", "event_time", "order_date"}:
            out["tsColumn"] = name
            out["hasTime"] = True
            break
    if out["tsColumn"] is None:
        for name in inferred.keys():
            if (_sem_val(inferred, name) or "") in _TIME:
                out["tsColumn"] = name
                out["hasTime"] = True
                break
    if out["tsColumn"] is None and hm:
        for _, meta_c in hm.items():
            canon = (meta_c.get("canonical_hint") if isinstance(meta_c, dict) else getattr(meta_c, "canonical_hint", None))
            norm  = (meta_c.get("normalized_name") if isinstance(meta_c, dict) else getattr(meta_c, "normalized_name", None))
            if isinstance(canon, str) and isinstance(norm, str) and canon.lower() in {"created_at", "updated_at", "order_date"}:
                out["tsColumn"] = norm
                out["hasTime"] = True
                break

    # ---- hasLabel / labelType, prefer explicit target if given
    def set_label(sem: str) -> bool:
        s = (sem or "").lower()
        if s in _BOOL:
            out["hasLabel"] = True
            out["labelType"] = "binary"
            return True
        if s in _NUM:
            out["hasLabel"] = True
            out["labelType"] = "numeric"
            return True
        if s in _CAT:
            out["hasLabel"] = True
            out["labelType"] = "multiclass"
            return True
        return False

    target_norm = None
    if target_column:
        for _, v in (hm or {}).items():
            if v.get("original_name") == target_column or v.get("normalized_name") == target_column:
                target_norm = v.get("normalized_name") or target_column
                break
        target_norm = target_norm or target_column
        sem = _sem_val(inferred, target_norm)
        if sem:
            set_label(sem)

    if out["hasLabel"] is None:
        for name in inferred.keys():
            sem = _sem_val(inferred, name)
            if sem and set_label(sem):
                break

    # default horizon for unlabeled time series
    if out["hasTime"] and not out["hasLabel"] and out.get("horizonDays") is None:
        out["horizonDays"] = 14

    # optional event rate if provided in meta
    clb = (meta.get("class_balance") or {})
    if isinstance(clb.get("positive_rate"), (int, float)):
        out["eventRate"] = float(clb["positive_rate"])

    return out

# --- derive basic signals from target + preview -------------------------------

def map_target_to_signals(target: "TargetSpecModel", preview_rows: Optional[List[Dict[str, Any]]]) -> "SignalsIn":
    """
    Minimal local implementation. Uses preview to infer label type, event rate, and time/id columns.
    """
    rows = preview_rows or []
    tgt_col = target.target_column
    ts_col = target.ts_column
    id_col = target.id_column

    label_type = None
    event_rate = None

    if tgt_col:
        label_type = _infer_label_type_from_preview(rows, tgt_col)
        if label_type == "binary":
            values = [r.get(tgt_col) for r in rows if tgt_col in r]
            event_rate = _compute_event_rate_binary(values, target.positive_class)

    has_label = tgt_col is not None
    has_time = ts_col is not None
    horizon = target.horizon_days

    # Dimensions from preview
    n_rows = len(rows)
    n_cols = len(rows[0]) if rows else None

    return SignalsIn(
        hasLabel=has_label,
        labelType=label_type,
        hasTime=has_time,
        horizonDays=horizon,
        eventRate=event_rate,
        rows=n_rows,
        cols=n_cols,
        tsColumn=ts_col,
        idColumn=id_col,
    )

def merge_signals(base: dict | None, fill: dict | "SignalsIn") -> dict:
    out = dict(base or {})
    if hasattr(fill, "model_dump"):
        src = fill.model_dump()
    elif hasattr(fill, "dict"):
        src = fill.dict()
    elif hasattr(fill, "__dataclass_fields__"):
        from dataclasses import asdict
        src = asdict(fill)
    else:
        src = dict(fill or {})

    for k in ["hasLabel","labelType","hasTime","isCensored","horizonDays",
              "eventRate","rows","cols","tsColumn","idColumn"]:
        if out.get(k) in (None, "", [], {}):
            v = src.get(k)
            if v is not None:
                out[k] = v
    return out


def map_target_to_signals_dict(t: Union[TargetSpecModel, Dict[str, Any], None],
                               preview_rows: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    if t is None:
        return {}

    if hasattr(t, "model_dump"):
        t = t.model_dump()

    task = (t.get("task") or "auto").lower()
    tgt  = t.get("target_column")
    pos  = t.get("positive_class")
    ts_c = t.get("ts_column")
    id_c = t.get("id_column")
    hz   = t.get("horizon_days")

    has_time = bool(ts_c) if ts_c else None
    is_cens  = True if task == "survival" else None
    has_label = None
    label_type = None

    if task == "segment":
        has_label = False
    elif task == "survival":
        has_label = True
        has_time = True if has_time is None else has_time
        hz = hz or 14
    elif task in ("forecast", "time_series", "ts_forecast"):
        has_time = True if has_time is None else has_time
        hz = hz or 14
        has_label = False if has_label is None else has_label
    else:
        if tgt:
            has_label = True
            if task == "regression":
                label_type = "numeric"
            elif task == "classification":
                label_type = _infer_label_type_from_preview(preview_rows or [], tgt) or "binary"
            else:
                label_type = _infer_label_type_from_preview(preview_rows or [], tgt)

    if has_time and task == "auto" and hz is None:
        hz = 14

    # eventRate if binary
    event_rate = None
    if has_label and label_type == "binary" and tgt and preview_rows:
        vals = [r.get(tgt) for r in preview_rows if tgt in r]

        def _is_pos(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                # treat nonzero as positive
                try:
                    return float(v) != 0.0
                except Exception:
                    return False
            if isinstance(v, str):
                return v.strip().lower() in {"true", "yes", "y", "1"}
            return bool(v)

        if pos is None:
            trues = sum(1 for v in vals if _is_pos(v))
        else:
            trues = sum(1 for v in vals if v == pos)
        denom = max(1, len(vals))
        event_rate = trues / denom

    return {
        "hasLabel": has_label,
        "labelType": label_type,
        "hasTime": has_time,
        "isCensored": is_cens,
        "horizonDays": hz,
        "eventRate": event_rate,
        "rows": len(preview_rows) if preview_rows else None,
        "cols": (len(preview_rows[0]) if preview_rows else None),
        "tsColumn": ts_c,
        "idColumn": id_c,
    }


# ---------------------------------------------------------------------
# Merge helpers & “derive_signals_for_compile”
# ---------------------------------------------------------------------

FILL_FIELDS = [
    "hasLabel", "labelType", "hasTime", "isCensored", "horizonDays",
    "eventRate", "rows", "cols", "tsColumn", "idColumn",
]


def _merge_missing(base: Optional[dict], fill: Union[dict, Any]) -> dict:
    out = dict(base or {})
    src = dict(fill or {})
    for k in FILL_FIELDS:
        if out.get(k) in (None, "", [], {}):
            v = src.get(k)
            if v is not None:
                out[k] = v
    return out


def load_preview_for_dataset(dataset_id: Optional[int]) -> Optional[List[Dict[str, Any]]]:
    store = globals().get("PREVIEW_STORE")
    if store and dataset_id is not None:
        try:
            return store.get(dataset_id)
        except Exception:
            return None
    return None
import pandas as pd     # ✅ valid
def ensure_minimal_event_schema(
    df: pd.DataFrame,
    kind: str,
    warnings: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Ensure df has at least: event_id, value, timestamp (optional), entity_id (optional).
    Returns (df_with_cols, meta_flags).
    """
    df = df.copy()
    meta = {
        "synthetic_event_id": False,
        "synthetic_entity_id": False,
        "synthetic_timestamp": False,
        "synthetic_value": False,
    }

    # 1) event_id (core key)
    event_candidates = ["order_id", "id", "event_id", "transaction_id"]
    for c in event_candidates:
        if c in df.columns:
            df["event_id"] = df[c].astype(str)
            break
    else:
        df["event_id"] = df.index.astype(str)
        meta["synthetic_event_id"] = True
        warnings.append("Synthetic event_id created from row index (no id-like column found).")

    # 2) value (amount)
    value_candidates = ["amount", "total", "revenue", "price", "value"]
    for c in value_candidates:
        if c in df.columns:
            df["value"] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            break
    else:
        # fallback: count-based value = 1 per row
        df["value"] = 1.0
        meta["synthetic_value"] = True
        warnings.append("No amount-like column found; using value=1.0 per event.")

    # 3) timestamp (optional but useful)
    ts_candidates = ["order_date", "created_at", "timestamp", "event_time", "date"]
    ts_col = None
    for c in ts_candidates:
        if c in df.columns:
            ts_col = c
            parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
            if parsed.notna().any():
                df["timestamp"] = parsed
                break
            else:
                ts_col = None

    if "timestamp" not in df.columns:
        # synthetic monotonic timeline
        base = pd.Timestamp("2020-01-01", tz="UTC")
        df["timestamp"] = base + pd.to_timedelta(range(len(df)), unit="D")
        meta["synthetic_timestamp"] = True
        warnings.append("No usable date column; synthetic timestamp sequence assigned.")

    # 4) entity_id (customer or group, optional)
    entity_candidates = ["customer_id", "user_id", "account_id", "email"]
    for c in entity_candidates:
        if c in df.columns:
            df["entity_id"] = df[c].astype(str)
            break
    else:
        # Optional: don't always force an entity; you can leave it missing
        df["entity_id"] = df["event_id"]
        meta["synthetic_entity_id"] = True
        warnings.append("No entity/customer column; using event_id as entity_id (no grouping).")

    return df, meta


def derive_signals_for_compile(
    *,
    payload,                     # CompileRequest-compatible (attrs: dataset_id, target, signals?)
    current_user,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (signals_dict, meta_used_debug)
    Merge priority: intake-meta -> stored -> target-map -> explicit payload.signals
    """
    # 0) inputs
    dataset_id = getattr(payload, "dataset_id", None)
    target_obj = getattr(payload, "target", None)

    # 1) load meta + auto-signals
    auto_signals = {}
    meta = None
    if dataset_id is not None:
        try:
            loader = globals().get("load_or_backfill_intake_meta")
            if callable(loader):
                meta = loader(
                    user_id=getattr(current_user, "id", None) or getattr(current_user, "user_id", None),
                    dataset_id=dataset_id,
                    current_user=current_user,
                )
            tcol = None
            if target_obj is not None:
                if hasattr(target_obj, "target_column"):
                    tcol = getattr(target_obj, "target_column", None)
                elif hasattr(target_obj, "model_dump"):
                    tcol = (target_obj.model_dump() or {}).get("target_column")
            auto_signals = signals_from_meta(meta or {}, target_column=tcol) or {}
        except Exception:
            auto_signals = {}

    # 2) stored signals (if you keep any) — optional, no storage required
    signals_dict: Dict[str, Any] = dict(auto_signals)
    if dataset_id is not None:
        mdl = globals().get("SIGNALS_STORE")
        if mdl:
            try:
                stored = mdl.get(dataset_id)
                if stored:
                    if isinstance(stored, dict):
                        signals_dict = _merge_missing(signals_dict, stored)
                    elif hasattr(stored, "dict"):
                        signals_dict = _merge_missing(signals_dict, stored.dict())
            except Exception:
                pass

    # 3) target-mapped + preview
    preview_rows = load_preview_for_dataset(dataset_id)
    signals_dict = _merge_missing(signals_dict, map_target_to_signals_dict(target_obj, preview_rows))

    # 4) explicit payload.signals (highest priority, direct update)
    if getattr(payload, "signals", None) is not None:
        s_in = payload.signals.model_dump() if hasattr(payload.signals, "model_dump") else dict(payload.signals)
        signals_dict.update({k: v for k, v in s_in.items() if v is not None})

    # 5) final nudge: if time present but unlabeled, default horizon
    if signals_dict.get("hasTime") and not signals_dict.get("hasLabel") and not signals_dict.get("horizonDays"):
        signals_dict["horizonDays"] = 14

    meta_used = {
        "rows": auto_signals.get("rows"),
        "cols": auto_signals.get("cols"),
        "tsColumn": auto_signals.get("tsColumn"),
        "idColumn": auto_signals.get("idColumn"),
        "hasLabel": auto_signals.get("hasLabel"),
        "labelType": auto_signals.get("labelType"),
        "hasTime": auto_signals.get("hasTime"),
        "horizonDays": auto_signals.get("horizonDays"),
    }
    return signals_dict, meta_used


# ---------------------------------------------------------------------
# Budget summary
# ---------------------------------------------------------------------

def _safe_num(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def summarize_budget(
    steps: List[Dict[str, Any]],
    balance: Optional[float] = None,
    requested: Optional[float] = None,
    cap: Optional[float] = None,
) -> Dict[str, Any]:
    est_total_cost = 0.0
    cap_used = _safe_num(cap) if cap is not None else None

    try:
        est_total_cost = sum(_safe_num(s.get("est_cost", 0.0)) for s in (steps or []))
    except Exception:
        est_total_cost = 0.0

    return {
        "balance": _safe_num(balance) if balance is not None else None,
        "requested": _safe_num(requested) if requested is not None else None,
        "cap_used": cap_used if cap_used is not None else est_total_cost,
        "est_total_cost": est_total_cost,
    }


# ---------------------------------------------------------------------
# Public API: compile_plan
# ---------------------------------------------------------------------

def compile_plan(
    intent: Dict[str, Any],
    signals: Dict[str, Any],
    artifacts: Dict[str, bool],
    top_k: int = 1,  # reserved for future multi-branch output
    gaps: Optional[Mapping[str, Union[GapRecord, Dict[str, Any]]]] = None,
) -> CompiledPlan:
    i = Intent(**intent)
    s = Signals(**signals)
    a = Artifacts(**artifacts)

    # Nudge signals from required families to keep them eligible
    req = set((i.constraints or {}).get("required") or [])
    if "classification" in req:
        s.hasLabel = True if s.hasLabel is None else s.hasLabel
        s.labelType = s.labelType or "binary"
    if "regression" in req:
        s.hasLabel = True if s.hasLabel is None else s.hasLabel
        s.labelType = "numeric"
    if "forecast" in req:
        s.hasTime = True if s.hasTime is None else s.hasTime
        s.horizonDays = s.horizonDays or 14
        if s.hasLabel is None:
            s.hasLabel = False

    reg = get_registry()
    W = weights_for_preset(i.risk_preset)
    cons = i.constraints or {}
    if cons.get("ignoreLatencyScore", False) and cons.get("latencyMs") is None:
        W.w_speed = 0.0

    # 1) Candidate core families based on signals
    families: List[List[str]] = []
    if i.mode == "train":
        if s.hasLabel and s.labelType in ("binary", "multiclass"):
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
        if a.classification:
            arts.append(["classification"])
        if a.regression:
            arts.append(["regression"])
        if a.forecast:
            arts.append(["forecast"])
        if a.survival:
            arts.append(["survival"])
        if not arts:
            # fallback like train
            return compile_plan({**intent, "mode": "train"}, signals, artifacts, top_k, gaps)
        families = arts

    if not families:
        families = [["clustering"]]  # default fallback

    # 2) Build branch candidates
    branches: List[List[StepSpec]] = []
    for fam in families:
        selected = [reg["derive"]]
        core = reg[fam[0]]
        if core.eligible(i, s, a):
            selected.append(core)

        # analysis add-ons (only if will be satisfied)
        if fam[0] in ("classification", "regression", "forecast", "survival"):
            if reg["feature_impact"].eligible(i, s, a):
                selected.append(reg["feature_impact"])
        if fam[0] == "classification":
            if reg["risk_analysis"].eligible(i, s, a):
                selected.append(reg["risk_analysis"])
            if reg["decision_paths"].eligible(i, s, a):
                selected.append(reg["decision_paths"])

        # segment analysis is broadly useful
        if reg["segment_analysis"].eligible(i, s, a):
            selected.append(reg["segment_analysis"])

        # inference (mode-aware)
        if i.mode in ("predict", "analyze"):
            if fam[0] == "classification":
                selected.append(reg["classification_predict"])
            elif fam[0] == "regression":
                selected.append(reg["regression_predict"])

            if fam[0] in ("classification", "regression"):
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

        # append required if eligible (and not already present)
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
        score, summary = _compute_plan_score(sel, W, i, a, gaps)
        # Cache bonus: if artifact exists for core step, boost
        core_name = summary.chosen_core
        if core_name and getattr(a, core_name, False):
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
