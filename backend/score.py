from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import math
import re



# from .storage import load_intake_artifacts
# from .auth import get_current_active_user
from storage import load_intake_artifacts
from auth import get_current_active_user


router = APIRouter()

# ---------- Models ----------

class Issue(BaseModel):
    category: str                     # e.g., "Schema", "DataQuality"
    severity: str                     # "low" | "medium" | "high"
    message: str                      # human-readable reasoning
    impact_on_tasks: List[str]        # e.g., ["classification", "regression", "forecast"]
    suggestions: List[str]            # concrete fixes
    evidence: Dict[str, Any] = {}     # small structured facts (rates, cols, etc.)

class ScoreResponse(BaseModel):
    ok: bool = True
    dataset_id: int
    overall: int                      # 0..100
    subscores: Dict[str, int]         # category → 0..weight
    task_viability: Dict[str, Any]    # viability notes per task
    issues: List[Issue]
    sample_size: int
    stats: Dict[str, Any]             # echo a few useful stats (rows/cols etc.)

# ---------- Helpers (PURE, fast) ----------

EXPECTED_KEYS = {
    # flexible via canonical hints from your header_map (from intake)
    "id": {"hints": ["id", "order_id", "customer_id", "appointment_id"]},
    "timestamp": {"hints": ["created_at", "updated_at", "timestamp", "date"]},
}

def _present_by_canonical(header_map: Dict[str, Any], hint: str) -> bool:
    return any((m.get("canonical_hint") == hint) for m in header_map.values())

def _dup_or_blank_headers(header_map: Dict[str, Any]) -> Dict[str, Any]:
    norm_counts = {}
    blanks = []
    for meta in header_map.values():
        norm = meta.get("normalized_name") or ""
        if not norm.strip():
            blanks.append(meta.get("original_name"))
        norm_counts[norm] = 1 + norm_counts.get(norm, 0)
    dups = [k for k, v in norm_counts.items() if v > 1 and k]
    return {"dups": dups, "blanks": blanks}

def _non_ascii_headers(header_map: Dict[str, Any]) -> List[str]:
    bad = []
    for meta in header_map.values():
        name = (meta.get("original_name") or "")
        try:
            name.encode("ascii")
        except Exception:
            bad.append(name)
    return bad

def _collect_semantics(inferred_types: Dict[str, Any]) -> Dict[str, List[str]]:
    by_sem = {}
    for col, ti in inferred_types.items():
        by_sem.setdefault(ti.get("semantic"), []).append(col)
    return by_sem

def _top_category_dominance(df_sample: Optional[list], cat_cols: List[str], threshold=0.9) -> List[Dict[str, Any]]:
    """
    df_sample is a list[dict] (e.g., preview.normalized).
    Very light dominance check to avoid heavy compute.
    """
    issues = []
    if not df_sample or not cat_cols:
        return issues
    # approximate with small frequency maps
    for c in cat_cols[:10]:
        freq = {}
        n = 0
        for row in df_sample[:500]:
            if c in row:
                v = row[c]
                if v is None:
                    continue
                freq[v] = freq.get(v, 0) + 1
                n += 1
        if n >= 10 and freq:
            top = max(freq.values())
            if top / n >= threshold:
                issues.append({"column": c, "dominance": round(top / max(1, n), 3), "n": n})
    return issues

def _outlier_burden(df_sample: Optional[list]) -> float:
    """
    If your normalization created *_capped columns, measure share of capped values in sample.
    """
    if not df_sample:
        return 0.0
    capped_counts = 0
    total = 0
    for row in df_sample[:500]:
        for k, v in row.items():
            if k.endswith("_capped"):
                total += 1
                # presence is enough, this is a heuristic
                if v is not None:
                    capped_counts += 1
    if total == 0:
        return 0.0
    return capped_counts / total

def _timestamp_quality(inferred_types: Dict[str, Any]) -> float:
    """
    Use valid_rate of timestamp columns as a quick proxy.
    """
    rates = [ti.get("valid_rate", 0.0) for ti in inferred_types.values() if ti.get("semantic") == "timestamp"]
    return 1.0 if not rates else sum(rates) / len(rates)

def _nullish_columns(df_sample: Optional[list]) -> List[str]:
    if not df_sample:
        return []
    cols = set()
    for row in df_sample:
        cols.update(row.keys())
    bad = []
    for c in list(cols)[:80]:
        seen = 0
        nulls = 0
        for row in df_sample:
            if c in row:
                seen += 1
                if row[c] in (None, "", "NaN", "nan"):
                    nulls += 1
        if seen >= 5 and (nulls / seen) >= 0.9:
            bad.append(c)
    return bad

def _mixed_type_hints(inferred_types: Dict[str, Any]) -> List[str]:
    """
    If semantic is 'numeric' but valid_rate is low, it's a hint of mixing.
    """
    bad = []
    for col, ti in inferred_types.items():
        if ti.get("semantic") in ("numeric", "money", "percent") and ti.get("valid_rate", 1) < 0.7:
            bad.append(col)
    return bad

# ---------- Main route ----------

@router.post("/datasets/{dataset_id}/score", response_model=ScoreResponse)
def score_dataset(dataset_id: int):  # keep your auth dep
    """
    Compute a Dataset Readiness Score from existing intake artifacts.
    Assumes you can fetch: stats, meta, and small normalized preview for this dataset.
    """
    # --- 0) Get intake artifacts you already store ---
    # Replace these with your actual lookups:
    User = Depends(get_current_active_user)
    intake = load_intake_artifacts(dataset_id)  # { "meta": {...}, "stats": {...}, "preview": {...} }
    if not intake:
        raise HTTPException(status_code=404, detail="No intake artifacts found for dataset")

    meta = intake["meta"]
    stats = intake["stats"]                           # {raw_rows, raw_cols, normalized_rows, normalized_cols}
    df_sample = intake["preview"]["normalized"] or [] # list[dict] (small)

    header_map = meta["header_map"]
    inferred = meta["inferred_types"]
    anomalies = meta.get("anomalies", [])
    number_format = meta.get("number_format", {})
    by_sem = _collect_semantics(inferred)

    # --- 1) Initialize subscore buckets ---
    subs = {
        "Schema": 30,
        "DataQuality": 30,
        "StatIntegrity": 20,
        "FileLevel": 10,
        "LowSignal": 5,
        "OpsRisk": 5,
    }
    issues: List[Issue] = []

    # --- 2) Schema checks ---
    # Missing expected canonicals
    if not any(_present_by_canonical(header_map, h) for h in EXPECTED_KEYS["id"]["hints"]):
        subs["Schema"] -= 8
        issues.append(Issue(
            category="Schema", severity="high",
            message="No obvious ID column detected.",
            impact_on_tasks=["classification", "regression", "forecast"],
            suggestions=["Add a stable identifier (e.g., order_id, customer_id)."],
            evidence={"expected": EXPECTED_KEYS["id"]["hints"]}
        ))
    if not any(_present_by_canonical(header_map, h) for h in EXPECTED_KEYS["timestamp"]["hints"]):
        subs["Schema"] -= 8
        issues.append(Issue(
            category="Schema", severity="medium",
            message="No timestamp-like column detected.",
            impact_on_tasks=["forecast", "temporal_features"],
            suggestions=["Include created_at/processed_at or another event time field."],
            evidence={"expected": EXPECTED_KEYS["timestamp"]["hints"]}
        ))

    dups_blanks = _dup_or_blank_headers(header_map)
    if dups_blanks["dups"]:
        subs["Schema"] -= 6
        issues.append(Issue(
            category="Schema", severity="medium",
            message=f"Duplicate normalized headers: {dups_blanks['dups']}",
            impact_on_tasks=["all"],
            suggestions=["Rename duplicates or disambiguate columns before training."],
            evidence={"dups": dups_blanks["dups"]}
        ))
    if dups_blanks["blanks"]:
        subs["Schema"] -= 4
        issues.append(Issue(
            category="Schema", severity="low",
            message="Blank/empty header names found.",
            impact_on_tasks=["all"],
            suggestions=["Give every column a descriptive name."],
            evidence={"blanks": dups_blanks["blanks"]}
        ))

    # Misleading headers: numeric semantics with poor valid_rate
    mis = _mixed_type_hints(inferred)
    if mis:
        subs["Schema"] -= 4
        issues.append(Issue(
            category="Schema", severity="medium",
            message=f"Header suggests numeric/money/percent but values often invalid: {mis}",
            impact_on_tasks=["regression"],
            suggestions=["Fix number format or strip symbols at source."],
            evidence={"columns": mis}
        ))

    # --- 3) Data Quality ---
    rows = int(stats.get("raw_rows", 0))
    cols = int(stats.get("raw_cols", 0))
    if rows == 0 or cols == 0:
        subs["DataQuality"] = max(0, subs["DataQuality"] - 30)
        subs["FileLevel"] = max(0, subs["FileLevel"] - 10)
        issues.append(Issue(
            category="FileLevel", severity="high",
            message="Empty or unreadable file.",
            impact_on_tasks=["all"],
            suggestions=["Upload a valid CSV/Excel/Parquet with at least a few dozen rows."],
            evidence={"rows": rows, "cols": cols}
        ))
    elif rows < 30:
        subs["DataQuality"] -= 10
        issues.append(Issue(
            category="DataQuality", severity="medium",
            message=f"Very few rows ({rows}); not enough for robust modeling.",
            impact_on_tasks=["classification", "regression"],
            suggestions=["Collect more data; aim for 100–1,000+ rows depending on task."],
            evidence={"rows": rows}
        ))
    elif rows < 100:
        subs["DataQuality"] -= 5
        issues.append(Issue(
            category="DataQuality", severity="low",
            message=f"Limited rows ({rows}); models may be unstable.",
            impact_on_tasks=["classification", "regression"],
            suggestions=["Gather more samples or use simpler models/regularization."],
            evidence={"rows": rows}
        ))

    nullish = _nullish_columns(df_sample)
    if nullish:
        subs["DataQuality"] -= 6
        issues.append(Issue(
            category="DataQuality", severity="medium",
            message=f"Columns mostly null in sample: {nullish}",
            impact_on_tasks=["all"],
            suggestions=["Drop or impute high-null columns."],
            evidence={"columns": nullish}
        ))

    # JSON-in-cell failures hinted via anomalies list
    json_flags = [a for a in anomalies if "json" in a.lower()]
    if json_flags:
        subs["DataQuality"] -= 4
        issues.append(Issue(
            category="DataQuality", severity="low",
            message="JSON-in-cell detected with parse issues.",
            impact_on_tasks=["feature_generation"],
            suggestions=["Fix malformed JSON at source or disable JSON expansion for this column."],
            evidence={"anomalies": json_flags}
        ))

    # --- 4) Statistical & Integrity ---
    outlier_rate = _outlier_burden(df_sample)  # 0..1
    if outlier_rate >= 0.3:
        subs["StatIntegrity"] -= 8
        issues.append(Issue(
            category="StatIntegrity", severity="medium",
            message=f"Heavy outlier capping observed (~{round(outlier_rate*100)}% of numeric caps).",
            impact_on_tasks=["regression"],
            suggestions=["Investigate upstream errors, consider robust scalers or domain caps."],
            evidence={"outlier_cap_share": outlier_rate}
        ))

    cat_cols = _collect_semantics(inferred).get("categorical", [])
    dom = _top_category_dominance(df_sample, cat_cols, threshold=0.9)
    if dom:
        subs["StatIntegrity"] -= 6
        issues.append(Issue(
            category="StatIntegrity", severity="low",
            message="Dominant single-category columns (≥90%).",
            impact_on_tasks=["classification"],
            suggestions=["Balance classes or downweight dominant class."],
            evidence={"columns": dom}
        ))

    ts_quality = _timestamp_quality(inferred)
    if ts_quality < 0.6 and any(ti.get("semantic") == "timestamp" for ti in inferred.values()):
        subs["StatIntegrity"] -= 6
        issues.append(Issue(
            category="StatIntegrity", severity="medium",
            message="Timestamp parsing is unreliable (low valid rate).",
            impact_on_tasks=["forecast", "temporal_features"],
            suggestions=["Normalize date formats (ISO 8601) or fix mixed locales/timezones."],
            evidence={"avg_valid_rate": ts_quality}
        ))

    # Locale mismatches (decimal/thousands)
    dec = number_format.get("decimal")
    thou = number_format.get("thousands")
    if dec and thou and dec == thou:
        subs["StatIntegrity"] -= 4
        issues.append(Issue(
            category="StatIntegrity", severity="low",
            message="Decimal and thousands separators appear conflicting.",
            impact_on_tasks=["regression"],
            suggestions=["Ensure decimal and thousands separators are distinct and consistent."],
            evidence={"decimal": dec, "thousands": thou}
        ))

    # --- 5) File-level (readability etc.) ---
    # Most of these would have been caught pre-intake; keep a small proxy:
    if "password" in " ".join(anomalies).lower():
        subs["FileLevel"] -= 5
        issues.append(Issue(
            category="FileLevel", severity="medium",
            message="Password-protected or unreadable Excel hinted.",
            impact_on_tasks=["all"],
            suggestions=["Export an unprotected version or use CSV/Parquet."],
            evidence={}
        ))

    # --- 6) Low-signal ---
    foreign = _non_ascii_headers(header_map)
    if foreign:
        subs["LowSignal"] -= 2
        issues.append(Issue(
            category="LowSignal", severity="low",
            message="Non-ASCII headers may not match lexicon.",
            impact_on_tasks=["feature_generation"],
            suggestions=["Add translations/aliases for headers or expand lexicon."],
            evidence={"headers": foreign[:10]}
        ))

    # --- 7) Ops / Patch risk (proxy) ---
    risky = [a for a in anomalies if re.search(r"(failed|error|mismatch|impossible)", a, re.I)]
    if risky:
        subs["OpsRisk"] -= 3
        issues.append(Issue(
            category="OpsRisk", severity="low",
            message="Intake anomalies suggest patch/apply risk.",
            impact_on_tasks=["pipeline_stability"],
            suggestions=["Address anomalies before automated patching; run in sandbox first."],
            evidence={"anomalies": risky[:5]}
        ))

    # --- 8) Task viability summary ---
    # Simple rules of thumb:
    viable_cls = rows >= 100 and bool(cat_cols)
    viable_reg = rows >= 100 and any(ti.get("semantic") in ("numeric","money","percent") for ti in inferred.values())
    viable_fc  = rows >= 200 and any(ti.get("semantic") == "timestamp" for ti in inferred.values())

    task_viability = {
        "classification": {
            "viable": bool(viable_cls),
            "reason": "Requires ≥100 rows and at least one categorical predictor/target.",
        },
        "regression": {
            "viable": bool(viable_reg),
            "reason": "Requires ≥100 rows and at least one numeric/money/percent target.",
        },
        "forecast": {
            "viable": bool(viable_fc),
            "reason": "Requires ≥200 rows and a reliable timestamp column.",
        },
    }

    # --- 9) Finalize ---
    # Clamp subs to >=0, then sum to overall
    for k in list(subs.keys()):
        subs[k] = max(0, int(round(subs[k])))

    overall = sum(subs.values())
    return ScoreResponse(
        dataset_id=dataset_id,
        overall=overall,
        subscores=subs,
        task_viability=task_viability,
        issues=issues,
        sample_size=len(df_sample),
        stats=stats,
    )
