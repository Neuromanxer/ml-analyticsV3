from __future__ import annotations
from openai import OpenAI  # Or whatever LLM or inference engine you use
import json
# Reuse your models / stores / auth
from planner_router import IntentIn, SignalsIn, ArtifactsIn, TargetSpecModel, PREVIEW_STORE, SIGNALS_STORE
from auth import get_current_active_user

# from .planner_router import IntentIn, SignalsIn, ArtifactsIn, TargetSpecModel, PREVIEW_STORE, SIGNALS_STORE
# from .auth import get_current_active_user
def generate_insights(task_type: str, payload: dict) -> str:
    """
    Generates business recommendations from model results.
    :param task_type: 'classification', 'regression', etc.
    :param payload: The structured result from your ML pipeline
    :return: Plain English business suggestions
    """
    prompt = f"""
You are a business data analyst. Based on the following {task_type} model results, give the user:
- A brief performance summary
- Plain-English interpretation of metrics (accuracy, F1, etc.)
- Insights into top features
- Business actions or follow-up suggestions

Here are the results:
{json.dumps(payload, indent=2)}
"""

    # Example OpenAI call
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful data science assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response["choices"][0]["message"]["content"]
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os, shutil, tempfile, subprocess, uuid, re, shlex, sys, json

# Optional POSIX-only resource limiting
try:
    import resource
except Exception:
    resource = None  # Windows / non-POSIX

router = APIRouter()

# ---------- Models ----------

class PatchRequest(BaseModel):
    patch: str = Field(..., description="Unified diff content")
    sandbox_id: Optional[str] = None
    # Where to copy a baseline project from (safe, read-only source you control)
    base_path: Optional[str] = Field(None, description="Path to a baseline project to copy into sandbox")
    # Commands to run after patch applies (each as a full command string)
    run: Optional[List[str]] = Field(default=None, description="Commands to run in sandbox, e.g. ['pytest -q']")
    # Fail-fast: if any command fails, stop and return
    stop_on_failure: bool = True
    # Max wall-clock time per command (seconds)
    timeout_sec: int = 60

class PatchResult(BaseModel):
    ok: bool
    sandbox_id: str
    message: str
    applied_files: List[str] = []
    logs: str = ""
    run_results: Optional[List[Dict[str, Any]]] = None

# ---------- Helpers ----------

def _prepare_sandbox_dir(sandbox_id: Optional[str]) -> str:
    sid = sandbox_id or str(uuid.uuid4())
    path = os.path.join(tempfile.gettempdir(), f"sandbox_{sid}")
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path

def _copy_base(path_from: Optional[str], path_to: str):
    if not path_from:
        return
    if not os.path.isdir(path_from):
        raise HTTPException(status_code=400, detail=f"base_path not found or not a directory: {path_from}")
    # Copy only your safe subtree (ignore venvs, .git, big dirs)
    def _ignore(dir, names):
        ignore_set = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules"}
        return [n for n in names if n in ignore_set]
    shutil.copytree(path_from, path_to, dirs_exist_ok=True, ignore=_ignore)

PATCH_FILE_RE = re.compile(r'^\+\+\+\s+[ab]/(.+)$', re.M)  # naive collector

def _parse_applied_files_from_patch(diff_text: str) -> List[str]:
    files = PATCH_FILE_RE.findall(diff_text or "")  # ‘+++ b/path/to/file’
    # De-duplicate and normalize
    out = []
    seen = set()
    for f in files:
        f2 = f.strip()
        if f2 and f2 not in seen:
            out.append(f2)
            seen.add(f2)
    return out

def _apply_patch(sandbox_dir: str, patch_text: str) -> str:
    patch_path = os.path.join(sandbox_dir, "patch.diff")
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(patch_text)

    # Try `patch` utility first
    try:
        proc = subprocess.run(
            ["patch", "-p1", "-i", patch_path],
            cwd=sandbox_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout
        patch_logs = proc.stdout or ""
    except FileNotFoundError:
        patch_logs = "patch utility not found; will try git apply.\n"
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Applying patch timed out")

    # Fallback to `git apply` if available
    try:
        # Initialize a git repo if not already
        if not os.path.isdir(os.path.join(sandbox_dir, ".git")):
            subprocess.run(["git", "init"], cwd=sandbox_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        proc2 = subprocess.run(
            ["git", "apply", "--whitespace=fix", "patch.diff"],
            cwd=sandbox_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        patch_logs += "\n" + (proc2.stdout or "")
        if proc2.returncode == 0:
            return patch_logs
        raise HTTPException(status_code=400, detail=f"Patch failed to apply.\n{patch_logs}")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Neither `patch` nor `git` available.\n{patch_logs}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Applying patch (git) timed out")

# Allowlist for commands we’ll run (you can expand this)
ALLOWED_BINARIES = {"pytest", "python", "python3", "pip", "ruff", "flake8", "pytest.exe", "python.exe"}

def _split_cmd(cmd: str) -> List[str]:
    return shlex.split(cmd, posix=os.name != "nt")

def _ensure_allowed(cmd: str):
    parts = _split_cmd(cmd)
    if not parts:
        raise HTTPException(status_code=400, detail="Empty command")
    bin_name = os.path.basename(parts[0])
    if bin_name not in ALLOWED_BINARIES:
        raise HTTPException(status_code=400, detail=f"Command not allowed: {bin_name}")

def _set_limits():
    # POSIX-only soft resource caps (best-effort)
    if resource is None:
        return
    # CPU seconds
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    except Exception:
        pass
    # Max resident set (bytes) ~ 512MB
    try:
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    except Exception:
        pass
    # File size output limit ~ 64MB
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    except Exception:
        pass

def _run_command(cmd: str, cwd: str, timeout_sec: int) -> Dict[str, Any]:
    _ensure_allowed(cmd)
    try:
        proc = subprocess.run(
            _split_cmd(cmd),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
            # On POSIX, add a preexec_fn to set limits inside child
            preexec_fn=_set_limits if (resource is not None and os.name != "nt") else None,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        return {
            "command": cmd,
            "returncode": proc.returncode,
            "output": proc.stdout,
            "ok": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired as e:
        return {"command": cmd, "returncode": None, "output": f"[TIMEOUT after {timeout_sec}s]\n{e}", "ok": False}
    except Exception as e:
        return {"command": cmd, "returncode": None, "output": f"[ERROR] {e}", "ok": False}


@router.post("/api/sandbox/apply", response_model=PatchResult)
def apply_patch(req: PatchRequest):  # keep your auth dep
    if not req.patch.strip():
        raise HTTPException(status_code=400, detail="Patch is empty")

    sandbox_dir = _prepare_sandbox_dir(req.sandbox_id)
    sandbox_id = os.path.basename(sandbox_dir).replace("sandbox_", "")

    # 1) Copy a safe baseline project (optional but recommended)
    _copy_base(req.base_path, sandbox_dir)

    # 2) Apply the patch
    patch_logs = _apply_patch(sandbox_dir, req.patch)
    applied_files = _parse_applied_files_from_patch(req.patch)

    # 3) Optionally run commands/tests
    run_results: List[Dict[str, Any]] = []
    if req.run:
        for cmd in req.run:
            res = _run_command(cmd, cwd=sandbox_dir, timeout_sec=req.timeout_sec)
            run_results.append(res)
            if req.stop_on_failure and not res["ok"]:
                break

    # Combined logs for convenience
    combined_logs = patch_logs
    if run_results:
        combined_logs += "\n\n=== RUN RESULTS ===\n" + "\n\n".join(
            f"$ {r['command']}\n(exit={r['returncode']})\n{r['output']}" for r in run_results
        )

    ok = (not req.run) or all(r.get("ok") for r in run_results)
    return PatchResult(
        ok=ok,
        sandbox_id=sandbox_id,
        message="Patch applied and commands executed" if req.run else "Patch applied",
        applied_files=applied_files,
        logs=combined_logs,
        run_results=run_results or None,
    )
# backend/ai_router.py

import hashlib, json, os, re, textwrap
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, validator
from auth import get_current_active_user  # your auth dep

# --- If you use OpenAI, wire your client here. Stub provided.
from openai import AsyncOpenAI
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Request/Response models ----------
class AIIntakeRequest(BaseModel):
    dataset_id: int
    header_map: Dict[str, str] = Field(default_factory=dict)
    inferred_types: Dict[str, str] = Field(default_factory=dict)
    anomalies: List[Any] = Field(default_factory=list)
    preview_raw: List[Dict[str, Any]] = Field(default_factory=list)
    preview_normalized: List[Dict[str, Any]] = Field(default_factory=list)
    instructions: Optional[str] = None

    @validator("preview_raw", "preview_normalized", pre=True)
    def _cap_rows(cls, v):
        # Hard cap to keep prompts small (frontend already slices, this is defense-in-depth)
        return list(v or [])[:8]

class AIIntakeResponse(BaseModel):
    patch: str
    summary: str
    warnings: List[str] = []
    usage: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None

# ---------- Helpers ----------
PII_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PII_PHONE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b")
PII_CC    = re.compile(r"\b(?:\d[ -]*?){13,19}\b")  # crude CC-like

def mask_pii(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value
    s = PII_EMAIL.sub("[email]", s)
    s = PII_PHONE.sub("[phone]", s)
    s = PII_CC.sub("[card]", s)
    return s

def rows_to_csv(rows: List[Dict[str, Any]], max_chars=2000) -> str:
    if not rows:
        return ""
    # Build header
    cols = list(rows[0].keys())
    lines = [",".join(map(str, cols))]
    for r in rows:
        vals = []
        for c in cols:
            v = mask_pii(r.get(c, ""))
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            s = str(v).replace("\n", " ").replace("\r", " ")
            # simple CSV escaping for commas/quotes
            if ("," in s) or ('"' in s):
                s = '"' + s.replace('"', '""') + '"'
            vals.append(s)
        lines.append(",".join(vals))
        if sum(len(x) for x in lines) > max_chars:
            lines.append("# …truncated…")
            break
    return "\n".join(lines)

def compact_json(obj: Any, max_chars=2000) -> str:
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if len(s) > max_chars:
        s = s[: max_chars - 20] + "… (truncated)"
    return s

UNIFIED_DIFF_RE = re.compile(r"(?m)^(---\s+a/.*\n\+\+\+\s+b/.*\n@@)", re.DOTALL)

def looks_like_unified_diff(text: str) -> bool:
    return bool(UNIFIED_DIFF_RE.search(text))

DEFAULT_INSTRUCTIONS = (
    "Suggest a minimal set of runnable code changes (patch) to improve robustness of "
    "intake/normalization. Output a single unified diff. Avoid boilerplate and commentary."
)

def make_prompt(payload: AIIntakeRequest) -> Dict[str, str]:
    user_msg = textwrap.dedent(f"""
    You are reviewing an ingestion/normalization pipeline. Propose a **single unified diff** that fixes robustness issues
    discovered during intake. Keep the diff minimal and runnable.

    ## Constraints
    - Output only one patch, in unified diff format with '--- a/…' / '+++ b/…' and @@ hunks.
    - Do not include prose inside the diff. Provide a separate short summary outside the diff.
    - Prefer small, targeted fixes (schema coercion, dtype guards, missing-column handling, duplicate ID resolution,
      timezone-safe parsing, CSV sniffing).
    - The repo is Python/FastAPI + pandas. Focus changes in files like `planner_router.py`, `intake.py`, `normalize.py`.

    ## Dataset context
    dataset_id: {payload.dataset_id}
    header_map: {compact_json(payload.header_map)}
    inferred_types: {compact_json(payload.inferred_types)}
    anomalies: {compact_json(payload.anomalies, 1200)}

    ### Preview (raw)
    {rows_to_csv(payload.preview_raw)}

    ### Preview (normalized)
    {rows_to_csv(payload.preview_normalized)}

    ## Task
    {payload.instructions or DEFAULT_INSTRUCTIONS}

    Return JSON with keys: summary (string), patch (string). The patch must be a single unified diff.
    """).strip()

    system_msg = (
        "You are a senior Python data engineer. Be precise, conservative, and ensure the patch applies cleanly. "
        "Never invent files that don't exist; modify plausible targets only."
    )
    return {"system": system_msg, "user": user_msg}

async def call_llm_for_patch(system: str, user: str) -> Dict[str, Any]:
    """
    Call your LLM. Return {"summary": str, "patch": str, "usage": {...}}.
    Replace stub with your provider (OpenAI, Anthropic, etc.).
    """
    # ---- STUB IMPLEMENTATION (replace with your actual client) ----
    # Example with OpenAI responses as JSON:
    
    resp = await openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    data = json.loads(resp.choices[0].message.content)
    return {"summary": data.get("summary",""), "patch": data.get("patch",""), "usage": resp.usage}
    
    # For now return an empty patch to keep the stub safe.
    return {"summary": "No-op stub (wire your LLM).", "patch": "", "usage": None}

# ---------- Endpoint ----------
@router.post("/api/ai/analyze-intake", response_model=AIIntakeResponse)
async def analyze_intake(payload: AIIntakeRequest, current_user=Depends(get_current_active_user)):
    # Build an idempotency key so repeated identical requests short-circuit upstream (optional)
    idem_body = json.dumps(payload.dict(), sort_keys=True, default=str).encode("utf-8")
    idem_key = "ai-intake:" + hashlib.sha256(idem_body).hexdigest()[:16]

    # Compose prompt
    msgs = make_prompt(payload)

    # Call the LLM
    try:
        out = await call_llm_for_patch(msgs["system"], msgs["user"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI provider error: {e}")

    patch = (out.get("patch") or "").strip()
    summary = (out.get("summary") or "").strip()
    usage = out.get("usage") or None

    # Validate + guardrails
    warnings: List[str] = []
    if not patch:
        warnings.append("Model returned an empty patch.")
    elif not looks_like_unified_diff(patch):
        warnings.append("Patch is not a valid unified diff (expected --- a/... +++ b/... with @@ hunks).")

    # Optional: tightly enforce diff format
    if not patch:
        # Hard fail if you prefer strictness
        # raise HTTPException(status_code=422, detail="Model did not return a unified diff.")
        pass

    if not summary:
        summary = "Targeted changes based on the dataset sample."

    return AIIntakeResponse(
        patch=patch,
        summary=summary,
        warnings=warnings,
        usage=usage,
        idempotency_key=idem_key,
    )
