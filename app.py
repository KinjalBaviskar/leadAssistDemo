"""LeadAssist prototype - AI lead analysis with Gemini.

Roles: client (uploads requirement documents, sees its estimate) and company (sees all leads,
approves / rejects / comments). Flow: documents -> text -> Gemini extraction -> deterministic estimate & feasibility
-> recommendation (numbers by backend, rationale by Gemini) -> scope change -> proposal.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import os
import re
import secrets
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

BASE = Path(__file__).parent
DB_PATH = BASE / "leadassist.db"
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.5-flash"
# Tried in order if the configured model is retired / not found for this API key.
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.8-flash"]
HOURLY_RATE = float(os.getenv("HOURLY_RATE", "1800"))  # INR, blended
COMPLEXITY_HOURS = {"low": 16, "medium": 36, "high": 72}

app = FastAPI(title="LeadAssist Prototype")


# --------------------------------------------------------------------------- storage
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS leads (id TEXT PRIMARY KEY, created_at TEXT, data TEXT);
        CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, email TEXT UNIQUE, name TEXT, company TEXT,
                                          role TEXT, pw_hash TEXT);
        CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, user_id TEXT, created_at TEXT);
    """)
    return conn


def save_lead(lead: dict[str, Any]) -> None:
    with db() as conn:
        conn.execute("INSERT OR REPLACE INTO leads VALUES (?, ?, ?)", (lead["id"], lead["created_at"], json.dumps(lead)))


def load_lead(lead_id: str, user: dict[str, Any] | None = None) -> dict[str, Any]:
    with db() as conn:
        row = conn.execute("SELECT data FROM leads WHERE id = ?", (lead_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Lead not found")
    lead = json.loads(row["data"])
    if user and user["role"] == "client" and lead.get("owner_id") != user["id"]:
        raise HTTPException(404, "Lead not found")  # clients only ever see their own leads
    return lead


# --------------------------------------------------------------------------- auth
def hash_pw(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()
    return f"{salt}${digest}"


def check_pw(password: str, stored: str) -> bool:
    salt = stored.split("$", 1)[0]
    return hmac.compare_digest(hash_pw(password, salt), stored)


def create_user(email: str, password: str, name: str, company: str, role: str) -> dict[str, Any]:
    user = {"id": uuid.uuid4().hex[:10], "email": email.lower().strip(), "name": name, "company": company, "role": role}
    with db() as conn:
        conn.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)",
                     (user["id"], user["email"], name, company, role, hash_pw(password)))
    return user


def seed_users() -> None:
    """Demo accounts (development only)."""
    with db() as conn:
        exists = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if not exists:
        create_user("company@leadassist.local", "company123", "Priya Sharma", "Straveda Tech", "company")
        create_user("client@leadassist.local", "client123", "Rahul Mehta", "UrbanCart Retail Pvt Ltd", "client")


def current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = (authorization or "").removeprefix("Bearer ").strip()
    with db() as conn:
        row = conn.execute("SELECT u.id, u.email, u.name, u.company, u.role FROM sessions s "
                           "JOIN users u ON u.id = s.user_id WHERE s.token = ?", (token,)).fetchone()
    if not token or row is None:
        raise HTTPException(401, "Please log in.")
    return dict(row)


def company_only(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    if user["role"] != "company":
        raise HTTPException(403, "Company access only.")
    return user


# --------------------------------------------------------------------------- documents
MAX_FILE_BYTES = 5 * 1024 * 1024


def extract_text(name: str, data: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext in (".txt", ".md"):
        return data.decode("utf-8", errors="ignore")
    if ext == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    if ext == ".docx":
        from docx import Document

        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    raise HTTPException(400, f"Unsupported file type '{ext}'. Upload PDF, DOCX, TXT or MD.")


# --------------------------------------------------------------------------- Gemini
class Feature(BaseModel):
    name: str
    complexity: str = "medium"  # low | medium | high


class Extraction(BaseModel):
    project_type: str | None = None
    summary: str = ""
    features: list[Feature] = Field(default_factory=list)
    integrations: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    budget_inr: float | None = None
    timeline_days: int | None = None
    missing_fields: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class ScopeDiff(BaseModel):
    added: list[Feature] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    modified: list[Feature] = Field(default_factory=list)
    summary: str = ""


_client = None


def gemini(prompt: str, json_mode: bool = True) -> str:
    """Call Gemini. Falls back to newer models on 404; raises HTTP 503 with a clear message otherwise."""
    global _client, GEMINI_MODEL
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise HTTPException(503, "GEMINI_API_KEY is not set. Add it to .env and restart the server.")
    from google import genai
    from google.genai import types

    if _client is None:
        _client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        temperature=0.2, response_mime_type="application/json" if json_mode else "text/plain")
    for model in [GEMINI_MODEL] + [m for m in FALLBACK_MODELS if m != GEMINI_MODEL]:
        try:
            resp = _client.models.generate_content(model=model, contents=prompt, config=config)
        except Exception as exc:  # noqa: BLE001
            if "404" in str(exc) or "NOT_FOUND" in str(exc):
                last_error = exc
                continue  # model retired / unavailable -> try the next one
            raise HTTPException(503, f"Gemini request failed: {exc}") from exc
        GEMINI_MODEL = model  # remember the model that works
        return (resp.text or "").strip()
    raise HTTPException(503, f"Gemini request failed: {last_error}")


def gemini_json(prompt: str, schema: type[BaseModel]) -> Any:
    text = gemini(f"{prompt}\n\nReturn ONLY JSON matching this schema:\n{json.dumps(schema.model_json_schema())}")
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    if not text:
        raise HTTPException(502, "The AI returned an empty response. Please try again.")
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        raise HTTPException(502, f"Gemini returned invalid JSON: {exc.errors()[:3]}") from exc


EXTRACT_PROMPT = """You are a business analyst at a software services company in India.
Extract structured requirements from the client text below.
Rules: only use facts in the text; never invent budget or timeline (use null);
budget_inr is a number in INR (1 lakh = 100000, 1 crore = 10000000);
timeline_days in calendar days (1 month = 30); feature complexity is low, medium or high;
list missing information needed for a reliable estimate and practical follow-up questions.

Client requirement:
\"\"\"{text}\"\"\""""


# --------------------------------------------------------------------------- deterministic engines
def estimate(features: list[Feature], integrations: list[str]) -> dict[str, Any]:
    feature_hours = sum(COMPLEXITY_HOURS.get(f.complexity.lower(), 36) for f in features)
    dev = 60 + feature_hours + 32 * len(integrations)  # base setup + features + integrations
    total = round(dev * 1.35)  # + QA, PM, design, deployment overhead
    team = max(2, min(8, round(math.sqrt(total / 160)) + 1))
    duration = math.ceil(total / (6 * team * 0.8) * 7 / 5)  # calendar days
    return {
        "hours": total,
        "cost": round(total * HOURLY_RATE),
        "team_size": team,
        "duration_days": duration,
        "breakdown": {"base": 60, "features": feature_hours, "integrations": 32 * len(integrations),
                      "overhead_35pct": total - dev},
    }


def smooth(ratio: float, mid: float, k: float) -> float:
    return round(100 / (1 + math.exp(-k * (ratio - mid))), 1)


def feasibility(ext: Extraction, est: dict[str, Any]) -> dict[str, Any]:
    high = sum(1 for f in ext.features if f.complexity.lower() == "high")
    technical = max(20.0, 92 - 4 * high - 3 * max(0, len(ext.integrations) - 2))
    budget = smooth(ext.budget_inr / est["cost"], 0.8, 6) if ext.budget_inr else 55.0
    needed = est["duration_days"] * 1.15
    timeline = smooth(ext.timeline_days / needed, 0.9, 7) if ext.timeline_days else 60.0
    overall = round(0.35 * technical + 0.35 * budget + 0.30 * timeline, 1)
    notes = []
    if ext.budget_inr:
        notes.append(f"Budget covers {ext.budget_inr / est['cost']:.0%} of the estimated cost.")
    if ext.timeline_days:
        notes.append(f"Requested {ext.timeline_days} days vs ~{needed:.0f} days needed (incl. 15% buffer).")
    return {"technical": technical, "budget": budget, "timeline": timeline, "overall": overall, "notes": notes}


def recommend(feas: dict[str, Any], ext: Extraction, est: dict[str, Any]) -> dict[str, Any]:
    revenue = ext.budget_inr or est["cost"]
    delivery_cost = est["cost"] * 0.6  # internal cost assumed at 60% of billing
    margin = (revenue - delivery_cost) / revenue
    profitability = max(0.0, min(100.0, (margin + 0.2) / 0.6 * 100))
    clarity = max(0, 100 - 10 * len(ext.missing_fields))
    score = round(0.45 * feas["overall"] + 0.35 * profitability + 0.20 * clarity, 1)
    order = ["PRIORITIZE", "PURSUE_WITH_CONDITIONS", "REVIEW", "DEPRIORITIZE"]
    category = ("PRIORITIZE" if score >= 70 else "PURSUE_WITH_CONDITIONS" if score >= 55
                else "REVIEW" if score >= 40 else "DEPRIORITIZE")
    # Guardrail: one very weak feasibility dimension (e.g. impossible deadline) caps the category.
    weakest = min(feas["technical"], feas["budget"], feas["timeline"])
    cap = "REVIEW" if weakest < 20 else "PURSUE_WITH_CONDITIONS" if weakest < 40 else "PRIORITIZE"
    cap_reason = None
    if order.index(category) < order.index(cap):
        category = cap
        score = min(score, 69.0 if cap == "PURSUE_WITH_CONDITIONS" else 54.0)  # keep score consistent with verdict
        weak = min((("technical", feas["technical"]), ("budget", feas["budget"]), ("timeline", feas["timeline"])),
                   key=lambda kv: kv[1])
        cap_reason = f"Capped at {cap.replace('_', ' ').lower()}: {weak[0]} feasibility is only {weak[1]:.0f}/100."
    return {"score": score, "category": category, "cap_reason": cap_reason, "expected_revenue": round(revenue),
            "delivery_cost": round(delivery_cost), "profit_margin": round(margin, 3),
            "profitability_score": round(profitability, 1), "clarity_score": clarity}


def parse_client_inputs(min_lakh: str, max_lakh: str, delivery: str) -> dict[str, Any]:
    """Validate the optional budget range (in lakh) and expected delivery date from the form."""
    def lakh(value: str, label: str) -> float | None:
        if not value.strip():
            return None
        try:
            amount = float(value)
        except ValueError as exc:
            raise HTTPException(400, f"{label} must be a number (in lakh).") from exc
        if amount <= 0 or amount > 100000:
            raise HTTPException(400, f"{label} must be between 0 and 1,00,000 lakh.")
        return round(amount * 100000)

    budget_min, budget_max = lakh(min_lakh, "Minimum budget"), lakh(max_lakh, "Maximum budget")
    if budget_min and budget_max and budget_min > budget_max:
        raise HTTPException(400, "Minimum budget cannot be more than the maximum budget.")
    due = None
    if delivery.strip():
        try:
            due = date.fromisoformat(delivery.strip())
        except ValueError as exc:
            raise HTTPException(400, "Expected delivery date must be a valid date.") from exc
        if due <= date.today():
            raise HTTPException(400, "Expected delivery date must be in the future.")
    parts = []
    if budget_min or budget_max:
        rng = " to ".join(f"INR {v:,}" for v in (budget_min, budget_max) if v)
        parts.append(f"Budget range: {rng}.")
    if due:
        parts.append(f"Expected delivery date: {due.isoformat()} ({(due - date.today()).days} days from today).")
    return {"budget_min": budget_min, "budget_max": budget_max,
            "delivery_date": due.isoformat() if due else None,
            "timeline_days": (due - date.today()).days if due else None, "summary": " ".join(parts)}


def apply_client_inputs(ext: Extraction, inputs: dict[str, Any]) -> None:
    """Explicit form values override what the AI extracted from the documents."""
    if inputs["budget_min"] or inputs["budget_max"]:
        ext.budget_inr = float(inputs["budget_max"] or inputs["budget_min"])  # upper bound = client's ceiling
        ext.missing_fields = [m for m in ext.missing_fields if "budget" not in m.lower()]
    if inputs["timeline_days"]:
        ext.timeline_days = inputs["timeline_days"]
        ext.missing_fields = [m for m in ext.missing_fields
                              if not any(w in m.lower() for w in ("timeline", "deadline", "delivery date", "launch"))]


# --------------------------------------------------------------------------- API
class LoginIn(BaseModel):
    email: str
    password: str


class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    company: str = Field(min_length=2, max_length=200)
    email: str = Field(min_length=5, max_length=200, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=6, max_length=100)


class ScopeIn(BaseModel):
    change_text: str = Field(min_length=5, max_length=10000)


class ReviewIn(BaseModel):
    decision: str = Field(pattern="^(approve|reject)$")
    comment: str = Field(default="", max_length=3000)


class CommentIn(BaseModel):
    text: str = Field(min_length=1, max_length=3000)


INTERNAL_KEYS = ("recommendation", "scope_changes")


def client_view(lead: dict[str, Any]) -> dict[str, Any]:
    """What a client may see: their analysis and estimate, not internal profitability or rationale."""
    view = {k: v for k, v in lead.items() if k not in INTERNAL_KEYS}
    if lead.get("status") != "APPROVED":
        view["proposal"] = None
    return view


def session_for(user: dict[str, Any]) -> dict[str, Any]:
    token = secrets.token_urlsafe(32)
    with db() as conn:
        conn.execute("INSERT INTO sessions VALUES (?, ?, ?)", (token, user["id"], datetime.now().isoformat()))
    return {"token": token, "user": {k: user[k] for k in ("id", "email", "name", "company", "role")}}


@app.post("/api/login")
def login(body: LoginIn) -> dict[str, Any]:
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (body.email.lower().strip(),)).fetchone()
    if row is None or not check_pw(body.password, row["pw_hash"]):
        raise HTTPException(401, "Invalid email or password.")
    return session_for(dict(row))


@app.post("/api/register")
def register(body: RegisterIn) -> dict[str, Any]:
    """Self-registration always creates a CLIENT account."""
    try:
        user = create_user(body.email, body.password, body.name, body.company, "client")
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, "An account with this email already exists.") from exc
    return session_for(user)


@app.post("/api/logout")
def logout(authorization: str | None = Header(default=None)) -> dict[str, str]:
    with db() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", ((authorization or "").removeprefix("Bearer ").strip(),))
    return {"status": "ok"}


@app.get("/api/me")
def me(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    return user


@app.post("/api/leads")
async def submit_lead(
    project_title: str = Form(..., min_length=1, max_length=200),
    notes: str = Form(default="", max_length=5000),
    client_name: str = Form(default="", max_length=200),
    budget_min_lakh: str = Form(default="", max_length=20),
    budget_max_lakh: str = Form(default="", max_length=20),
    delivery_date: str = Form(default="", max_length=10),
    files: list[UploadFile] = File(default=[]),
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    """Upload requirement document(s) -> text extraction -> Gemini analysis -> estimate.

    Optional form inputs (budget range in lakh, expected delivery date) take precedence over values
    the AI finds in the documents.
    """
    inputs = parse_client_inputs(budget_min_lakh, budget_max_lakh, delivery_date)
    texts, names = [], []
    for upload in files:
        data = await upload.read()
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(400, f"{upload.filename} is larger than 5 MB.")
        if data:
            texts.append(f"--- {upload.filename} ---\n{extract_text(upload.filename or '', data)}")
            names.append(upload.filename)
    if notes.strip():
        texts.append(f"--- Additional notes ---\n{notes}")
    if inputs["summary"]:
        texts.append(f"--- Client-provided details ---\n{inputs['summary']}")
    requirement = "\n\n".join(texts).strip()[:20000]
    if len(requirement) < 20:
        raise HTTPException(400, "Upload at least one document with readable text (or add notes).")

    ext: Extraction = gemini_json(EXTRACT_PROMPT.format(text=requirement), Extraction)
    apply_client_inputs(ext, inputs)
    est = estimate(ext.features, ext.integrations)
    feas = feasibility(ext, est)
    rec = recommend(feas, ext, est)
    rec["rationale"] = gemini(
        "In 3-4 sentences, explain this sales recommendation to a manager and give ONE concrete next best action. "
        "Use the numbers exactly as given, do not invent new ones.\n" + json.dumps(
            {"project": project_title, "summary": ext.summary, "estimate": est, "feasibility": feas,
             "recommendation": rec, "missing_info": ext.missing_fields, "risks": ext.risks}),
        json_mode=False)
    is_client = user["role"] == "client"
    lead = {"id": uuid.uuid4().hex[:10], "created_at": datetime.now().isoformat(timespec="seconds"),
            "owner_id": user["id"] if is_client else None,
            "client_name": user["company"] if is_client else (client_name or "Walk-in client"),
            "submitted_by": user["name"], "project_title": project_title, "documents": names,
            "client_inputs": {k: inputs[k] for k in ("budget_min", "budget_max", "delivery_date")},
            "requirement_text": requirement, "extraction": ext.model_dump(), "estimate": est, "feasibility": feas,
            "recommendation": rec, "status": "PENDING_REVIEW", "comments": [], "scope_changes": [],
            "proposal": None}
    save_lead(lead)
    return client_view(lead) if is_client else lead


@app.get("/api/leads")
def list_leads(user: dict[str, Any] = Depends(current_user)) -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute("SELECT data FROM leads ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        lead = json.loads(r["data"])
        if user["role"] == "client" and lead.get("owner_id") != user["id"]:
            continue
        item = {k: lead.get(k) for k in ("id", "client_name", "project_title", "created_at", "status")}
        item |= {"estimated_cost": lead["estimate"]["cost"], "hours": lead["estimate"]["hours"],
                 "duration_days": lead["estimate"]["duration_days"], "feasibility": lead["feasibility"]["overall"],
                 "documents": len(lead.get("documents", [])), "comments": len(lead.get("comments", [])),
                 "project_type": lead["extraction"].get("project_type")}
        if user["role"] == "company":
            item |= {"category": lead["recommendation"]["category"], "score": lead["recommendation"]["score"]}
        out.append(item)
    return out


@app.get("/api/leads/{lead_id}")
def get_lead(lead_id: str, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    lead = load_lead(lead_id, user)
    return client_view(lead) if user["role"] == "client" else lead


@app.post("/api/leads/{lead_id}/review")
def review(lead_id: str, body: ReviewIn, user: dict[str, Any] = Depends(company_only)) -> dict[str, Any]:
    lead = load_lead(lead_id)
    lead["status"] = "APPROVED" if body.decision == "approve" else "REJECTED"
    lead["comments"].append({"author": user["name"], "role": "company", "at": datetime.now().isoformat(timespec="seconds"),
                             "text": (("Approved" if body.decision == "approve" else "Not taken forward")
                                      + (f": {body.comment}" if body.comment.strip() else "."))})
    save_lead(lead)
    return lead


@app.post("/api/leads/{lead_id}/comments")
def comment(lead_id: str, body: CommentIn, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    lead = load_lead(lead_id, user)
    lead["comments"].append({"author": user["name"], "role": user["role"],
                             "at": datetime.now().isoformat(timespec="seconds"), "text": body.text})
    save_lead(lead)
    return client_view(lead) if user["role"] == "client" else lead


@app.post("/api/leads/{lead_id}/scope-change")
def scope_change(lead_id: str, body: ScopeIn, user: dict[str, Any] = Depends(company_only)) -> dict[str, Any]:
    lead = load_lead(lead_id)
    current = [f["name"] for f in lead["extraction"]["features"]]
    diff: ScopeDiff = gemini_json(
        "Compare the approved project scope with a client change request. Identify features that are added, "
        "removed, or modified (give complexity low/medium/high for added and modified). Only report real changes.\n"
        f"Approved features: {json.dumps(current)}\nChange request:\n\"\"\"{body.change_text}\"\"\"", ScopeDiff)
    added = sum(COMPLEXITY_HOURS.get(f.complexity.lower(), 36) for f in diff.added)
    modified = 0.5 * sum(COMPLEXITY_HOURS.get(f.complexity.lower(), 36) for f in diff.modified)
    removed = 0.7 * 36 * len(diff.removed)
    hours = round((added + modified - removed) * 1.35)
    team = lead["estimate"]["team_size"]
    days = math.ceil(hours / (6 * team * 0.8) * 7 / 5) if hours > 0 else 0
    ratio = hours / lead["estimate"]["hours"]
    severity = "CRITICAL" if ratio >= 0.5 else "HIGH" if ratio >= 0.25 else "MEDIUM" if ratio >= 0.1 else "LOW"
    change = {"id": uuid.uuid4().hex[:8], "created_at": datetime.now().isoformat(timespec="seconds"),
              "change_text": body.change_text, "diff": diff.model_dump(), "additional_hours": hours,
              "additional_cost": round(hours * HOURLY_RATE), "additional_days": days, "severity": severity,
              "status": "PENDING"}
    lead["scope_changes"].insert(0, change)
    save_lead(lead)
    return change


@app.post("/api/leads/{lead_id}/scope-change/{change_id}/{decision}")
def decide_scope(lead_id: str, change_id: str, decision: str,
                 user: dict[str, Any] = Depends(company_only)) -> dict[str, Any]:
    if decision not in ("approve", "reject"):
        raise HTTPException(400, "decision must be approve or reject")
    lead = load_lead(lead_id)
    change = next((c for c in lead["scope_changes"] if c["id"] == change_id), None)
    if change is None or change["status"] != "PENDING":
        raise HTTPException(400, "Change not found or already decided")
    change["status"] = "APPROVED" if decision == "approve" else "REJECTED"
    if decision == "approve":  # update scope + estimate
        removed = {n.lower() for n in change["diff"]["removed"]}
        feats = [f for f in lead["extraction"]["features"] if f["name"].lower() not in removed]
        feats += change["diff"]["added"]
        lead["extraction"]["features"] = feats
        for key, add in (("hours", "additional_hours"), ("cost", "additional_cost"), ("duration_days", "additional_days")):
            lead["estimate"][key] += change[add]
    save_lead(lead)
    return lead


@app.post("/api/leads/{lead_id}/proposal")
def proposal(lead_id: str, user: dict[str, Any] = Depends(company_only)) -> dict[str, Any]:
    lead = load_lead(lead_id)
    est = lead["estimate"]
    lead["proposal"] = gemini(
        "Write a concise, professional project proposal in Markdown for the client below. Sections: Overview, Scope "
        "(feature list), Deliverables, Technology, Timeline, Estimated Effort & Cost, Assumptions, Exclusions, "
        "Validity (30 days), Approval. Use ONLY these numbers: "
        f"cost INR {est['cost']:,}, effort {est['hours']} hours, team of {est['team_size']}, "
        f"duration {est['duration_days']} calendar days.\n"
        + json.dumps({"client": lead["client_name"], "project": lead["project_title"],
                      "summary": lead["extraction"]["summary"], "features": lead["extraction"]["features"],
                      "technologies": lead["extraction"]["technologies"],
                      "integrations": lead["extraction"]["integrations"]}),
        json_mode=False)
    save_lead(lead)
    return lead


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "gemini_key_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
            "model": GEMINI_MODEL}


seed_users()
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE / "static" / "index.html")
