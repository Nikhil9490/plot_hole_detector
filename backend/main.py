# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from dotenv import load_dotenv
import os
import json
import httpx
import traceback
import re

load_dotenv()

# ---------------------------
# App + CORS (dev)
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Config (OpenAI)
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_JSON_MODE = os.getenv("OPENAI_JSON_MODE", "1").strip() == "1"

# ---------------------------
# Schemas
# ---------------------------
class AnalyzeRequest(BaseModel):
    docId: str
    text: str

Severity = Literal["low", "medium", "high"]

class Issue(BaseModel):
    type: str = "logic"
    severity: Severity = "low"
    message: str = ""
    evidence: List[str] = Field(default_factory=list)

class AnalyzeResponse(BaseModel):
    docId: str
    issues: List[Issue] = Field(default_factory=list)

# ---------------------------
# Prompt
# ---------------------------
SYSTEM_PROMPT = """
You are an expert fiction editor and a conservative plot-hole detector.

You must detect ONLY *true* logical/continuity contradictions, not style issues.

Formatting markers:
- [[THOUGHT]] ... [[/THOUGHT]] = internal monologue / mind voice.
  Treat these as subjective and non-binding unless contradicted by objective facts later.
- [[LETTER]] ... [[/LETTER]] = an in-world letter/note (can be misleading).
- [[META]] ... [[/META]] = author notes, ignore for plot holes.

Do NOT flag as plot holes:
- Hyperbole (“no finer boy anywhere”), opinions, narrator voice, sarcasm.
- Missing scene details unless it creates an actual contradiction.
- Time compression unless explicit timestamps conflict.
- Normal story transitions.
- Do NOT infer object state changes unless explicitly stated.
  Example: location (“under his bed”) does NOT imply “unwrapped”, “already used”, or any other state.

Only flag if there is a hard contradiction (explicit in the text), e.g.:
- Explicit impossible travel (“5 minutes later, New York -> Tokyo”)
- Age/time math contradictions with explicit times
- Character facts that negate earlier facts
- Mutually exclusive locations/states at the same time (explicitly stated)

Before emitting an issue, perform this test:
"Is there a plausible interpretation where BOTH statements can be true without adding new facts?"
If YES, do NOT flag.

Return STRICT JSON only. No markdown. No extra text.

Schema:
{
  "issues": [
    {"type": "...", "severity": "low|medium|high", "message": "...", "evidence": ["...","..."]}
  ]
}

Rules:
- Evidence must be DIRECT quotes from the provided text (verbatim substrings).
- Prefer fewer, higher-confidence issues.
- If no real issues: {"issues": []}
""".strip()

# ---------------------------
# Helpers
# ---------------------------
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_json_block(s: str) -> str:
    s = (s or "").strip()
    m = _JSON_RE.search(s)
    if not m:
        raise ValueError(f"Model did not return JSON. Got: {s[:200]}")
    return m.group(0)

def normalize_severity(s: str) -> Severity:
    s = (s or "").strip().lower()
    if s in ("low", "medium", "high"):
        return s  # type: ignore
    return "low"

def evidence_is_direct_quote(evidence: List[str], original_text: str) -> bool:
    if not evidence:
        return False
    return all(ev and (ev in original_text) for ev in evidence)

NUMERIC_TYPES = {"age", "time", "math", "timeline", "date"}

def sanitize_issue(item: dict, original_text: str) -> Optional[Issue]:
    try:
        itype = str(item.get("type", "logic")).strip() or "logic"
        sev = normalize_severity(str(item.get("severity", "low")))
        msg = str(item.get("message", "")).strip()
        ev = [str(x) for x in (item.get("evidence", []) or [])][:4]

        if not msg:
            return None

        # Allow numeric types even if evidence strings aren't perfect;
        # we also run symbolic guards below.
        if itype.lower() in NUMERIC_TYPES:
            return Issue(type=itype, severity=sev, message=msg, evidence=ev)

        # Narrative issues must have verbatim evidence
        if not evidence_is_direct_quote(ev, original_text):
            return None

        return Issue(type=itype, severity=sev, message=msg, evidence=ev)
    except Exception:
        return None

# ---------------------------
# Hybrid numeric guard: age vs birthday
# ---------------------------
AGE_RE = re.compile(r"\b(\d{1,3})\s*(?:year|years)[ -]?old\b", re.IGNORECASE)
BIRTHDAY_RE = re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)\s+birthday\b", re.IGNORECASE)

def detect_explicit_age_contradiction(text: str) -> Optional[Issue]:
    ages = [int(m.group(1)) for m in AGE_RE.finditer(text)]
    birthdays = [int(m.group(1)) for m in BIRTHDAY_RE.finditer(text)]
    if not ages or not birthdays:
        return None

    for a in ages:
        for b in birthdays:
            if a != b:
                age_match = AGE_RE.search(text)
                bday_match = BIRTHDAY_RE.search(text)
                evidence = []
                if age_match:
                    evidence.append(age_match.group(0))
                if bday_match:
                    evidence.append(bday_match.group(0))
                return Issue(
                    type="age",
                    severity="high",
                    message="Explicit age contradiction: stated age does not match stated birthday.",
                    evidence=evidence[:4],
                )
    return None

# ---------------------------
# OpenAI calls
# ---------------------------
async def openai_chat(payload: dict) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in backend/.env")

    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

async def openai_analyze(text: str) -> dict:
    base_payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
    }

    # Prefer strict JSON if available
    if OPENAI_JSON_MODE:
        payload = {**base_payload, "response_format": {"type": "json_object"}}
        data = await openai_chat(payload)
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)

    # Fallback: extract JSON from content
    data = await openai_chat(base_payload)
    content = data["choices"][0]["message"]["content"]
    json_str = extract_json_block(content)
    return json.loads(json_str)

# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    try:
        text = (req.text or "").strip()
        issues: List[Issue] = []

        if text:
            # 1) LLM detection
            result = await openai_analyze(text)
            issues_raw = result.get("issues", []) or []

            for it in issues_raw:
                if isinstance(it, dict):
                    cleaned = sanitize_issue(it, text)
                    if cleaned is not None:
                        issues.append(cleaned)

            # 2) Hybrid hard-guard for explicit age contradiction
            age_issue = detect_explicit_age_contradiction(text)
            if age_issue is not None:
                already = any(i.type.lower() == "age" and i.severity == "high" for i in issues)
                if not already:
                    issues.append(age_issue)

        return AnalyzeResponse(docId=req.docId, issues=issues)

    except Exception as e:
        print("Analyze error:", repr(e))
        traceback.print_exc()
        return AnalyzeResponse(docId=req.docId, issues=[])

# Run backend:
# python -m uvicorn main:app --reload --port 8000
# Run frontend:
# python -m http.server 5500
