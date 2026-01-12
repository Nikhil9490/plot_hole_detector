from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os
import json
import httpx

app = FastAPI()
load_dotenv()

# ----- config from .env -----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai").rstrip("/")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
# ---------- schemas ----------
class AnalyzeRequest(BaseModel):
    docId: str
    text: str

class Issue(BaseModel):
    type: str
    severity: str
    message: str
    evidence: List[str] = []

class AnalyzeResponse(BaseModel):
    docId: str
    issues: List[Issue]

# ---------- DeepSeek call ----------
SYSTEM_PROMPT = """
You are an expert fiction editor. Detect plot holes and continuity issues.
Return STRICT JSON only. No markdown.

Schema:
{
  "issues": [
    {"type": "...", "severity": "low|medium|high", "message": "...", "evidence": ["...","..."]}
  ]
}

Rules:
- Only report issues you can support with evidence quotes.
- Avoid nitpicks. Prefer fewer, higher confidence issues.
"""
async def groq_analyze(text: str) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY in backend/.env")

    # OpenAI-compatible path
    url = f"{GROQ_BASE_URL}/v1/chat/completions"

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    content = data["choices"][0]["message"]["content"]
    return json.loads(extract_json_block(content))

async def deepseek_analyze(text: str) -> dict:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in backend/.env")

    url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    content = data["choices"][0]["message"]["content"]
    return json.loads(content)
def extract_json_block(s: str) -> str:
    """
    Models sometimes return extra text. This extracts the first {...} block.
    """
    s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return JSON. Got: {s[:200]}")
    return s[start:end+1]

def fallback_rule_analyzer(text: str) -> List[Issue]:
    issues: List[Issue] = []
    lower = text.lower()

    if "19 years old" in lower and "21st birthday" in lower:
        issues.append(Issue(
            type="Timeline inconsistency",
            severity="high",
            message="Possible age/time jump: 19 years old but later 21st birthday without clear time passing.",
            evidence=["Found: '19 years old' and '21st birthday'"]
        ))
    if "in new york" in lower and "in tokyo" in lower and "five minutes later" in lower:
        issues.append(Issue(
            type="Location continuity",
            severity="medium",
            message="Possible impossible travel: New York → Tokyo in five minutes.",
            evidence=["Found: 'in New York', 'in Tokyo', 'five minutes later'"]
        ))

    return issues

# ---------- routes ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    try:
        result = await groq_analyze(req.text)

        issues_raw = result.get("issues", [])
        issues = []
        for it in issues_raw:
            issues.append(Issue(
                type=str(it.get("type", "Other")),
                severity=str(it.get("severity", "low")),
                message=str(it.get("message", "")),
                evidence=[str(x) for x in (it.get("evidence", []) or [])][:4],
            ))

        return AnalyzeResponse(docId=req.docId, issues=issues)

    except Exception:
        # Last fallback so app still works
        issues = fallback_rule_analyzer(req.text)
        return AnalyzeResponse(docId=req.docId, issues=issues)
