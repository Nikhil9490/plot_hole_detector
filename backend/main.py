from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
import os
import json
import httpx

# -----------------------------
# App + CORS
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-only; restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# -----------------------------
# Config from .env
# -----------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai").rstrip("/")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# -----------------------------
# In-memory story memory per docId (MVP)
# NOTE: resets when server restarts
# -----------------------------
STORY_MEMORY: Dict[str, str] = {}

# -----------------------------
# Schemas
# -----------------------------
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
    memory: str = ""  # NEW: return memory to frontend if you want


# -----------------------------
# Prompt (issues + updated memory)
# -----------------------------
SYSTEM_PROMPT = """
You are an expert fiction editor and continuity checker.

You will be given:
1) STORY_MEMORY: a running memory of established facts so far
2) NEW_TEXT: the current paragraph

Task:
A) Detect plot holes / continuity issues in NEW_TEXT, especially contradictions with STORY_MEMORY.
B) Update STORY_MEMORY by adding new stable facts from NEW_TEXT (characters, ages, relationships, locations, dates, objects, rules).
Only include facts that are explicitly stated or strongly implied.

Return STRICT JSON only. No markdown.

Schema:
{
  "issues": [
    {"type": "...", "severity": "low|medium|high", "message": "...", "evidence": ["...","..."]}
  ],
  "memory": "updated story memory as plain text bullet points"
}

Rules:
- Only report issues you can support with evidence quotes from NEW_TEXT or memory.
- Prefer fewer, higher-confidence issues.
- memory should be short (max ~15 bullets), deduplicated, and consistent.
""".strip()


# -----------------------------
# Helpers
# -----------------------------
def extract_json_block(s: str) -> str:
    """
    Models sometimes return extra text. Extract the first {...} block.
    """
    s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return JSON. Got: {s[:200]}")
    return s[start:end + 1]


def fallback_rule_analyzer(text: str) -> List[Issue]:
    """
    If model fails, still return something.
    """
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


# -----------------------------
# LLM calls
# -----------------------------
async def groq_analyze(doc_id: str, text: str) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY in backend/.env")

    url = f"{GROQ_BASE_URL}/v1/chat/completions"

    memory = STORY_MEMORY.get(doc_id, "")
    user_content = f"""
STORY_MEMORY:
{memory if memory else "(empty)"}

NEW_TEXT:
{text}
""".strip()

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
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
    parsed = json.loads(extract_json_block(content))

    # Save updated memory if present
    new_memory = str(parsed.get("memory", "")).strip()
    if new_memory:
        STORY_MEMORY[doc_id] = new_memory

    return parsed


async def deepseek_analyze(doc_id: str, text: str) -> dict:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in backend/.env")

    url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"

    memory = STORY_MEMORY.get(doc_id, "")
    user_content = f"""
STORY_MEMORY:
{memory if memory else "(empty)"}

NEW_TEXT:
{text}
""".strip()

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
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
    parsed = json.loads(extract_json_block(content))

    new_memory = str(parsed.get("memory", "")).strip()
    if new_memory:
        STORY_MEMORY[doc_id] = new_memory

    return parsed


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    Uses Groq by default, updates memory per docId,
    returns issues + memory. Falls back to rules if anything fails.
    """
    try:
        result = await groq_analyze(req.docId, req.text)

        issues_raw = result.get("issues", [])
        issues: List[Issue] = []

        for it in issues_raw:
            issues.append(Issue(
                type=str(it.get("type", "Other")),
                severity=str(it.get("severity", "low")),
                message=str(it.get("message", "")),
                evidence=[str(x) for x in (it.get("evidence", []) or [])][:4],
            ))

        return AnalyzeResponse(
            docId=req.docId,
            issues=issues,
            memory=STORY_MEMORY.get(req.docId, "")
        )

    except Exception:
        issues = fallback_rule_analyzer(req.text)
        return AnalyzeResponse(
            docId=req.docId,
            issues=issues,
            memory=STORY_MEMORY.get(req.docId, "")
        )


@app.post("/reset/{doc_id}")
def reset_memory(doc_id: str):
    """
    Clear story memory for a given docId.
    """
    STORY_MEMORY.pop(doc_id, None)
    return {"ok": True, "docId": doc_id}
