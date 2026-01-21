# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import os
import json
import httpx


# ----------------------------
# App + CORS
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-only; tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()


# ----------------------------
# Config
# ----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai").rstrip("/")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Simple in-memory story memory (per docId). Resets when server restarts.
STORY_MEMORY: Dict[str, str] = {}


# ----------------------------
# Schemas
# ----------------------------
class AnalyzeRequest(BaseModel):
    docId: str = Field(..., description="Document ID / chapter ID")
    text: str = Field(..., description="Story text to analyze")


class Issue(BaseModel):
    type: str
    severity: str
    message: str
    evidence: List[str] = []


class AnalyzeResponse(BaseModel):
    docId: str
    issues: List[Issue]


class MemoryUpsertRequest(BaseModel):
    memory: str = Field(..., description="Facts / context to remember for this docId")


# ----------------------------
# Prompts
# ----------------------------
SYSTEM_PROMPT = """
You are an expert fiction editor and continuity checker.

Task:
- Detect plot holes, timeline inconsistencies, spatial/setting contradictions, character inconsistencies, causal logic gaps, continuity errors, and unclear/confusing references.
- You must rely on the provided text + story memory (if any). Do NOT invent facts.

Output format:
Return STRICT JSON only. No markdown. No extra text.

Schema:
{
  "issues": [
    {
      "type": "string",
      "severity": "low|medium|high",
      "message": "string",
      "evidence": ["exact quote 1", "exact quote 2"]
    }
  ]
}

Rules:
- Evidence MUST be exact short quotes copied from the input.
- Only report issues you can support with evidence.
- Avoid nitpicks; prefer fewer, higher-confidence issues.
Do NOT report issues for:

Missing details (e.g., unspecified location, time, or appearance)

Vague narration or stylistic choices

Normal human assumptions (e.g., clocks exist in many places)

Information that could reasonably be inferred without contradiction

Only report an issue if two or more statements cannot logically coexist.
""".strip()

REPAIR_PROMPT = """
You must output STRICT JSON only (no markdown, no extra text).
Take the content below and convert it into valid JSON that matches this schema exactly:

{
  "issues": [
    {
      "type": "string",
      "severity": "low|medium|high",
      "message": "string",
      "evidence": ["exact quote 1", "exact quote 2"]
    }
  ]
}

If the content has no issues, output:
{"issues":[]}

CONTENT TO CONVERT:
""".strip()


# ----------------------------
# Helpers
# ----------------------------
def extract_first_json_object(text: str) -> str:
    """
    Extract the first top-level JSON object from a string by scanning braces.
    Handles model outputs that include extra text before/after JSON.
    """
    s = text.strip()
    start = s.find("{")
    if start == -1:
        raise ValueError(f"No JSON object start found. Got: {s[:200]}")

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    raise ValueError(f"Unbalanced braces; could not extract JSON. Got: {s[:200]}")


def normalize_issues(payload: Dict[str, Any]) -> List[Issue]:
    """
    Normalize whatever the model returned into Issue objects safely.
    """
    raw = payload.get("issues", [])
    issues: List[Issue] = []

    if not isinstance(raw, list):
        return issues

    for it in raw:
        if not isinstance(it, dict):
            continue
        issues.append(
            Issue(
                type=str(it.get("type", "Other")),
                severity=str(it.get("severity", "low")),
                message=str(it.get("message", "")).strip(),
                evidence=[str(x) for x in (it.get("evidence", []) or [])][:4],
            )
        )
    return issues


def fallback_error_issue(err: Exception) -> List[Issue]:
    """
    IMPORTANT: We do NOT pretend we found 'no issues' if the LLM output broke.
    This makes failures visible instead of silently lying.
    """
    return [
        Issue(
            type="LLM_Output_Error",
            severity="high",
            message="LLM response was not valid JSON, so analysis could not be completed.",
            evidence=[repr(err)[:200]],
        )
    ]


async def groq_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY in backend/.env")

    url = f"{GROQ_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    return data["choices"][0]["message"]["content"]


async def groq_analyze(doc_id: str, text: str) -> Dict[str, Any]:
    memory = STORY_MEMORY.get(doc_id, "").strip()

    user_content = f"""
STORY_MEMORY:
{memory if memory else "(empty)"}

NEW_TEXT:
{text}
""".strip()

    content = await groq_chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )

    # Try direct JSON parse; if fails, extract JSON block; if still fails -> repair.
    try:
        return json.loads(content)
    except Exception:
        json_block = extract_first_json_object(content)
        return json.loads(json_block)


async def groq_repair_to_json(bad_content: str) -> Dict[str, Any]:
    """
    Second-chance: ask Groq to convert its previous output into strict JSON.
    """
    repaired = await groq_chat(
        messages=[
            {"role": "system", "content": "You are a strict JSON formatter."},
            {"role": "user", "content": f"{REPAIR_PROMPT}\n\n{bad_content}"},
        ],
        temperature=0.0,
    )

    try:
        return json.loads(repaired)
    except Exception:
        json_block = extract_first_json_object(repaired)
        return json.loads(json_block)


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True, "model": GROQ_MODEL}


@app.get("/memory/{docId}")
def get_memory(docId: str):
    return {"docId": docId, "memory": STORY_MEMORY.get(docId, "")}


@app.post("/memory/{docId}")
def set_memory(docId: str, req: MemoryUpsertRequest):
    STORY_MEMORY[docId] = req.memory.strip()
    return {"ok": True, "docId": docId, "memory_len": len(STORY_MEMORY[docId])}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    # 1) First attempt
    try:
        result = await groq_analyze(req.docId, req.text)
        issues = normalize_issues(result)
        return AnalyzeResponse(docId=req.docId, issues=issues)

    except Exception as e1:
        # 2) Repair attempt (use the error string to show in evidence if needed)
        try:
            # We don't have the raw model text here unless we capture it.
            # So we re-run once and if it breaks again, we at least show failure.
            # (If you want perfect repair: change groq_analyze() to return raw content too.)
            content = await groq_chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"STORY_MEMORY:\n{STORY_MEMORY.get(req.docId,'') or '(empty)'}\n\nNEW_TEXT:\n{req.text}",
                    },
                ],
                temperature=0.2,
            )
            repaired = await groq_repair_to_json(content)
            issues = normalize_issues(repaired)
            return AnalyzeResponse(docId=req.docId, issues=issues)

        except Exception as e2:
            # 3) Visible fallback (do NOT lie with empty issues)
            return AnalyzeResponse(docId=req.docId, issues=fallback_error_issue(e2))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
import os
import json
import httpx

# ----------------------------
# App + CORS
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-only; lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# ----------------------------
# Config (.env)
# ----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai").rstrip("/")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Optional: lightweight per-doc memory (in-process)
STORY_MEMORY: Dict[str, str] = {}

# ----------------------------
# Schemas
# ----------------------------
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

# ----------------------------
# Prompt (updated)
# ----------------------------
SYSTEM_PROMPT = """
You are a strict continuity & plot-hole checker for fiction.

INPUT CONVENTIONS:
- Text may include mind-voice / internal monologue wrapped in:
  [[THOUGHT]] ... [[/THOUGHT]]
  Treat those as thoughts, not literal timeline anchors.
  Do NOT claim a plot hole based only on thoughts.

OUTPUT:
Return STRICT JSON only (no markdown, no extra text).

Schema:
{
  "issues": [
    {
      "type": "timeline|location|character|causal|object|logic|dialogue|other",
      "severity": "low|medium|high",
      "message": "short, specific",
      "evidence": ["quote 1", "quote 2"]
    }
  ]
}

HARD RULES (IMPORTANT):
- Only report issues supported by direct evidence quotes from the text.
- Do NOT infer missing facts (no guessing like “Sunday implies morning”).
- Do NOT flag “unclear” or “ambiguous” as an issue unless it creates a direct contradiction.
- Prefer fewer, higher-confidence issues over many weak ones.
- If there is no clear contradiction, return {"issues": []}.
""".strip()

# ----------------------------
# Helpers
# ----------------------------
def extract_json_block(s: str) -> str:
    """
    Extract first {...} JSON object from model output in case it adds extra text.
    """
    s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return JSON. Got: {s[:300]}")
    return s[start:end + 1]

async def groq_analyze(doc_id: str, text: str) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY in backend/.env")

    url = f"{GROQ_BASE_URL}/v1/chat/completions"

    memory = STORY_MEMORY.get(doc_id, "").strip()

    user_content = f"""
STORY_MEMORY (facts from earlier text; may be empty):
{memory if memory else "(empty)"}

NEW_TEXT (analyze for contradictions against itself and STORY_MEMORY):
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
    json_str = extract_json_block(content)
    return json.loads(json_str)

def safe_issues(result: dict) -> List[Issue]:
    issues_raw = (result or {}).get("issues", []) or []
    issues: List[Issue] = []
    for it in issues_raw:
        issues.append(
            Issue(
                type=str(it.get("type", "other")),
                severity=str(it.get("severity", "low")),
                message=str(it.get("message", "")),
                evidence=[str(x) for x in (it.get("evidence", []) or [])][:4],
            )
        )
    return issues

# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    # Update memory with the latest text (simple approach)
    # (Later we’ll store extracted facts instead of raw text)
    prior = STORY_MEMORY.get(req.docId, "")
    STORY_MEMORY[req.docId] = (prior + "\n\n" + req.text).strip()[-15000:]  # cap size

    try:
        result = await groq_analyze(req.docId, req.text)
        issues = safe_issues(result)
        return AnalyzeResponse(docId=req.docId, issues=issues)
    except Exception as e:
        # If Groq fails, return a safe error as an "issue" instead of making up rules
        return AnalyzeResponse(
            docId=req.docId,
            issues=[
                Issue(
                    type="other",
                    severity="low",
                    message=f"LLM call failed; returning no plot-hole analysis. Error: {type(e).__name__}",
                    evidence=[],
                )
            ],
        )






# to run backend 
# python -m uvicorn main:app --reload --port 8000


# to run frontend
# python -m http.server 5500










