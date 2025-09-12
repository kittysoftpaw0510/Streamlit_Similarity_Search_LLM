import os
import httpx
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a strict similarity judge. "
    "Given a TYPED_PREFIX and a CANDIDATE_SENTENCE, return JSON with fields: "
    '{"similar": true|false}.\n'
    "Consider semantic similarity; be conservative on short prefixes."
)

def _build_user(typed_prefix: str, sentence: str) -> str:
    return (
        "TYPED_PREFIX:\n"
        f"{typed_prefix}\n\n"
        "CANDIDATE_SENTENCE:\n"
        f"{sentence}\n\n"
        "Return ONLY JSON like {\"similar\": true/false}"
    )

async def llm_is_similar(client: httpx.AsyncClient, typed_prefix: str, sentence: str) -> bool:
    # Fallback heuristic if no API key (dev/demo only)
    if not OPENAI_API_KEY:
        tp = set(w.lower().strip(",.!?") for w in typed_prefix.split() if len(w) > 3)
        sn = set(w.lower().strip(",.!?") for w in sentence.split() if len(w) > 3)
        return len(tp) > 0 and len(tp.intersection(sn)) / max(1, len(tp)) > 0.6

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user(typed_prefix, sentence)},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    try:
        import json
        parsed = json.loads(content)
        return bool(parsed.get("similar", False))
    except Exception:
        return False
