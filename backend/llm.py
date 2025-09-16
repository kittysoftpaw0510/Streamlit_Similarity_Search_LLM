import os
import json
import math
import asyncio
from typing import Iterable, List, Tuple, Union, Optional

import httpx
from fastapi import HTTPException
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Concurrency & HTTP limits
MAX_LLM_CONCURRENCY = int(os.getenv("MAX_LLM_CONCURRENCY", "8"))
REQUEST_TIMEOUT_SECS = float(os.getenv("LLM_REQUEST_TIMEOUT_SECS", "30"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
INITIAL_BACKOFF = float(os.getenv("LLM_INITIAL_BACKOFF", "0.5"))
BACKOFF_FACTOR = float(os.getenv("LLM_BACKOFF_FACTOR", "2.0"))

_LLM_SEM = asyncio.Semaphore(MAX_LLM_CONCURRENCY)

_HTTP_LIMITS = httpx.Limits(
    max_connections=MAX_LLM_CONCURRENCY * 2,
    max_keepalive_connections=MAX_LLM_CONCURRENCY,
)
_SHARED_CLIENT: Optional[httpx.AsyncClient] = httpx.AsyncClient(
    timeout=REQUEST_TIMEOUT_SECS,
    limits=_HTTP_LIMITS,
)

SYSTEM_PROMPT = (
    "You are a strict semantic similarity judge. "
    "Given a TYPED_PREFIX and a CANDIDATE_SENTENCE, return JSON with fields: "
    '{"score": float} where score ∈ [0,1].'
)

YESNO_SYSTEM = (
    "You must answer with exactly ONE token: Yes or No. Nothing else."
)

def _shared_client_or_default(client: Optional[httpx.AsyncClient]) -> httpx.AsyncClient:
    global _SHARED_CLIENT
    if client is not None:
        return client
    if _SHARED_CLIENT is None:
        _SHARED_CLIENT = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECS, limits=_HTTP_LIMITS)
    return _SHARED_CLIENT

async def close_shared_client():
    global _SHARED_CLIENT
    if _SHARED_CLIENT is not None:
        await _SHARED_CLIENT.aclose()
        _SHARED_CLIENT = None

async def _retry_request_json(client: httpx.AsyncClient, payload: dict) -> dict:
    """
    POST /v1/chat/completions with retries on 429/5xx.
    Raises HTTPException with specific error codes/messages.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    backoff = INITIAL_BACKOFF
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 429:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(backoff)
                    backoff *= BACKOFF_FACTOR
                    continue
                raise HTTPException(status_code=429, detail="LLM rate limit exceeded")
            if 500 <= resp.status_code < 600:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(backoff)
                    backoff *= BACKOFF_FACTOR
                    continue
                raise HTTPException(status_code=resp.status_code, detail=f"LLM service unavailable (code: {resp.status_code})")
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(backoff)
                backoff *= BACKOFF_FACTOR
                continue
            raise HTTPException(status_code=504, detail="LLM request timed out")
        except httpx.RequestError as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                await asyncio.sleep(backoff)
                backoff *= BACKOFF_FACTOR
                continue
            raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")
        except Exception as e:
            last_exc = e
            break

    raise HTTPException(status_code=500, detail=f"LLM call failed: {last_exc}")

def _build_user(typed_prefix: str, sentence: str) -> str:
    return (
        f"TYPED_PREFIX:\n{typed_prefix}\n\nCANDIDATE_SENTENCE:\n{sentence}\n\n"
        'Return ONLY JSON like {"score": 0.0-1.0}'
    )

def _build_yesno_user(typed_prefix: str, sentence: str) -> str:
    return f"Typed prefix: {typed_prefix}\nCandidate: {sentence}\nAnswer strictly Yes or No."

async def llm_similarity_score_json(
    client: Optional[httpx.AsyncClient],
    typed_prefix: str,
    sentence: str
) -> float:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="LLM API key not configured")

    payload = {
        "model": OPENAI_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user(typed_prefix, sentence)},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    client = _shared_client_or_default(client)
    data = await _retry_request_json(client, payload)

    try:
        content = data["choices"][0]["message"]["content"].strip()
        parsed = json.loads(content)
        score = float(parsed.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception:
        raise HTTPException(status_code=500, detail="LLM response parse error")

async def llm_similarity_score_yesno_logprob(
    client: Optional[httpx.AsyncClient],
    typed_prefix: str,
    sentence: str
) -> float:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="LLM API key not configured")

    payload = {
        "model": OPENAI_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": YESNO_SYSTEM},
            {"role": "user", "content": _build_yesno_user(typed_prefix, sentence)},
        ],
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
    }

    client = _shared_client_or_default(client)
    data = await _retry_request_json(client, payload)

    try:
        choice = data["choices"][0]
        content_tokens = (choice.get("logprobs") or {}).get("content") or []
        if not content_tokens:
            raise HTTPException(status_code=500, detail="LLM did not return logprobs")

        # probability mass extraction
        first_tok = content_tokens[0]
        def canon(token: str) -> Optional[str]:
            t = (token or "").strip().lower()
            return t if t in ("yes", "no") else None

        def safe_exp(lp) -> float:
            try:
                return math.exp(lp)
            except Exception:
                return 0.0

        yes_mass, no_mass, total_mass = 0.0, 0.0, 0.0
        tok = first_tok.get("token"); lp = first_tok.get("logprob")
        total_mass += safe_exp(lp)
        c = canon(tok)
        if c == "yes": yes_mass += safe_exp(lp)
        if c == "no": no_mass += safe_exp(lp)

        for alt in first_tok.get("top_logprobs") or []:
            c = canon(alt.get("token"))
            m = safe_exp(alt.get("logprob"))
            total_mass += m
            if c == "yes": yes_mass += m
            if c == "no": no_mass += m

        if yes_mass == 0.0 and no_mass == 0.0:
            raise HTTPException(status_code=500, detail="LLM logprobs missing Yes/No")

        if yes_mass > 0 and no_mass > 0:
            p_yes = yes_mass / (yes_mass + no_mass)
        else:
            p_yes = yes_mass / total_mass if total_mass > 0 else 0.5

        return max(0.0, min(1.0, p_yes))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="LLM logprob parse error")

async def llm_similarity_score(
    client: Optional[httpx.AsyncClient],
    typed_prefix: str,
    sentence: str,
    method: str = "json"
) -> float:
    client = _shared_client_or_default(client)
    if method == "logprob":
        return await llm_similarity_score_yesno_logprob(client, typed_prefix, sentence)
    return await llm_similarity_score_json(client, typed_prefix, sentence)

async def _with_semaphore(coro):
    async with _LLM_SEM:
        return await coro

async def most_similar(
    client: Optional[httpx.AsyncClient],
    typed_prefix: str,
    candidates: Iterable[str],
    top_k: int = 1,
    method: str = "json",
) -> Union[Tuple[str, float], List[Tuple[str, float]]]:
    cand_list = list(candidates)
    if not cand_list:
        return [] if top_k > 1 else ("", 0.0)

    client = _shared_client_or_default(client)
    tasks = [_with_semaphore(llm_similarity_score(client, typed_prefix, c, method=method)) for c in cand_list]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    scores: List[float] = []
    for r in results:
        if isinstance(r, HTTPException):
            raise r  # bubble specific HTTPException
        if isinstance(r, Exception):
            raise HTTPException(status_code=500, detail=f"LLM call error: {str(r)}")
        scores.append(float(r))

    ranked = sorted(zip(cand_list, scores), key=lambda x: x[1], reverse=True)
    return ranked[0] if top_k <= 1 else ranked[: min(top_k, len(ranked))], zip(cand_list, scores)
