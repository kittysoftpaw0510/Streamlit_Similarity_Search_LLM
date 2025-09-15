import os
import json
import httpx
from typing import Iterable, List, Tuple, Union
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a strict semantic similarity judge. "
    "Given a TYPED_PREFIX and a CANDIDATE_SENTENCE, return JSON with fields: "
    '{"score": float} where score ∈ [0,1].\n'
    "Scoring rubric:\n"
    "- 1.0 = essentially the same intent/meaning; perfect continuation.\n"
    "- 0.8–0.99 = very strong semantic match; highly likely intended.\n"
    "- 0.5–0.79 = related but possibly broader/narrower or partially off.\n"
    "- 0.2–0.49 = weakly related; shares topic but not the same intent.\n"
    "- 0.0–0.19 = unrelated.\n"
    "Important: Be conservative for short prefixes (<= 3 words). "
    "Penalize mismatched named entities, numbers, dates, or negations.\n"
    "Return ONLY JSON like {\"score\": 0.0-1.0}"
)

def _build_user(typed_prefix: str, sentence: str) -> str:
    return (
        "TYPED_PREFIX:\n"
        f"{typed_prefix}\n\n"
        "CANDIDATE_SENTENCE:\n"
        f"{sentence}\n\n"
        "Return ONLY JSON like {\"score\": 0.0-1.0}"
    )

def _fallback_score(typed_prefix: str, sentence: str) -> float:
    """
    Conservative lexical overlap fallback in [0,1] if no API key or API fails.
    Uses token overlap weighted by token length and lightly boosts for order.
    """
    def tokens(s: str) -> List[str]:
        return [w.lower().strip(",.!?:;()[]\"'") for w in s.split() if w.strip()]

    tp = tokens(typed_prefix)
    sn = tokens(sentence)
    if not tp or not sn:
        return 0.0

    # Length-weighted overlap
    tp_set = set(tp)
    inter = sum(len(t) for t in tp_set.intersection(sn))
    denom = sum(len(t) for t in set(tp_set)) or 1
    base = inter / denom

    # Order/cohesion bonus (very light)
    idx = {t: i for i, t in enumerate(sn)}
    in_order_pairs = 0
    total_pairs = max(0, len(tp) - 1)
    for i in range(len(tp) - 1):
        if tp[i] in idx and tp[i+1] in idx and idx[tp[i]] < idx[tp[i+1]]:
            in_order_pairs += 1
    order_bonus = (in_order_pairs / total_pairs) * 0.1 if total_pairs else 0.0

    score = max(0.0, min(1.0, base + order_bonus))

    # Be conservative for short prefixes (<= 3 words)
    if len(tp) <= 3:
        score *= 0.7

    return float(f"{score:.4f}")

async def llm_similarity_score(
    client: httpx.AsyncClient, typed_prefix: str, sentence: str
) -> float:
    """
    Returns a semantic similarity score in [0,1] between typed_prefix and sentence.
    Falls back to a conservative lexical score if API key is missing or on failure.
    """
    if not OPENAI_API_KEY:
        return _fallback_score(typed_prefix, sentence)

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

    try:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        parsed = json.loads(content)
        score = float(parsed.get("score", 0.0))
        # Clamp and tidy
        score = max(0.0, min(1.0, score))
        return float(f"{score:.4f}")
    except Exception:
        # Any error -> fallback
        return _fallback_score(typed_prefix, sentence)

async def most_similar(
    client: httpx.AsyncClient,
    typed_prefix: str,
    candidates: Iterable[str],
    top_k: int = 1,
) -> Union[Tuple[str, float], List[Tuple[str, float]]]:
    """
    Compute scores for all candidates and return the most similar (or top-K) sorted desc.
    - top_k=1 -> returns (best_candidate, best_score)
    - top_k>1 -> returns [(candidate, score), ...] length=min(top_k, len(candidates))
    """
    # Materialize list once
    cand_list = list(candidates)
    if not cand_list:
        return [] if top_k > 1 else ("", 0.0)

    # Score in parallel
    import asyncio
    tasks = [llm_similarity_score(client, typed_prefix, c) for c in cand_list]
    scores = await asyncio.gather(*tasks, return_exceptions=False)

    ranked = sorted(zip(cand_list, scores), key=lambda x: x[1], reverse=True)

    if top_k <= 1:
        return ranked[0]
    return ranked[: min(top_k, len(ranked))]
