import os
import json
import math
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

YESNO_SYSTEM = (
    "You are a strict semantic similarity judge.\n\n"
    "You must answer with exactly ONE token: \"Yes\" or \"No\". Nothing else.\n\n"
    "Interpretation rule:\n"
    "- \"Yes\" = The candidate sentence strongly continues or matches the typed prefix’s meaning/intention.\n"
    "- \"No\"  = The candidate sentence does not strongly match or continue the typed prefix’s meaning/intention.\n\n"
    "Do not include punctuation, explanations, or any additional text. "
    "Output must be a single token \"Yes\" or \"No\"."
)



def _build_user(typed_prefix: str, sentence: str) -> str:
    return (
        "TYPED_PREFIX:\n"
        f"{typed_prefix}\n\n"
        "CANDIDATE_SENTENCE:\n"
        f"{sentence}\n\n"
        "Return ONLY JSON like {\"score\": 0.0-1.0}"
    )


def _build_yesno_user(typed_prefix: str, sentence: str) -> str:
    return (
        "Question: Does this sentence strongly match/continue the typed prefix's intent?\n\n"
        f"Typed prefix: {typed_prefix}\n"
        f"Candidate sentence: {sentence}\n\n"
        "Answer strictly 'Yes' or 'No'."
    )


def _fallback_score(typed_prefix: str, sentence: str) -> float:
    """
    Conservative lexical overlap fallback in [0,1] if no API key or API fails.
    Uses token overlap weighted by token length and lightly boosts for order.
    """
    print("[INFO] Fallback used")

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


async def llm_similarity_score_json(
    client: httpx.AsyncClient, typed_prefix: str, sentence: str
) -> float:
    """
    Method 1: Ask the model to return a scalar score via JSON.
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
        score = max(0.0, min(1.0, score))
        return float(f"{score:.4f}")
    except Exception:
        return _fallback_score(typed_prefix, sentence)


async def llm_similarity_score_yesno_logprob(
    client: httpx.AsyncClient, typed_prefix: str, sentence: str
) -> float:
    """
    Yes/No scoring using first-token logprobs only.
    Primary: P(Yes | {Yes, No}) when both labels are present in the distribution.
    Fallback: If only one label appears, return a conservative lower-bound P(Yes)
              by normalizing over total first-token mass (chosen + all alts).
    No heuristics; if logprobs unavailable, fall back to lexical scorer.
    """
    if not OPENAI_API_KEY:
        return _fallback_score(typed_prefix, sentence)

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": YESNO_SYSTEM},
            {"role": "user", "content": _build_yesno_user(typed_prefix, sentence)},
        ],
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,  # ← ask for more alts so 'No' is more likely to appear
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
        choice = data["choices"][0]

        # print(data)

        content_tokens = (choice.get("logprobs") or {}).get("content") or []
        if not content_tokens:
            return _fallback_score(typed_prefix, sentence)

        first_tok = content_tokens[0]

        import re
        def canon(token: str | None) -> str | None:
            if not token:
                return None
            t = re.sub(r"^\s+|^[^A-Za-z]+", "", token)  # remove leading spaces/non-letters
            t = t.rstrip(" .,!?:;\"'")                  # strip common trailing punctuation
            t = t.lower()
            return t if t in ("yes", "no") else None

        def safe_exp(lp) -> float:
            try:
                return math.exp(lp) if isinstance(lp, (int, float)) else 0.0
            except Exception:
                return 0.0

        # Collect masses
        yes_mass = 0.0
        no_mass  = 0.0
        total_mass = 0.0

        # chosen token
        tok = first_tok.get("token")
        lp  = first_tok.get("logprob")
        mass = safe_exp(lp)
        total_mass += mass
        c = canon(tok)
        if c == "yes":
            yes_mass += mass
        elif c == "no":
            no_mass += mass

        # alternatives
        for alt in first_tok.get("top_logprobs") or []:
            t = alt.get("token")
            l = alt.get("logprob")
            m = safe_exp(l)
            total_mass += m
            c = canon(t)
            if c == "yes":
                yes_mass += m
            elif c == "no":
                no_mass += m

        # print(yes_mass, no_mass)

        # If no mass on either label, we can't derive a Yes/No probability
        if yes_mass == 0.0 and no_mass == 0.0:
            return _fallback_score(typed_prefix, sentence)

        if yes_mass > 0.0 and no_mass > 0.0:
            # proper binary normalization
            p_yes = yes_mass / (yes_mass + no_mass)
        else:
            # conservative lower-bound: normalize by total first-token mass
            # (prevents hard 1.000 when only one label appears)
            denom = total_mass if total_mass > 0.0 else (yes_mass + no_mass)
            p_yes = yes_mass / denom if denom > 0.0 else 0.5

        print(p_yes)

        return float(f"{max(0.0, min(1.0, p_yes)):.4f}")

    except Exception:
        return _fallback_score(typed_prefix, sentence)




async def llm_similarity_score(
    client: httpx.AsyncClient, typed_prefix: str, sentence: str, method: str = "json"
) -> float:
    """
    Dispatch between the two methods.
    method ∈ {"json", "logprob"}
    """
    if method == "logprob":
        return await llm_similarity_score_yesno_logprob(client, typed_prefix, sentence)
    return await llm_similarity_score_json(client, typed_prefix, sentence)


async def most_similar(
    client: httpx.AsyncClient,
    typed_prefix: str,
    candidates: Iterable[str],
    top_k: int = 1,
    method: str = "json",
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
    tasks = [llm_similarity_score(client, typed_prefix, c, method=method) for c in cand_list]
    scores = await asyncio.gather(*tasks, return_exceptions=False)

    ranked = sorted(zip(cand_list, scores), key=lambda x: x[1], reverse=True)

    if top_k <= 1:
        return ranked[0]
    return ranked[: min(top_k, len(ranked))]