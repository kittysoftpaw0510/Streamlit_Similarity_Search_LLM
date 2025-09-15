# llm.py
import os
import json
import httpx
import numpy as np
from typing import Iterable, List, Tuple, Union
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# --------- Fallbacks (no API key / errors) ---------
def _lexical_fallback_score(a: str, b: str) -> float:
    """
    Conservative token overlap in [0,1] for dev/demo fallback.
    """
    def toks(s: str) -> List[str]:
        return [w.lower().strip(",.!?:;()[]\"'") for w in s.split() if w.strip()]

    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    inter = len(sa & sb)
    denom = min(len(sa), len(sb)) or 1
    base = inter / denom
    if len(ta) <= 3:  # be conservative for short prefixes
        base *= 0.7
    return float(f"{max(0.0, min(1.0, base)):.4f}")

def _trivial_vec(s: str) -> np.ndarray:
    """
    Tiny hashed bag-of-words vector for fallback-only (8 dims).
    """
    toks = [w.lower().strip(",.!?:;()[]\"'") for w in s.split() if w.strip()]
    v = np.zeros(8, dtype=np.float32)
    for t in toks:
        v[hash(t) % 8] += 1.0
    n = np.linalg.norm(v)
    return (v / n) if n else v

# --------- Embeddings helpers ---------
async def _embed_many(client: httpx.AsyncClient, texts: List[str]) -> List[np.ndarray]:
    if not OPENAI_API_KEY:
        return [_trivial_vec(t) for t in texts]

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"input": texts, "model": OPENAI_EMBED_MODEL}
    resp = await client.post("https://api.openai.com/v1/embeddings", json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]
    return [np.array(item["embedding"], dtype=np.float32) for item in data]

def _cosine01(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    cos = float(np.dot(a, b) / denom)  # [-1, 1]
    return float(f"{((cos + 1.0) / 2.0):.4f}")  # [0,1]

# --------- Public API ---------
async def similarity_score(client: httpx.AsyncClient, query: str, text: str) -> float:
    """
    Embeddings + cosine in [0,1]; falls back to lexical heuristic if needed.
    """
    try:
        vecs = await _embed_many(client, [query, text])
        return _cosine01(vecs[0], vecs[1])
    except Exception:
        return _lexical_fallback_score(query, text)

async def most_similar(
    client: httpx.AsyncClient,
    query: str,
    candidates: Iterable[str],
    top_k: int = 1,
) -> Union[Tuple[int, str, float], List[Tuple[int, str, float]]]:
    """
    Scores all candidates in one batch and returns:
      - top_k=1 -> (index, sentence, score)
      - top_k>1 -> [(index, sentence, score), ...] sorted desc by score
    """
    cand_list = list(candidates)
    if not cand_list:
        return [] if top_k > 1 else (-1, "", 0.0)

    try:
        vecs = await _embed_many(client, [query] + cand_list)
        qv = vecs[0]
        cvecs = vecs[1:]
        scored = [(i, s, _cosine01(qv, v)) for i, (s, v) in enumerate(zip(cand_list, cvecs))]
    except Exception:
        # Fallback: lexical heuristic per candidate
        scored = [(i, s, _lexical_fallback_score(query, s)) for i, s in enumerate(cand_list)]

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[0] if top_k <= 1 else scored[: min(top_k, len(scored))]
