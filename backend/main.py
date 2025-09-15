import time
import httpx
from fastapi import FastAPI, Response, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Set

from llm import most_similar  # <- our scoring helper

app = FastAPI(title="Stateless Similarity Backend", version="1.3.0")

# simple in-memory storage
USER_FILES = {}  # key: (user_id, user) -> {"filename": str, "content": str}

class SimilarityRequest(BaseModel):
    user_id: str
    user: int = Field(..., description="1 or 2; 1 -> a.txt, 2 -> b.txt")
    text: str = ""
    min_prefix_words: int = 5
    top_k: int = 1
    method: str = Field("json", description="'json' (prompt returns number) or 'logprob' (Yes/No token prob)")
    threshold: float = Field(0.0, description="Minimum similarity score threshold for match_found (0.0-1.0)")

class SimilarityResult(BaseModel):
    match_found: bool
    # What the frontend reads to highlight
    match_index: Optional[int] = None
    # Alias (optional; handy for debugging)
    best_index: Optional[int] = None
    best_sentence: Optional[str] = None
    best_score: Optional[float] = None
    # For top-k mode: list of (global_index, score)
    topk: Optional[List[Tuple[int, float]]] = None
    color: Optional[str] = None  # 'red' for user1, 'blue' for user2
    llm_elapsed_ms: float = 0.0

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload/{user_id}/{user}")
async def upload_file(user_id: str, user: int, file: UploadFile = File(...)):
    if user not in (1, 2):
        raise HTTPException(400, "user must be 1 or 2")
    content = (await file.read()).decode("utf-8", errors="replace")
    USER_FILES[(user_id, user)] = {"filename": file.filename, "content": content}
    return {"ok": True, "filename": file.filename, "bytes": len(content)}

@app.get("/file/{user_id}/{user}")
def get_file(user_id: str, user: int):
    if (user_id, user) not in USER_FILES:
        raise HTTPException(404, "No file uploaded for this user")
    entry = USER_FILES[(user_id, user)]
    headers = {"Content-Disposition": f'attachment; filename="{entry["filename"]}"'}
    return Response(entry["content"], media_type="text/plain; charset=utf-8", headers=headers)

# Tunables
MAX_SENTENCES = 200         # guardrail for very large files
MIN_TOKEN_OVERLAP = 0.15    # quick filter before LLM
PREFIX_STRATEGY = "few"     # "few" or "longest"


def norm_tokens(s: str) -> Set[str]:
    return {t.lower() for t in s.strip().split() if t.strip()}


def token_overlap(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    denom = min(len(a), len(b))
    return inter / denom


def build_prefixes(words: List[str], min_prefix_words: int) -> List[str]:
    if PREFIX_STRATEGY == "longest":
        return [" ".join(words)] if len(words) >= min_prefix_words else []
    # "few": use 2–3 robust prefixes (full, last 24, last 12)
    variants = []
    if len(words) >= min_prefix_words:
        variants.append(" ".join(words))
        if len(words) > 24:
            variants.append(" ".join(words[-24:]))
        if len(words) > 12:
            variants.append(" ".join(words[-12:]))
    # de-dup while preserving order
    seen = set(); out = []
    for p in variants:
        if p not in seen:
            out.append(p); seen.add(p)
    return out


@app.post("/similarity", response_model=SimilarityResult)
async def similarity(req: SimilarityRequest):
    if req.user not in (1, 2):
        raise HTTPException(400, "user must be 1 or 2")
    color = "red" if req.user == 1 else "blue"

    key = (req.user_id, req.user)
    if key not in USER_FILES:
        raise HTTPException(404, "No file uploaded for this user")

    # Split file into lines (sentences)
    text = USER_FILES[key]["content"]
    sentences = [s.strip() for s in text.splitlines() if s.strip()][:MAX_SENTENCES]

    words = (req.text or "").strip().split()
    if len(words) < req.min_prefix_words:
        return SimilarityResult(match_found=False, color=color, llm_elapsed_ms=0.0)

    prefixes = build_prefixes(words, req.min_prefix_words)
    if not prefixes:
        return SimilarityResult(match_found=False, color=color, llm_elapsed_ms=0.0)

    # Prefilter by cheap token overlap
    prefix_token_sets = [norm_tokens(p) for p in prefixes]
    candidates: List[Tuple[int, str]] = []
    for idx, sent in enumerate(sentences):
        ts = norm_tokens(sent)
        if any(token_overlap(ts, ps) >= MIN_TOKEN_OVERLAP for ps in prefix_token_sets):
            candidates.append((idx, sent))
    if not candidates:
        # Fallback: first 30 lines as a small pool
        for idx, sent in enumerate(sentences[: min(30, len(sentences))]):
            candidates.append((idx, sent))

    # Build pool of candidate strings
    pool = [s for _, s in candidates]

    # ---- Scoring (LLM) ----
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=None) as client:
        query = prefixes[0]  # use first viable prefix
        ranked = await most_similar(
            client, query, pool, top_k=max(1, req.top_k), method=req.method
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Helper: map sentences back to first unused matching global index (handles duplicates)
    def first_unused_index_for(sentence: str, used_locals: set) -> Optional[int]:
        for local_i, (global_i, s) in enumerate(candidates):
            if local_i in used_locals:
                continue
            if s == sentence:
                used_locals.add(local_i)
                return global_i
        return None

    if req.top_k <= 1:
        # most_similar returns: (best_sentence, best_score)
        best_sentence, best_score = ranked
        used = set()
        global_idx = first_unused_index_for(best_sentence, used)
        # If somehow None, default to 0
        if global_idx is None:
            global_idx = 0
        best_score = round(float(best_score), 4)
        return SimilarityResult(
            match_found=best_score >= req.threshold,
            match_index=global_idx,
            best_index=global_idx,              # optional alias
            best_sentence=best_sentence,
            best_score=best_score,
            color=color,
            llm_elapsed_ms=elapsed_ms,
        )
    else:
        # ranked: [(sentence, score), ...]
        used = set()
        topk_global: List[Tuple[int, float]] = []
        for sent, sc in ranked:
            gi = first_unused_index_for(sent, used)
            if gi is not None:
                topk_global.append((gi, round(float(sc), 4)))
            if len(topk_global) >= req.top_k:
                break

        if not topk_global:
            return SimilarityResult(match_found=False, color=color, llm_elapsed_ms=elapsed_ms)

        best_global_idx, best_score = topk_global[0]
        return SimilarityResult(
            match_found=best_score >= req.threshold,
            match_index=best_global_idx,
            best_index=best_global_idx,
            best_sentence=sentences[best_global_idx],
            best_score=best_score,
            topk=topk_global,
            color=color,
            llm_elapsed_ms=elapsed_ms,
        )