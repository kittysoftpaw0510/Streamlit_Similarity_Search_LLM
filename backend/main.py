import httpx
import asyncio
from fastapi import FastAPI, Response, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Set

from textgen import generate_sentences, simulate_file_content
from llm import llm_is_similar

app = FastAPI(title="Stateless Similarity Backend", version="1.0.0")

# simple in-memory storage
USER_FILES = {}  # key: (user_id, user) -> {"filename": str, "content": str}

class SimilarityRequest(BaseModel):
    user_id: str
    user: int = Field(..., description="1 or 2; 1 -> a.txt, 2 -> b.txt")
    text: str = ""
    min_prefix_words: int = 5

class SimilarityResult(BaseModel):
    match_found: bool
    match_index: Optional[int] = None
    match_sentence: Optional[str] = None
    color: Optional[str] = None  # 'red' for user1, 'blue' for user2

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload/{user_id}/{user}")
async def upload_file(user: int, file: UploadFile = File(...)):
    user_id = f"user{user}"
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
    headers = {"Content-Disposition": f'attachment; filename="{entry['filename']}\"'}
    return Response(entry["content"], media_type="text/plain; charset=utf-8", headers=headers)

MAX_SENTENCES = 200         # guardrail for very large files
MAX_CONCURRENCY = 6         # # of parallel LLM checks
LLM_CALL_TIMEOUT = 6.0      # per-call timeout seconds
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
        if len(words) > 12:
            variants.append(" ".join(words[-12:]))
        if len(words) > 24:
            variants.append(" ".join(words[-24:]))
    # de-dup while preserving order
    seen = set()
    out = []
    for p in variants:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

@app.post("/similarity", response_model=SimilarityResult)
async def similarity(req: SimilarityRequest):
    if req.user not in (1, 2):
        raise HTTPException(400, "user must be 1 or 2")
    color = "red" if req.user == 1 else "blue"

    # ---- get sentences from your uploaded storage ----
    key = (req.user_id, req.user)
    if key not in USER_FILES:
        # if you still support synthetic defaults, swap this to simulate_file_content
        raise HTTPException(404, "No file uploaded for this user")
    text = USER_FILES[key]["content"]
    sentences = [s.strip() for s in text.splitlines() if s.strip()][:MAX_SENTENCES]

    words = (req.text or "").strip().split()
    if len(words) < req.min_prefix_words:
        return SimilarityResult(match_found=False, color=color)

    # robust prefixes (few only)
    prefixes = build_prefixes(words, req.min_prefix_words)
    if not prefixes:
        return SimilarityResult(match_found=False, color=color)

    # quick token-screening: only keep candidates with minimal overlap
    prefix_token_sets = [norm_tokens(p) for p in prefixes]
    candidates: List[Tuple[int, str]] = []
    for idx, sent in enumerate(sentences):
        ts = norm_tokens(sent)
        if any(token_overlap(ts, ps) >= MIN_TOKEN_OVERLAP for ps in prefix_token_sets):
            candidates.append((idx, sent))

    # if everything filtered out, pick a small fallback subset to try
    if not candidates:
        for idx, sent in enumerate(sentences[: min(30, len(sentences))]):
            candidates.append((idx, sent))

    # concurrent LLM checks with a semaphore + per-call timeout
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    async with httpx.AsyncClient(timeout=None) as client:
        async def check_one(idx: int, sent: str) -> Optional[int]:
            async with sem:
                # try each prefix; early return on first hit
                for pref in prefixes:
                    try:
                        ok = await asyncio.wait_for(llm_is_similar(client, pref, sent), timeout=LLM_CALL_TIMEOUT)
                    except asyncio.TimeoutError:
                        ok = False
                    if ok:
                        return idx
            return None

        tasks = [asyncio.create_task(check_one(idx, sent)) for idx, sent in candidates]
        for fut in asyncio.as_completed(tasks):
            res_idx = await fut
            if res_idx is not None:
                # Cancel pending tasks to save budget
                for t in tasks:
                    if not t.done():
                        t.cancel()
                return SimilarityResult(
                    match_found=True,
                    match_index=res_idx,
                    match_sentence=sentences[res_idx],
                    color=color,
                )

    return SimilarityResult(match_found=False, color=color)