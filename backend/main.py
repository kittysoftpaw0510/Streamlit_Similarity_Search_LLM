import time
import logging
import json
from datetime import datetime
from fastapi import FastAPI, Response, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Set

from llm import most_similar, close_shared_client  # updated llm helpers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('similarity_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stateless Similarity Backend", version="1.5.0")

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
    # Error details for frontend
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    # Debug information
    debug_info: Optional[dict] = None

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
MAX_LLM_CANDIDATES = 60     # hard cap on calls to LLM scorer after prefilter


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
    start_time = time.perf_counter()
    request_id = f"{req.user_id}_{req.user}_{int(time.time())}"

    logger.info(f"[{request_id}] Similarity request started")
    logger.info(f"[{request_id}] Request: user={req.user}, text='{req.text[:50]}...', method={req.method}, threshold={req.threshold}")

    try:
        if req.user not in (1, 2):
            raise HTTPException(400, "user must be 1 or 2")
        color = "red" if req.user == 1 else "blue"

        key = (req.user_id, req.user)
        if key not in USER_FILES:
            raise HTTPException(404, "No file uploaded for this user")

        # Split file into lines (sentences)
        text = USER_FILES[key]["content"]
        sentences = [s.strip() for s in text.splitlines() if s.strip()][:MAX_SENTENCES]
        logger.info(f"[{request_id}] Loaded {len(sentences)} sentences from file")

        words = (req.text or "").strip().split()
        if len(words) < req.min_prefix_words:
            logger.info(f"[{request_id}] Text too short: {len(words)} words < {req.min_prefix_words} minimum")
            return SimilarityResult(match_found=False, color=color, llm_elapsed_ms=0.0)

        prefixes = build_prefixes(words, req.min_prefix_words)
        if not prefixes:
            logger.info(f"[{request_id}] No valid prefixes generated")
            return SimilarityResult(match_found=False, color=color, llm_elapsed_ms=0.0)

        logger.info(f"[{request_id}] Generated {len(prefixes)} prefixes: {prefixes}")

        # Prefilter by cheap token overlap
        prefix_token_sets = [norm_tokens(p) for p in prefixes]
        candidates: List[Tuple[int, str]] = []
        for idx, sent in enumerate(sentences):
            ts = norm_tokens(sent)
            if any(token_overlap(ts, ps) >= MIN_TOKEN_OVERLAP for ps in prefix_token_sets):
                candidates.append((idx, sent))

        # Early exit instead of fallback:
        if not candidates:
            logger.info(f"[{request_id}] No candidates after overlap filter; returning no match")
            return SimilarityResult(
                match_found=False,
                color=color,
                llm_elapsed_ms=0.0,
                debug_info={
                    "query": prefixes[0],
                    "candidates_count": 0,
                    "threshold": req.threshold,
                    "method": req.method,
                    "reason": "no token overlap"
                },
            )

        # Hard cap LLM workload
        if len(candidates) > MAX_LLM_CANDIDATES:
            candidates = candidates[:MAX_LLM_CANDIDATES]

        logger.info(f"[{request_id}] Found {len(candidates)} candidate sentences after filtering")

        # Build pool of candidate strings
        pool = [s for _, s in candidates]

        # ---- Scoring (LLM) ----
        t0 = time.perf_counter()
        query = prefixes[0]  # use first viable prefix
        logger.info(f"[{request_id}] Starting LLM scoring with query: '{query}'")

        ranked, all_scores = await most_similar(
            client=None,  # None -> use shared AsyncClient inside llm.py
            typed_prefix=query,
            candidates=pool,
            top_k=max(1, req.top_k),
            method=req.method,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"[{request_id}] LLM scoring completed in {elapsed_ms:.1f}ms")

        # Log all scores for debugging
        if req.top_k <= 1:
            best_sentence, best_score = ranked
            logger.info(f"[{request_id}] Best result: score={best_score:.4f}, sentence='{best_sentence[:100]}...'")
        else:
            logger.info(f"[{request_id}] Top results:")
            for i, (sent, score) in enumerate(ranked[:5]):  # Log top 5
                logger.info(f"[{request_id}]   {i+1}. score={score:.4f}, sentence='{sent[:100]}...'")

        # Helper: map sentences back to first unused matching global index (handles duplicates)
        def first_unused_index_for(sentence: str, used_locals: set) -> Optional[int]:
            for local_i, (global_i, s) in enumerate(candidates):
                if local_i in used_locals:
                    continue
                if s == sentence:
                    used_locals.add(local_i)
                    return global_i
            return None

        # Create debug info
        debug_info = {
            "query": query,
            "candidates_count": len(candidates),
            "threshold": req.threshold,
            "method": req.method,
            "template_used": "JSON scoring" if req.method == "json" else "Yes/No logprob",
        }

        if req.top_k <= 1:
            # most_similar returns: (best_sentence, best_score)
            best_sentence, best_score = ranked
            used = set()
            global_idx = first_unused_index_for(best_sentence, used)
            if global_idx is None:
                global_idx = 0
            best_score = round(float(best_score), 4)

            match_found = best_score >= req.threshold
            logger.info(f"[{request_id}] Final result: match_found={match_found}, score={best_score}, threshold={req.threshold}")

            debug_info.update({
                "best_score": best_score,
                "best_sentence": best_sentence,
                "match_found": match_found,
                # "all_scores": [round(float(s), 4) for _, s in (ranked if isinstance(ranked, list) else [(best_sentence, best_score)])]
                "all_scores": all_scores
            })

            return SimilarityResult(
                match_found=match_found,
                match_index=global_idx,
                best_index=global_idx,              # optional alias
                best_sentence=best_sentence,
                best_score=best_score,
                color=color,
                llm_elapsed_ms=elapsed_ms,
                debug_info=debug_info,
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
                logger.info(f"[{request_id}] No valid results found in top-k mode")
                return SimilarityResult(match_found=False, color=color, llm_elapsed_ms=elapsed_ms)

            best_global_idx, best_score = topk_global[0]
            match_found = best_score >= req.threshold
            logger.info(f"[{request_id}] Top-k result: match_found={match_found}, best_score={best_score}")

            debug_info.update({
                "best_score": best_score,
                "topk_scores": [score for _, score in topk_global],
                "match_found": match_found,
            })

            return SimilarityResult(
                match_found=match_found,
                match_index=best_global_idx,
                best_index=best_global_idx,
                best_sentence=sentences[best_global_idx],
                best_score=best_score,
                topk=topk_global,
                color=color,
                llm_elapsed_ms=elapsed_ms,
                debug_info=debug_info,
            )

    except HTTPException as e:
        logger.error(f"[{request_id}] HTTP error: {e.detail}")
        return SimilarityResult(
            match_found=False,
            color=color,
            llm_elapsed_ms=0.0,
            error_message=e.detail,
            error_type="HTTPException"
        )
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)
        return SimilarityResult(
            match_found=False,
            color=color,
            llm_elapsed_ms=0.0,
            error_message=f"Internal server error: {str(e)}",
            error_type="Exception"
        )

# Cleanly close shared HTTP resources when FastAPI shuts down
@app.on_event("shutdown")
async def _shutdown_event():
    await close_shared_client()
