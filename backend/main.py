import time
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Response, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Set

from llm import most_similar, close_shared_client  # updated llm helpers

# Create logs directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure main logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'similarity_debug.log', encoding='utf-8'),
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
    top_k: int = 1
    method: str = Field("json", description="'json' (prompt returns number) or 'logprob' (Yes/No token prob)")
    threshold: float = Field(0.0, description="Minimum similarity score threshold for match_found (0.0-1.0)")
    batch_size: int = Field(20, description="Number of sentences to process in each batch (1-100)")

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
MIN_BATCH_SIZE = 1          # minimum batch size
MAX_BATCH_SIZE = 100        # maximum batch size


def create_request_logger(request_id: str) -> logging.Logger:
    """Create a dedicated logger for a specific request"""
    request_logger = logging.getLogger(f"request_{request_id}")
    request_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    for handler in request_logger.handlers[:]:
        request_logger.removeHandler(handler)

    # Create file handler for this specific request with UTF-8 encoding
    log_file = LOG_DIR / f"request_{request_id}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    request_logger.addHandler(file_handler)
    request_logger.propagate = False  # Don't propagate to parent logger

    return request_logger


def safe_text_for_logging(text: str, max_length: int = 100) -> str:
    """Safely prepare text for logging by handling Unicode and length"""
    try:
        # Remove or replace problematic Unicode characters
        safe_text = text.encode('ascii', errors='replace').decode('ascii')
        # Truncate if too long
        if len(safe_text) > max_length:
            safe_text = safe_text[:max_length] + "..."
        return safe_text
    except Exception:
        return f"<text-encoding-error-length-{len(text)}>"





async def process_sentences_in_batches(
    request_logger: logging.Logger,
    request_id: str,
    query: str,
    sentences: List[str],
    batch_size: int,
    method: str
) -> List[Tuple[int, str, float]]:
    """Process all sentences in batches and return scored results"""
    all_results = []
    total_batches = (len(sentences) + batch_size - 1) // batch_size

    request_logger.info(f"Processing {len(sentences)} sentences in {total_batches} batches of size {batch_size}")

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sentences))
        batch_sentences = sentences[start_idx:end_idx]

        request_logger.info(f"Processing batch {batch_idx + 1}/{total_batches} (sentences {start_idx}-{end_idx-1})")

        # Score this batch
        batch_results = await most_similar(
            client=None,
            typed_prefix=query,
            candidates=batch_sentences,
            top_k=len(batch_sentences),  # Get all scores
            method=method,
        )

        # Convert to global indices and add to results
        # NOTE: batch_results is sorted by score, not original order!
        # We need to find the original index of each sentence in the batch
        for sentence, score in batch_results:
            # Find the original index of this sentence in the batch
            try:
                local_idx = batch_sentences.index(sentence)
                global_idx = start_idx + local_idx
            except ValueError:
                # Fallback: if exact match fails, find by content
                global_idx = None
                for i, orig_sentence in enumerate(batch_sentences):
                    if orig_sentence.strip() == sentence.strip():
                        global_idx = start_idx + i
                        break
                if global_idx is None:
                    request_logger.error(f"Could not find original index for sentence: '{sentence[:50]}...'")
                    continue

            all_results.append((global_idx, sentence, score))
            safe_sentence = safe_text_for_logging(sentence, 100)
            request_logger.info(f"Sentence {global_idx}: score={score:.4f}, text='{safe_sentence}'")

    return all_results


@app.post("/similarity", response_model=SimilarityResult)
async def similarity(req: SimilarityRequest):
    request_id = f"{req.user_id}_{req.user}_{int(time.time())}"

    # Create dedicated logger for this request
    request_logger = create_request_logger(request_id)

    # Also log to main logger
    logger.info(f"[{request_id}] Similarity request started")
    logger.info(f"[{request_id}] Request: user={req.user}, text='{req.text[:50]}...', method={req.method}, threshold={req.threshold}, batch_size={req.batch_size}")

    # Log to request-specific file
    request_logger.info("=== SIMILARITY REQUEST STARTED ===")
    request_logger.info(f"Request ID: {request_id}")
    request_logger.info(f"User: {req.user}")
    request_logger.info(f"Input text: '{req.text}'")
    request_logger.info(f"Method: {req.method}")
    request_logger.info(f"Threshold: {req.threshold}")
    request_logger.info(f"Batch size: {req.batch_size}")

    try:
        if req.user not in (1, 2):
            raise HTTPException(400, "user must be 1 or 2")
        color = "red" if req.user == 1 else "blue"

        # Validate batch size
        batch_size = max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, req.batch_size))
        if batch_size != req.batch_size:
            request_logger.info(f"Batch size adjusted from {req.batch_size} to {batch_size}")

        key = (req.user_id, req.user)
        if key not in USER_FILES:
            raise HTTPException(404, "No file uploaded for this user")

        # Split file into lines (sentences)
        text = USER_FILES[key]["content"]
        sentences = [s.strip() for s in text.splitlines() if s.strip()][:MAX_SENTENCES]
        request_logger.info(f"Loaded {len(sentences)} sentences from file")
        logger.info(f"[{request_id}] Loaded {len(sentences)} sentences from file")

        # Use the input text directly as query (no prefix logic or token filtering)
        query = (req.text or "").strip()
        if not query:
            request_logger.info("Empty query text")
            logger.info(f"[{request_id}] Empty query text")
            return SimilarityResult(match_found=False, color=color, llm_elapsed_ms=0.0)

        request_logger.info(f"Using query: '{query}'")
        logger.info(f"[{request_id}] Using query: '{query}'")

        # ---- Scoring (LLM) - Process ALL sentences ----
        t0 = time.perf_counter()
        request_logger.info("=== STARTING LLM SCORING ===")
        logger.info(f"[{request_id}] Starting LLM scoring for all {len(sentences)} sentences")

        all_scored_results = await process_sentences_in_batches(
            request_logger, request_id, query, sentences, batch_size, req.method
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        request_logger.info(f"=== LLM SCORING COMPLETED in {elapsed_ms:.1f}ms ===")
        logger.info(f"[{request_id}] LLM scoring completed in {elapsed_ms:.1f}ms")

        # Sort all results by score (descending)
        all_scored_results.sort(key=lambda x: x[2], reverse=True)

        # Log all scores to request log (sorted by score)
        request_logger.info("=== ALL SENTENCE SCORES (SORTED BY RELEVANCE) ===")
        for rank, (idx, sentence, score) in enumerate(all_scored_results, 1):
            safe_sentence = safe_text_for_logging(sentence, 200)
            rank_indicator = "🎯" if rank == 1 else f"#{rank}"
            request_logger.info(f"{rank_indicator} Sentence {idx}: score={score:.4f}, text='{safe_sentence}'")

        # Log top results to main log
        logger.info(f"[{request_id}] === TOP 5 RESULTS (SORTED BY SCORE) ===")
        for i, (idx, sentence, score) in enumerate(all_scored_results[:5]):
            safe_sentence = safe_text_for_logging(sentence, 100)
            rank_indicator = "🎯 BEST" if i == 0 else f"#{i+1}"
            logger.info(f"[{request_id}] {rank_indicator} → Index {idx}, Score {score:.4f}: '{safe_sentence}'")

        if not all_scored_results:
            request_logger.info("No results found")
            logger.info(f"[{request_id}] No results found")
            return SimilarityResult(match_found=False, color=color, llm_elapsed_ms=elapsed_ms)

        # Get the best result
        best_idx, best_sentence, best_score = all_scored_results[0]
        best_score = round(float(best_score), 4)
        match_found = best_score >= req.threshold

        request_logger.info(f"=== FINAL RESULT ===")
        request_logger.info(f"Best match: sentence {best_idx}")
        request_logger.info(f"Best score: {best_score}")
        request_logger.info(f"Threshold: {req.threshold}")
        request_logger.info(f"Match found: {match_found}")
        safe_best_sentence = safe_text_for_logging(best_sentence, 200)
        request_logger.info(f"Best sentence: '{safe_best_sentence}'")

        # Debug: verify the index is correct
        if 0 <= best_idx < len(sentences):
            actual_sentence = sentences[best_idx]
            if actual_sentence.strip() != best_sentence.strip():
                request_logger.error(f"INDEX MISMATCH! best_idx={best_idx} points to different sentence")
                request_logger.error(f"Expected: '{safe_text_for_logging(best_sentence, 200)}'")
                request_logger.error(f"Actual at index {best_idx}: '{safe_text_for_logging(actual_sentence, 200)}'")
            else:
                request_logger.info(f"Index verification: ✓ Index {best_idx} correctly points to the best sentence")
        else:
            request_logger.error(f"INVALID INDEX! best_idx={best_idx} is out of range for {len(sentences)} sentences")

        logger.info(f"[{request_id}] Final result: match_found={match_found}, score={best_score}, threshold={req.threshold}")

        # Create debug info
        debug_info = {
            "query": query,
            "total_sentences": len(sentences),
            "batch_size": batch_size,
            "threshold": req.threshold,
            "method": req.method,
            "template_used": "JSON scoring" if req.method == "json" else "Yes/No logprob",
            "best_score": best_score,
            "best_sentence": best_sentence,
            "match_found": match_found,
            "all_scores": [round(float(score), 4) for _, _, score in all_scored_results],
            "top_5_results": [
                {"index": idx, "score": round(float(score), 4), "sentence": sentence[:200]}
                for idx, sentence, score in all_scored_results[:5]
            ]
        }

        if req.top_k <= 1:
            return SimilarityResult(
                match_found=match_found,
                match_index=best_idx,
                best_index=best_idx,
                best_sentence=best_sentence,
                best_score=best_score,
                color=color,
                llm_elapsed_ms=elapsed_ms,
                debug_info=debug_info,
            )
        else:
            # For top-k mode, return multiple results
            topk_results = []
            for i, (idx, sentence, score) in enumerate(all_scored_results[:req.top_k]):
                topk_results.append((idx, round(float(score), 4)))

            if not topk_results:
                request_logger.info("No valid results found in top-k mode")
                logger.info(f"[{request_id}] No valid results found in top-k mode")
                return SimilarityResult(match_found=False, color=color, llm_elapsed_ms=elapsed_ms)

            debug_info.update({
                "topk_scores": [score for _, score in topk_results],
            })

            return SimilarityResult(
                match_found=match_found,
                match_index=best_idx,
                best_index=best_idx,
                best_sentence=best_sentence,
                best_score=best_score,
                topk=topk_results,
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
