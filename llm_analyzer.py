"""
LLM integration module for analyzing product reviews.

Uses the OpenAI-compatible chat completions API to generate:
- Sentiment classification (positive / negative / mixed / neutral)
- A concise summary of each review
- Key points extracted from the review

Supports any OpenAI-compatible API (OpenAI, Azure, Ollama, LM Studio, etc.)
via the OPENAI_API_BASE environment variable.
"""

import logging
import time

from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from config import (
    API_CALL_DELAY,
    API_MAX_RETRIES,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)


# ─── Client Setup ────────────────────────────────────────────────────────────────


def _get_client() -> OpenAI:
    """Create an OpenAI client with configured base URL and API key."""
    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_API_BASE:
        kwargs["base_url"] = OPENAI_API_BASE
    return OpenAI(**kwargs)


# ─── Prompt Template ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a product review analyst. Your task is to analyze a customer review and provide:
1. **Sentiment**: Classify as one of: POSITIVE, NEGATIVE, MIXED, or NEUTRAL.
2. **Summary**: A 1-2 sentence concise summary of the review's main point.
3. **Key Points**: A list of 2-5 key takeaways from the review.

Respond in exactly this JSON format (no markdown fencing, just raw JSON):
{
  "sentiment": "POSITIVE|NEGATIVE|MIXED|NEUTRAL",
  "summary": "Concise summary here.",
  "key_points": ["point 1", "point 2", "point 3"]
}"""


def _build_user_prompt(review_text: str, rating: float | None = None) -> str:
    """Build the user prompt for a single review."""
    parts = ["Analyze the following product review:\n"]
    if rating is not None:
        parts.append(f"Rating: {rating}/5 stars\n")
    parts.append(f"Review:\n{review_text}")
    return "\n".join(parts)


# ─── Single Review Analysis ─────────────────────────────────────────────────────


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    stop=stop_after_attempt(API_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_llm(client: OpenAI, messages: list[dict], model: str) -> str:
    """
    Make a single LLM API call with retry logic for rate limits and connection errors.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def _parse_llm_response(response_text: str) -> dict:
    """
    Parse the JSON response from the LLM.
    Falls back to a basic structure if JSON parsing fails.
    """
    import json

    # Try to extract JSON from the response (handle markdown fencing)
    text = response_text.strip()
    if text.startswith("```"):
        # Remove markdown code fencing
        lines = text.split("\n")
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )

    try:
        result = json.loads(text)
        # Validate expected keys
        return {
            "sentiment": result.get("sentiment", "UNKNOWN"),
            "summary": result.get("summary", ""),
            "key_points": result.get("key_points", []),
        }
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM JSON response: {text[:200]}...")
        # Fallback: use the raw text as the summary
        return {
            "sentiment": "UNKNOWN",
            "summary": text[:500],
            "key_points": [],
        }


def analyze_review(
    review_text: str,
    rating: float | None = None,
    model: str = OPENAI_MODEL,
    client: OpenAI | None = None,
) -> dict:
    """
    Analyze a single review using the LLM.

    Args:
        review_text: The cleaned review text.
        rating: Optional star rating for additional context.
        model: LLM model name.
        client: Optional pre-created OpenAI client.

    Returns:
        Dict with 'sentiment', 'summary', and 'key_points'.
    """
    if client is None:
        client = _get_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(review_text, rating)},
    ]

    try:
        raw_response = _call_llm(client, messages, model)
        return _parse_llm_response(raw_response)
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded after retries: {e}")
        return {"sentiment": "ERROR", "summary": f"Rate limit error: {e}", "key_points": []}
    except APIConnectionError as e:
        logger.error(f"API connection failed: {e}")
        return {"sentiment": "ERROR", "summary": f"Connection error: {e}", "key_points": []}
    except APIError as e:
        logger.error(f"API error: {e}")
        return {"sentiment": "ERROR", "summary": f"API error: {e}", "key_points": []}
    except Exception as e:
        logger.error(f"Unexpected error during LLM analysis: {e}")
        return {"sentiment": "ERROR", "summary": f"Unexpected error: {e}", "key_points": []}


def analyze_chunked_review(
    chunks: list[str],
    rating: float | None = None,
    model: str = OPENAI_MODEL,
    client: OpenAI | None = None,
) -> dict:
    """
    Analyze a review that was split into multiple chunks.
    Analyzes each chunk individually and then merges results.

    Args:
        chunks: List of text chunks for a single review.
        rating: Optional star rating.
        model: LLM model name.
        client: Optional pre-created OpenAI client.

    Returns:
        Merged analysis dict.
    """
    if not chunks:
        return {"sentiment": "UNKNOWN", "summary": "", "key_points": []}

    if len(chunks) == 1:
        return analyze_review(chunks[0], rating, model, client)

    # Analyze each chunk
    chunk_results = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Analyzing chunk {i + 1}/{len(chunks)}")
        result = analyze_review(chunk, rating, model, client)
        chunk_results.append(result)
        if i < len(chunks) - 1:
            time.sleep(API_CALL_DELAY)

    # Merge results
    sentiments = [r["sentiment"] for r in chunk_results if r["sentiment"] != "ERROR"]
    all_key_points = []
    for r in chunk_results:
        all_key_points.extend(r.get("key_points", []))

    summaries = [r["summary"] for r in chunk_results if r["summary"]]

    # Determine overall sentiment by majority vote
    if sentiments:
        from collections import Counter
        sentiment = Counter(sentiments).most_common(1)[0][0]
    else:
        sentiment = "UNKNOWN"

    return {
        "sentiment": sentiment,
        "summary": " | ".join(summaries) if summaries else "",
        "key_points": all_key_points[:5],  # Keep top 5
    }


# ─── Batch Processing ───────────────────────────────────────────────────────────


def analyze_reviews_batch(
    reviews: list[dict],
    model: str = OPENAI_MODEL,
    progress_callback=None,
) -> list[dict]:
    """
    Analyze a batch of preprocessed reviews.

    Each review dict should have 'cleaned_text' or 'chunks' and optionally 'rating'.

    Args:
        reviews: List of preprocessed review dicts.
        model: LLM model name.
        progress_callback: Optional callable(current, total) for progress updates.

    Returns:
        List of review dicts enriched with 'sentiment', 'summary', 'key_points'.
    """
    client = _get_client()
    results = []
    total = len(reviews)

    for i, review in enumerate(reviews):
        logger.info(f"Analyzing review {i + 1}/{total}")

        chunks = review.get("chunks", [])
        rating = review.get("rating")

        if chunks and len(chunks) > 1:
            analysis = analyze_chunked_review(chunks, rating, model, client)
        else:
            text = chunks[0] if chunks else review.get("cleaned_text", review.get("review_text", ""))
            analysis = analyze_review(text, rating, model, client)

        # Merge analysis into the review dict
        enriched_review = review.copy()
        enriched_review["sentiment"] = analysis["sentiment"]
        enriched_review["summary"] = analysis["summary"]
        enriched_review["key_points"] = analysis["key_points"]
        results.append(enriched_review)

        if progress_callback:
            progress_callback(i + 1, total)

        # Rate limit delay between reviews (skip after the last one)
        if i < total - 1:
            time.sleep(API_CALL_DELAY)

    logger.info(f"Batch analysis complete. Processed {len(results)} reviews.")
    return results
