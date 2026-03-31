"""
Text preprocessing module for cleaning and chunking review text
before sending to an LLM API.

Features:
- HTML entity decoding
- Unicode normalization
- Whitespace cleanup
- Token-aware chunking via tiktoken
"""

import html
import logging
import re
import unicodedata

import tiktoken

from config import MAX_TOKENS_PER_CHUNK, OPENAI_MODEL

logger = logging.getLogger(__name__)


# ─── Text Cleaning ───────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """
    Clean a raw review text string.

    Steps:
        1. Decode HTML entities (&amp; → &, etc.)
        2. Normalize Unicode (e.g., curly quotes → straight quotes)
        3. Remove non-printable / control characters
        4. Collapse multiple whitespace into single spaces
        5. Strip leading/trailing whitespace

    Args:
        text: Raw review text.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # 1. Decode HTML entities
    text = html.unescape(text)

    # 2. Unicode normalize (NFKC maps compatibility chars to canonical form)
    text = unicodedata.normalize("NFKC", text)

    # 3. Remove non-printable / control characters (keep newlines / tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # 4. Replace various dash / quote unicode variants with ASCII equivalents
    replacements = {
        "\u2018": "'",   # left single quotation mark
        "\u2019": "'",   # right single quotation mark
        "\u201c": '"',   # left double quotation mark
        "\u201d": '"',   # right double quotation mark
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u2026": "...", # ellipsis
        "\u00a0": " ",   # non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # 5. Collapse multiple spaces / newlines
    text = re.sub(r"\n{3,}", "\n\n", text)    # max two consecutive newlines
    text = re.sub(r"[ \t]+", " ", text)         # collapse horizontal whitespace
    text = re.sub(r" *\n *", "\n", text)        # trim spaces around newlines

    # 6. Strip
    text = text.strip()

    return text


# ─── Token Counting & Chunking ──────────────────────────────────────────────────


def _get_encoding(model: str = OPENAI_MODEL) -> tiktoken.Encoding:
    """
    Get the tiktoken encoding for the specified model.
    Falls back to cl100k_base if the model is unknown.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.debug(
            f"Model '{model}' not found in tiktoken. Using cl100k_base encoding."
        )
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """Count the number of tokens in a text string."""
    enc = _get_encoding(model)
    return len(enc.encode(text))


def chunk_text(
    text: str,
    max_tokens: int = MAX_TOKENS_PER_CHUNK,
    model: str = OPENAI_MODEL,
) -> list[str]:
    """
    Split text into chunks that each fit within `max_tokens`.

    Strategy:
        1. If the text fits in one chunk, return it as-is.
        2. Otherwise, split on paragraph boundaries first.
        3. If individual paragraphs exceed the limit, split on sentence boundaries.
        4. As a last resort, split on word boundaries.

    Args:
        text: The text to chunk.
        max_tokens: Maximum tokens per chunk.
        model: Model name for tokenizer selection.

    Returns:
        List of text chunks, each within the token limit.
    """
    if not text:
        return []

    total_tokens = count_tokens(text, model)
    if total_tokens <= max_tokens:
        return [text]

    logger.info(
        f"Text has {total_tokens} tokens, splitting into chunks of ≤{max_tokens}"
    )

    # Split into paragraphs first
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_chunk = ""

    for para in paragraphs:
        para_tokens = count_tokens(para, model)

        # If a single paragraph exceeds the limit, split it further
        if para_tokens > max_tokens:
            # Flush current chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Split the big paragraph on sentences
            chunks.extend(_split_paragraph(para, max_tokens, model))
            continue

        # Check if adding this paragraph would exceed the limit
        candidate = f"{current_chunk}\n\n{para}" if current_chunk else para
        if count_tokens(candidate, model) <= max_tokens:
            current_chunk = candidate
        else:
            # Flush and start a new chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks


def _split_paragraph(
    paragraph: str, max_tokens: int, model: str
) -> list[str]:
    """Split a single long paragraph into sentence-level chunks."""
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence, model)

        # If a single sentence exceeds the limit, split on words
        if sentence_tokens > max_tokens:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_by_words(sentence, max_tokens, model))
            continue

        candidate = f"{current} {sentence}" if current else sentence
        if count_tokens(candidate, model) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


def _split_by_words(text: str, max_tokens: int, model: str) -> list[str]:
    """Last-resort splitting: break on word boundaries."""
    words = text.split()
    chunks: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}" if current else word
        if count_tokens(candidate, model) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = word

    if current:
        chunks.append(current.strip())

    return chunks


# ─── Pipeline ────────────────────────────────────────────────────────────────────


def preprocess_reviews(reviews: list[dict]) -> list[dict]:
    """
    Run the full preprocessing pipeline on a list of review dicts.

    Each review dict is expected to have a 'review_text' key.
    This function:
        1. Cleans the text
        2. Counts tokens
        3. Chunks if needed
        4. Adds 'cleaned_text', 'token_count', and 'chunks' keys

    Args:
        reviews: List of raw review dicts from the scraper.

    Returns:
        List of review dicts enriched with preprocessing data.
    """
    processed = []
    for review in reviews:
        review = review.copy()  # Don't mutate the original

        raw_text = review.get("review_text", "")
        cleaned = clean_text(raw_text)
        tokens = count_tokens(cleaned)
        chunks = chunk_text(cleaned)

        review["cleaned_text"] = cleaned
        review["token_count"] = tokens
        review["chunks"] = chunks

        # Also clean the title
        review["title"] = clean_text(review.get("title", ""))

        processed.append(review)

    logger.info(f"Preprocessed {len(processed)} reviews")
    return processed
