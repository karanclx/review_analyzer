"""
Central configuration for the Product Review Scraper & Analyzer.
All settings are overridable via environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── LLM Configuration ─────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)  # None = use default
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Maximum tokens per chunk when splitting long reviews for the LLM
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "3000"))

# ─── Scraper Configuration ──────────────────────────────────────────────────────

# Rotating User-Agents to reduce blocking
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
    "Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

# Delay range (seconds) between requests to avoid rate limiting
REQUEST_DELAY_MIN = float(os.getenv("REQUEST_DELAY_MIN", "1.0"))
REQUEST_DELAY_MAX = float(os.getenv("REQUEST_DELAY_MAX", "3.0"))

# Maximum number of review pages to scrape (0 = unlimited)
MAX_PAGES = int(os.getenv("MAX_PAGES", "10"))

# HTTP request timeout in seconds
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))

# Maximum retry attempts for failed HTTP requests
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# ─── Rate Limit Configuration ───────────────────────────────────────────────────

# Delay between LLM API calls (seconds) to respect rate limits
API_CALL_DELAY = float(os.getenv("API_CALL_DELAY", "1.0"))

# Maximum retries for LLM API calls (with exponential backoff)
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "5"))

# ─── Output Configuration ───────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
