"""
Web scraper for extracting product reviews from Amazon product pages.

Supports:
- Live scraping with rotating User-Agents and retry logic
- Selenium headless browser fallback for CAPTCHA bypass
- Loading from local HTML files for testing
- Pagination across multiple review pages
- robots.txt compliance checking
"""

import logging
import random
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

try:
    from playwright.sync_api import sync_playwright
    BROWSER_AVAILABLE = True
except ImportError:
    BROWSER_AVAILABLE = False

from config import (
    MAX_PAGES,
    MAX_RETRIES,
    REQUEST_DELAY_MAX,
    REQUEST_DELAY_MIN,
    REQUEST_TIMEOUT,
    USER_AGENTS,
)

logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────────


def _get_random_headers() -> dict:
    """Return HTTP headers with a randomly selected User-Agent."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }


def _polite_delay() -> None:
    """Sleep for a random duration between configured min and max seconds."""
    delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
    logger.debug(f"Sleeping for {delay:.1f}s before next request")
    time.sleep(delay)


# ─── Playwright Browser Helpers ──────────────────────────────────────────────────


def _fetch_page_browser(url: str, page) -> str | None:
    """Fetch a page using Playwright browser. Bypasses most CAPTCHAs."""
    try:
        logger.info(f"Browser: fetching {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        # Human-like delay to let page render
        time.sleep(random.uniform(3, 5))

        page_source = page.content()

        if "captcha" in page_source.lower():
            logger.warning("CAPTCHA detected even in browser mode")
            return None

        return page_source
    except Exception as e:
        logger.error(f"Browser fetch error: {e}")
        return None


def _fetch_page(url: str, session: requests.Session) -> str | None:
    """
    Fetch a single URL with retries and random delays.
    Returns the page HTML or None on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Fetching {url} (attempt {attempt}/{MAX_RETRIES})")
            response = session.get(
                url,
                headers=_get_random_headers(),
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            # Amazon sometimes returns a CAPTCHA page
            if "captcha" in response.text.lower() or len(response.text) < 1000:
                logger.warning(
                    f"Possible CAPTCHA or empty page on attempt {attempt}"
                )
                if attempt < MAX_RETRIES:
                    _polite_delay()
                    continue
                return None

            return response.text

        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP error on attempt {attempt}: {e}")
            if response.status_code == 503:
                logger.info("Service unavailable — retrying after delay")
                time.sleep(5 * attempt)  # longer backoff for 503
            elif response.status_code == 404:
                logger.error("Page not found (404). Aborting.")
                return None
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error on attempt {attempt}: {e}")
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Unexpected request error: {e}")
            return None

        if attempt < MAX_RETRIES:
            _polite_delay()

    logger.error(f"Failed to fetch {url} after {MAX_RETRIES} attempts")
    return None


# ─── robots.txt ──────────────────────────────────────────────────────────────────


def check_robots_txt(url: str) -> bool:
    """Check if scraping the given URL is allowed by robots.txt."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch("*", url)
        if not allowed:
            logger.warning(f"robots.txt disallows fetching {url}")
        return allowed
    except Exception as e:
        logger.warning(f"Could not check robots.txt: {e}. Proceeding anyway.")
        return True  # Proceed if robots.txt is unavailable


# ─── Amazon Review Parsing ───────────────────────────────────────────────────────


def _build_reviews_url(product_url: str, page: int = 1) -> str:
    """
    Convert an Amazon product URL to its all-reviews page URL.
    Example:
      https://www.amazon.com/dp/B09V3KXJPB → …/product-reviews/B09V3KXJPB?pageNumber=1
    """
    parsed = urlparse(product_url)

    # Extract ASIN from various Amazon URL formats
    path_parts: list[str] = parsed.path.strip("/").split("/")
    asin: str | None = None
    for i, part in enumerate(path_parts):
        if part in ("dp", "gp", "product-reviews") and i + 1 < len(path_parts):
            asin = path_parts[i + 1]
            break

    if not asin:
        # Fallback: last path segment might be the ASIN
        asin = path_parts[-1] if path_parts else None

    if not asin:
        raise ValueError(f"Could not extract product ASIN from URL: {product_url}")

    base = f"{parsed.scheme}://{parsed.netloc}"
    return f"{base}/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber={page}&sortBy=recent"


def _parse_amazon_reviews(html: str) -> list[dict]:
    """
    Parse review data from an Amazon reviews HTML page.
    Returns a list of dicts with keys: title, author, date, rating, review_text.
    """
    soup = BeautifulSoup(html, "lxml")
    reviews = []

    review_divs = soup.select('div[data-hook="review"]')

    if not review_divs:
        # Fallback selectors for different Amazon layouts
        review_divs = soup.select(".review")

    for div in review_divs:
        review = {}

        # ── Title ────────────────────────────────────────────────────────
        title_el = div.select_one('a[data-hook="review-title"] span')
        if not title_el:
            title_el = div.select_one('a[data-hook="review-title"]')
        if not title_el:
            title_el = div.select_one(".review-title span")
        review["title"] = title_el.get_text(strip=True) if title_el else ""

        # ── Author ───────────────────────────────────────────────────────
        author_el = div.select_one('span.a-profile-name')
        review["author"] = author_el.get_text(strip=True) if author_el else "Anonymous"

        # ── Date ─────────────────────────────────────────────────────────
        date_el = div.select_one('span[data-hook="review-date"]')
        review["date"] = date_el.get_text(strip=True) if date_el else ""

        # ── Rating ───────────────────────────────────────────────────────
        rating_el = div.select_one('i[data-hook="review-star-rating"] span')
        if not rating_el:
            rating_el = div.select_one('i[data-hook="cmps-review-star-rating"] span')
        if rating_el:
            rating_text = rating_el.get_text(strip=True)
            try:
                review["rating"] = float(rating_text.split(" ")[0])
            except (ValueError, IndexError):
                review["rating"] = None
        else:
            review["rating"] = None

        # ── Review body ──────────────────────────────────────────────────
        body_el = div.select_one('span[data-hook="review-body"] span')
        if not body_el:
            body_el = div.select_one('span[data-hook="review-body"]')
        if not body_el:
            body_el = div.select_one(".review-text-content span")
        review["review_text"] = body_el.get_text(strip=True) if body_el else ""

        # Only include reviews that have actual text content
        if review["review_text"]:
            reviews.append(review)

    return reviews


def _has_next_page(html: str) -> bool:
    """Check if there is a next page of reviews."""
    soup = BeautifulSoup(html, "lxml")
    next_btn = soup.select_one("li.a-last:not(.a-disabled)")
    return next_btn is not None


# ─── Public API ──────────────────────────────────────────────────────────────────


def scrape_reviews_from_html(html: str) -> list[dict]:
    """
    Parse reviews from raw HTML string.
    Useful for testing with local HTML files.
    """
    logger.info("Parsing reviews from provided HTML")
    reviews = _parse_amazon_reviews(html)
    logger.info(f"Found {len(reviews)} reviews in HTML")
    return reviews


def scrape_reviews(url: str, max_pages: int | None = None) -> list[dict]:
    """
    Scrape reviews from an Amazon product page URL.

    First tries with requests. If Amazon returns CAPTCHA, automatically
    falls back to Playwright headless browser.

    Args:
        url: Amazon product page or reviews page URL.
        max_pages: Maximum number of review pages to fetch. Defaults to config.MAX_PAGES.

    Returns:
        List of review dicts with keys: title, author, date, rating, review_text.
    """
    _max_pages: int = max_pages if max_pages is not None else MAX_PAGES

    # Check robots.txt
    check_robots_txt(url)

    all_reviews: list[dict] = []
    session = requests.Session()
    use_browser = False

    if _max_pages > 0:
        page_limit = _max_pages + 1
    else:
        page_limit = 10000

    playwright = None
    browser = None
    page = None

    try:
        for page_num in range(1, page_limit):
            reviews_url = _build_reviews_url(url, page=page_num)
            logger.info(f"Scraping review page {page_num}: {reviews_url}")

            html = None

            if not use_browser:
                html = _fetch_page(reviews_url, session)
                if html is None and BROWSER_AVAILABLE:
                    logger.info("Requests failed — switching to browser mode")
                    use_browser = True

            # Use browser if requests failed or we already switched
            if use_browser:
                if BROWSER_AVAILABLE and playwright is None:
                    logger.info("Starting headless Chromium browser via Playwright...")
                    playwright = sync_playwright().start()
                    browser = playwright.chromium.launch(headless=True)
                    context = browser.new_context(
                        user_agent=random.choice(USER_AGENTS),
                        viewport={"width": 1920, "height": 1080}
                    )
                    page = context.new_page()
                
                if page:
                    html = _fetch_page_browser(reviews_url, page)

            if html is None:
                logger.warning(f"Could not fetch page {page_num}. Stopping pagination.")
                break

            page_reviews = _parse_amazon_reviews(html)

            # If requests succeeded but found 0 reviews, it might be a silent block page (e.g. sign in wall).
            if not page_reviews and not use_browser and BROWSER_AVAILABLE:
                logger.info("Requests fetched a page but found 0 reviews (possible silent anti-bot). Switching to browser mode.")
                use_browser = True
                
                if playwright is None:
                    logger.info("Starting headless Chromium browser via Playwright...")
                    playwright = sync_playwright().start()
                    browser = playwright.chromium.launch(headless=True)
                    context = browser.new_context(
                        user_agent=random.choice(USER_AGENTS),
                        viewport={"width": 1920, "height": 1080}
                    )
                    page = context.new_page()
                
                if page:
                    html = _fetch_page_browser(reviews_url, page)
                    if html:
                        page_reviews = _parse_amazon_reviews(html)

            if not page_reviews:
                logger.info(f"No reviews found on page {page_num}. Reached the end.")
                break

            all_reviews.extend(page_reviews)
            logger.info(
                f"Page {page_num}: found {len(page_reviews)} reviews "
                f"(total: {len(all_reviews)})"
            )

            if not _has_next_page(html):
                logger.info("No next page found. Pagination complete.")
                break

            _polite_delay()

    finally:
        if browser:
            browser.close()
        if playwright:
            playwright.stop()

    logger.info(f"Scraping complete. Total reviews collected: {len(all_reviews)}")
    return all_reviews


def scrape_reviews_from_file(filepath: str) -> list[dict]:
    """
    Load and parse reviews from a local HTML file.

    Args:
        filepath: Path to the HTML file.

    Returns:
        List of review dicts.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {filepath}")

    html = path.read_text(encoding="utf-8")
    return scrape_reviews_from_html(html)
