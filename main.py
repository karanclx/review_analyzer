#!/usr/bin/env python3
"""
Product Review Scraper & LLM Sentiment Analyzer

CLI entry point that orchestrates:
  scrape → preprocess → analyze → store

Usage:
  python main.py <product_url> [options]
  python main.py --local-html sample.html [options]
"""

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from config import DEFAULT_OUTPUT_DIR, OPENAI_API_KEY, OPENAI_MODEL
from scraper import scrape_reviews, scrape_reviews_from_file
from preprocessor import preprocess_reviews
from llm_analyzer import analyze_reviews_batch
from storage import save_to_json, save_to_csv, display_summary


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-8s │ %(name)-18s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Scrape product reviews and analyze sentiment using an LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape Amazon reviews and analyze
  python main.py "https://www.amazon.com/dp/B09V3KXJPB"

  # Use a local HTML file for testing
  python main.py --local-html reviews.html

  # Specify output format and model
  python main.py "https://www.amazon.com/dp/B09V3KXJPB" --format both --model gpt-4

  # Limit to 3 pages of reviews
  python main.py "https://www.amazon.com/dp/B09V3KXJPB" --max-pages 3
        """,
    )

    # Input source (URL or local file)
    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="Product page URL (e.g., Amazon product page)",
    )
    parser.add_argument(
        "--local-html",
        type=str,
        default=None,
        help="Path to a local HTML file containing reviews (for testing)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="both",
        help="Output format (default: both)",
    )

    # LLM options
    parser.add_argument(
        "--model",
        type=str,
        default=OPENAI_MODEL,
        help=f"LLM model to use (default: {OPENAI_MODEL})",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip LLM analysis (only scrape and preprocess)",
    )

    # Scraper options
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of review pages to scrape",
    )

    # General options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    args = parser.parse_args()

    # Validation
    if not args.url and not args.local_html:
        parser.error("You must provide either a product URL or --local-html <file>")

    return args


def main() -> int:
    """Main application entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║     🛒  Product Review Scraper & LLM Analyzer  🤖         ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # ── Step 1: Scrape reviews ───────────────────────────────────────────────
    print("📥 Step 1: Scraping reviews...")

    try:
        if args.local_html:
            print(f"   Loading from local file: {args.local_html}")
            reviews = scrape_reviews_from_file(args.local_html)
        else:
            print(f"   URL: {args.url}")
            reviews = scrape_reviews(args.url, max_pages=args.max_pages)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Scraping failed: {e}", exc_info=True)
        print(f"\n❌ Scraping failed: {e}")
        return 1

    if not reviews:
        print("\n⚠️  No reviews found. This could be due to:")
        print("   • The page structure has changed")
        print("   • Anti-bot protection blocked the request")
        print("   • The product has no reviews")
        print("   Try using --local-html with a saved HTML file for testing.")
        return 1

    print(f"   ✅ Found {len(reviews)} reviews\n")

    # ── Step 2: Preprocess ───────────────────────────────────────────────────
    print("🧹 Step 2: Preprocessing review text...")

    reviews = preprocess_reviews(reviews)
    total_tokens = sum(r.get("token_count", 0) for r in reviews)
    print(f"   ✅ Preprocessed {len(reviews)} reviews ({total_tokens:,} total tokens)\n")

    # ── Step 3: LLM Analysis ────────────────────────────────────────────────
    if args.skip_analysis:
        print("⏭️  Step 3: Skipping LLM analysis (--skip-analysis flag)\n")
    else:
        if not OPENAI_API_KEY:
            print("⚠️  OPENAI_API_KEY not set. Skipping LLM analysis.")
            print("   Set it via: export OPENAI_API_KEY='your-key-here'\n")
        else:
            print(f"🤖 Step 3: Analyzing reviews with {args.model}...")

            # Create a tqdm progress bar
            pbar = tqdm(total=len(reviews), desc="   Analyzing", unit="review")

            def progress_callback(current, total):
                pbar.update(1)

            try:
                reviews = analyze_reviews_batch(
                    reviews,
                    model=args.model,
                    progress_callback=progress_callback,
                )
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}", exc_info=True)
                print(f"\n⚠️  LLM analysis error: {e}")
                print("   Saving scraped data without analysis.\n")
            finally:
                pbar.close()

            print(f"   ✅ Analysis complete\n")

    # ── Step 4: Save results ────────────────────────────────────────────────
    print("💾 Step 4: Saving results...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ("json", "both"):
        json_path = save_to_json(reviews, str(output_dir / "reviews.json"))
        print(f"   📄 JSON: {json_path}")

    if args.format in ("csv", "both"):
        csv_path = save_to_csv(reviews, str(output_dir / "reviews.csv"))
        print(f"   📊 CSV:  {csv_path}")

    print()

    # ── Step 5: Display summary ─────────────────────────────────────────────
    display_summary(reviews)

    print("✅ Done! All results saved to:", str(output_dir.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
