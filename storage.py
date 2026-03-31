"""
Data storage module for persisting scraped and analyzed review data.

Supports:
- JSON output (full structured data including nested fields)
- CSV output via Pandas (flattened for tabular consumption)
- Console summary display
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_dir(filepath: str) -> None:
    """Create parent directories for the given filepath if they don't exist."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def save_to_json(data: list[dict], filepath: str) -> str:
    """
    Save review data to a JSON file.

    Args:
        data: List of review dicts (may contain nested fields like 'chunks', 'key_points').
        filepath: Output file path.

    Returns:
        Absolute path to the saved file.
    """
    _ensure_dir(filepath)

    # Serialize, converting any non-serializable types
    def _default(obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_default)

    abs_path = str(Path(filepath).resolve())
    logger.info(f"Saved {len(data)} reviews to JSON: {abs_path}")
    return abs_path


def save_to_csv(data: list[dict], filepath: str) -> str:
    """
    Save review data to a CSV file.

    Nested fields (chunks, key_points) are serialized to JSON strings
    so the CSV remains a flat table.

    Args:
        data: List of review dicts.
        filepath: Output file path.

    Returns:
        Absolute path to the saved file.
    """
    _ensure_dir(filepath)

    # Flatten nested fields for CSV
    flat_data = []
    for review in data:
        row = {}
        for key, value in review.items():
            if isinstance(value, (list, dict)):
                row[key] = json.dumps(value, ensure_ascii=False)
            else:
                row[key] = value
        flat_data.append(row)

    df = pd.DataFrame(flat_data)

    # Reorder columns for readability
    preferred_order = [
        "title", "author", "date", "rating",
        "review_text", "cleaned_text", "token_count",
        "sentiment", "summary", "key_points",
        "chunks",
    ]
    existing_cols = [c for c in preferred_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in preferred_order]
    df = df[existing_cols + remaining_cols]

    df.to_csv(filepath, index=False, encoding="utf-8")

    abs_path = str(Path(filepath).resolve())
    logger.info(f"Saved {len(data)} reviews to CSV: {abs_path}")
    return abs_path


def display_summary(data: list[dict]) -> None:
    """
    Print a formatted summary table of the analyzed reviews to the console.

    Shows: title (truncated), rating, sentiment, and summary (truncated).
    """
    if not data:
        print("\nNo reviews to display.")
        return

    summary_rows = []
    for review in data:
        summary_rows.append({
            "Title": (review.get("title", "")[:40] + "...") if len(review.get("title", "")) > 40 else review.get("title", ""),
            "Rating": review.get("rating", "N/A"),
            "Sentiment": review.get("sentiment", "N/A"),
            "Summary": (review.get("summary", "")[:80] + "...") if len(review.get("summary", "")) > 80 else review.get("summary", ""),
        })

    df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 100)
    print("  REVIEW ANALYSIS SUMMARY")
    print("=" * 100)
    print(df.to_string(index=True))
    print("=" * 100)

    # Sentiment distribution
    sentiments = [r.get("sentiment", "UNKNOWN") for r in data]
    print("\nSentiment Distribution:")
    for sentiment in sorted(set(sentiments)):
        count = sentiments.count(sentiment)
        pct = count / len(sentiments) * 100
        bar = "#" * int(pct / 5)
        print(f"  {sentiment:10s} : {count:3d} ({pct:5.1f}%) {bar}")

    # Average rating
    ratings = [r["rating"] for r in data if r.get("rating") is not None]
    if ratings:
        avg = sum(ratings) / len(ratings)
        print(f"\nAverage Rating: {avg:.1f}/5 (from {len(ratings)} rated reviews)")

    print()
