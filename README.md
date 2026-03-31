# 🛒 Product Review Scraper & LLM Sentiment Analyzer

A robust Python CLI application that scrapes product reviews from Amazon, preprocesses the text, and uses an OpenAI-compatible LLM to generate sentiment analysis and summaries for each review.

##  Chosen Product URL for Testing

```
https://www.amazon.com/dp/B09V3KXJPB
```

*(Apple AirPods Pro 2nd Generation — a popular product with many reviews)*

> **Note**: Amazon actively blocks automated scrapers. If live scraping is blocked, use the `--local-html` flag with a saved HTML file (see [Testing with Local HTML](#-testing-with-local-html)).

---

##  Quick Start

### 1. Clone & install dependencies

```bash
cd scraper
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

Or export directly:

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### 3. Run the scraper

```bash
# Scrape reviews from an Amazon product page
python main.py "https://www.amazon.com/dp/B09V3KXJPB"

# Or use a local HTML file for testing
python main.py --local-html tests/fixtures/sample_reviews.html
```

---

## 📖 Usage

```
python main.py <product_url> [options]
```

| Option | Description | Default |
|---|---|---|
| `url` | Amazon product page URL | — |
| `--local-html FILE` | Load reviews from a local HTML file | — |
| `--output-dir DIR` | Output directory | `output/` |
| `--format {json,csv,both}` | Output format | `both` |
| `--model MODEL` | LLM model name | `gpt-3.5-turbo` |
| `--max-pages N` | Max review pages to scrape | `10` |
| `--skip-analysis` | Skip LLM step (scrape only) | `false` |
| `-v, --verbose` | Debug logging | `false` |

### Examples

```bash
# Scrape and analyze with GPT-4
python main.py "https://www.amazon.com/dp/B09V3KXJPB" --model gpt-4

# Only scrape, skip LLM analysis
python main.py "https://www.amazon.com/dp/B09V3KXJPB" --skip-analysis

# Limit to 3 pages, output JSON only
python main.py "https://www.amazon.com/dp/B09V3KXJPB" --max-pages 3 --format json

# Use a local Ollama model
export OPENAI_API_BASE="http://localhost:11434/v1"
export OPENAI_MODEL="llama3"
python main.py --local-html reviews.html
```

---

##  Testing with Local HTML

Since Amazon aggressively blocks automated scrapers, you can test with saved HTML:

1. Open an Amazon product's reviews page in your browser
2. Right-click → **Save As** → save as `.html`
3. Run:
   ```bash
   python main.py --local-html saved_reviews.html
   ```

---

##  Architecture

```
scraper/
├── main.py              # CLI entry point & pipeline orchestration
├── scraper.py           # Web scraping with retries & User-Agent rotation
├── preprocessor.py      # Text cleaning & token-aware chunking
├── llm_analyzer.py      # OpenAI API integration with rate limiting
├── storage.py           # JSON/CSV output & console summary
├── config.py            # Centralized configuration (env vars)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── output/              # Generated output (gitignored)
    ├── reviews.json
    └── reviews.csv
```

### Pipeline Flow

```
Product URL → Scraper → Raw Reviews → Preprocessor → Cleaned/Chunked Text
                                                           ↓
Console Summary ← Storage ← Enriched Reviews ← LLM Analyzer
```

---

##  Design Choices

### Scraping Strategy
- **Rotating User-Agents**: 5 different browser signatures to reduce blocking
- **Random delays**: Configurable 1–3s delay between requests
- **Retry logic**: Up to 3 retries per page with CAPTCHA detection
- **Pagination**: Follows review pages until no more are found (configurable max)
- **robots.txt compliance**: Checks `robots.txt` before scraping

### Text Preprocessing
- **HTML entity decoding**: Handles `&amp;`, `&lt;`, etc.
- **Unicode normalization**: NFKC normalization, smart quote → ASCII conversion
- **Token-aware chunking**: Uses `tiktoken` to split long reviews into LLM-safe chunks, splitting at paragraph → sentence → word boundaries

### LLM Integration
- **Structured JSON prompting**: Asks the LLM to return `{sentiment, summary, key_points}`
- **Exponential backoff**: Via `tenacity` for rate limit errors (up to 5 retries)
- **Provider flexibility**: Works with OpenAI, Azure, Ollama, LM Studio, or any OpenAI-compatible API
- **Chunked review merging**: For very long reviews, analyzes chunks independently and merges with majority-vote sentiment

### Error Handling
- Network errors: Retry with backoff, graceful degradation
- API rate limits: Exponential backoff via `tenacity`
- Invalid HTML: Fallback selectors for different Amazon layouts
- Missing API key: Skips analysis, saves scraped data only

---

##  Output Format

### JSON (`output/reviews.json`)
```json
[
  {
    "title": "Amazing sound quality!",
    "author": "John D.",
    "date": "Reviewed in the United States on January 15, 2024",
    "rating": 5.0,
    "review_text": "These are the best earbuds I've ever owned...",
    "cleaned_text": "These are the best earbuds I've ever owned...",
    "token_count": 85,
    "chunks": ["These are the best earbuds I've ever owned..."],
    "sentiment": "POSITIVE",
    "summary": "Highly impressed with the sound quality and noise cancellation.",
    "key_points": [
      "Excellent sound quality",
      "Effective noise cancellation",
      "Comfortable fit for extended use"
    ]
  }
]
```

### CSV (`output/reviews.csv`)
Flat table with all fields. Nested fields (`chunks`, `key_points`) are serialized as JSON strings.

### Console Summary
Includes a formatted table and sentiment distribution chart.

---

## Limitations

1. **Amazon anti-bot protection**: Amazon may block requests or serve CAPTCHAs. Use `--local-html` as a reliable fallback.
2. **Review format changes**: If Amazon changes their HTML structure, the CSS selectors in `scraper.py` may need updating.
3. **Rate limits**: The LLM API calls are rate-limited to ~1 request/second by default. Adjust `API_CALL_DELAY` in `.env` for faster processing.
4. **Single site**: Currently supports Amazon only. The modular architecture makes adding new sites straightforward.
5. **No JavaScript rendering**: Uses `requests` (no browser). JavaScript-rendered reviews won't be captured.

---

##  License

MIT
