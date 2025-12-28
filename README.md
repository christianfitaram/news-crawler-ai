# NewsFeeder-IA: Automated News Article Ingestion & Analysis Pipeline

A comprehensive, multi-source news aggregation and analysis system that automatically scrapes articles from 10+ global news outlets, deduplicates them, cleans the text, extracts entities, generates summaries, and classifies articles by sentiment and topic using state-of-the-art transformer models. All data is persisted in MongoDB with detailed metadata and operational insights.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Architecture](#project-architecture)
4. [Tech Stack](#tech-stack)
5. [Prerequisites](#prerequisites)
6. [Installation & Setup](#installation--setup)
7. [Environment Configuration](#environment-configuration)
8. [Model Preparation](#model-preparation)
9. [Running the Pipeline](#running-the-pipeline)
10. [Directory Structure](#directory-structure)
11. [Data Models & MongoDB Collections](#data-models--mongodb-collections)
12. [Scripts & Utilities](#scripts--utilities)
13. [Advanced: Webhooks & Integrations](#advanced-webhooks--integrations)
14. [Troubleshooting](#troubleshooting)
15. [Development & Contributing](#development--contributing)

---

## Overview

**NewsFeeder-IA** is an intelligent news aggregation platform that:

- **Ingests** articles from 10+ sources (BBC, CNN, WSJ, Al Jazeera, DW, The Guardian, Reuters, France24, NPR, and NewsAPI)
- **Deduplicates** URLs to avoid reprocessing
- **Cleans** article text using Google Gemini API (removes boilerplate, ads, navigation)
- **Extracts** named entities (persons, organizations, locations) via Gemini with structured JSON
- **Summarizes** long-form articles using Facebook BART abstractive models
- **Classifies** sentiment (DistilBERT SST-2) and topics (12 predefined categories via Zero-Shot classification with BART-MNLI)
- **Stores** enriched articles in MongoDB with metadata, statistics, and audit trails
- **Sends webhooks** to downstream consumers for embedding, vectorization, and real-time alerts

All transformers run on **CPU by default** (CUDA/MPS support for GPU acceleration), with **offline-cached models** that can work without internet connectivity after bootstrap.

---

## Key Features

### 1. **Hybrid Multi-Source Ingestion**
- **Custom scrapers** for BBC, CNN, WSJ, Al Jazeera, DW, The Guardian, Reuters using RSS feeds, HTML parsing, and Selenium
- **Selenium-powered dynamic scrapers** for JavaScript-heavy sites (The Guardian, France24, NPR)
- **NewsAPI integration** for 100+ international news sources
- **Automatic deduplication** via URL hash in `link_pool`

### 2. **Text Processing Pipeline**
- **Trafilatura** for robust HTML-to-text extraction
- **Google Gemini API** for deterministic text cleaning (removal of ads, bylines, cookies, layout artifacts)
- **BART-large-CNN** abstractive summarization (intelligent truncation for long articles)
- **Fallback mechanisms** when APIs are unavailable

### 3. **NLP Classification**
- **Sentiment Analysis**: DistilBERT fine-tuned on SST-2 (positive/negative/neutral)
- **Topic Classification**: Zero-shot classification with BART-MNLI across 12 predefined topics:
  - Politics & Government
  - Sports & Athletics
  - Science & Research
  - Technology & Innovation
  - Health & Medicine
  - Business & Finance
  - Entertainment & Celebrity
  - Crime & Justice
  - Climate & Environment
  - Education & Schools
  - War & Conflict
  - Travel & Tourism

### 4. **Entity Extraction**
- Named Entity Recognition via Gemini API with strict verbatim matching
- Extracts: **Persons**, **Organizations**, **Locations**
- Deduplicates by normalized key to avoid near-duplicates
- Fallback to empty arrays if entities are absent

### 5. **Metadata & Auditing**
- Per-batch statistics: article counts, topic/sentiment distributions, processing times
- Global metadata tracking total articles, topic trends
- Sample tracking: group articles by UUID batch identifier
- Source attribution and processing timestamps

### 6. **Webhook Integration**
- POST webhooks to downstream microservices for each classified article
- **Embedding webhook**: full article + metadata for vectorization
- **Thread-events webhook**: lightweight notifications for real-time aggregators
- Automatic retry with exponential backoff (3 retries by default)
- Signature-based authentication (HMAC-compatible)

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      NEWS SOURCES                               │
│  BBC  CNN  WSJ  Al Jazeera  DW  Guardian  Reuters  NewsAPI...  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              INGESTION & DEDUPLICATION                           │
│  ingest/custom_scrapers.py + ingest/news_api_scrapper.py       │
│  - Fetches articles and extracts text with trafilatura         │
│  - Checks LinkPoolRepository for duplicate URLs                │
│  - Yields unique {title, url, text, source, scraped_at}       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│               TEXT CLEANING & ENTITY EXTRACTION                 │
│  ingest/call_to_genai.py                                        │
│  - Pass 1: Clean with Google Gemini (remove noise/ads)         │
│  - Pass 2: Extract entities (persons/orgs/locations)           │
│  - JSON-structured output with fallback to empty arrays        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│            NLP CLASSIFICATION & SUMMARIZATION                    │
│  ingest/classifier.py + ingest/summarizer.py                   │
│  - Summarize long articles with BART-large-CNN                 │
│  - Sentiment: DistilBERT SST-2 (positive/negative)            │
│  - Topic: Zero-shot with BART-MNLI (12 categories)             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                STORAGE & METADATA                                │
│  MongoDB + lib/repositories/*.py                                │
│  - articles: classified articles with embeddings metadata      │
│  - link_pool: processed URLs                                    │
│  - metadata: per-batch statistics                               │
│  - global_metadata: cumulative trends                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│               WEBHOOK NOTIFICATIONS                              │
│  ingest/call_to_webhook.py                                      │
│  - Embedding service (vectorization)                            │
│  - Thread-event service (aggregation)                           │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Scraping Phase** (`get_all_articles()`)
   - Iterate through all configured scrapers
   - Check URL against `link_pool` to avoid duplicates
   - Extract text with `trafilatura`
   - Yield unique articles

2. **Cleaning & Enrichment Phase**
   - Send raw text to Gemini API for cleaning
   - Extract named entities from cleaned text
   - Format as JSON with fallbacks

3. **Classification Phase**
   - Summarize if text > 200 tokens
   - Run through sentiment pipeline
   - Run through zero-shot topic pipeline
   - Store result in MongoDB

4. **Metadata & Integration Phase**
   - Update per-batch metadata (counts, distributions)
   - Increment global topic counters
   - Send webhook to downstream services
   - Mark URL as processed in `link_pool`

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Language** | Python | 3.10+ |
| **Database** | MongoDB | 4.x+ |
| **Web Scraping** | Trafilatura, BeautifulSoup4, Selenium, Feedparser | Latest |
| **NLP Transformers** | HuggingFace Transformers | ≥4.30, <4.37 |
| **Deep Learning** | PyTorch | 2.2.2 |
| **ML Models** | DistilBERT, BART-large-cnn, BART-large-mnli | Hugging Face Hub |
| **API Integration** | Google Gemini, NewsAPI | Latest SDKs |
| **ODM** | PyMongo | 4.15.0 |
| **CLI** | Typer, Rich, tqdm | Latest |
| **HTTP** | Requests, urllib3 | Latest |
| **Browser Automation** | Selenium + ChromeDriver | 4.27.1 + webdriver-manager |

**See `requirements.txt` for pinned versions.**

---

## Prerequisites

### System Requirements
- **Python**: 3.10 or higher
- **Disk Space**: ~4–5 GB (models + cache)
- **Memory**: 4 GB RAM minimum (8 GB recommended for smooth TF model inference)
- **OS**: macOS, Linux, or Windows (WSL recommended)

### External Services
- **MongoDB**: Local or cloud instance (e.g., MongoDB Atlas)
  - Read/write permissions required
  - Collections: `articles`, `link_pool`, `metadata`, `global_metadata`, `clean_articles`, `summaries`, `daily_trends`, `trend_threads`

- **Google Gemini API** (optional but recommended)
  - API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
  - Used for text cleaning and entity extraction

- **NewsAPI.org** (optional)
  - API key for accessing 100+ global news sources
  - [Get free key](https://newsapi.org/)

- **Web Browser** (for Selenium)
  - Google Chrome (recommended) or Firefox
  - Install via Homebrew on macOS: `brew install --cask google-chrome`

---

## Installation & Setup

### 1. Clone or Download the Repository

```bash
cd /Users/christianfita/Desktop/Projects/NewsFeeder-IA
```

### 2. Create a Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Upgrade pip and Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Optional**: Install development/debugging tools:
```bash
pip install ipython jupyter
```

### 4. Create `.env` File

Copy the template (if provided) or create `.env` in the project root:

```bash
touch .env
```

Then populate it with your configuration (see **Environment Configuration** below).

---

## Environment Configuration

Create a `.env` file in the project root with the following variables:

### **Required**

```env
# MongoDB connection
MONGO_URI=mongodb+srv://<user>:<password>@<host>/<database>?retryWrites=true&w=majority
MONGODB_DB=news_feeder_db

# Google Gemini API (for text cleaning & entity extraction)
GEMINI_API_KEY=your-gemini-api-key-here
GOOGLE_API_KEY=your-google-api-key-here  # Fallback if GEMINI_API_KEY not set
```

### **Optional (with Defaults)**

```env
# Application metadata
APP_NAME=trend-app

# Model caching
TRANSFORMERS_CACHE=/custom/path/to/models  # Default: ./models/transformers

# NewsAPI (if using scrape_newsapi_stream or scrape_all_categories)
NEWSAPI_KEY=your-newsapi-key-here

# Webhook integration
WEBHOOK_URL=http://localhost:8080/webhook/news
WEBHOOK_URL_THREAD_EVENTS=http://localhost:8000/webhooks/article-vectorized
WEBHOOK_SIGNATURE=your-secret-signature-key  # For HMAC validation
WEBHOOK_TIMEOUT=60  # seconds
NEWS_FETCH_TIMEOUT=20  # seconds

# Gemini API tuning (advanced)
GENAI_MODEL=gemini-2.0-flash
GENAI_TEMPERATURE=0.0
GENAI_TOP_P=1.0
GENAI_MAX_CHUNK_CHARS=12000

# Selenium / Browser automation
CHROME_BIN=/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome  # macOS
CHROMEDRIVER_PATH=/usr/local/bin/chromedriver
CHROMEDRIVER_VERSION=latest  # or specific version
CHROME_USER_DATA_DIR=/tmp/chrome-user-data

# General logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### Configuration Notes

- **MONGO_URI**: Must be a valid MongoDB connection string. For local MongoDB:
  ```
  MONGO_URI=mongodb://localhost:27017
  MONGODB_DB=news_feeder_db
  ```

- **GEMINI_API_KEY**: Required if you want text cleaning and entity extraction. Free tier available at [Google AI Studio](https://makersuite.google.com/app/apikey).

- **TRANSFORMERS_CACHE**: If not set, models default to `./models/transformers` (must be writable).

- **Selenium paths**: Only needed if using custom Chrome installations or driver paths. `webdriver-manager` handles auto-download by default.

**Verification**: After setting `.env`, run:
```bash
python -c "from lib.db.mongo_client import get_db; print(get_db().name)"
```
Should print your database name without errors.

---

## Model Preparation

### Automatic Download (Recommended)

Before running the classifier, download and cache all required models:

```bash
python scripts/bootstrap_models.py
```

This script downloads and caches:
- `distilbert-base-uncased-finetuned-sst-2-english` (sentiment analysis)
- `facebook/bart-large-cnn` (summarization)
- `facebook/bart-large-mnli` (zero-shot topic classification)

**Output**: Models stored in `./models/transformers/` (or custom `TRANSFORMERS_CACHE` path).

### Manual / Custom Cache

If you prefer a custom cache location:

```bash
export TRANSFORMERS_CACHE=/path/to/custom/cache
python scripts/bootstrap_models.py
```

### Offline Mode

After bootstrap, the pipeline can run **without internet** if all models are cached. Useful for air-gapped deployments.

### Model Details

| Model | Size | Purpose | Device Support |
|-------|------|---------|-----------------|
| distilbert-base-uncased-finetuned-sst-2-english | ~268 MB | Sentiment (pos/neg/neutral) | CPU, CUDA, MPS |
| facebook/bart-large-cnn | ~1.6 GB | Abstractive summarization | CPU, CUDA, MPS |
| facebook/bart-large-mnli | ~1.6 GB | Zero-shot classification | CPU, CUDA, MPS |
| **Total** | **~3.5 GB** | — | — |

---

## Running the Pipeline

### Main Classification Pipeline

The core workflow that ingests, cleans, classifies, and stores articles:

```bash
python -m ingest.classifier
```

**What happens:**
1. Generates a unique sample UUID
2. Calls `get_all_articles()` to fetch from all sources
3. For each article:
   - Cleans text with Gemini API
   - Extracts entities
   - Summarizes if > 200 tokens
   - Classifies sentiment & topic
   - Stores in MongoDB `articles` collection
4. Updates `link_pool` to mark URLs as processed
5. Computes and stores batch statistics in `metadata`
6. Sends webhook notifications to downstream consumers

**Typical runtime**: 2–10 minutes for ~50–200 articles (depends on article length and API response times).

**Output example**:
```
[1] ✅ BBC Report: Climate Summit Reaches Agreement
[2] ✅ CNN Breaking: Market Update on Tech Stocks
[3] ⏭️ Skipping static/boilerplate article: Legal Notice
...
Total documents in the database: 347
```

### Inspect MongoDB Data

To explore collected articles:

```bash
python -m outputs.main
```

Then uncomment/modify the function call at the end:

```python
# articles()  # List all articles
# access_metadata()  # View batch metadata
# get_links()  # View processed URLs
# countArticles()  # Count total articles
# articles_documents_grouped_by_source()  # Group by source
```

### Generate Representative Sample

Extract a proportional sample of articles from MongoDB (useful for evaluation):

```bash
python scripts/select_representative_sample.py
```

Output: `outputs/representative_sample.jsonl` (JSON Lines format, one article per line).

**Configuration** (in script):
- `TARGET_SAMPLE_SIZE = 150` — adjust as needed
- Allocates samples proportionally to each source

### Run from Cron / Scheduler

To run the pipeline automatically on a schedule, use `systemd` timer or cron:

#### **Linux / macOS Cron**

```bash
crontab -e
```

Add:
```
0 6 * * * cd /Users/christianfita/Desktop/Projects/NewsFeeder-IA && source .venv/bin/activate && python -m ingest.classifier >> /var/log/newsfeed.log 2>&1
```

(Runs at 6 AM daily)

#### **systemd Timer** (included)

On systemd systems, use the provided service file:

```bash
sudo cp scripts/systemd/news-crawler-ai.service /etc/systemd/system/
sudo cp scripts/systemd/news-crawler-ai.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable news-crawler-ai.timer
sudo systemctl start news-crawler-ai.timer
sudo systemctl status news-crawler-ai.timer
```

View logs:
```bash
journalctl -u news-crawler-ai.service -f
```

---

## Directory Structure

```
NewsFeeder-IA/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── main.py                            # Placeholder entry point
├── .env.example                       # Example environment file (create .env from this)
│
├── ingest/                            # Ingestion & classification pipeline
│   ├── __init__.py
│   ├── classifier.py                  # Main classification orchestrator
│   ├── get_all_articles.py            # Aggregates all scraper outputs
│   ├── custom_scrapers.py             # BBC, CNN, WSJ, Al Jazeera, DW, Guardian, Reuters, France24, NPR
│   ├── news_api_scrapper.py           # NewsAPI integration (streamers)
│   ├── summarizer.py                  # BART-large-cnn abstractive summarization
│   ├── sentiment_detector.py           # DistilBERT sentiment analysis (optional)
│   ├── topic_classifier.py             # BART-MNLI zero-shot classification (optional)
│   ├── call_to_genai.py               # Google Gemini API calls for cleaning & entity extraction
│   ├── call_to_webhook.py             # Webhook POST integrations
│   ├── call_to_ollama.py              # Ollama LLM integration (legacy)
│   ├── utils.py                       # fetch_and_extract, is_urls_processed_already
│   ├── crawler_dw.py                  # Deutsche Welle (DW) Selenium crawler
│   ├── test.py                        # Unit test placeholders
│   └── __pycache__/                   # Python bytecode cache
│
├── lib/                               # Data access layer
│   ├── __init__.py
│   ├── db/
│   │   ├── __init__.py
│   │   └── mongo_client.py            # MongoDB connection & client singleton
│   │
│   └── repositories/                  # Repository pattern (CRUD abstractions)
│       ├── __init__.py
│       ├── articles_repository.py     # CRUD for 'articles' collection
│       ├── link_pool_repository.py    # CRUD for 'link_pool' (URL deduplication)
│       ├── metadata_repository.py     # CRUD for 'metadata' (per-batch stats)
│       ├── global_metadata_repository.py # CRUD for 'global_metadata' (cumulative stats)
│       ├── clean_articles_repository.py  # CRUD for 'clean_articles'
│       ├── summaries_repository.py    # CRUD for 'summaries'
│       ├── daily_trends_repository.py # CRUD for 'daily_trends'
│       └── trend_threads_repository.py # CRUD for 'trend_threads'
│
├── models/                            # Model cache (created after bootstrap_models.py)
│   └── transformers/
│       ├── models--distilbert-base-uncased-finetuned-sst-2-english/
│       ├── models--facebook--bart-large-cnn/
│       └── models--facebook--bart-large-mnli/
│
├── outputs/                           # Inspection & debugging utilities
│   ├── __init__.py
│   ├── main.py                        # MongoDB data viewer functions
│   └── representative_sample.jsonl    # Generated sample output
│
├── scripts/                           # Utility & bootstrap scripts
│   ├── __init__.py
│   ├── bootstrap_models.py            # Download & cache all transformers models
│   ├── select_representative_sample.py # Extract proportional article sample
│   ├── fix_text.py                    # Text normalization utilities
│   ├── get_ids.py                     # ID lookup helpers
│   │
│   └── systemd/                       # Linux systemd service files
│       ├── news-crawler-ai.service    # Service definition
│       └── news-crawler-ai.timer      # Timer trigger
│
├── utils/                             # Shared utilities
│   ├── __init__.py
│   └── validation.py                  # Input validation helpers
│
└── docs/                              # Documentation
    ├── selenium.md                    # Selenium setup & troubleshooting
    ├── selenium-troubleshooting.md    # Common Selenium issues
    └── entity-extraction-troubleshooting.md # Entity extraction fixes
```

---

## Data Models & MongoDB Collections

### `articles`
**Purpose**: Stores classified articles with full metadata.

**Schema**:
```json
{
  "_id": ObjectId,
  "title": "Article Title",
  "url": "https://example.com/article",
  "text": "Cleaned article body...",
  "summary": "Article summary...",
  "source": "bbc-news",
  "sample": "uuid-batch-id",
  "scraped_at": ISODate("2025-01-28T12:34:56Z"),
  "topic": "politics and government",
  "sentiment": {
    "label": "POSITIVE",
    "score": 0.98
  },
  "locations": ["London", "United Kingdom"],
  "organizations": ["BBC", "Parliament"],
  "persons": ["Boris Johnson"],
  "isCleaned": false
}
```

**Indexes**:
- Compound: `(isCleaned, sample)`

### `link_pool`
**Purpose**: Tracks processed URLs to prevent duplicate ingestion.

**Schema**:
```json
{
  "_id": ObjectId,
  "url": "https://example.com/article",
  "is_articles_processed": true,
  "in_sample": "uuid-batch-id",
  "sample": "uuid-batch-id"
}
```

**Indexes**:
- Unique: `url`
- Regular: `is_articles_processed`

### `metadata`
**Purpose**: Per-batch statistics and processing logs.

**Schema**:
```json
{
  "_id": "uuid-batch-id",
  "gathering_sample_startedAt": ISODate("2025-01-28T12:00:00Z"),
  "gathering_sample_finishedAt": ISODate("2025-01-28T12:30:00Z"),
  "articles_processed": {
    "successfully": 145,
    "unsuccessfully": 3
  },
  "topic_distribution": [
    { "label": "politics and government", "percentage": 22.5 },
    { "label": "technology and innovation", "percentage": 18.3 },
    ...
  ],
  "sentiment_distribution": [
    { "label": "POSITIVE", "percentage": 55.2 },
    { "label": "NEGATIVE", "percentage": 25.1 },
    { "label": "NEUTRAL", "percentage": 19.7 }
  ]
}
```

### `global_metadata`
**Purpose**: Cumulative statistics across all batches.

**Schema**:
```json
{
  "_id": ObjectId("6923b800f3d19f7c28f53a6d"),
  "total_articles": 1234,
  "topics_data": [
    { "topic": "politics and government", "document_count": 275 },
    { "topic": "technology and innovation", "document_count": 225 },
    ...
  ]
}
```

### Other Collections (Optional)

- **`clean_articles`**: Deduped & cleaned versions of `articles`
- **`summaries`**: Grouped summaries by batch or thread
- **`daily_trends`**: Daily aggregated statistics
- **`trend_threads`**: Conversation threads of related articles

---

## Scripts & Utilities

### `scripts/bootstrap_models.py`

**Purpose**: Download and cache all required transformer models.

**Usage**:
```bash
python scripts/bootstrap_models.py
```

**Downloads**:
- DistilBERT SST-2 (sentiment)
- BART-large-CNN (summarization)
- BART-large-MNLI (zero-shot classification)

**Output**: Models cached in `./models/transformers/` (or custom path via `TRANSFORMERS_CACHE`).

---

### `scripts/select_representative_sample.py`

**Purpose**: Extract a stratified sample of articles from MongoDB for evaluation/review.

**Usage**:
```bash
python scripts/select_representative_sample.py
```

**Configuration** (modify in script):
```python
TARGET_SAMPLE_SIZE = 150  # Adjust sample size
OUTPUT_PATH = Path("outputs/representative_sample.jsonl")
```

**Strategy**:
- Groups articles by source
- Allocates sample slots proportionally to each source's share
- Uses MongoDB `$sample` aggregation for randomness
- Exports to JSONL format

**Output example**:
```json
{"_id": "...", "title": "Article 1", "source": "bbc-news", ...}
{"_id": "...", "title": "Article 2", "source": "cnn", ...}
...
```

---

### `scripts/fix_text.py`

**Purpose**: Text normalization helpers (whitespace, encoding, etc.).

**Usage**:
```bash
python scripts/fix_text.py <input_text>
```

---

### `scripts/get_ids.py`

**Purpose**: Lookup and extract MongoDB object IDs.

**Usage** (example):
```bash
python scripts/get_ids.py
```

---

### `outputs/main.py`

**Purpose**: Inspect and debug MongoDB data.

**Available functions**:

| Function | Description |
|----------|-------------|
| `articles()` | List all articles |
| `access_metadata()` | View batch metadata |
| `delete_metadata()` | Delete metadata by ID |
| `get_links()` | List all processed URLs |
| `countArticles()` | Count total articles |
| `articles_documents_grouped_by_source()` | Group articles by source |

**Usage**:
```python
# Edit end of outputs/main.py:
if __name__ == "__main__":
    articles()  # or any other function
```

Then run:
```bash
python -m outputs.main
```

---

## Advanced: Webhooks & Integrations

### Overview

After each article is classified, the system sends POST webhooks to downstream microservices for:
- **Vectorization/Embedding**: Full article for semantic search
- **Real-time aggregation**: Thread-event notifications

### Configuration

Set webhook endpoints in `.env`:

```env
WEBHOOK_URL=http://localhost:8080/webhook/news
WEBHOOK_URL_THREAD_EVENTS=http://localhost:8000/webhooks/article-vectorized
WEBHOOK_SIGNATURE=your-shared-secret-key
WEBHOOK_TIMEOUT=60
```

### Webhook Payloads

#### **Embedding Webhook** (`WEBHOOK_URL`)

**Method**: POST  
**Content-Type**: `application/json`

**Payload**:
```json
{
  "article_id": "507f1f77bcf86cd799439011",
  "url": "https://example.com/article",
  "title": "Article Title",
  "text": "Full cleaned text...",
  "topic": "technology and innovation",
  "source": "bbc-news",
  "sentiment": {
    "label": "POSITIVE",
    "score": 0.98
  },
  "scraped_at": "2025-01-28T12:34:56Z"
}
```

**Headers**:
```
Content-Type: application/json
X-Signature: <HMAC-SHA256(payload, WEBHOOK_SIGNATURE)>
```

#### **Thread-Events Webhook** (`WEBHOOK_URL_THREAD_EVENTS`)

**Method**: POST  
**Content-Type**: `application/json`

**Payload**:
```json
{
  "article_id": "507f1f77bcf86cd799439011",
  "source": "bbc-news",
  "scraped_at": "2025-01-28T12:34:56Z"
}
```

### Retry Logic

- **3 retries** with exponential backoff (0.5s, 1s, 2s)
- **Status codes**: 429, 500, 502, 503, 504 trigger retry
- **Timeout**: 60 seconds (configurable)
- **Errors**: Logged but don't block classifier

### Testing Webhooks Locally

Use `nc` (netcat) or `ngrok`:

```bash
# Terminal 1: Listen for POST
nc -l 8080

# Terminal 2: Trigger classifier (will attempt webhook)
python -m ingest.classifier
```

Or use a mock server:
```bash
python -m http.server 8080 --directory .
```

---

## Troubleshooting

### MongoDB Connection Issues

**Error**: `RuntimeError: Environment variable 'MONGO_URI' is not set or empty`

**Solution**:
1. Verify `.env` file exists in project root
2. Check `MONGO_URI` is not blank:
   ```bash
   grep MONGO_URI .env
   ```
3. Test connection:
   ```bash
   python -c "from lib.db.mongo_client import get_db; print(get_db().list_collection_names())"
   ```

---

### Model Download Failures

**Error**: `ConnectionError: HTTPError 403 Forbidden` or timeout

**Solution**:
1. Check internet connectivity
2. Verify Hugging Face API is accessible
3. Set custom cache path:
   ```bash
   export TRANSFORMERS_CACHE=/tmp/models
   python scripts/bootstrap_models.py
   ```
4. If offline, manually download models (see Hugging Face docs)

---

### Duplicate Articles in MongoDB

**Error**: Same URL appears multiple times in `articles` collection

**Solution**:
1. Check `link_pool` indexes:
   ```python
   from lib.repositories.link_pool_repository import LinkPoolRepository
   repo = LinkPoolRepository()
   repo.setup_indexes()
   ```
2. Verify `is_articles_processed` is set to `true` for all processed URLs
3. Clean old duplicates:
   ```bash
   python -c "
   from lib.repositories.articles_repository import ArticlesRepository
   from lib.repositories.link_pool_repository import LinkPoolRepository
   repo = ArticlesRepository()
   pool = LinkPoolRepository()
   for doc in repo.get_articles({}):
       if not pool.is_processed(doc['url']):
           repo.delete_articles({'_id': doc['_id']})
           print(f'Removed duplicate: {doc[\"url\"]}')
   "
   ```

---

### Entity Extraction Returns Empty Lists

**Error**: Articles stored with empty `locations`, `organizations`, `persons`

**Root cause**: Gemini API timeout or connection issue.

**Solution**:
1. Check `GEMINI_API_KEY` is set:
   ```bash
   grep GEMINI_API_KEY .env
   ```
2. Test API connectivity:
   ```bash
   python -c "
   from ingest.call_to_genai import call_to_genai_sdk
   result = call_to_genai_sdk('President Biden visited London yesterday.')
   print(result)
   "
   ```
3. If API unavailable, fallback returns `{"locations": [], "organizations": [], "persons": []}`
4. See `docs/entity-extraction-troubleshooting.md` for detailed fixes

---

### Sentiment/Topic Classification Fails

**Error**: `sentiment` or `topic` fields are `None` or missing

**Solution**:
1. Verify models cached:
   ```bash
   python scripts/bootstrap_models.py
   ```
2. Check GPU/CPU availability:
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
   ```
3. Check model load logs:
   ```bash
   python -c "from ingest.classifier import sentiment_pipeline; print(sentiment_pipeline)"
   ```

---

### Selenium Scraper Issues

**Error**: `chromedriver` not found or Chrome browser missing

**Solution**:
1. Install Chrome:
   ```bash
   brew install --cask google-chrome  # macOS
   sudo apt-get install google-chrome-stable  # Linux
   ```
2. Let `webdriver-manager` auto-install driver:
   ```bash
   # No action needed; it auto-installs
   ```
3. Or specify manual path:
   ```bash
   export CHROMEDRIVER_PATH=/usr/local/bin/chromedriver
   python -m ingest.classifier
   ```
4. See `docs/selenium.md` for detailed configuration

---

### Webhook POST Failures

**Error**: Webhook returns 500 or timeout

**Solution**:
1. Verify webhook endpoint is running:
   ```bash
   curl -X POST http://localhost:8080/webhook/news -H "Content-Type: application/json" -d '{"test": true}'
   ```
2. Check signature header matches:
   ```bash
   # In webhook consumer, verify X-Signature HMAC matches payload
   ```
3. Increase timeout:
   ```env
   WEBHOOK_TIMEOUT=120
   ```
4. Review logs for detailed POST errors

---

### High Memory Usage During Classification

**Issue**: Python process consumes 6+ GB RAM

**Solution**:
1. Limit batch size (modify `ingest/classifier.py`):
   ```python
   # Process articles in smaller batches
   batch_size = 25
   for i, article in enumerate(get_all_articles()):
       if i % batch_size == 0:
           gc.collect()  # Force garbage collection
   ```
2. Disable GPU:
   ```env
   # In ingest/classifier.py, force TORCH_DEVICE = torch.device("cpu")
   ```
3. Reduce model precision (use `float16`):
   ```python
   model = AutoModelForSequenceClassification.from_pretrained(...).half()
   ```

---

## Development & Contributing

### Testing Individual Components

#### Test Scraper Output

```python
from ingest.get_all_articles import get_all_articles

articles = list(get_all_articles())
print(f"Fetched {len(articles)} articles")
for article in articles[:3]:
    print(f"  - {article['title']} ({article['source']})")
```

#### Test Text Cleaning & Entities

```python
from ingest.call_to_genai import call_to_genai_sdk

text = "President Biden visited London yesterday to meet with Prime Minister Sunak."
result = call_to_genai_sdk(text)
print(result)
# Output: {
#   "cleaned_text": "President Biden visited London yesterday to meet with Prime Minister Sunak.",
#   "locations": ["London"],
#   "organizations": [],
#   "persons": ["Biden", "Sunak"]
# }
```

#### Test Summarization

```python
from ingest.summarizer import smart_summarize

long_text = """
Your long article text here...
(Must be > 200 tokens to trigger summarization)
"""

summary = smart_summarize(long_text)
print(summary)
```

#### Test Sentiment Analysis

```python
from ingest.classifier import sentiment_pipeline

text = "This is a fantastic news story!"
result = sentiment_pipeline(text)
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9987}]
```

#### Test Topic Classification

```python
from ingest.classifier import topic_pipeline, CANDIDATE_TOPICS

text = "Stock markets climbed 2% today on positive earnings reports."
result = topic_pipeline(text, candidate_labels=CANDIDATE_TOPICS)
print(result)
# Output: {
#   'labels': ['business and finance', 'technology and innovation', ...],
#   'scores': [0.92, 0.04, ...]
# }
```

### Adding a New News Source

1. **Create scraper function** in `ingest/custom_scrapers.py`:
   ```python
   def scrape_mysource_stream() -> Iterable[Dict]:
       # Fetch articles from your source
       # Check link_pool for duplicates
       # Yield {title, url, text, source, scraped_at}
       pass
   ```

2. **Register in `get_all_articles()`**:
   ```python
   def get_all_articles():
       for scrape_func in [
           scrape_bbc_stream,
           scrape_cnn_stream,
           scrape_mysource_stream,  # ← Add here
           ...
       ]:
           # ...
   ```

3. **Test**:
   ```bash
   python -c "
   from ingest.custom_scrapers import scrape_mysource_stream
   for article in scrape_mysource_stream():
       print(f'✅ {article[\"title\"]}')
   "
   ```

### Adding a New Classification Model

1. Update `ingest/classifier.py`:
   ```python
   # Load new model
   new_model = AutoModelForXYZ.from_pretrained("model/name", cache_dir=CACHE_DIR)
   new_pipeline = pipeline("task", model=new_model, device=PIPELINE_DEVICE)
   ```

2. Update `scripts/bootstrap_models.py` to download it:
   ```python
   def dl_newmodel():
       name = "model/name"
       print(f"⬇️ {name}")
       AutoTokenizer.from_pretrained(name, cache_dir=CACHE_DIR)
       AutoModelForXYZ.from_pretrained(name, cache_dir=CACHE_DIR)
   
   def main():
       dl_sentiment()
       dl_topic()
       dl_summarizer()
       dl_newmodel()  # ← Add
   ```

3. Test and add to `requirements.txt` if needed

### Code Style & Best Practices

- **Linting**: Use `black` and `flake8` (not enforced currently)
- **Type hints**: Encouraged for new code
- **Docstrings**: Add for public functions
- **Error handling**: Always try/except around API calls and DB operations
- **Logging**: Use `print()` or Python `logging` module

---

## License

[Add your license here, e.g., MIT, Apache 2.0]

## Contact

For questions or issues:
- Create a GitHub issue
- Contact: Christian Fita

---

## Acknowledgments

- **Hugging Face**: Transformer models (DistilBERT, BART)
- **Google**: Gemini API for text cleaning & entity extraction
- **MongoDB**: Data persistence layer
- **Selenium**: Web scraping for dynamic content
- **PyTorch**: Deep learning framework

---

**Last Updated**: December 28, 2025
