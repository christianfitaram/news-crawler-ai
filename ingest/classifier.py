# classifier.py
import os

from dotenv import load_dotenv

from ingest.call_to_webhook import send_to_all_webhooks

load_dotenv()
from collections import Counter
from datetime import datetime, timezone
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import re
from bson import ObjectId
from ingest.get_all_articles import get_all_articles
from ingest.summarizer import smart_summarize
from lib.repositories.articles_repository import ArticlesRepository
from lib.repositories.link_pool_repository import LinkPoolRepository
from lib.repositories.metadata_repository import MetadataRepository
from lib.repositories.global_metadata_repository import GlobalMetadataRepository
import requests
import uuid
from google import genai

client = genai.Client()
load_dotenv()

CACHE_DIR_FROM_ENV = os.getenv('TRANSFORMERS_CACHE')

# tzinfo constant for UTC
TZ_UTC = timezone.utc


def generate_uuid4():
    return str(uuid.uuid4())


# from utils.validation import is_valid_sample
# batch-YYYY-MM-DD (batch is numeric)
_PATTERN = r'^(\d+)-(202[5-9]|20[3-9]\d)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$'
_rx = re.compile(_PATTERN)

# Injecting repositories
repo_articles = ArticlesRepository()
repo_link_pool = LinkPoolRepository()
repo_metadata = MetadataRepository()
repo_global_metadata = GlobalMetadataRepository()

# Define your candidate labels (topics)
CANDIDATE_TOPICS = [
    "politics and government",
    "sports and athletics",
    "science and research",
    "technology and innovation",
    "health and medicine",
    "business and finance",
    "entertainment and celebrity",
    "crime and justice",
    "climate and environment",
    "education and schools",
    "war and conflict",
    "travel and tourism"
]

# Titles to skip (case-insensitive substring match)
SKIP_TITLE_PHRASES = [
    "data privacy policy",
    "legal notice",
    "accessibility statement",
    "Top stories in 90 seconds",
    "Accessibility statement"
]

# Load HuggingFace sentiment_pipeline
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
CACHE_DIR = CACHE_DIR_FROM_ENV if CACHE_DIR_FROM_ENV else "/home/christianfita/news-scrawler-ai/models/transformers"

# Device detection: prefer CUDA, then MPS (Apple), else CPU
# For transformers.pipeline pass an integer device index (0 for first CUDA GPU, -1 for CPU)
TORCH_DEVICE = torch.device("cpu")
PIPELINE_DEVICE = -1
if torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda:0")
    PIPELINE_DEVICE = 0
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    TORCH_DEVICE = torch.device("mps")
    # some HF pipeline versions don't accept 'mps' as device arg; use CPU device index (-1)
    PIPELINE_DEVICE = -1

print(f"Using torch device: {TORCH_DEVICE}  | pipeline device index: {PIPELINE_DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
# move model weights to torch device when possible
try:
    model.to(TORCH_DEVICE)
except Exception:
    pass

# Debug: report where model parameters live and torch/CUDA info
try:
    param_device = next(model.parameters()).device
    print(f"Model 1 first parameter device: {param_device}")
except Exception:
    print("Model 1 device: unknown")

print(
    f"torch version: {torch.__version__}, torch.cuda.is_available: {torch.cuda.is_available()}, torch.version.cuda: {torch.version.cuda}")

# Create sentiment pipeline robustly; fall back to CPU if GPU pipeline creation fails
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=PIPELINE_DEVICE,
        max_length=512,
        truncation=True,
    )
except Exception as e:
    print(
        f"Warning: failed to create sentiment pipeline on device {PIPELINE_DEVICE}: {e}. Falling back to CPU pipeline.")
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            max_length=512,
            truncation=True,
        )
    except Exception as e2:
        print(f"Error: failed to create fallback CPU sentiment pipeline: {e2}")
        sentiment_pipeline = None

# Load HuggingFace topic_pipeline
MODEL_NAME_TOPIC = "facebook/bart-large-mnli"
CACHE_DIR_TOPIC = CACHE_DIR_FROM_ENV if CACHE_DIR_FROM_ENV else "/home/christianfita/news-scrawler-ai/models/transformers"

tokenizer_topic = AutoTokenizer.from_pretrained(MODEL_NAME_TOPIC, cache_dir=CACHE_DIR_TOPIC)
model_topic = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_TOPIC, cache_dir=CACHE_DIR_TOPIC)
try:
    model_topic.to(TORCH_DEVICE)
except Exception:
    pass

# Debug: report where topic model parameters live
try:
    param_device_topic = next(model_topic.parameters()).device
    print(f"Model topic first parameter device: {param_device_topic}")
except Exception:
    print("Model topic device: unknown")

try:
    topic_pipeline = pipeline(
        "zero-shot-classification",
        model=model_topic,
        tokenizer=tokenizer_topic,
        device=PIPELINE_DEVICE,
        max_length=512,
        truncation=True
    )
except Exception as e:
    print(f"Warning: failed to create topic pipeline on device {PIPELINE_DEVICE}: {e}. Falling back to CPU pipeline.")
    try:
        topic_pipeline = pipeline(
            "zero-shot-classification",
            model=model_topic,
            tokenizer=tokenizer_topic,
            device=-1,
            max_length=512,
            truncation=True
        )
    except Exception as e2:
        print(f"Error: failed to create fallback CPU topic pipeline: {e2}")
        topic_pipeline = None


def is_valid_sample(sample: str) -> bool:
    m = _rx.match(sample)
    if not m:
        return False
    _, year, month, day = m.groups()
    try:
        datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
        return True
    except ValueError:
        return False


def classify_articles():
    id_for_metadata = generate_uuid4()
    # Initialize counters
    sentiment_counter = Counter()
    topic_counter = Counter()
    num_well_classified = 0
    num_failed_classified = 0
    try:
        repo_metadata.insert_metadata(
            {
                "_id": id_for_metadata,
                "gathering_sample_startedAt": datetime.now(TZ_UTC),
            }
        )
    except Exception as e:
        # Log and continue; do not recurse on failure
        print(f"Error inserting metadata: {e}")

    for i, article in enumerate(get_all_articles(), start=1):
        title = (article.get("title") or "").strip()
        # Skip undesired static pages by title
        title_lower = title.lower()
        if any(phrase in title_lower for phrase in SKIP_TITLE_PHRASES):
            # mark link as processed to avoid re-processing
            try:
                repo_link_pool.update_link_in_pool({"url": article.get("url")},
                                                   {"$set": {"is_articles_processed": True, "sample": id_for_metadata}})
            except Exception:
                pass
            print(f"[{i}] ⏭️ Skipping static/boilerplate article: {title}")
            continue

        text = article.get("text", "")
        text_len = len(text)
        if not text:
            continue

        try:
            if text_len > 200:
                summary = smart_summarize(text)
            else:
                summary = text

            topic = topic_pipeline(summary, candidate_labels=CANDIDATE_TOPICS)
            sentiment = sentiment_pipeline(summary)[0]
            try:
                text_cleaned = call_to_gpt_api(article.get("text"), timeout=60)  # 60 second timeout
            except Exception as e:
                print(f"[{i}] ⚠️ Text cleaning failed: {e}, using original text")
                text_cleaned = article.get("text", "")

            classified_article = {
                "title": article.get("title"),
                "url": article.get("url"),
                "summary": summary,
                "text": text_cleaned,
                "source": article.get("source"),
                "sample": id_for_metadata,
                "scraped_at": article.get("scraped_at"),
                "topic": topic["labels"][0],
                "isCleaned": False,
                "sentiment": {
                    "label": sentiment["label"],
                    "score": sentiment["score"]
                },
            }
            # set data for metadata
            num_well_classified += 1
            topic_label = topic["labels"][0]
            sentiment_label = sentiment["label"]
            topic_counter[topic_label] += 1
            sentiment_counter[sentiment_label] += 1

            # inserting data into mongoDB
            insert_id = repo_articles.create_articles(classified_article)
            send_to_all_webhooks(insert_id)
            add_one_to_total_articles_in_documents()
            add_one_to_topic_data_in_documents(topic_label)
            repo_link_pool.update_link_in_pool({"url": article.get("url")},
                                               {"$set": {"is_articles_processed": True, "sample": id_for_metadata}})

            print(f"[{i}] ✅ {classified_article['title']}")

        except Exception as e:
            num_failed_classified += 1
            repo_link_pool.update_link_in_pool({"url": article.get("url")},
                                               {"$set": {"is_articles_processed": True, "sample": id_for_metadata}})
            print(f"[{i}] ❌ Error classifying article: {e}")

    # Total number of successfully classified articles
    total_classified = sum(topic_counter.values())

    # Compute sorted percentages
    topic_percentages = [
        {"label": label, "percentage": round((count / total_classified) * 100, 2)}
        for label, count in topic_counter.most_common()
    ]

    sentiment_percentages = [
        {"label": label, "percentage": round((count / total_classified) * 100, 2)}
        for label, count in sentiment_counter.most_common()
    ]
    repo_metadata.update_metadata({"_id": id_for_metadata}, {
        "$set": {
            "articles_processed": {
                "successfully": num_well_classified,
                "unsuccessfully": num_failed_classified
            },
            "topic_distribution": topic_percentages,
            "sentiment_distribution": sentiment_percentages,
            "gathering_sample_finishedAt": datetime.now(TZ_UTC)
        }
    })

    return id_for_metadata


def call_to_gpt_api(prompt: str, timeout: int = 60) -> str:
    prompt_final = """You are a professional text cleaner.
Your task:
- Remove any reference to news outlets, authors, publication names, URLs, or web layout artifacts.
- Discard malformed, incomplete, or irrelevant fragments.
- Do not include explanations, comments, or formatting — only return the clean text.
Text to rewrite:
""" + prompt

    api_url = "http://localhost:11434/api/generate"

    payload = {
        "model": "gpt-oss:20b",
        "prompt": prompt_final,
        "stream": False
    }

    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        data = response.json()
        return data["response"].strip()
    except requests.exceptions.Timeout:
        print(f"GPT API timeout after {timeout}s, using original text")
        return prompt  # Return original text as fallback
    except requests.exceptions.RequestException as e:
        print(f"GPT API error: {e}, using original text")
        return prompt  # Return original text as fallback

def call_to_genai_api(prompt: str, timeout: int = 60) -> str:
    prompt_final = """You are a professional text cleaner.
Your task:
- Remove any reference to news outlets, authors, publication names, URLs, or web layout artifacts.
- Discard malformed, incomplete, or irrelevant fragments.
- Do not include explanations, comments, or formatting — only return the clean text.
Text to rewrite:""" + prompt
    
    try:
        response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=prompt_final,
        )
        return response.text.strip()
    except Exception as e:
        print(f"GenAI API error: {e}, using original text")
        return prompt  # Return original text as fallback                           


def add_one_to_total_articles_in_documents():
    selector = {"_id": ObjectId("6923b800f3d19f7c28f53a6d")}
    update_data = {"$inc": {"total_articles": 1}}
    repo_global_metadata.update_metadata(selector, update_data)
    total_articles = repo_articles.count_articles({})
    print(f"Total documents in the database: {total_articles}")

def add_one_to_topic_data_in_documents(topic: str):
    selector = {"_id": ObjectId("6923b800f3d19f7c28f53a6d"), "topics_data.topic": topic}
    update_data = {"$inc": {"topics_data.$.document_count": 1}}
    repo_global_metadata.update_metadata(selector, update_data)
    print(f"Incremented topic count for topic: {topic}")

if __name__ == "__main__":
    classify_articles()
