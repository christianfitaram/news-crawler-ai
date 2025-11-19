# python
# file: 'ingest/summarizer.py'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
from pathlib import Path
import re
import torch
from dotenv import load_dotenv

load_dotenv()

UNWANTED_KEYWORDS = [
    "Copyright",
    "©",
    "all rights reserved",
    "terms of use",
    "bbc is not responsible",
    "AP Photo",
    "r",
]

MODEL_NAME = "facebook/bart-large-cnn"

# -------------------------------------------------------------------
# CACHE DIRECTORY RESOLUTION (FIXED)
# -------------------------------------------------------------------

# Base dir of the project (assuming: <root>/ingest/summarizer.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# Default cache directory inside the project
DEFAULT_CACHE = (BASE_DIR / "models" / "transformers").resolve()

# Use TRANSFORMERS_CACHE if set, otherwise DEFAULT_CACHE
env_cache = os.getenv("TRANSFORMERS_CACHE")

if env_cache:
    cache_path = Path(env_cache).expanduser()
else:
    cache_path = DEFAULT_CACHE

# Ensure it's an absolute Path
if not cache_path.is_absolute():
    cache_path = DEFAULT_CACHE

# Create directory if needed
cache_path.mkdir(parents=True, exist_ok=True)
CACHE_DIR = str(cache_path)

print(f"Summarizer: using transformers cache at {CACHE_DIR}")

# -------------------------------------------------------------------
# MODEL & TOKENIZER LOADING
# -------------------------------------------------------------------

# Load from cache (will download if not present, once)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    local_files_only=False,
    use_fast=True,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    local_files_only=False,
)

# -------------------------------------------------------------------
# DEVICE SELECTION (GLOBAL)
# -------------------------------------------------------------------

if torch.cuda.is_available():
    _torch_dev = torch.device("cuda:0")
    _pipeline_device = 0
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    _torch_dev = torch.device("mps")
    # HF pipeline may not accept 'mps' as device index; use CPU index but keep model on mps
    _pipeline_device = -1
else:
    _torch_dev = torch.device("cpu")
    _pipeline_device = -1

try:
    model.to(_torch_dev)
except Exception:
    # Don't crash if device move fails (e.g. no MPS backend in this build)
    pass

print(
    f"Summarizer: torch version={torch.__version__}, "
    f"cuda_available={torch.cuda.is_available()}, "
    f"device={_torch_dev}, pipeline_device={_pipeline_device}"
)

# Create the summarization pipeline once and reuse
_summarizer_pipeline = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=_pipeline_device,
)

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def is_photo_credit(text: str) -> bool:
    return bool(re.search(r"\(AP Photo/.*?\)", text, flags=re.IGNORECASE))


def chunk_text(text: str, max_tokens: int = 512):
    """
    Split text into chunks based on sentence boundaries, limited by token count.
    """
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks, current, cur_len = [], "", 0

    for s in sentences:
        tlen = len(tokenizer.encode(s, add_special_tokens=False))
        if cur_len + tlen > max_tokens:
            if current:
                chunks.append(current.strip())
            current, cur_len = s, tlen
        else:
            current += (" " if current else "") + s
            cur_len += tlen

    if current:
        chunks.append(current.strip())

    return chunks


# -------------------------------------------------------------------
# MAIN SUMMARIZE FUNCTION
# -------------------------------------------------------------------

def smart_summarize(text: str, device: str = "auto") -> str:
    text = text.strip()
    if len(text) < 200:
        return text

    # Select device dynamically if requested
    if device == "auto":
        if torch.cuda.is_available():
            torch_dev = torch.device("cuda:0")
            pipeline_device = 0
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch_dev = torch.device("mps")
            pipeline_device = -1
        else:
            torch_dev = torch.device("cpu")
            pipeline_device = -1
    else:
        # allow explicit strings or ints: 'cpu', 'cuda', 'mps', or integer GPU index
        if isinstance(device, int):
            pipeline_device = device
            torch_dev = torch.device("cuda:0") if device >= 0 else torch.device("cpu")
        else:
            d = str(device).lower()
            if d == "cpu":
                torch_dev = torch.device("cpu")
                pipeline_device = -1
            elif d.startswith("cuda") or d == "gpu":
                torch_dev = torch.device("cuda:0")
                pipeline_device = 0
            elif d == "mps":
                torch_dev = torch.device("mps")
                pipeline_device = -1
            else:
                torch_dev = torch.device("cpu")
                pipeline_device = -1

    # Move model to desired device (pipeline is already created)
    try:
        model.to(torch_dev)
    except Exception:
        pass

    summarizer = _summarizer_pipeline

    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        try:
            in_len = len(tokenizer.encode(chunk, add_special_tokens=False))

            if in_len < 200:
                max_len = max(int(in_len * 0.8), 20)
                min_len = min(10, max_len // 2)
            else:
                max_len, min_len = 200, 80

            out = summarizer(
                chunk,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True,
            )[0]["summary_text"]

            summaries.append(out)

            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:
            print(f"Error summarizing chunk: {e}")

    result = "\n".join(summaries)

    # If the final summary is still too long, recursively summarize again
    if len(tokenizer.encode(result, add_special_tokens=False)) > 512:
        return smart_summarize(result, device=device)

    return result
