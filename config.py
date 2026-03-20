"""Global configuration for ST5230 experiment pipeline."""

import json
import os
import time
import logging

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("st5230")

# ── Reproducibility ──────────────────────────────────────────────
RANDOM_SEED = 42
SAMPLE_SIZE = 1000

# ── API Configuration ────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Shared OpenRouter Client ────────────────────────────────────
client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

# ── Models ───────────────────────────────────────────────────────
PARAPHRASE_MODEL = "meta-llama/llama-3.3-70b-instruct"

EVAL_MODELS = {
    "gpt-4o-mini": {
        "model_id": "openai/gpt-4o-mini",
        "top_logprobs": 5,
    },
    "qwen3.5-27b": {
        "model_id": "qwen/qwen3.5-27b",
        "top_logprobs": 5,
    },
    "kimi-k2": {
        "model_id": "moonshotai/kimi-k2",
        "top_logprobs": 5,
    },
}

# ── Datasets ─────────────────────────────────────────────────────
DATASETS = {
    "commonsense_qa": {
        "path": "commonsense_qa",
        "name": None,
        "split": "validation",
    },
    "arc_challenge": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "split": "test",
    },
    "mmlu": {
        "path": "cais/mmlu",
        "name": "all",
        "split": "test",
    },
}

# ── Paraphrase Types ─────────────────────────────────────────────
PARAPHRASE_TYPES = ["lexical", "syntactic", "style", "context", "translation"]

PARAPHRASE_PROMPTS = {
    "lexical": (
        "Rewrite the following question by replacing key words with synonyms. "
        "Keep the meaning identical. Output ONLY the rewritten question, nothing else."
    ),
    "syntactic": (
        "Restructure the following question's syntax (e.g., active↔passive, reorder clauses). "
        "Keep the meaning identical. Output ONLY the rewritten question, nothing else."
    ),
    "style": (
        "Rewrite the following question in a different style (e.g., formal↔casual, concise↔verbose). "
        "Keep the meaning identical. Output ONLY the rewritten question, nothing else."
    ),
    "context": (
        "Add a brief, semantically irrelevant introductory sentence before the following question. "
        "Keep the original question and meaning unchanged. Output ONLY the result (intro + question), nothing else."
    ),
    "translation": (
        "Translate the following question into Chinese. "
        "Keep the answer options in English letters (A/B/C/D/E). "
        "Output ONLY the translated question, nothing else."
    ),
}

# ── Paths ────────────────────────────────────────────────────────
_ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(_ROOT, "data", "sampled")
PARAPHRASED_DIR = os.path.join(_ROOT, "data_paraphrased")
RESULTS_DIR = os.path.join(_ROOT, "results")

# ── Retry ────────────────────────────────────────────────────────
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2  # seconds, exponential backoff base


# ── Helper Functions ─────────────────────────────────────────────

def call_llm(model_id, messages, max_tokens=1, temperature=0,
             logprobs=False, top_logprobs=None):
    """Call an LLM via OpenRouter with retry logic."""
    kwargs = dict(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if logprobs:
        kwargs["logprobs"] = True
        if top_logprobs is not None:
            kwargs["top_logprobs"] = top_logprobs

    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                f"Retrying in {delay}s..."
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
            else:
                logger.error(f"Failed after {MAX_RETRIES} attempts.")
                raise


def load_json(path):
    """Load a JSON file (expected to be a list of dicts)."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    """Save a list of dicts as a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_completed_ids(path):
    """Load IDs of already-processed items for checkpoint/resume."""
    return {item["id"] for item in load_json(path)}


def append_result(result, path):
    """Append a single result to a JSON file (load-append-save)."""
    data = load_json(path)
    data.append(result)
    save_json(data, path)
