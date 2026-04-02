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

# Suppress noisy HTTP request logs from OpenAI SDK
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Reproducibility ──────────────────────────────────────────────
RANDOM_SEED = 42
SAMPLE_SIZE = 1000

# ── API Configuration ────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Shared OpenRouter Client ────────────────────────────────────
client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

# ── Models ───────────────────────────────────────────────────────
PARAPHRASE_MODEL = "openai/gpt-4o"

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
        "Rewrite the following question by replacing some content words with synonyms.\n\n"
        "CRITICAL: Your output MUST contain the SAME number of sentences as the input. "
        "Count the sentences first. If the input has 5 sentences, your output must also have 5 sentences. "
        "If the input contains a passage or background information, you MUST rewrite the ENTIRE passage — "
        "do NOT skip or summarize any part.\n\n"
        "Rules:\n"
        "1. ONLY replace individual words with true synonyms of the SAME specificity "
        "(e.g., 'hamburger' stays 'hamburger', NOT 'sandwich'; 'ferret' stays 'ferret', NOT 'weasel'; "
        "'warm' stays 'warm', NOT 'hot' or 'tropical'; 'mountie' stays 'mountie', NOT 'constable').\n"
        "2. Keep the EXACT same sentence structure, word order, and grammar.\n"
        "3. Keep all question words (what, where, who, how, etc.) exactly as they are. "
        "If the original ends with '...the what?' or '...a what?', the rewrite must also end with '...the what?' or '...a what?'.\n"
        "4. Do NOT fill in blanks, answer the question, or add any new information.\n"
        "5. Output ONLY the rewritten question, nothing else."
    ),
    "syntactic": (
        "Restructure the following question's syntax while keeping all original words.\n\n"
        "CRITICAL: Your output MUST contain the SAME number of sentences as the input. "
        "Count the sentences first. If the input has 5 sentences, your output must also have 5 sentences. "
        "If the input contains a passage, experiment description, or background information, "
        "you MUST restructure EVERY sentence — do NOT skip, merge, or summarize any part.\n\n"
        "Rules:\n"
        "1. ONLY change sentence structure: reorder clauses, switch active↔passive, "
        "move prepositional phrases, etc.\n"
        "2. Keep the EXACT same vocabulary — do NOT replace any words with synonyms.\n"
        "3. Keep all question words (what, where, who, how, etc.) — the rewrite must remain a question.\n"
        "4. Do NOT fill in blanks, answer the question, or add new words that were not in the original.\n"
        "5. Output ONLY the rewritten question, nothing else."
    ),
    "style": (
        "Rewrite the following question in a more formal or more casual tone.\n\n"
        "CRITICAL: Your output MUST contain the SAME number of sentences as the input. "
        "Count the sentences first. If the input has 5 sentences, your output must also have 5 sentences. "
        "If the input contains a passage, experiment description, or background information, "
        "you MUST rewrite EVERY sentence — do NOT skip, merge, or summarize any part.\n\n"
        "Rules:\n"
        "1. PRIMARILY change tone and register (e.g., casual→formal or formal→casual). "
        "Minor vocabulary changes are allowed ONLY when necessary for the tone shift "
        "(e.g., 'booze'→'alcohol' for formalization), but do NOT do extensive synonym replacement.\n"
        "2. Keep the same sentence structure as much as possible.\n"
        "3. The rewrite MUST remain a question. Keep all question words (what, where, who, etc.).\n"
        "4. Do NOT introduce any words that could be an answer to the question. "
        "For example, if the question asks 'when what struck him?', do NOT write 'when inspiration struck him'.\n"
        "5. Do NOT fill in blanks or answer the question.\n"
        "6. Output ONLY the rewritten question, nothing else."
    ),
    "context": (
        "Your task: prepend ONE short irrelevant sentence before the question below, "
        "then output the irrelevant sentence followed by the COMPLETE original question.\n\n"
        "CRITICAL: You must copy the ENTIRE original text word-for-word. "
        "If the input has 5 sentences, your output must have 6 sentences (1 intro + 5 original). "
        "If the input contains a passage, experiment description, or quoted material, "
        "you MUST copy ALL of it — do NOT skip, summarize, or shorten any part.\n\n"
        "Rules:\n"
        "1. The introductory sentence MUST be completely unrelated to the question's topic. "
        "Pick a RANDOM topic each time — vary across weather, geography, history, science, sports, food, etc. "
        "Do NOT repeat the same fact (e.g., do NOT always use 'honey never spoils'). "
        "The intro sentence MUST be a declarative statement, NOT a question.\n"
        "2. Copy the ENTIRE original question word-for-word after the intro sentence. "
        "Do NOT shorten, summarize, merge, or modify the original question in ANY way.\n"
        "3. Do NOT fill in blanks or answer the question.\n"
        "4. Output format: [one intro sentence] [complete original question, all sentences]"
    ),
    "translation": (
        "Translate the following English question into natural Chinese.\n\n"
        "CRITICAL: You MUST translate EVERY sentence. "
        "Count the sentences first. If the input has 5 sentences, your output must also have 5 sentences in Chinese. "
        "If the input contains a passage, experiment description, quoted material, or background information, "
        "you MUST translate ALL of it — do NOT skip or summarize any part. "
        "The Chinese output should be roughly the same length as the English input.\n\n"
        "Rules:\n"
        "1. Translate faithfully — do NOT add, remove, or reinterpret any content.\n"
        "2. Keep the answer options in English letters (A/B/C/D/E) if present.\n"
        "3. If the question contains a blank or question word like 'what', translate it as a question, "
        "NOT as a statement. For example, '...the what?' → '……的什么？'\n"
        "4. Do NOT answer the question or fill in blanks.\n"
        "5. Output ONLY the translated question in Chinese, nothing else."
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
