"""Step 1: Load datasets from HuggingFace and sample."""

import os
import random

from datasets import load_dataset

import config


def _normalize_commonsense_qa(example):
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    choices = [f"{l}) {t}" for l, t in zip(labels, texts)]
    return {
        "id": example["id"],
        "question": example["question"],
        "choices": choices,
        "answer": example["answerKey"],
        "source": "commonsense_qa",
    }


def _normalize_arc(example):
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    choices = [f"{l}) {t}" for l, t in zip(labels, texts)]
    return {
        "id": example["id"],
        "question": example["question"],
        "choices": choices,
        "answer": example["answerKey"],
        "source": "arc_challenge",
    }


def _normalize_mmlu(example, idx):
    labels = ["A", "B", "C", "D"]
    choices_text = example["choices"]
    choices = [f"{l}) {t}" for l, t in zip(labels, choices_text)]
    answer_idx = example["answer"]
    return {
        "id": f"mmlu_{idx}",
        "question": example["question"],
        "choices": choices,
        "answer": labels[answer_idx],
        "source": "mmlu",
    }


def load_and_sample(dataset_name):
    """Load a dataset from HuggingFace, sample, and normalize."""
    ds_config = config.DATASETS[dataset_name]
    config.logger.info(f"Loading {dataset_name} from HuggingFace...")

    kwargs = {"path": ds_config["path"], "split": ds_config["split"]}
    if ds_config["name"]:
        kwargs["name"] = ds_config["name"]

    ds = load_dataset(**kwargs)
    config.logger.info(f"  Total examples: {len(ds)}")

    rng = random.Random(config.RANDOM_SEED)
    n = min(config.SAMPLE_SIZE, len(ds))
    indices = rng.sample(range(len(ds)), n)
    indices.sort()

    normalizers = {
        "commonsense_qa": lambda ex, _i: _normalize_commonsense_qa(ex),
        "arc_challenge": lambda ex, _i: _normalize_arc(ex),
        "mmlu": lambda ex, i: _normalize_mmlu(ex, i),
    }
    normalize = normalizers[dataset_name]
    results = [normalize(ds[i], i) for i in indices]

    config.logger.info(f"  Sampled {len(results)} examples.")
    return results


def prepare_all():
    """Load and sample all datasets, save to data/sampled/."""
    os.makedirs(config.DATA_DIR, exist_ok=True)

    for dataset_name in config.DATASETS:
        out_path = os.path.join(config.DATA_DIR, f"{dataset_name}.json")
        data = load_and_sample(dataset_name)
        config.save_json(data, out_path)
        config.logger.info(f"  Saved to {out_path}")
