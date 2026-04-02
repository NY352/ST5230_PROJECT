#!/usr/bin/env python3
"""Filter paraphrased data by removing items with hard quality issues.

Usage:
    python run.py filter                    # filter all datasets
    python run.py filter commonsense_qa     # filter one dataset

Filtered outputs are written to data_paraphrased/{dataset}/{type}_filtered.json.
A report is printed to stdout.
"""

import os
import re
import sys

import config


# ── Sentence counting ────────────────────────────────────────────

def _count_sentences_en(text):
    """Count English sentences by splitting on sentence-ending punctuation
    followed by whitespace and an uppercase letter or quote."""
    if not text.strip():
        return 0
    # Split on . ! ? followed by space+uppercase (handles abbreviations better)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text.strip())
    return len(parts)


def _count_sentences_zh(text):
    """Count Chinese sentences by splitting on Chinese sentence-ending punctuation."""
    if not text.strip():
        return 0
    # Split on 。！？ (Chinese period, exclamation, question mark)
    parts = re.split(r'[。！？]+', text.strip())
    # Remove empty parts from trailing punctuation
    parts = [p for p in parts if p.strip()]
    return max(len(parts), 1)


# ── Common checks ────────────────────────────────────────────────

def check_answer_leaked(item):
    """Check if the correct answer text leaked into the paraphrase but wasn't in the original."""
    orig = item["original_question"].lower()
    para = item["paraphrased_question"].lower()
    answer_letter = item["answer"]
    for choice in item["choices"]:
        if choice.startswith(f"{answer_letter})"):
            answer_text = choice[len(f"{answer_letter})"):].strip().lower()
            if len(answer_text) > 3 and answer_text in para and answer_text not in orig:
                return f"answer '{answer_text}' leaked"
    return None


def check_question_form(item):
    """Check if question mark was lost."""
    orig = item["original_question"].rstrip()
    para = item["paraphrased_question"].rstrip()
    if orig.endswith("?") and not para.endswith("?") and not para.endswith("？"):
        return "question form lost"
    return None


def check_sentence_dropped_en(item):
    """Check if English paraphrase has fewer sentences than original."""
    orig_count = _count_sentences_en(item["original_question"])
    para_count = _count_sentences_en(item["paraphrased_question"])
    if para_count < orig_count:
        return f"sentence dropped ({orig_count} -> {para_count})"
    return None


# ── Per-type checks ──────────────────────────────────────────────

def check_lexical(item):
    """Lexical-specific checks."""
    reasons = []

    r = check_question_form(item)
    if r:
        reasons.append(r)

    r = check_answer_leaked(item)
    if r:
        reasons.append(r)

    r = check_sentence_dropped_en(item)
    if r:
        reasons.append(r)

    # Fill-in-blank pattern lost
    orig = item["original_question"]
    para = item["paraphrased_question"]
    if re.search(r'\b(the|a|his|her|your|its|their)\s+what\s*\?', orig, re.IGNORECASE):
        if not re.search(r'\b(the|a|his|her|your|its|their)\s+what\s*\?', para, re.IGNORECASE):
            reasons.append("fill-in-blank lost")

    return reasons


def check_syntactic(item):
    """Syntactic-specific checks."""
    reasons = []

    r = check_question_form(item)
    if r:
        reasons.append(r)

    r = check_answer_leaked(item)
    if r:
        reasons.append(r)

    r = check_sentence_dropped_en(item)
    if r:
        reasons.append(r)

    return reasons


def check_style(item):
    """Style-specific checks."""
    reasons = []

    r = check_question_form(item)
    if r:
        reasons.append(r)

    r = check_answer_leaked(item)
    if r:
        reasons.append(r)

    r = check_sentence_dropped_en(item)
    if r:
        reasons.append(r)

    return reasons


def check_context(item):
    """Context-specific checks."""
    reasons = []

    r = check_answer_leaked(item)
    if r:
        reasons.append(r)

    orig = item["original_question"]
    para = item["paraphrased_question"]

    # Original question must be preserved verbatim
    orig_normalized = " ".join(orig.split())
    para_normalized = " ".join(para.split())
    if orig_normalized not in para_normalized:
        reasons.append("original question modified")

    return reasons


def check_translation(item):
    """Translation-specific checks."""
    reasons = []

    orig = item["original_question"]
    para = item["paraphrased_question"]

    # Must contain Chinese characters
    cjk_count = sum(1 for c in para if '\u4e00' <= c <= '\u9fff')
    if cjk_count < 3:
        reasons.append(f"not Chinese (only {cjk_count} CJK chars)")

    # Question form
    if orig.rstrip().endswith("?") and not para.rstrip().endswith("？") and not para.rstrip().endswith("?"):
        reasons.append("question form lost")

    # Too short
    if len(para) < 5:
        reasons.append(f"too short ({len(para)} chars)")

    # Content loss: if Chinese length < 15% of English length, content was dropped
    # (normal Chinese translations are 20-40% of English length due to character density)
    if len(orig) > 0:
        ratio = len(para) / len(orig)
        if ratio < 0.15:
            reasons.append(f"content lost (length ratio {ratio:.2f})")

    return reasons


CHECKERS = {
    "lexical": check_lexical,
    "syntactic": check_syntactic,
    "style": check_style,
    "context": check_context,
    "translation": check_translation,
}


# ── Filter & intersect ───────────────────────────────────────────

def filter_dataset(dataset_name):
    """Filter one dataset's paraphrases and write filtered outputs."""
    print(f"\n{'=' * 70}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'=' * 70}")

    for ptype in config.PARAPHRASE_TYPES:
        input_path = os.path.join(
            config.PARAPHRASED_DIR, dataset_name, f"{ptype}.json"
        )
        data = config.load_json(input_path)
        if not data:
            print(f"\n  [{ptype}] No data found at {input_path}")
            continue

        checker = CHECKERS[ptype]
        kept = []
        removed = []

        for item in data:
            reasons = checker(item)
            if reasons:
                removed.append({"id": item["id"], "reasons": reasons})
            else:
                kept.append(item)

        # Write filtered output
        output_path = os.path.join(
            config.PARAPHRASED_DIR, dataset_name, f"{ptype}_filtered.json"
        )
        config.save_json(kept, output_path)

        # Report
        print(f"\n  [{ptype}] {len(data)} -> {len(kept)} kept, {len(removed)} removed")
        if removed:
            # Count by reason
            reason_counts = {}
            for r in removed:
                for reason in r["reasons"]:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                print(f"    - {reason}: {count}")


def intersect_dataset(dataset_name):
    """Take intersection of valid IDs across all 5 paraphrase types,
    then rewrite all filtered files to keep only the common IDs."""

    # Collect valid IDs from each type's filtered file
    id_sets = {}
    for ptype in config.PARAPHRASE_TYPES:
        path = os.path.join(
            config.PARAPHRASED_DIR, dataset_name, f"{ptype}_filtered.json"
        )
        data = config.load_json(path)
        if not data:
            print(f"  [{ptype}] No filtered data found, skipping intersection")
            return
        id_sets[ptype] = {item["id"] for item in data}

    # Intersection
    common_ids = set.intersection(*id_sets.values())

    print(f"\n{'=' * 70}")
    print(f"  Intersection: {dataset_name}")
    print(f"{'=' * 70}")
    for ptype in config.PARAPHRASE_TYPES:
        print(f"  [{ptype}] {len(id_sets[ptype])} -> {len(common_ids)}")

    # Rewrite filtered files with only common IDs
    for ptype in config.PARAPHRASE_TYPES:
        path = os.path.join(
            config.PARAPHRASED_DIR, dataset_name, f"{ptype}_filtered.json"
        )
        data = config.load_json(path)
        kept = [item for item in data if item["id"] in common_ids]
        config.save_json(kept, path)

    print(f"\n  Final: {len(common_ids)} items per type")


def filter_and_intersect(datasets=None):
    """Filter datasets and take intersection across all types per dataset."""
    if datasets is None:
        datasets = list(config.DATASETS.keys())

    for dataset_name in datasets:
        filter_dataset(dataset_name)

    for dataset_name in datasets:
        intersect_dataset(dataset_name)

    print(f"\n{'=' * 70}")
    print("  Done. Filtered files saved as *_filtered.json (intersected)")
    print(f"{'=' * 70}")


def main():
    if len(sys.argv) > 1:
        datasets = [sys.argv[1]]
    else:
        datasets = None
    filter_and_intersect(datasets)


if __name__ == "__main__":
    main()
