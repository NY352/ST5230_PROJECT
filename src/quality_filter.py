#!/usr/bin/env python3
"""Filter paraphrased data by removing items with hard quality issues.

Usage:
    python3 src/quality_filter.py              # filter all datasets
    python3 src/quality_filter.py commonsense_qa  # filter one dataset

Filtered outputs are written to data_paraphrased/ with suffix _filtered.json.
A report is printed to stdout.
"""

import os
import re
import sys

import config


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


def check_lexical(item):
    """Lexical-specific checks."""
    reasons = []

    r = check_question_form(item)
    if r:
        reasons.append(r)

    r = check_answer_leaked(item)
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
    """Syntactic-specific checks: only hard errors."""
    reasons = []

    r = check_question_form(item)
    if r:
        reasons.append(r)

    r = check_answer_leaked(item)
    if r:
        reasons.append(r)

    return reasons


def check_style(item):
    """Style-specific checks: only hard errors."""
    reasons = []

    r = check_question_form(item)
    if r:
        reasons.append(r)

    r = check_answer_leaked(item)
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

    para = item["paraphrased_question"]

    # Must contain Chinese characters
    cjk_count = sum(1 for c in para if '\u4e00' <= c <= '\u9fff')
    if cjk_count < 3:
        reasons.append(f"not Chinese (only {cjk_count} CJK chars)")

    # Question form
    orig = item["original_question"].rstrip()
    if orig.endswith("?") and not para.rstrip().endswith("？") and not para.rstrip().endswith("?"):
        reasons.append("question form lost")

    # Too short
    if len(para) < 5:
        reasons.append(f"too short ({len(para)} chars)")

    return reasons


CHECKERS = {
    "lexical": check_lexical,
    "syntactic": check_syntactic,
    "style": check_style,
    "context": check_context,
    "translation": check_translation,
}


def filter_dataset(dataset_name):
    """Filter one dataset's paraphrases and write filtered outputs."""
    print(f"\n{'=' * 70}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'=' * 70}")

    for ptype in config.PARAPHRASE_TYPES:
        input_path = os.path.join(
            config.PARAPHRASED_DIR, f"{dataset_name}_{ptype}.json"
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
            config.PARAPHRASED_DIR, f"{dataset_name}_{ptype}_filtered.json"
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


def main():
    if len(sys.argv) > 1:
        datasets = [sys.argv[1]]
    else:
        datasets = list(config.DATASETS.keys())

    for dataset_name in datasets:
        filter_dataset(dataset_name)

    print(f"\n{'=' * 70}")
    print("  Done. Filtered files saved as *_filtered.json")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
