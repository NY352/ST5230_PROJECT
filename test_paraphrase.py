#!/usr/bin/env python3
"""Test paraphrase quality: run 60 items per type, then auto-analyze."""

import json
import os
import re
import sys

from tqdm import tqdm

import config
from src.paraphraser import paraphrase_question

TEST_LIMIT = 60
DATASET = "commonsense_qa"


def run_paraphrases():
    """Run 60 paraphrases for each type."""
    input_path = os.path.join(config.DATA_DIR, f"{DATASET}.json")
    data = config.load_json(input_path)[:TEST_LIMIT]

    os.makedirs(config.PARAPHRASED_DIR, exist_ok=True)

    for ptype in config.PARAPHRASE_TYPES:
        output_path = os.path.join(
            config.PARAPHRASED_DIR, f"{DATASET}_{ptype}.json"
        )
        # Skip if already has enough
        existing = config.load_json(output_path)
        if len(existing) >= TEST_LIMIT:
            print(f"[{ptype}] Already has {len(existing)} items, skipping.")
            continue

        completed_ids = {item["id"] for item in existing}
        remaining = [item for item in data if item["id"] not in completed_ids]

        print(f"\n[{ptype}] Running {len(remaining)} items...")
        for item in tqdm(remaining, desc=ptype):
            try:
                paraphrased = paraphrase_question(item["question"], ptype)
                result = {
                    "id": item["id"],
                    "original_question": item["question"],
                    "paraphrased_question": paraphrased,
                    "paraphrase_type": ptype,
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "source": item["source"],
                }
                config.append_result(result, output_path)
            except Exception as e:
                print(f"  Failed {item['id']}: {e}")
                continue


def analyze_all():
    """Analyze quality of all paraphrase types."""
    print("\n" + "=" * 80)
    print("QUALITY ANALYSIS REPORT")
    print("=" * 80)

    summary = {}

    for ptype in config.PARAPHRASE_TYPES:
        output_path = os.path.join(
            config.PARAPHRASED_DIR, f"{DATASET}_{ptype}.json"
        )
        data = config.load_json(output_path)
        if not data:
            print(f"\n[{ptype}] No data found.")
            continue

        if ptype == "lexical":
            issues = analyze_lexical(data)
        elif ptype == "syntactic":
            issues = analyze_syntactic(data)
        elif ptype == "style":
            issues = analyze_style(data)
        elif ptype == "context":
            issues = analyze_context(data)
        elif ptype == "translation":
            issues = analyze_translation(data)

        summary[ptype] = {
            "total": len(data),
            "issues": issues,
        }

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Type':<15} {'Total':<8} {'Issues':<8} {'Rate':<8} {'Details'}")
    print("-" * 80)
    for ptype, info in summary.items():
        total = info["total"]
        issue_list = info["issues"]
        total_issues = len(set(i["id"] for i in issue_list))
        rate = f"{total_issues / total * 100:.1f}%"

        # Count by category
        cats = {}
        for i in issue_list:
            cats[i["type"]] = cats.get(i["type"], 0) + 1
        detail_str = ", ".join(f"{k}:{v}" for k, v in cats.items()) if cats else "none"

        print(f"{ptype:<15} {total:<8} {total_issues:<8} {rate:<8} {detail_str}")


def analyze_lexical(data):
    """Check lexical paraphrases."""
    issues = []
    for item in data:
        orig = item["original_question"]
        para = item["paraphrased_question"]
        iid = item["id"]

        # Check: question form lost (original ends with ? but paraphrase doesn't)
        if orig.rstrip().endswith("?") and not para.rstrip().endswith("?"):
            issues.append({"id": iid, "type": "question_form_lost",
                           "detail": f"Original ends with '?', paraphrase: ...{para[-40:]}"})

        # Check: fill-in-blank answered ("...the what?" pattern lost)
        what_pattern = re.search(r'\b(the|a|his|her|your|its|their)\s+what\s*\?', orig, re.IGNORECASE)
        if what_pattern:
            if not re.search(r'\b(the|a|his|her|your|its|their)\s+what\s*\?', para, re.IGNORECASE):
                issues.append({"id": iid, "type": "blank_filled",
                               "detail": f"Original has fill-in-blank, paraphrase: ...{para[-50:]}"})

        # Check: answer leaked
        answer_letter = item["answer"]
        for choice in item["choices"]:
            if choice.startswith(f"{answer_letter})"):
                answer_text = choice[len(f"{answer_letter})"):].strip().lower()
                if len(answer_text) > 3 and answer_text in para.lower() and answer_text not in orig.lower():
                    issues.append({"id": iid, "type": "answer_leaked",
                                   "detail": f"Answer '{answer_text}' found in paraphrase but not original"})
                break

        # Check: structure changed too much (sentence count changed)
        orig_sents = len([s for s in re.split(r'[.!?]+', orig) if s.strip()])
        para_sents = len([s for s in re.split(r'[.!?]+', para) if s.strip()])
        if orig_sents >= 2 and para_sents < orig_sents:
            issues.append({"id": iid, "type": "sentence_dropped",
                           "detail": f"Original has {orig_sents} sentences, paraphrase has {para_sents}"})

    _print_type_report("LEXICAL", data, issues)
    return issues


def analyze_syntactic(data):
    """Check syntactic paraphrases."""
    issues = []
    for item in data:
        orig = item["original_question"]
        para = item["paraphrased_question"]
        iid = item["id"]

        # Check: question form lost
        if orig.rstrip().endswith("?") and not para.rstrip().endswith("?"):
            issues.append({"id": iid, "type": "question_form_lost",
                           "detail": f"Paraphrase: ...{para[-40:]}"})

        # Check: sentence dropped
        orig_sents = len([s for s in re.split(r'[.!?]+', orig) if s.strip()])
        para_sents = len([s for s in re.split(r'[.!?]+', para) if s.strip()])
        if orig_sents >= 2 and para_sents < orig_sents:
            issues.append({"id": iid, "type": "sentence_dropped",
                           "detail": f"Original has {orig_sents} sentences, paraphrase has {para_sents}"})

        # Check: answer leaked
        answer_letter = item["answer"]
        for choice in item["choices"]:
            if choice.startswith(f"{answer_letter})"):
                answer_text = choice[len(f"{answer_letter})"):].strip().lower()
                if len(answer_text) > 3 and answer_text in para.lower() and answer_text not in orig.lower():
                    issues.append({"id": iid, "type": "answer_leaked",
                                   "detail": f"Answer '{answer_text}' found in paraphrase but not original"})
                break

    _print_type_report("SYNTACTIC", data, issues)
    return issues


def analyze_style(data):
    """Check style paraphrases."""
    issues = []
    for item in data:
        orig = item["original_question"]
        para = item["paraphrased_question"]
        iid = item["id"]

        # Check: question form lost
        if orig.rstrip().endswith("?") and not para.rstrip().endswith("?"):
            issues.append({"id": iid, "type": "question_form_lost",
                           "detail": f"Paraphrase: ...{para[-40:]}"})

        # Check: sentence dropped
        orig_sents = len([s for s in re.split(r'[.!?]+', orig) if s.strip()])
        para_sents = len([s for s in re.split(r'[.!?]+', para) if s.strip()])
        if orig_sents >= 2 and para_sents < orig_sents:
            issues.append({"id": iid, "type": "sentence_dropped",
                           "detail": f"Original has {orig_sents} sentences, paraphrase has {para_sents}"})

        # Check: answer leaked
        answer_letter = item["answer"]
        for choice in item["choices"]:
            if choice.startswith(f"{answer_letter})"):
                answer_text = choice[len(f"{answer_letter})"):].strip().lower()
                if len(answer_text) > 3 and answer_text in para.lower() and answer_text not in orig.lower():
                    issues.append({"id": iid, "type": "answer_leaked",
                                   "detail": f"Answer '{answer_text}' found in paraphrase but not original"})
                break

    _print_type_report("STYLE", data, issues)
    return issues


def analyze_context(data):
    """Check context paraphrases."""
    issues = []
    # Track intro diversity
    intros = []

    for item in data:
        orig = item["original_question"]
        para = item["paraphrased_question"]
        iid = item["id"]

        # Check: original question preserved (must contain original text)
        # Normalize whitespace for comparison
        orig_normalized = " ".join(orig.split())
        para_normalized = " ".join(para.split())
        if orig_normalized not in para_normalized:
            # Try checking if at least the last sentence is preserved
            last_sent = orig.split(".")[-1].strip() if "." in orig else orig.strip()
            last_sent = last_sent.rstrip("?").strip()
            if len(last_sent) > 10 and last_sent not in para:
                issues.append({"id": iid, "type": "original_modified",
                               "detail": f"Original not found verbatim in output"})
            elif orig_normalized not in para_normalized:
                # Partial preservation - first part might be dropped
                first_sent = orig.split(".")[0].strip()
                if len(first_sent) > 10 and first_sent not in para:
                    issues.append({"id": iid, "type": "context_dropped",
                                   "detail": f"First sentence of original dropped: '{first_sent[:50]}...'"})

        # Track intro sentence (everything before the original question)
        if orig_normalized in para_normalized:
            intro = para_normalized[:para_normalized.index(orig_normalized)].strip()
            intros.append(intro)

    # Check intro diversity
    if intros:
        from collections import Counter
        # Extract key phrase from each intro
        intro_counter = Counter()
        for intro in intros:
            # Simplify to first few key words
            key = intro[:50].lower()
            intro_counter[key] += 1

        most_common = intro_counter.most_common(1)[0]
        if most_common[1] > len(intros) * 0.3:
            issues.append({"id": "GLOBAL", "type": "intro_repetitive",
                           "detail": f"Most common intro used {most_common[1]}/{len(intros)} times: '{most_common[0][:60]}...'"})

        unique_ratio = len(intro_counter) / len(intros) * 100
        print(f"  Intro diversity: {len(intro_counter)} unique / {len(intros)} total ({unique_ratio:.0f}%)")

    _print_type_report("CONTEXT", data, issues)
    return issues


def analyze_translation(data):
    """Check translation paraphrases."""
    issues = []
    for item in data:
        orig = item["original_question"]
        para = item["paraphrased_question"]
        iid = item["id"]

        # Check: output is actually Chinese (has CJK characters)
        cjk_count = sum(1 for c in para if '\u4e00' <= c <= '\u9fff')
        if cjk_count < 3:
            issues.append({"id": iid, "type": "not_chinese",
                           "detail": f"Output has only {cjk_count} CJK chars: {para[:60]}"})

        # Check: question form (should end with ？)
        if orig.rstrip().endswith("?") and not para.rstrip().endswith("？") and not para.rstrip().endswith("?"):
            issues.append({"id": iid, "type": "question_form_lost",
                           "detail": f"Paraphrase: ...{para[-30:]}"})

        # Check: hallucination (output much longer or shorter than expected)
        if len(para) < 5:
            issues.append({"id": iid, "type": "too_short",
                           "detail": f"Output only {len(para)} chars: {para}"})

        # Check: sentence dropped for multi-sentence
        orig_sents = len([s for s in re.split(r'[.!?]+', orig) if s.strip()])
        if orig_sents >= 2:
            # Check Chinese sentence count (split by 。！？)
            para_sents = len([s for s in re.split(r'[。！？?]+', para) if s.strip()])
            if para_sents < orig_sents - 1:  # Allow some flexibility
                issues.append({"id": iid, "type": "sentence_dropped",
                               "detail": f"Original has {orig_sents} sentences, translation has {para_sents}"})

    _print_type_report("TRANSLATION", data, issues)
    return issues


def _print_type_report(name, data, issues):
    """Print a formatted report for one paraphrase type."""
    total = len(data)
    unique_issues = len(set(i["id"] for i in issues))
    rate = unique_issues / total * 100 if total > 0 else 0

    print(f"\n{'─' * 80}")
    print(f"  {name}: {total} items, {unique_issues} with issues ({rate:.1f}%)")
    print(f"{'─' * 80}")

    if not issues:
        print("  No issues found.")
        return

    # Group by type
    by_type = {}
    for i in issues:
        by_type.setdefault(i["type"], []).append(i)

    for issue_type, items in by_type.items():
        print(f"\n  [{issue_type}] — {len(items)} case(s)")
        for item in items[:5]:  # Show up to 5 examples
            print(f"    ID: {item['id'][:20]}...  {item['detail'][:80]}")
        if len(items) > 5:
            print(f"    ... and {len(items) - 5} more")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_all()
    else:
        run_paraphrases()
        analyze_all()
