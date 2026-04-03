#!/usr/bin/env python3
"""Statistical analysis — strictly follows ST5230 Project Proposal.

Proposal Section 4.2: Confidence Degradation  Δt = L̄t − L̄base, 95% CI, paired t-test.
Proposal Section 5:
  - Summary table: Accuracy, mean/median Logprob, Δt with 95% CI.
  - Failure modes: Robust / Hidden Hesitation / Total Collapse.
  - H1 (Stability): no significant distributional shift in Li,t.
  - H2 (Heterogeneity): different paraphrase types → distinct shift patterns.

Usage:
    python run.py analyze
    python src/analysis.py
"""

import os
import sys
import json
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

# Threshold for "substantial" logprob decline in failure mode classification.
# Items whose per-item ΔL < -HESITATION_THRESHOLD are "Hidden Hesitation".
HESITATION_THRESHOLD = 0.5


# ── Data Loading ─────────────────────────────────────────────────

def load_results(model_name, dataset_name, condition):
    """Load evaluation results, keyed by item ID."""
    path = os.path.join(
        config.RESULTS_DIR, f"{model_name}_{dataset_name}_{condition}.json"
    )
    return {item["id"]: item for item in config.load_json(path)}


def build_paired_data(model_name, dataset_name, condition):
    """Build paired (baseline, paraphrase) data.
    Only includes IDs present in BOTH baseline and condition results."""
    baseline = load_results(model_name, dataset_name, "baseline")
    paraphrase = load_results(model_name, dataset_name, condition)

    common_ids = set(baseline.keys()) & set(paraphrase.keys())
    if not common_ids:
        return None

    pairs = []
    for item_id in sorted(common_ids):
        b = baseline[item_id]
        p = paraphrase[item_id]
        pairs.append({
            "id": item_id,
            "baseline_correct": b["is_correct"],
            "paraphrase_correct": p["is_correct"],
            "baseline_logprob": b["gt_logprob"],
            "paraphrase_logprob": p["gt_logprob"],
            "baseline_question": b.get("question", ""),
            "paraphrase_question": p.get("question", ""),
            "answer": b.get("answer", ""),
        })
    return pairs


# ── Proposal Section 5: Summary Table ────────────────────────────

def summary_statistics(pairs):
    """Compute Accuracy, mean/median Logprob, Confidence Degradation Δt
    with 95% CI and paired t-test (Proposal §4.2, §5)."""
    n = len(pairs)
    base_acc = sum(1 for p in pairs if p["baseline_correct"]) / n
    para_acc = sum(1 for p in pairs if p["paraphrase_correct"]) / n

    # Filter pairs with valid logprobs
    valid = [p for p in pairs
             if p["baseline_logprob"] is not None
             and p["paraphrase_logprob"] is not None]
    n_valid = len(valid)

    if n_valid < 5:
        return {
            "n": n, "n_valid_logprob": n_valid,
            "baseline_acc": base_acc, "paraphrase_acc": para_acc,
            "logprob": None,
        }

    base_lp = np.array([p["baseline_logprob"] for p in valid])
    para_lp = np.array([p["paraphrase_logprob"] for p in valid])
    delta_lp = para_lp - base_lp  # per-item Δ

    # Confidence Degradation: Δt = L̄t − L̄base  (Proposal §4.2)
    delta_t = float(np.mean(delta_lp))
    se = float(np.std(delta_lp, ddof=1) / np.sqrt(n_valid))
    ci_low = delta_t - 1.96 * se
    ci_high = delta_t + 1.96 * se

    # Paired t-test: H1 (Stability) — is Δt significantly ≠ 0?
    t_stat, t_p = stats.ttest_rel(para_lp, base_lp)

    return {
        "n": n,
        "n_valid_logprob": n_valid,
        "baseline_acc": float(base_acc),
        "paraphrase_acc": float(para_acc),
        "logprob": {
            "baseline_mean": float(np.mean(base_lp)),
            "baseline_median": float(np.median(base_lp)),
            "paraphrase_mean": float(np.mean(para_lp)),
            "paraphrase_median": float(np.median(para_lp)),
            "baseline_std": float(np.std(base_lp, ddof=1)),
            "paraphrase_std": float(np.std(para_lp, ddof=1)),
            "delta_t": delta_t,
            "delta_t_ci95": (float(ci_low), float(ci_high)),
            "ttest_stat": float(t_stat),
            "ttest_p": float(t_p),
        },
    }


# ── Proposal Section 5: Failure Mode Diagnosis ──────────────────
#
#   Robust:            Accuracy correct → correct, Logprob stable
#   Hidden Hesitation: Accuracy correct → correct, Logprob declines substantially
#   Total Collapse:    Accuracy correct → wrong   (+ Logprob deteriorates)
#
# Only items where baseline is CORRECT are classified.

def failure_mode_analysis(pairs, threshold=HESITATION_THRESHOLD):
    """Classify each baseline-correct item into Robust / Hidden Hesitation
    / Total Collapse (Proposal §5)."""
    robust = []
    hidden_hesitation = []
    total_collapse = []
    baseline_wrong = 0

    for p in pairs:
        if not p["baseline_correct"]:
            baseline_wrong += 1
            continue

        bl = p["baseline_logprob"]
        pl = p["paraphrase_logprob"]

        if not p["paraphrase_correct"]:
            # Accuracy flipped: Total Collapse
            total_collapse.append(p)
        elif bl is not None and pl is not None and (pl - bl) < -threshold:
            # Still correct but logprob dropped substantially
            hidden_hesitation.append(p)
        else:
            # Still correct and logprob stable
            robust.append(p)

    return {
        "robust": robust,
        "hidden_hesitation": hidden_hesitation,
        "total_collapse": total_collapse,
        "baseline_wrong_count": baseline_wrong,
    }


def failure_mode_counts(fm):
    """Return just the counts (for table display / JSON)."""
    return {
        "robust": len(fm["robust"]),
        "hidden_hesitation": len(fm["hidden_hesitation"]),
        "total_collapse": len(fm["total_collapse"]),
        "baseline_wrong": fm["baseline_wrong_count"],
    }


def failure_mode_examples(fm, n_examples=2):
    """Pick representative examples for each failure mode (Proposal §5)."""
    examples = {}
    for mode_name, items in [("robust", fm["robust"]),
                              ("hidden_hesitation", fm["hidden_hesitation"]),
                              ("total_collapse", fm["total_collapse"])]:
        if not items:
            examples[mode_name] = []
            continue
        # Sort by magnitude of logprob change to pick illustrative cases
        valid = [p for p in items
                 if p["baseline_logprob"] is not None
                 and p["paraphrase_logprob"] is not None]
        if not valid:
            examples[mode_name] = []
            continue
        valid.sort(key=lambda p: p["paraphrase_logprob"] - p["baseline_logprob"])
        if mode_name == "robust":
            selected = valid[-n_examples:]  # least change
        else:
            selected = valid[:n_examples]   # most dramatic change
        examples[mode_name] = [
            {
                "id": p["id"],
                "baseline_q": p["baseline_question"][:120],
                "paraphrase_q": p["paraphrase_question"][:120],
                "answer": p["answer"],
                "baseline_logprob": p["baseline_logprob"],
                "paraphrase_logprob": p["paraphrase_logprob"],
                "delta_logprob": round(p["paraphrase_logprob"] - p["baseline_logprob"], 4),
                "baseline_correct": p["baseline_correct"],
                "paraphrase_correct": p["paraphrase_correct"],
            }
            for p in selected
        ]
    return examples


# ── Main Analysis ────────────────────────────────────────────────

def analyze_all():
    """Run full analysis across all models, datasets, and conditions."""
    results = []
    for model_name in config.EVAL_MODELS:
        for dataset_name in config.DATASETS:
            for condition in config.PARAPHRASE_TYPES:
                pairs = build_paired_data(model_name, dataset_name, condition)
                if not pairs:
                    continue

                summary = summary_statistics(pairs)
                fm = failure_mode_analysis(pairs)

                entry = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "condition": condition,
                    "summary": summary,
                    "failure_modes": failure_mode_counts(fm),
                    "failure_examples": failure_mode_examples(fm),
                }
                results.append(entry)
    return results


# ── Print Report ─────────────────────────────────────────────────

def print_report(results):
    """Print analysis report matching Proposal §5 structure."""
    if not results:
        print("No results to analyze yet.")
        return

    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    for model_name, entries in by_model.items():
        print(f"\n{'=' * 100}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 100}")

        # ── Summary Table (Proposal §5) ──
        print(f"\n  Summary Table: Accuracy, Logprob, Confidence Degradation (Δt)")
        header = (f"  {'Dataset':<18} {'Condition':<13} {'N':>5} "
                  f"{'Acc_base':>9} {'Acc_para':>9} "
                  f"{'LP_base':>9} {'LP_para':>9} "
                  f"{'Med_base':>9} {'Med_para':>9} "
                  f"{'Δt':>8} {'95% CI':>20} {'p-value':>10}")
        print(header)
        print(f"  {'-' * 96}")

        for e in sorted(entries, key=lambda x: (x["dataset"], x["condition"])):
            s = e["summary"]
            lp = s["logprob"]
            if lp is None:
                print(f"  {e['dataset']:<18} {e['condition']:<13} {s['n']:>5} "
                      f"{s['baseline_acc']:>9.3f} {s['paraphrase_acc']:>9.3f} "
                      f"{'N/A':>9} {'N/A':>9} {'N/A':>9} {'N/A':>9} "
                      f"{'N/A':>8} {'N/A':>20} {'N/A':>10}")
                continue
            sig = "***" if lp["ttest_p"] < 0.001 else "**" if lp["ttest_p"] < 0.01 else "*" if lp["ttest_p"] < 0.05 else ""
            ci_str = f"[{lp['delta_t_ci95'][0]:+.3f}, {lp['delta_t_ci95'][1]:+.3f}]"
            print(f"  {e['dataset']:<18} {e['condition']:<13} {s['n']:>5} "
                  f"{s['baseline_acc']:>9.3f} {s['paraphrase_acc']:>9.3f} "
                  f"{lp['baseline_mean']:>9.3f} {lp['paraphrase_mean']:>9.3f} "
                  f"{lp['baseline_median']:>9.3f} {lp['paraphrase_median']:>9.3f} "
                  f"{lp['delta_t']:>+8.3f} {ci_str:>20} {lp['ttest_p']:>8.4f} {sig}")

        # ── Failure Mode Table (Proposal §5) ──
        print(f"\n  Failure Mode Diagnosis (threshold={HESITATION_THRESHOLD})")
        print(f"  {'Dataset':<18} {'Condition':<13} {'Robust':>10} {'Hidden Hes.':>12} {'Total Coll.':>12} {'Base Wrong':>11}")
        print(f"  {'-' * 76}")

        for e in sorted(entries, key=lambda x: (x["dataset"], x["condition"])):
            fm = e["failure_modes"]
            total_base_correct = fm["robust"] + fm["hidden_hesitation"] + fm["total_collapse"]
            print(f"  {e['dataset']:<18} {e['condition']:<13} "
                  f"{fm['robust']:>10} ({fm['robust']/total_base_correct*100:4.1f}%) "
                  f"{fm['hidden_hesitation']:>6} ({fm['hidden_hesitation']/total_base_correct*100:4.1f}%) "
                  f"{fm['total_collapse']:>6} ({fm['total_collapse']/total_base_correct*100:4.1f}%) "
                  f"{fm['baseline_wrong']:>10}"
                  if total_base_correct > 0 else
                  f"  {e['dataset']:<18} {e['condition']:<13} N/A")

        # ── Representative Examples (Proposal §5) ──
        print(f"\n  Representative Failure Examples:")
        # Show examples for one dataset only to keep output concise
        for e in sorted(entries, key=lambda x: (x["dataset"], x["condition"])):
            exs = e["failure_examples"]
            has_examples = any(exs[k] for k in exs)
            if not has_examples:
                continue
            print(f"\n  [{e['dataset']} / {e['condition']}]")
            for mode_name, label in [("hidden_hesitation", "Hidden Hesitation"),
                                      ("total_collapse", "Total Collapse")]:
                for ex in exs.get(mode_name, []):
                    print(f"    {label}: ΔLP={ex['delta_logprob']:+.4f} "
                          f"(base={ex['baseline_logprob']:.4f} → para={ex['paraphrase_logprob']:.4f}) "
                          f"correct: {ex['baseline_correct']}→{ex['paraphrase_correct']}")
                    print(f"      Base: {ex['baseline_q']}")
                    print(f"      Para: {ex['paraphrase_q']}")
            # Only show first dataset with examples per model to avoid flooding
            break


# ── Save ─────────────────────────────────────────────────────────

def save_results(results):
    """Save analysis results to JSON."""
    output_path = os.path.join(config.RESULTS_DIR, "analysis_summary.json")

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        return obj

    serializable = json.loads(json.dumps(results, default=convert))
    config.save_json(serializable, output_path)
    print(f"\n  Analysis saved to {output_path}")


# ── Entry Point ──────────────────────────────────────────────────

def main():
    results = analyze_all()
    print_report(results)
    save_results(results)

    from src.visualize import generate_all_figures
    generate_all_figures(results)


if __name__ == "__main__":
    main()
