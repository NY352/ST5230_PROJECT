#!/usr/bin/env python3
"""Unified entry point for ST5230 experiment pipeline.

Usage:
    python run.py prepare                                      # download & sample datasets
    python run.py paraphrase                                   # all datasets × all types
    python run.py paraphrase commonsense_qa                    # one dataset, all types
    python run.py paraphrase commonsense_qa lexical            # one dataset, one type
    python run.py filter                                       # filter all datasets
    python run.py filter commonsense_qa                        # filter one dataset
    python run.py evaluate                                     # all models × datasets × conditions
    python run.py evaluate gpt-4o-mini                         # one model, all datasets
    python run.py evaluate gpt-4o-mini commonsense_qa          # one model, one dataset
    python run.py evaluate gpt-4o-mini commonsense_qa baseline # specific condition
    python run.py analyze                                     # run statistical analysis
"""

import sys

import config
from src.data_loader import prepare_all, expand_sample
from src.paraphraser import paraphrase_dataset, paraphrase_all
from src.evaluator import evaluate_condition, evaluate_all
from src.quality_filter import filter_and_intersect
from src.analysis import main as run_analysis


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "prepare":
        prepare_all()

    elif command == "expand":
        if len(sys.argv) >= 3:
            expand_sample(sys.argv[2])
        else:
            for ds in config.DATASETS:
                expand_sample(ds)

    elif command == "paraphrase":
        if len(sys.argv) >= 4:
            paraphrase_dataset(sys.argv[2], sys.argv[3])
        elif len(sys.argv) == 3:
            for ptype in config.PARAPHRASE_TYPES:
                paraphrase_dataset(sys.argv[2], ptype)
        else:
            paraphrase_all()

    elif command == "filter":
        if len(sys.argv) >= 3:
            filter_and_intersect([sys.argv[2]])
        else:
            filter_and_intersect()

    elif command == "evaluate":
        conditions = ["baseline"] + config.PARAPHRASE_TYPES
        if len(sys.argv) >= 5:
            evaluate_condition(sys.argv[3], sys.argv[4], sys.argv[2])
        elif len(sys.argv) == 4:
            for cond in conditions:
                evaluate_condition(sys.argv[3], cond, sys.argv[2])
        elif len(sys.argv) == 3:
            for ds in config.DATASETS:
                for cond in conditions:
                    evaluate_condition(ds, cond, sys.argv[2])
        else:
            evaluate_all()

    elif command == "analyze":
        run_analysis()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
