# ST5230 Project: Benchmark Reliability under Semantic Paraphrasing

Investigating whether semantically equivalent paraphrases of benchmark questions cause performance shifts in LLMs, and which types of paraphrasing are most disruptive.

**Theme**: Reliability of LLM Evaluation (Theme 1)

## Key Findings

- Paraphrasing systematically degrades model confidence: 11/15 conditions show significant logprob decline (p < 0.05)
- **Translation** (English → Chinese) causes the largest degradation (Δt up to -0.496)
- **Syntactic** restructuring has minimal effect on ARC-Challenge and MMLU
- CommonsenseQA is the most sensitive benchmark; ARC-Challenge is the most robust

## Setup

```bash
conda create -n st5230 python=3.11 -y
conda activate st5230
pip install -r requirements.txt
# Create .env file with: OPENROUTER_API_KEY=your_key_here
```

## Pipeline

```bash
python run.py prepare      # Step 1: Download & sample datasets
python run.py paraphrase   # Step 2: Generate 5 types of paraphrases (GPT-4o)
python run.py filter       # Step 3: Quality filtering + ID intersection
python run.py evaluate     # Step 4: Evaluate model (GPT-4o-mini)
python run.py analyze      # Step 5: Statistical analysis + visualization
```

All steps support checkpoint/resume — safe to interrupt and re-run.

## Experimental Design

| Component | Details |
|-----------|---------|
| Datasets | CommonsenseQA (1085), ARC-Challenge (732), MMLU (1474) — after filtering |
| Paraphrase types | Lexical, Syntactic, Style, Context, Translation |
| Paraphrase model | GPT-4o via OpenRouter |
| Evaluation model | GPT-4o-mini via OpenRouter |
| Metrics | Accuracy, Ground-truth Logprob, Confidence Degradation (Δt) |
| Statistical test | Paired t-test with 95% CI |
| Failure modes | Robust / Hidden Hesitation / Total Collapse |

## Project Structure

```
ST5230_Project/
├── config.py               # Configuration, API client, prompts
├── run.py                  # CLI entry point
├── src/
│   ├── data_loader.py      # Dataset loading & sampling
│   ├── paraphraser.py      # Paraphrase generation
│   ├── evaluator.py        # Model evaluation
│   ├── quality_filter.py   # Quality filtering + ID intersection
│   ├── analysis.py         # Statistical analysis
│   └── visualize.py        # Violin + boxplot visualization
├── data/sampled/           # Sampled datasets
├── data_paraphrased/       # Paraphrased datasets (per-dataset subdirs)
├── results/                # Evaluation results + analysis_summary.json
├── figures/                # Visualization outputs
├── EXPERIMENT_LOG.md       # Detailed experiment log (Chinese)
└── test_paraphrase.py      # Pilot test script
```

## Documentation

- `EXPERIMENT_LOG.md` — Full experiment log with all results, prompt engineering history, filtering details, and key findings (in Chinese)
- `results/analysis_summary.json` — Machine-readable analysis results
