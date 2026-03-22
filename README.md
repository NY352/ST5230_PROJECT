# ST5230 Project: LLM Robustness to Paraphrased Inputs

Investigating how large language models respond to semantically equivalent but differently phrased questions across multiple-choice benchmarks.

## Setup

```bash
# 1. Create conda environment
conda create -n st5230 python=3.11 -y
conda activate st5230

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
# Create .env file with: OPENROUTER_API_KEY=your_key_here
```

## Pipeline

The experiment runs in 4 steps:

### Step 1: Prepare Data

Download datasets from HuggingFace and sample 1000 examples each.

```bash
python run.py prepare
```

- **Datasets**: CommonsenseQA, ARC-Challenge, MMLU
- **Output**: `data/sampled/{dataset_name}.json`

### Step 2: Paraphrase

Generate 5 types of paraphrases for each dataset using GPT-4o.

```bash
python run.py paraphrase                          # all datasets × all types
python run.py paraphrase commonsense_qa            # one dataset, all types
python run.py paraphrase commonsense_qa lexical    # one dataset, one type
```

- **Paraphrase types**: `lexical`, `syntactic`, `style`, `context`, `translation`
- **Output**: `data_paraphrased/{dataset}_{type}.json`
- Supports checkpoint/resume — safe to interrupt and re-run

### Step 3: Quality Filter

Filter out items with quality issues (answer leakage, question form lost, etc.).

```bash
python run.py filter                               # filter all datasets
python run.py filter commonsense_qa                # filter one dataset
```

- **Output**: `data_paraphrased/{dataset}_{type}_filtered.json`

### Step 4: Evaluate

Evaluate 3 models on original (baseline) and paraphrased questions. Automatically uses filtered files when available.

```bash
python run.py evaluate                                         # all models × datasets × conditions
python run.py evaluate gpt-4o-mini                             # one model, all datasets
python run.py evaluate gpt-4o-mini commonsense_qa              # one model, one dataset
python run.py evaluate gpt-4o-mini commonsense_qa baseline     # specific condition
```

- **Models**: `gpt-4o-mini`, `qwen3.5-27b`, `kimi-k2` (all via OpenRouter)
- **Conditions**: `baseline` + 5 paraphrase types
- **Output**: `results/{model}_{dataset}_{condition}.json`
- Supports checkpoint/resume — safe to interrupt and re-run

## Project Structure

```
ST5230_Project/
├── .env                    # API key (not tracked by git)
├── .gitignore
├── requirements.txt
├── README.md
├── EXPERIMENT_LOG.md       # Detailed experiment log
├── config.py               # Configuration + shared utilities
├── run.py                  # Unified CLI entry point
├── src/
│   ├── data_loader.py      # Dataset loading & sampling
│   ├── paraphraser.py      # Paraphrase generation
│   ├── evaluator.py        # Model evaluation
│   └── quality_filter.py   # Rule-based quality filtering
├── test_paraphrase.py      # Pilot test script (60 items/type)
├── data/sampled/           # Sampled datasets (generated)
├── data_paraphrased/       # Paraphrased datasets (generated)
└── results/                # Evaluation results (generated)
```

## Models

| Role | Model | Provider |
|------|-------|----------|
| Paraphraser | `openai/gpt-4o` | OpenRouter |
| Evaluator 1 | `openai/gpt-4o-mini` | OpenRouter |
| Evaluator 2 | `qwen/qwen3.5-27b` | OpenRouter |
| Evaluator 3 | `moonshotai/kimi-k2` | OpenRouter |
