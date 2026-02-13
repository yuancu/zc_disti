# DISTI — Knowledge Distillation for Embedding Models

This project implements a knowledge distillation pipeline to improve the [BGE-M3](https://huggingface.co/BAAI/bge-m3) multilingual embedding model for domain-specific information retrieval. It uses cross-encoder rerankers as teachers to generate soft relevance labels, then distills that knowledge into the student embedding model via KL-divergence loss.

Three training approaches are supported and compared:
- **Knowledge distillation** using teacher reranker scores
- **LoRA** (parameter-efficient fine-tuning)
- **Standard fine-tuning** (full model weight updates)

## Pipeline Overview

The core distillation pipeline has three stages:

```
Bedrock Batch Inference (synthetic query generation)
        │
        ▼
1. Mine Hard Negatives ─── BM25 retrieval with BGE-M3 tokenizer
        │
        ▼
2. Teacher Scoring ─────── Cross-encoder reranker scores (query, doc) pairs
        │
        ▼
3. Distillation Training ─ Fine-tune BGE-M3 with KL-divergence loss
        │
        ▼
4. Evaluation ──────────── NDCG@10, MRR@10, Recall@10 on held-out sets
```

## Repository Structure

```
disti/
├── scripts/          # All executable scripts (see below)
├── data/             # Training and evaluation datasets
├── artifacts/        # Trained models and intermediate outputs
├── results/          # Evaluation results and comparison tables
├── docs/             # Detailed technical documentation
├── mlruns/           # MLflow experiment tracking
└── requirements.txt
```

## Scripts

Scripts are organized by function into subdirectories. Each subdirectory contains Python scripts and corresponding `run_*.sh` shell scripts that orchestrate execution (multi-GPU, dataset iteration, etc.).

### `scripts/distill/` — Distillation Pipeline

The main pipeline scripts, run in order:

| Script | Purpose |
|--------|---------|
| `mine_hard_negative.py` | Mines hard negatives from Bedrock synthetic query output using BM25 retrieval |
| `mine_hard_negative_semantic.py` | Alternative: mines hard negatives using semantic similarity instead of BM25 |
| `teacher_score.py` | Scores (query, document) pairs with a cross-encoder reranker (gemma2 or minicpm) |
| `student_score.py` | Scores (query, document) pairs with the BGE-M3 student embedding model |
| `finetune.py` | Distillation training — fine-tunes BGE-M3 using teacher scores and KL-divergence loss |
| `generate_gap_report.py` | Generates a gap analysis between teacher and student scores |

Shell scripts: `run_mine_hard_negative.sh`, `run_mine_hard_negative_semantic.sh`, `run_teacher_score.sh`, `run_student_score.sh`, `run_finetune.sh`, `run_gap_report.sh`

### `scripts/eval/` — Evaluation

| Script | Purpose |
|--------|---------|
| `evaluate_model.py` | Evaluates a dense embedding model on IR benchmarks (NDCG@10, MRR@10, Recall@10) |
| `evaluate_bm25.py` | Evaluates BM25 baseline using BGE-M3 tokenizer |
| `generate_report.py` | Generates evaluation reports from result files |
| `generate_comparison_table.py` | Generates comparison tables across model variants |

Shell scripts: `run_evaluate_distilled.sh`, `run_evaluate_lora.sh`, `run_evaluate_tuned.sh`, `run_evaluate_bm25.sh`

### `scripts/lora/` — LoRA Fine-Tuning

| Script | Purpose |
|--------|---------|
| `finetune_lora.py` | Parameter-efficient fine-tuning using LoRA adapters with `MultipleNegativesRankingLoss` |

Shell script: `run_finetune_lora.sh`

### `scripts/finetune/` — Standard Fine-Tuning

| Script | Purpose |
|--------|---------|
| `finetune.py` | Full-weight fine-tuning of BGE-M3 |

Shell script: `run_finetune.sh`

### `scripts/case_study/` — Analysis

| Script | Purpose |
|--------|---------|
| `case_study.py` | Per-query comparative analysis between models (includes BM25 baseline comparison) |
| `sample_typical_cases.py` | Samples representative cases for manual inspection |

Shell scripts: `run_case_study.sh`, `run_case_study_legal_summ.sh`

### `scripts/benchmark_quantization.py`

Benchmarks INT4/INT8 quantization impact on reranker scoring accuracy and VRAM usage.

## Data

### `data/finetune/` — Training Data

JSONL files used for fine-tuning and distillation, one per dataset:

```
aila-casedocs.jsonl    cuad.jsonl          financebench.jsonl
finqa.jsonl            hc3finance.jsonl    lecard-v2.jsonl
legal-summ.jsonl       legalquad.jsonl     multi-cpr-video.jsonl
```

Each line follows the format:
```json
{"anchor": "query text", "positive": "relevant document", "negatives": ["hard neg 1", "hard neg 2", ...]}
```

### `data/eval/` — Evaluation Datasets

Each subdirectory contains an IR evaluation benchmark with three files:

```
data/eval/
├── AILA_casedocs/       # Legal case document retrieval
├── AILA_statutes/       # Legal statute retrieval
├── CUREv1_en/           # Medical Q&A (multi-specialty)
├── HC3Finance/          # Finance Q&A
├── FinQA/               # Financial numerical reasoning
├── financebench/        # Financial benchmarks
├── legal_summarization/ # Legal contract summarization
├── LegalQuAD/           # Legal question answering
└── multi-cpr-video/     # Multi-modal CPR video
```

Each dataset directory contains:
- `corpus.jsonl` — `{"_id": str, "text": str, "title": str}`
- `queries.jsonl` — `{"_id": str, "text": str}`
- `relevance.jsonl` — `{"query-id": str, "corpus-id": str, "score": float}`

### `data/bedrock/` and `data/bedrock-seeded/`

Raw output from Amazon Bedrock batch inference (synthetic query generation). These are the input to the hard negative mining stage.

## Artifacts

Intermediate and final outputs produced by the pipeline.

```
artifacts/
├── models/
│   ├── distilled/                          # Distilled models (main)
│   ├── distilled-{200,500}steps/           # Step-count variants
│   ├── distilled-200steps-scale0.05/       # Label scale factor variants
│   ├── distilled-200steps-scale0.05-seeded/
│   ├── distilled-200steps-scale0.05-semantic/  # Semantic hard negatives
│   ├── distilled-300steps-scale0.2/
│   ├── distilled-1epoch/
│   ├── lora/                               # LoRA adapter models
│   └── tuned/                              # Fully fine-tuned models
│
├── hard-negative/              # BM25-mined hard negatives
├── hard-negative-seeded/       # Seeded variant
├── hard-negative-semantic/     # Semantic hard negatives
│
├── teacher-score-gemma2/       # Teacher scores (gemma2 reranker)
├── teacher-score-gemma2-seeded/
├── teacher-score-gemma2-semantic/
├── teacher-score-minicpm/      # Teacher scores (minicpm reranker)
│
├── student-score-bge-m3/       # Student embedding scores
├── case_study/                 # Case study outputs
└── benchmark/                  # Quantization benchmarks
```

## Results

Evaluation results are stored in `results/`, with one subdirectory per model variant. See [`results/comparison.md`](results/comparison.md) for a summary table comparing Finetuned, LoRA, and Distilled models across all datasets.

## Documentation

Detailed technical documentation for each pipeline component is available in `docs/`:

| Document | Topic |
|----------|-------|
| [`docs/finetune.md`](docs/finetune.md) | Distillation training script (data loading, KL-divergence loss, CLI flags) |
| [`docs/teacher_score.md`](docs/teacher_score.md) | Teacher scoring pipeline (reranker models, input/output formats) |
| [`docs/mine_hard_negative.md`](docs/mine_hard_negative.md) | Hard negative mining (BM25 indexing, retrieval, batching) |
| [`docs/bm25_baseline.md`](docs/bm25_baseline.md) | BM25 vs BGE-M3 baseline comparison |
| [`docs/quantization-score-comparison.md`](docs/quantization-score-comparison.md) | INT4/INT8 quantization analysis |
| [`docs/teacher-student-gap.md`](docs/teacher-student-gap.md) | Teacher-student score gap analysis |
| [`docs/case_study/`](docs/case_study/) | Per-dataset case studies |

## Getting Started

### Requirements

```
pip install -r requirements.txt
```

Key dependencies (installed transitively):
- `sentence-transformers` — model training and inference
- `transformers` / `torch` / `accelerate` — model loading and distributed training
- `peft` — LoRA adapters
- `bm25s` — BM25 retrieval
- `tensorboardX` — training logging

### Running the Pipeline

1. **Mine hard negatives** from Bedrock output:
   ```bash
   bash scripts/distill/run_mine_hard_negative.sh
   ```

2. **Generate teacher scores** using a cross-encoder reranker:
   ```bash
   bash scripts/distill/run_teacher_score.sh
   ```

3. **Run distillation training**:
   ```bash
   bash scripts/distill/run_finetune.sh
   ```

4. **Evaluate** the distilled model:
   ```bash
   bash scripts/eval/run_evaluate_distilled.sh
   ```

See the `run_*.sh` scripts for the full set of CLI arguments and multi-GPU configuration.
