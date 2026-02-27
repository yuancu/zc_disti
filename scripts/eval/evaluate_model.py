#!/usr/bin/env python3
"""Evaluate trained embedding models on information retrieval datasets.

This script evaluates dense embedding models (e.g., BGE-M3) on IR tasks.

Metrics computed:
- NDCG@10: Normalized Discounted Cumulative Gain at rank 10
- MRR@10: Mean Reciprocal Rank at rank 10
- Recall@10: Fraction of relevant documents retrieved in the top 10

Data format expected:
- corpus.jsonl: One JSON object per line with "_id", "text", and optional "title"
- queries.jsonl: One JSON object per line with "_id" and "text"
- relevance.jsonl: One JSON object per line with "query-id", "corpus-id", and "score"

Usage:
    # Evaluate a dense model
    python evaluate_model.py --model-path /path/to/model --dataset my_dataset

    # Evaluate on multiple datasets
    python evaluate_model.py --model-path /path/to/model --datasets dataset1 dataset2
"""

import argparse
import functools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.evaluation import InformationRetrievalEvaluator


# ============================================
# Data Loading
# ============================================

def load_eval_data(data_dir: Path, dataset_name: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Load evaluation data (corpus, queries, qrels) for test set.

    Args:
        data_dir: Base directory containing dataset folders
        dataset_name: Name of the dataset

    Returns:
        Tuple of (queries, corpus, qrels)
        - queries: Dict mapping query_id -> query_text
        - corpus: Dict mapping doc_id -> doc_text
        - qrels: Dict mapping query_id -> {doc_id: relevance_score}
    """
    dataset_dir = data_dir / dataset_name

    # Load corpus
    corpus = {}
    corpus_path = dataset_dir / "corpus.jsonl"
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line.strip())
                # Support both "_id" and "id" field names
                doc_id = doc.get("_id") or doc.get("id")
                text = doc.get("text", "")
                title = doc.get("title", "")
                corpus[doc_id] = f"{title} {text}".strip() if title else text
    else:
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    # Load queries
    queries = {}
    queries_path = dataset_dir / "queries.jsonl"
    if queries_path.exists():
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                query = json.loads(line.strip())
                # Support both "_id" and "id" field names
                query_id = query.get("_id") or query.get("id")
                queries[query_id] = query["text"]
    else:
        raise FileNotFoundError(f"Queries file not found: {queries_path}")

    # Load qrels/relevance
    qrels = {}
    relevance_path = dataset_dir / "relevance.jsonl"
    if relevance_path.exists():
        with open(relevance_path, "r", encoding="utf-8") as f:
            for line in f:
                rel_data = json.loads(line.strip())
                qid = str(rel_data["query-id"])
                did = str(rel_data["corpus-id"])
                score = int(rel_data["score"])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][did] = score
    else:
        raise FileNotFoundError(f"Relevance file not found: {relevance_path}")

    return queries, corpus, qrels


# ============================================
# Evaluation
# ============================================

@dataclass
class EvalResult:
    """Evaluation result container."""
    dataset: str
    ndcg_at_10: float
    mrr_at_10: float
    recall_at_10: float
    num_queries: int
    num_corpus: int


def evaluate_model(
    model,
    dataset_name: str,
    data_dir: Path,
    batch_size: int,
    pool=None,
) -> EvalResult:
    """Evaluate a model on a dataset.

    Args:
        model: The SentenceTransformer model to evaluate
        dataset_name: Name of the dataset
        data_dir: Directory containing the dataset
        batch_size: Batch size for encoding
        pool: Multi-process pool from model.start_multi_process_pool() for multi-GPU encoding

    Returns:
        EvalResult with metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on: {dataset_name}")
    print(f"{'='*60}")

    # Load evaluation data
    queries, corpus, qrels = load_eval_data(data_dir, dataset_name)

    # Filter queries that have qrels
    queries = {qid: q for qid, q in queries.items() if qid in qrels}

    if not queries:
        raise ValueError(f"No queries with relevance judgments for {dataset_name}")

    print(f"Queries: {len(queries)}, Corpus: {len(corpus)} documents")

    # Create evaluator. For sparse models, reduce corpus_chunk_size because
    # each embedding is vocab-sized (30,522 dims) and densified before scoring.
    is_sparse = isinstance(model, SparseEncoder)
    corpus_chunk_size = 10_000 if is_sparse else 50_000
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=qrels,
        name=f"{dataset_name}-test",
        show_progress_bar=True,
        batch_size=batch_size,
        corpus_chunk_size=corpus_chunk_size,
    )

    # If a multi-GPU pool is available, patch the evaluator's embed_inputs
    # to route encode calls through the pool for distributed encoding.
    if pool is not None:
        original_embed_inputs = evaluator.embed_inputs

        def embed_inputs_with_pool(model, sentences, **kwargs):
            # Only pass kwargs that model.encode() accepts
            encode_kwargs = {}
            for key in ("prompt_name", "prompt", "normalize_embeddings", "truncate_dim"):
                if key in kwargs:
                    encode_kwargs[key] = kwargs[key]
            embs = model.encode(
                sentences,
                pool=pool,
                batch_size=batch_size,
                show_progress_bar=True,
                **encode_kwargs,
            )
            return torch.tensor(embs)

        evaluator.embed_inputs = embed_inputs_with_pool

    # SparseEncoder.encode() does not accept truncate_dim, and returns sparse
    # COO tensors. The default evaluator passes truncate_dim (rejected by
    # SparseEncoder), and dot_score's torch.mm on two sparse COO tensors
    # triggers cusparseSpGEMM which needs huge workspace buffers, causing OOM.
    # Fix both: strip truncate_dim and densify embeddings so dot_score uses
    # efficient dense GEMM instead.
    if is_sparse and pool is None:

        def embed_inputs_sparse(model, sentences, *, is_query=False, convert_to_tensor=True, **kwargs):
            encode_fn = model.encode_query if is_query else model.encode
            embs = encode_fn(
                sentences,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=convert_to_tensor,
            )
            if embs.is_sparse:
                embs = embs.to_dense()
            return embs

        evaluator.embed_inputs = embed_inputs_sparse

    # Run evaluation
    results = evaluator(model)

    # Extract metrics -- key prefix comes from model.similarity_fn_name (e.g. "cosine" or "dot")
    sim_name = model.similarity_fn_name.value if hasattr(model.similarity_fn_name, "value") else str(model.similarity_fn_name)
    ndcg_key = f"{dataset_name}-test_{sim_name}_ndcg@10"
    mrr_key = f"{dataset_name}-test_{sim_name}_mrr@10"
    recall_key = f"{dataset_name}-test_{sim_name}_recall@10"

    ndcg = results.get(ndcg_key, 0.0)
    mrr = results.get(mrr_key, 0.0)
    recall = results.get(recall_key, 0.0)

    print(f"\nResults for {dataset_name}:")
    print(f"  NDCG@10:   {ndcg:.4f}")
    print(f"  MRR@10:    {mrr:.4f}")
    print(f"  Recall@10: {recall:.4f}")

    return EvalResult(
        dataset=dataset_name,
        ndcg_at_10=ndcg,
        mrr_at_10=mrr,
        recall_at_10=recall,
        num_queries=len(queries),
        num_corpus=len(corpus),
    )


def load_model(model_path: str, max_seq_length: int, sparse: bool = False):
    """Load a model for evaluation.

    Args:
        model_path: Path to saved model or HuggingFace model name
        max_seq_length: Maximum sequence length
        sparse: If True, load as SparseEncoder instead of SentenceTransformer

    Returns:
        Loaded SentenceTransformer or SparseEncoder model
    """
    print(f"Loading {'sparse' if sparse else 'dense'} model: {model_path}")

    cls = SparseEncoder if sparse else SentenceTransformer
    model = cls(model_path, trust_remote_code=True)
    model.max_seq_length = min(max_seq_length, model.max_seq_length)

    return model


def print_summary(results: list[EvalResult]) -> None:
    """Print summary of all evaluation results."""
    print("\n" + "=" * 82)
    print("EVALUATION SUMMARY")
    print("=" * 82)
    print(f"\n{'Dataset':<20} {'NDCG@10':<12} {'MRR@10':<12} {'Recall@10':<12} {'Queries':<10} {'Corpus':<10}")
    print("-" * 82)

    for r in results:
        print(f"{r.dataset:<20} {r.ndcg_at_10:<12.4f} {r.mrr_at_10:<12.4f} {r.recall_at_10:<12.4f} {r.num_queries:<10} {r.num_corpus:<10}")

    print("-" * 82)

    # Compute averages
    if results:
        avg_ndcg = sum(r.ndcg_at_10 for r in results) / len(results)
        avg_mrr = sum(r.mrr_at_10 for r in results) / len(results)
        avg_recall = sum(r.recall_at_10 for r in results) / len(results)
        print(f"{'Average':<20} {avg_ndcg:<12.4f} {avg_mrr:<12.4f} {avg_recall:<12.4f}")

    print("=" * 82)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained embedding models on IR datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Format:
  The evaluation data should be organized as:
    data_dir/
      dataset_name/
        corpus.jsonl         - Documents: {"_id": "doc1", "text": "...", "title": "..."}
        queries.jsonl        - Queries: {"_id": "q1", "text": "..."}
        relevance.jsonl      - Relevance: {"query-id": "q1", "corpus-id": "doc1", "score": 1}

Examples:
  # Evaluate a fine-tuned model on a single dataset
  python evaluate_model.py --model-path ./output/final_model --dataset my_dataset

  # Evaluate on multiple datasets
  python evaluate_model.py --model-path ./output/final_model --datasets dataset1 dataset2

  # Evaluate the base BGE-M3 model (no fine-tuning)
  python evaluate_model.py --model-path BAAI/bge-m3 --datasets my_dataset
"""
    )

    # Get project directory for relative defaults
    project_dir = Path(__file__).parent.parent.parent

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved model or HuggingFace model name (e.g., BAAI/bge-m3)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "data",
        help="Directory containing prepared evaluation datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Datasets to evaluate on"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results (optional)"
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Load as a SparseEncoder (SPLADE) model instead of a dense SentenceTransformer"
    )

    args = parser.parse_args()

    # Check data directory
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Load model
    model = load_model(args.model_path, args.max_seq_length, sparse=args.sparse)

    # Start multi-GPU pool if multiple GPUs are available.
    # Skipped for sparse models: the pool bypasses asymmetric query/document routing.
    num_gpus = torch.cuda.device_count()
    pool = None
    if num_gpus > 1 and not args.sparse:
        print(f"Starting multi-GPU encoding pool on {num_gpus} GPUs")
        pool = model.start_multi_process_pool(
            target_devices=[f"cuda:{i}" for i in range(num_gpus)]
        )

    # Evaluate on each dataset
    results = []
    try:
        for dataset_name in args.datasets:
            try:
                result = evaluate_model(
                    model=model,
                    dataset_name=dataset_name,
                    data_dir=args.data_dir,
                    batch_size=args.batch_size,
                    pool=pool,
                )
                results.append(result)
            except Exception as e:
                print(f"Error evaluating {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
    finally:
        if pool is not None:
            model.stop_multi_process_pool(pool)

    # Print summary
    if results:
        print_summary(results)

    # Save results if output specified
    if args.output and results:
        output_data = {
            "model_path": args.model_path,
            "datasets": {
                r.dataset: {
                    "ndcg@10": r.ndcg_at_10,
                    "mrr@10": r.mrr_at_10,
                    "recall@10": r.recall_at_10,
                    "num_queries": r.num_queries,
                    "num_corpus": r.num_corpus,
                }
                for r in results
            }
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
