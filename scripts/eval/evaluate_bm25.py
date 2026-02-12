#!/usr/bin/env python3
"""Evaluate BM25 baseline on information retrieval datasets.

This script evaluates BM25 retrieval (via bm25s) on the same IR tasks
used for dense model evaluation, providing a lexical baseline for comparison.

Uses BGE-M3 tokenizer for tokenization (consistent with the training pipeline).

Metrics computed:
- NDCG@10: Normalized Discounted Cumulative Gain at rank 10
- MRR@10: Mean Reciprocal Rank at rank 10
- Recall@10: Fraction of relevant documents retrieved in the top 10

Data format expected (same as evaluate_model.py):
- corpus.jsonl: One JSON object per line with "_id", "text", and optional "title"
- queries.jsonl: One JSON object per line with "_id" and "text"
- relevance.jsonl: One JSON object per line with "query-id", "corpus-id", and "score"

Usage:
    python evaluate_bm25.py --datasets dataset1 dataset2
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import bm25s
from transformers import AutoTokenizer
from tqdm import tqdm


# ============================================
# Tokenization (matches mine_hard_negative.py)
# ============================================

_tokenizer = None


def get_tokenizer():
    """Get or initialize the BGE-M3 tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        print("Loading BGE-M3 tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return _tokenizer


def tokenize_texts_batch(texts: list[str], batch_size: int = 1000) -> list[list[str]]:
    """Tokenize a list of texts in batches using BGE-M3 tokenizer."""
    tokenizer = get_tokenizer()
    all_tokenized = []

    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Tokenizing"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        batch_encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=8192,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        for input_ids in batch_encoded["input_ids"]:
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            all_tokenized.append(tokens)

    return all_tokenized


# ============================================
# Data Loading (matches evaluate_model.py)
# ============================================

def load_eval_data(data_dir: Path, dataset_name: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Load evaluation data (corpus, queries, qrels) for test set."""
    dataset_dir = data_dir / dataset_name

    # Load corpus
    corpus = {}
    corpus_path = dataset_dir / "corpus.jsonl"
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line.strip())
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
# Metrics
# ============================================

def compute_ndcg_at_k(retrieved_ids: List[str], qrel: Dict[str, int], k: int = 10) -> float:
    """Compute NDCG@k for a single query."""
    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = qrel.get(doc_id, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because rank starts at 1

    # Ideal DCG
    ideal_rels = sorted(qrel.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += (2 ** rel - 1) / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_mrr_at_k(retrieved_ids: List[str], qrel: Dict[str, int], k: int = 10) -> float:
    """Compute MRR@k for a single query."""
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if qrel.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(retrieved_ids: List[str], qrel: Dict[str, int], k: int = 10) -> float:
    """Compute Recall@k for a single query."""
    relevant = {did for did, score in qrel.items() if score > 0}
    if not relevant:
        return 0.0
    retrieved_relevant = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant)
    return retrieved_relevant / len(relevant)


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


def evaluate_bm25(
    dataset_name: str,
    data_dir: Path,
    batch_size: int = 1000,
) -> EvalResult:
    """Evaluate BM25 on a dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating BM25 on: {dataset_name}")
    print(f"{'='*60}")

    # Load evaluation data
    queries, corpus, qrels = load_eval_data(data_dir, dataset_name)

    # Filter queries that have qrels
    queries = {qid: q for qid, q in queries.items() if qid in qrels}

    if not queries:
        raise ValueError(f"No queries with relevance judgments for {dataset_name}")

    print(f"Queries: {len(queries)}, Corpus: {len(corpus)} documents")

    # Build ordered lists for corpus
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]

    # Build BM25 index
    print("Tokenizing corpus...")
    tokenized_corpus = tokenize_texts_batch(doc_texts, batch_size=batch_size)

    print("Building BM25 index...")
    retriever = bm25s.BM25()
    retriever.index(tokenized_corpus)

    # Retrieve for all queries
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print("Tokenizing queries...")
    tokenized_queries = tokenize_texts_batch(query_texts, batch_size=batch_size)

    k = 10
    print(f"Retrieving top-{k} documents for {len(query_ids)} queries...")
    results, scores = retriever.retrieve(tokenized_queries, k=min(k, len(doc_ids)))

    # Compute metrics
    ndcg_scores = []
    mrr_scores = []
    recall_scores = []

    for i, qid in enumerate(query_ids):
        retrieved_doc_ids = [doc_ids[idx] for idx in results[i]]
        qrel = qrels[qid]

        ndcg_scores.append(compute_ndcg_at_k(retrieved_doc_ids, qrel, k))
        mrr_scores.append(compute_mrr_at_k(retrieved_doc_ids, qrel, k))
        recall_scores.append(compute_recall_at_k(retrieved_doc_ids, qrel, k))

    ndcg = sum(ndcg_scores) / len(ndcg_scores)
    mrr = sum(mrr_scores) / len(mrr_scores)
    recall = sum(recall_scores) / len(recall_scores)

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


def print_summary(results: list[EvalResult]) -> None:
    """Print summary of all evaluation results."""
    print("\n" + "=" * 82)
    print("BM25 BASELINE EVALUATION SUMMARY")
    print("=" * 82)
    print(f"\n{'Dataset':<25} {'NDCG@10':<12} {'MRR@10':<12} {'Recall@10':<12} {'Queries':<10} {'Corpus':<10}")
    print("-" * 82)

    for r in results:
        print(f"{r.dataset:<25} {r.ndcg_at_10:<12.4f} {r.mrr_at_10:<12.4f} {r.recall_at_10:<12.4f} {r.num_queries:<10} {r.num_corpus:<10}")

    print("-" * 82)

    if results:
        avg_ndcg = sum(r.ndcg_at_10 for r in results) / len(results)
        avg_mrr = sum(r.mrr_at_10 for r in results) / len(results)
        avg_recall = sum(r.recall_at_10 for r in results) / len(results)
        print(f"{'Average':<25} {avg_ndcg:<12.4f} {avg_mrr:<12.4f} {avg_recall:<12.4f}")

    print("=" * 82)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 baseline on IR datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_bm25.py --datasets HC3Finance CUREv1_en
  python evaluate_bm25.py --datasets HC3Finance --output results/bm25_results.json
"""
    )

    project_dir = Path(__file__).parent.parent.parent

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
        default=1000,
        help="Batch size for tokenization (default: 1000)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results (optional)"
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Pre-load tokenizer
    get_tokenizer()

    # Evaluate on each dataset
    results = []
    for dataset_name in args.datasets:
        try:
            result = evaluate_bm25(
                dataset_name=dataset_name,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
            )
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    if results:
        print_summary(results)

    # Save results
    if args.output and results:
        output_data = {
            "model_path": "BM25 (bm25s + BGE-M3 tokenizer)",
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
