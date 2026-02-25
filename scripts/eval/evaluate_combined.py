#!/usr/bin/env python3
"""Evaluate combined BM25 + semantic search on information retrieval datasets.

This script combines BM25 (lexical) and dense/semantic retrieval scores using
normalization and combination methods from the OpenSearch blog post
"The ABCs of semantic search in OpenSearch" (Section 3).

This enables benchmarking whether combining retrieval methods outperforms
either one alone.

Normalization methods:
- min_max: Per-query min-max normalization
- l2: Per-query L2 normalization

Combination methods (weighted):
- arithmetic: Weighted arithmetic mean
- geometric: Weighted geometric mean (requires both signals > 0)
- harmonic: Weighted harmonic mean (requires both signals > 0)

Metrics computed:
- NDCG@10: Normalized Discounted Cumulative Gain at rank 10
- MRR@10: Mean Reciprocal Rank at rank 10
- Recall@10: Fraction of relevant documents retrieved in the top 10

Data format expected (same as evaluate_model.py / evaluate_bm25.py):
- corpus.jsonl: One JSON object per line with "_id", "text", and optional "title"
- queries.jsonl: One JSON object per line with "_id" and "text"
- relevance.jsonl: One JSON object per line with "query-id", "corpus-id", and "score"

Usage:
    python evaluate_combined.py --model-path BAAI/bge-m3 --datasets HC3Finance \\
        --normalization min_max --combination arithmetic --alpha 0.5
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import bm25s
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from transformers import AutoTokenizer
from tqdm import tqdm


# ============================================
# Score Caching
# ============================================

def _model_path_to_cache_name(model_path: str) -> str:
    """Convert model path to a safe directory name for caching."""
    return model_path.strip("/").replace("/", "__")


def _get_cache_path(cache_dir: Path, score_type: str, dataset: str,
                    model_path: str | None, retrieval_depth: int) -> Path:
    """Get the cache file path for a score type."""
    if score_type == "bm25":
        return cache_dir / "bm25" / dataset / f"k{retrieval_depth}.json"
    else:  # semantic
        model_name = _model_path_to_cache_name(model_path)
        return cache_dir / "semantic" / model_name / dataset / f"k{retrieval_depth}.json"


def _load_score_cache(path: Path) -> Dict[str, Dict[str, float]] | None:
    """Load cached scores from JSON. Returns None if cache doesn't exist."""
    if not path.exists():
        return None
    print(f"  Loading cached scores from: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data["scores"]


def _save_score_cache(path: Path, scores: Dict[str, Dict[str, float]], metadata: dict) -> None:
    """Save scores to JSON cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"metadata": metadata, "scores": scores}
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Cached scores saved to: {path}")


# ============================================
# Tokenization (matches evaluate_bm25.py)
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
# Metrics (matches evaluate_bm25.py)
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
# Model Loading (matches evaluate_model.py)
# ============================================

def load_model(model_path: str, max_seq_length: int):
    """Load a SentenceTransformer model for evaluation."""
    print(f"Loading model: {model_path}")
    model = SentenceTransformer(model_path, trust_remote_code=True)
    model.max_seq_length = max_seq_length
    return model


# ============================================
# Normalization
# ============================================

def normalize_min_max(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize scores per query result set.

    normalized = (score - min) / (max - min)
    If all scores are identical (max == min), returns 0.0 for all.
    """
    if not scores:
        return {}
    vals = list(scores.values())
    min_s = min(vals)
    max_s = max(vals)
    denom = max_s - min_s
    if denom == 0:
        return {doc_id: 0.0 for doc_id in scores}
    return {doc_id: (s - min_s) / denom for doc_id, s in scores.items()}


def normalize_l2(scores: Dict[str, float]) -> Dict[str, float]:
    """L2 normalize scores per query result set.

    normalized = score / sqrt(sum(score_i^2))
    If L2 norm is zero, returns 0.0 for all.
    """
    if not scores:
        return {}
    l2_norm = math.sqrt(sum(s * s for s in scores.values()))
    if l2_norm == 0:
        return {doc_id: 0.0 for doc_id in scores}
    return {doc_id: s / l2_norm for doc_id, s in scores.items()}


# ============================================
# Combination Methods
# ============================================

def combine_arithmetic(norm_bm25: float, norm_semantic: float, w_bm25: float, w_semantic: float) -> float:
    """Weighted arithmetic mean: w_bm25 * norm_bm25 + w_semantic * norm_semantic"""
    return w_bm25 * norm_bm25 + w_semantic * norm_semantic


def combine_geometric(norm_bm25: float, norm_semantic: float, w_bm25: float, w_semantic: float) -> float:
    """Weighted geometric mean: norm_bm25^w_bm25 * norm_semantic^w_semantic

    Returns 0.0 if either score is 0 (requires both signals).
    """
    if norm_bm25 <= 0 or norm_semantic <= 0:
        return 0.0
    return (norm_bm25 ** w_bm25) * (norm_semantic ** w_semantic)


def combine_harmonic(norm_bm25: float, norm_semantic: float, w_bm25: float, w_semantic: float) -> float:
    """Weighted harmonic mean: (w_bm25 + w_semantic) / (w_bm25/norm_bm25 + w_semantic/norm_semantic)

    Returns 0.0 if either score is 0 (requires both signals).
    """
    if norm_bm25 <= 0 or norm_semantic <= 0:
        return 0.0
    return (w_bm25 + w_semantic) / (w_bm25 / norm_bm25 + w_semantic / norm_semantic)


COMBINATION_METHODS = {
    "arithmetic": combine_arithmetic,
    "geometric": combine_geometric,
    "harmonic": combine_harmonic,
}

NORMALIZATION_METHODS = {
    "min_max": normalize_min_max,
    "l2": normalize_l2,
}


# ============================================
# Retrieval
# ============================================

def retrieve_bm25(
    query_ids: List[str],
    tokenized_queries: list[list[str]],
    retriever: bm25s.BM25,
    doc_ids: List[str],
    k: int,
) -> Dict[str, Dict[str, float]]:
    """Retrieve top-K documents per query using BM25.

    Returns:
        Dict mapping query_id -> {doc_id: bm25_score}
    """
    print(f"BM25: Retrieving top-{k} documents for {len(query_ids)} queries...")
    results_idx, scores = retriever.retrieve(tokenized_queries, k=min(k, len(doc_ids)))

    bm25_results = {}
    for i, qid in enumerate(query_ids):
        bm25_results[qid] = {}
        for j in range(results_idx.shape[1]):
            did = doc_ids[results_idx[i, j]]
            bm25_results[qid][did] = float(scores[i, j])

    return bm25_results


def retrieve_semantic(
    query_ids: List[str],
    query_texts: List[str],
    doc_ids: List[str],
    doc_texts: List[str],
    model: SentenceTransformer,
    k: int,
    batch_size: int,
    pool=None,
) -> Dict[str, Dict[str, float]]:
    """Retrieve top-K documents per query using dense semantic search.

    Returns:
        Dict mapping query_id -> {doc_id: cosine_similarity_score}
    """
    print(f"Semantic: Encoding {len(doc_texts)} corpus documents...")
    encode_kwargs = dict(batch_size=batch_size, show_progress_bar=True)
    if pool is not None:
        encode_kwargs["pool"] = pool
    corpus_embeddings = model.encode(doc_texts, **encode_kwargs)

    print(f"Semantic: Encoding {len(query_texts)} queries...")
    query_embeddings = model.encode(query_texts, **encode_kwargs)

    # Convert to tensors for semantic_search
    corpus_tensor = torch.tensor(corpus_embeddings)
    query_tensor = torch.tensor(query_embeddings)

    print(f"Semantic: Retrieving top-{k} documents for {len(query_ids)} queries...")
    hits = semantic_search(query_tensor, corpus_tensor, top_k=min(k, len(doc_ids)))

    semantic_results = {}
    for i, qid in enumerate(query_ids):
        semantic_results[qid] = {}
        for hit in hits[i]:
            did = doc_ids[hit["corpus_id"]]
            semantic_results[qid][did] = float(hit["score"])

    return semantic_results


# ============================================
# Score Combination
# ============================================

def combine_scores(
    bm25_results: Dict[str, Dict[str, float]],
    semantic_results: Dict[str, Dict[str, float]],
    normalization: str,
    combination: str,
    alpha: float,
) -> Dict[str, List[Tuple[str, float]]]:
    """Combine BM25 and semantic scores for all queries.

    Args:
        bm25_results: query_id -> {doc_id: bm25_score}
        semantic_results: query_id -> {doc_id: semantic_score}
        normalization: "min_max" or "l2"
        combination: "arithmetic", "geometric", or "harmonic"
        alpha: BM25 weight (semantic weight = 1 - alpha)

    Returns:
        Dict mapping query_id -> sorted list of (doc_id, combined_score) tuples
    """
    normalize_fn = NORMALIZATION_METHODS[normalization]
    combine_fn = COMBINATION_METHODS[combination]

    w_bm25 = alpha
    w_semantic = 1.0 - alpha

    all_query_ids = set(bm25_results.keys()) | set(semantic_results.keys())
    combined = {}

    for qid in all_query_ids:
        bm25_scores = bm25_results.get(qid, {})
        sem_scores = semantic_results.get(qid, {})

        # Normalize each system's scores
        norm_bm25 = normalize_fn(bm25_scores)
        norm_sem = normalize_fn(sem_scores)

        # Merge candidate sets
        all_doc_ids = set(norm_bm25.keys()) | set(norm_sem.keys())

        doc_combined = {}
        for did in all_doc_ids:
            b_score = norm_bm25.get(did, 0.0)
            s_score = norm_sem.get(did, 0.0)
            doc_combined[did] = combine_fn(b_score, s_score, w_bm25, w_semantic)

        # Sort by combined score descending
        sorted_docs = sorted(doc_combined.items(), key=lambda x: x[1], reverse=True)
        combined[qid] = sorted_docs

    return combined


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


def evaluate_combined(
    dataset_name: str,
    data_dir: Path,
    model: SentenceTransformer,
    model_path: str,
    normalization: str,
    combination: str,
    alpha: float,
    retrieval_depth: int,
    batch_size: int,
    tokenizer_batch_size: int,
    pool=None,
    cache_dir: Path | None = None,
) -> EvalResult:
    """Evaluate combined BM25 + semantic search on a dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating combined on: {dataset_name}")
    print(f"  Normalization: {normalization}, Combination: {combination}, Alpha: {alpha}")
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

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    # --- BM25 Retrieval (with caching) ---
    bm25_results = None
    if cache_dir is not None:
        bm25_cache_path = _get_cache_path(cache_dir, "bm25", dataset_name, None, retrieval_depth)
        bm25_results = _load_score_cache(bm25_cache_path)

    if bm25_results is None:
        print("\nBuilding BM25 index...")
        print("Tokenizing corpus...")
        tokenized_corpus = tokenize_texts_batch(doc_texts, batch_size=tokenizer_batch_size)

        retriever = bm25s.BM25()
        retriever.index(tokenized_corpus)

        print("Tokenizing queries...")
        tokenized_queries = tokenize_texts_batch(query_texts, batch_size=tokenizer_batch_size)

        bm25_results = retrieve_bm25(query_ids, tokenized_queries, retriever, doc_ids, retrieval_depth)

        if cache_dir is not None:
            _save_score_cache(bm25_cache_path, bm25_results, {
                "score_type": "bm25",
                "dataset": dataset_name,
                "retrieval_depth": retrieval_depth,
                "num_queries": len(queries),
                "num_corpus": len(corpus),
            })

    # --- Semantic Retrieval (with caching) ---
    semantic_results = None
    if cache_dir is not None:
        semantic_cache_path = _get_cache_path(cache_dir, "semantic", dataset_name, model_path, retrieval_depth)
        semantic_results = _load_score_cache(semantic_cache_path)

    if semantic_results is None:
        semantic_results = retrieve_semantic(
            query_ids, query_texts, doc_ids, doc_texts,
            model, retrieval_depth, batch_size, pool,
        )

        if cache_dir is not None:
            _save_score_cache(semantic_cache_path, semantic_results, {
                "score_type": "semantic",
                "dataset": dataset_name,
                "model_path": model_path,
                "retrieval_depth": retrieval_depth,
                "num_queries": len(queries),
                "num_corpus": len(corpus),
            })

    # --- Combine Scores ---
    print(f"\nCombining scores (normalization={normalization}, combination={combination}, alpha={alpha})...")
    combined_results = combine_scores(
        bm25_results, semantic_results,
        normalization, combination, alpha,
    )

    # --- Compute Metrics ---
    k = 10
    ndcg_scores = []
    mrr_scores = []
    recall_scores = []

    for qid in query_ids:
        ranked_docs = combined_results[qid]
        retrieved_doc_ids = [did for did, _ in ranked_docs]
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


def print_summary(results: list[EvalResult], normalization: str, combination: str, alpha: float) -> None:
    """Print summary of all evaluation results."""
    print("\n" + "=" * 82)
    print(f"COMBINED BM25 + SEMANTIC EVALUATION SUMMARY")
    print(f"  Normalization: {normalization}, Combination: {combination}, Alpha: {alpha}")
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
        description="Evaluate combined BM25 + semantic search on IR datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Normalization methods:
  min_max  - Per-query min-max normalization (default)
  l2       - Per-query L2 normalization

Combination methods (weighted by alpha):
  arithmetic - Weighted arithmetic mean (default)
  geometric  - Weighted geometric mean (requires both signals > 0)
  harmonic   - Weighted harmonic mean (requires both signals > 0)

Examples:
  python evaluate_combined.py --model-path BAAI/bge-m3 --datasets HC3Finance
  python evaluate_combined.py --model-path BAAI/bge-m3 --datasets HC3Finance CUREv1_en \\
      --normalization min_max --combination arithmetic --alpha 0.5
  python evaluate_combined.py --model-path BAAI/bge-m3 --datasets HC3Finance \\
      --combination geometric --alpha 0.3
"""
    )

    project_dir = Path(__file__).parent.parent.parent

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to dense model or HuggingFace model name (e.g., BAAI/bge-m3)"
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
        "--normalization",
        type=str,
        default="min_max",
        choices=["min_max", "l2"],
        help="Score normalization method (default: min_max)"
    )
    parser.add_argument(
        "--combination",
        type=str,
        default="arithmetic",
        choices=["arithmetic", "geometric", "harmonic"],
        help="Score combination method (default: arithmetic)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="BM25 weight; semantic weight = 1 - alpha (default: 0.5)"
    )
    parser.add_argument(
        "--retrieval-depth",
        type=int,
        default=1000,
        help="Top-K from each retrieval system before combining (default: 1000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Dense model encoding batch size (default: 32)"
    )
    parser.add_argument(
        "--tokenizer-batch-size",
        type=int,
        default=1000,
        help="BM25 tokenization batch size (default: 1000)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Dense model max sequence length (default: 4096)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results (optional)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=project_dir / "cache" / "scores",
        help="Directory for caching BM25/semantic scores (default: cache/scores)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable score caching; recompute everything"
    )

    args = parser.parse_args()

    # Validate alpha
    if not 0.0 <= args.alpha <= 1.0:
        print(f"Error: alpha must be in [0.0, 1.0], got {args.alpha}")
        sys.exit(1)

    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Resolve cache directory
    cache_dir = None if args.no_cache else args.cache_dir

    # Pre-load BM25 tokenizer
    get_tokenizer()

    # Load dense model
    model = load_model(args.model_path, args.max_seq_length)

    # Start multi-GPU pool if multiple GPUs are available
    num_gpus = torch.cuda.device_count()
    pool = None
    if num_gpus > 1:
        print(f"Starting multi-GPU encoding pool on {num_gpus} GPUs")
        pool = model.start_multi_process_pool(
            target_devices=[f"cuda:{i}" for i in range(num_gpus)]
        )

    # Evaluate on each dataset
    results = []
    try:
        for dataset_name in args.datasets:
            try:
                result = evaluate_combined(
                    dataset_name=dataset_name,
                    data_dir=args.data_dir,
                    model=model,
                    model_path=args.model_path,
                    normalization=args.normalization,
                    combination=args.combination,
                    alpha=args.alpha,
                    retrieval_depth=args.retrieval_depth,
                    batch_size=args.batch_size,
                    tokenizer_batch_size=args.tokenizer_batch_size,
                    pool=pool,
                    cache_dir=cache_dir,
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
        print_summary(results, args.normalization, args.combination, args.alpha)

    # Save results
    if args.output and results:
        output_data = {
            "model_path": args.model_path,
            "normalization": args.normalization,
            "combination": args.combination,
            "alpha": args.alpha,
            "retrieval_depth": args.retrieval_depth,
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
