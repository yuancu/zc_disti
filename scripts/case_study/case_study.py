#!/usr/bin/env python3
"""
Case Study: Compare fine-tuned vs baseline embedding model on a retrieval dataset.

This script:
1. Generates embeddings for both models using multi-GPU
2. Computes retrieval results with detailed per-query analysis
3. Runs BM25 baseline
4. Saves detailed results for bad case analysis

Expected data format (in --eval-data-dir):
  - corpus.jsonl:    {"id": "...", "text": "..."}
  - queries.jsonl:   {"id": "...", "text": "..."}
  - relevance.jsonl: {"query-id": "...", "corpus-id": "...", "score": 1}

Usage:
    python case_study.py --eval-data-dir DATA --finetuned-model MODEL --output-dir OUT
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm




def load_eval_data(eval_data_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """Load corpus, queries, and relevance judgments.

    Expected files:
      - corpus.jsonl:    {"id": "...", "text": "..."}
      - queries.jsonl:   {"id": "...", "text": "..."}
      - relevance.jsonl: {"query-id": "...", "corpus-id": "...", "score": int}
    """
    # Load corpus
    corpus = {}
    corpus_path = eval_data_dir / "corpus.jsonl"
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading corpus"):
            doc = json.loads(line.strip())
            text = doc.get("text", "")
            title = doc.get("title", "")
            corpus[doc["id"]] = {
                "text": text,
                "title": title,
                "full_text": f"{title} {text}".strip() if title else text
            }

    # Load queries
    queries = {}
    queries_path = eval_data_dir / "queries.jsonl"
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            query = json.loads(line.strip())
            queries[query["id"]] = query["text"]

    # Load relevance judgments
    qrels = {}
    qrels_path = eval_data_dir / "relevance.jsonl"
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            qid = entry["query-id"]
            did = entry["corpus-id"]
            rel = int(entry["score"])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = rel

    return corpus, queries, qrels


def generate_embeddings_multigpu(
    model_path: str,
    texts: List[str],
    text_ids: List[str],
    batch_size: int = 64,
    output_path: Path = None,
    desc: str = "Encoding"
) -> np.ndarray:
    """Generate embeddings using multi-GPU with sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    
    print(f"\nLoading model: {model_path}")
    model = SentenceTransformer(model_path, trust_remote_code=True)
    
    # Use all available GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        # sentence-transformers handles multi-GPU internally with encode_multi_process
        pool = model.start_multi_process_pool()
        embeddings = model.encode_multi_process(
            texts,
            pool,
            batch_size=batch_size,
            show_progress_bar=True
        )
        model.stop_multi_process_pool(pool)
    else:
        print("Using single GPU")
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    
    # Save embeddings
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            embeddings=embeddings,
            ids=np.array(text_ids)
        )
        print(f"Saved embeddings to: {output_path}")
    
    return embeddings


def compute_similarity_scores(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    query_ids: List[str],
    corpus_ids: List[str],
    top_k: int = 100
) -> Dict[str, List[Tuple[str, float]]]:
    """Compute cosine similarity and return top-k results per query."""
    # Normalize embeddings
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    
    results = {}
    
    # Process in batches to avoid memory issues
    batch_size = 100
    for i in tqdm(range(0, len(query_ids), batch_size), desc="Computing similarities"):
        batch_queries = query_embeddings[i:i+batch_size]
        batch_qids = query_ids[i:i+batch_size]
        
        # Compute similarities: (batch_size, corpus_size)
        similarities = np.dot(batch_queries, corpus_embeddings.T)
        
        for j, qid in enumerate(batch_qids):
            scores = similarities[j]
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            results[qid] = [(corpus_ids[idx], float(scores[idx])) for idx in top_indices]
    
    return results


def compute_metrics(
    results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10
) -> Dict[str, float]:
    """Compute NDCG@k, MRR@k, Recall@k."""
    ndcg_scores = []
    mrr_scores = []
    recall_scores = []
    
    for qid, ranked_list in results.items():
        if qid not in qrels:
            continue
        
        relevant = qrels[qid]
        
        # Get top-k
        top_k_docs = [doc_id for doc_id, _ in ranked_list[:k]]
        
        # MRR
        mrr = 0.0
        for rank, doc_id in enumerate(top_k_docs, 1):
            if doc_id in relevant and relevant[doc_id] > 0:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)
        
        # Recall
        relevant_in_top_k = sum(1 for doc_id in top_k_docs if doc_id in relevant and relevant[doc_id] > 0)
        total_relevant = sum(1 for rel in relevant.values() if rel > 0)
        recall = relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0
        recall_scores.append(recall)
        
        # NDCG
        dcg = 0.0
        for rank, doc_id in enumerate(top_k_docs, 1):
            if doc_id in relevant:
                rel = relevant[doc_id]
                dcg += (2**rel - 1) / np.log2(rank + 1)
        
        # Ideal DCG
        ideal_rels = sorted([rel for rel in relevant.values() if rel > 0], reverse=True)[:k]
        idcg = sum((2**rel - 1) / np.log2(rank + 1) for rank, rel in enumerate(ideal_rels, 1))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    return {
        f"ndcg@{k}": np.mean(ndcg_scores),
        f"mrr@{k}": np.mean(mrr_scores),
        f"recall@{k}": np.mean(recall_scores),
    }


def save_detailed_results(
    results: Dict[str, List[Tuple[str, float]]],
    queries: Dict[str, str],
    corpus: Dict[str, Dict],
    qrels: Dict[str, Dict[str, int]],
    output_path: Path,
    model_name: str
):
    """Save detailed per-query results for analysis."""
    detailed = []
    
    for qid in tqdm(queries.keys(), desc=f"Saving {model_name} results"):
        if qid not in qrels:
            continue
        
        query_text = queries[qid]
        relevant_docs = qrels[qid]
        retrieved = results.get(qid, [])[:20]  # Top 20 for analysis
        
        # Find ground truth docs
        gt_docs = []
        for doc_id, rel in relevant_docs.items():
            if rel > 0 and doc_id in corpus:
                gt_docs.append({
                    "doc_id": doc_id,
                    "relevance": rel,
                    "text": corpus[doc_id]["full_text"][:500]  # Truncate for readability
                })
        
        # Retrieved docs with scores
        retrieved_docs = []
        for doc_id, score in retrieved:
            is_relevant = doc_id in relevant_docs and relevant_docs[doc_id] > 0
            retrieved_docs.append({
                "doc_id": doc_id,
                "score": score,
                "is_relevant": is_relevant,
                "relevance": relevant_docs.get(doc_id, 0),
                "text": corpus[doc_id]["full_text"][:500] if doc_id in corpus else ""
            })
        
        # Compute per-query metrics
        top_10 = [d["doc_id"] for d in retrieved_docs[:10]]
        mrr = 0.0
        for rank, doc_id in enumerate(top_10, 1):
            if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
                mrr = 1.0 / rank
                break
        
        recall_10 = sum(1 for d in retrieved_docs[:10] if d["is_relevant"]) / len(gt_docs) if gt_docs else 0
        
        detailed.append({
            "query_id": qid,
            "query_text": query_text,
            "mrr@10": mrr,
            "recall@10": recall_10,
            "num_relevant": len(gt_docs),
            "ground_truth": gt_docs,
            "retrieved": retrieved_docs
        })
    
    # Sort by MRR (worst first for bad case analysis)
    detailed.sort(key=lambda x: x["mrr@10"])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)
    
    print(f"Saved detailed results to: {output_path}")


def run_bm25_baseline(
    corpus: Dict[str, Dict],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    output_path: Path
):
    """Run BM25 baseline using rank_bm25."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("Installing rank_bm25...")
        os.system("pip install rank_bm25 -q")
        from rank_bm25 import BM25Okapi

    print("\nRunning BM25 baseline...")

    # Tokenize corpus (simple whitespace tokenization, lowercased)
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid]["full_text"] for cid in corpus_ids]

    print("Tokenizing corpus...")
    tokenized_corpus = [text.lower().split() for text in tqdm(corpus_texts, desc="Tokenizing")]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Search
    results = {}
    for qid, query_text in tqdm(queries.items(), desc="BM25 search"):
        if qid not in qrels:
            continue

        tokenized_query = query_text.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Get top 100
        top_indices = np.argsort(scores)[::-1][:100]
        results[qid] = [(corpus_ids[idx], float(scores[idx])) for idx in top_indices]
    
    # Compute metrics
    metrics = compute_metrics(results, qrels, k=10)
    print(f"\nBM25 Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save detailed results
    save_detailed_results(
        results, queries, corpus, qrels,
        output_path / "bm25_detailed_results.json",
        "BM25"
    )
    
    return metrics, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-data-dir", type=str, required=True, help="Path to evaluation data directory (corpus.jsonl, queries.jsonl, relevance.jsonl)")
    parser.add_argument("--finetuned-model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--baseline-model", type=str, default="BAAI/bge-m3", help="Baseline model name or path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results and embeddings")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding generation if already exists")
    args = parser.parse_args()

    EVAL_DATA_DIR = Path(args.eval_data_dir)
    FINETUNED_MODEL = Path(args.finetuned_model)
    BASELINE_MODEL = args.baseline_model
    OUTPUT_DIR = Path(args.output_dir)

    # Set GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.num_gpus))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*60)
    print("Case Study: Fine-tuned vs Baseline BGE-M3")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Timestamp: {timestamp}")
    
    # Load data
    print("\n" + "="*60)
    print("Loading evaluation data...")
    print("="*60)
    corpus, queries, qrels = load_eval_data(EVAL_DATA_DIR)
    
    # Filter queries with qrels
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    
    print(f"Corpus size: {len(corpus)}")
    print(f"Queries: {len(queries)}")
    print(f"Qrels: {len(qrels)}")
    
    # Prepare texts for embedding
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid]["full_text"] for cid in corpus_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    # ============================================
    # Step 1: Generate embeddings
    # ============================================
    print("\n" + "="*60)
    print("Step 1: Generating Embeddings")
    print("="*60)
    
    # Baseline model embeddings
    baseline_corpus_path = OUTPUT_DIR / "baseline_corpus_embeddings.npz"
    baseline_query_path = OUTPUT_DIR / "baseline_query_embeddings.npz"
    
    if args.skip_embedding and baseline_corpus_path.exists() and baseline_query_path.exists():
        print("Loading existing baseline embeddings...")
        baseline_corpus_data = np.load(baseline_corpus_path)
        baseline_corpus_emb = baseline_corpus_data["embeddings"]
        baseline_query_data = np.load(baseline_query_path)
        baseline_query_emb = baseline_query_data["embeddings"]
    else:
        print("\n--- Baseline Model (BAAI/bge-m3) ---")
        baseline_corpus_emb = generate_embeddings_multigpu(
            BASELINE_MODEL, corpus_texts, corpus_ids,
            batch_size=args.batch_size,
            output_path=baseline_corpus_path,
            desc="Baseline corpus"
        )
        baseline_query_emb = generate_embeddings_multigpu(
            BASELINE_MODEL, query_texts, query_ids,
            batch_size=args.batch_size,
            output_path=baseline_query_path,
            desc="Baseline queries"
        )
    
    # Fine-tuned model embeddings
    finetuned_corpus_path = OUTPUT_DIR / "finetuned_corpus_embeddings.npz"
    finetuned_query_path = OUTPUT_DIR / "finetuned_query_embeddings.npz"
    
    if args.skip_embedding and finetuned_corpus_path.exists() and finetuned_query_path.exists():
        print("Loading existing fine-tuned embeddings...")
        finetuned_corpus_data = np.load(finetuned_corpus_path)
        finetuned_corpus_emb = finetuned_corpus_data["embeddings"]
        finetuned_query_data = np.load(finetuned_query_path)
        finetuned_query_emb = finetuned_query_data["embeddings"]
    else:
        print("\n--- Fine-tuned Model ---")
        finetuned_corpus_emb = generate_embeddings_multigpu(
            str(FINETUNED_MODEL), corpus_texts, corpus_ids,
            batch_size=args.batch_size,
            output_path=finetuned_corpus_path,
            desc="Fine-tuned corpus"
        )
        finetuned_query_emb = generate_embeddings_multigpu(
            str(FINETUNED_MODEL), query_texts, query_ids,
            batch_size=args.batch_size,
            output_path=finetuned_query_path,
            desc="Fine-tuned queries"
        )
    
    # ============================================
    # Step 2: Compute retrieval results
    # ============================================
    print("\n" + "="*60)
    print("Step 2: Computing Retrieval Results")
    print("="*60)
    
    print("\n--- Baseline Model ---")
    baseline_results = compute_similarity_scores(
        baseline_query_emb, baseline_corpus_emb,
        query_ids, corpus_ids, top_k=100
    )
    baseline_metrics = compute_metrics(baseline_results, qrels, k=10)
    print("Baseline Metrics:")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n--- Fine-tuned Model ---")
    finetuned_results = compute_similarity_scores(
        finetuned_query_emb, finetuned_corpus_emb,
        query_ids, corpus_ids, top_k=100
    )
    finetuned_metrics = compute_metrics(finetuned_results, qrels, k=10)
    print("Fine-tuned Metrics:")
    for k, v in finetuned_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # ============================================
    # Step 3: Save detailed results
    # ============================================
    print("\n" + "="*60)
    print("Step 3: Saving Detailed Results")
    print("="*60)
    
    save_detailed_results(
        baseline_results, queries, corpus, qrels,
        OUTPUT_DIR / "baseline_detailed_results.json",
        "Baseline"
    )
    
    save_detailed_results(
        finetuned_results, queries, corpus, qrels,
        OUTPUT_DIR / "finetuned_detailed_results.json",
        "Fine-tuned"
    )
    
    # ============================================
    # Step 4: BM25 Baseline
    # ============================================
    print("\n" + "="*60)
    print("Step 4: BM25 Baseline")
    print("="*60)
    
    bm25_metrics, bm25_results = run_bm25_baseline(corpus, queries, qrels, OUTPUT_DIR)
    
    # ============================================
    # Summary
    # ============================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    summary = {
        "timestamp": timestamp,
        "dataset": EVAL_DATA_DIR.name,
        "num_queries": len(queries),
        "num_corpus": len(corpus),
        "baseline_model": BASELINE_MODEL,
        "finetuned_model": str(FINETUNED_MODEL),
        "metrics": {
            "baseline": baseline_metrics,
            "finetuned": finetuned_metrics,
            "bm25": bm25_metrics,
        },
        "improvement": {
            metric: {
                "finetuned_vs_baseline": finetuned_metrics[metric] - baseline_metrics[metric],
                "finetuned_vs_bm25": finetuned_metrics[metric] - bm25_metrics[metric],
            }
            for metric in baseline_metrics.keys()
        }
    }
    
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'Method':<15} {'NDCG@10':<10} {'MRR@10':<10} {'Recall@10':<10}")
    print("-"*45)
    print(f"{'Baseline':<15} {baseline_metrics['ndcg@10']:<10.4f} {baseline_metrics['mrr@10']:<10.4f} {baseline_metrics['recall@10']:<10.4f}")
    print(f"{'Fine-tuned':<15} {finetuned_metrics['ndcg@10']:<10.4f} {finetuned_metrics['mrr@10']:<10.4f} {finetuned_metrics['recall@10']:<10.4f}")
    print(f"{'BM25':<15} {bm25_metrics['ndcg@10']:<10.4f} {bm25_metrics['mrr@10']:<10.4f} {bm25_metrics['recall@10']:<10.4f}")
    
    print(f"\nImprovement (Fine-tuned vs Baseline):")
    for metric in baseline_metrics.keys():
        diff = finetuned_metrics[metric] - baseline_metrics[metric]
        pct = (diff / baseline_metrics[metric] * 100) if baseline_metrics[metric] > 0 else 0
        print(f"  {metric}: {diff:+.4f} ({pct:+.2f}%)")
    
    print(f"\n" + "="*60)
    print("Output files:")
    print(f"  - Summary: {OUTPUT_DIR / 'summary.json'}")
    print(f"  - Baseline detailed: {OUTPUT_DIR / 'baseline_detailed_results.json'}")
    print(f"  - Fine-tuned detailed: {OUTPUT_DIR / 'finetuned_detailed_results.json'}")
    print(f"  - BM25 detailed: {OUTPUT_DIR / 'bm25_detailed_results.json'}")
    print(f"  - Embeddings: {OUTPUT_DIR / '*.npz'}")
    print("="*60)


if __name__ == "__main__":
    main()
