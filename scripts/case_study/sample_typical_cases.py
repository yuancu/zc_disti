#!/usr/bin/env python3
"""
Sample typical cases comparing baseline vs fine-tuned model.

Categories:
1. Baseline wins: baseline found relevant doc in top-10, fine-tuned didn't
2. Fine-tuned wins: fine-tuned found relevant doc in top-10, baseline didn't

Usage:
    python sample_typical_cases.py
    python sample_typical_cases.py --num-samples 10
"""

import argparse
import json
import random
from pathlib import Path



def load_results(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return {r["query_id"]: r for r in json.load(f)}


def print_case(baseline, finetuned, category, idx):
    """Pretty print a case comparison."""
    print(f"\n{'='*100}")
    print(f"[{category}] Case {idx}")
    print(f"{'='*100}")
    print(f"Query ID: {baseline['query_id']}")
    print(f"Query: {baseline['query_text']}")
    print(f"\nBaseline MRR@10: {baseline['mrr@10']:.4f} | Fine-tuned MRR@10: {finetuned['mrr@10']:.4f}")
    print(f"Baseline Recall@10: {baseline['recall@10']:.4f} | Fine-tuned Recall@10: {finetuned['recall@10']:.4f}")
    
    print(f"\n--- Ground Truth ({baseline['num_relevant']} relevant) ---")
    for gt in baseline["ground_truth"][:2]:
        print(f"  [Doc {gt['doc_id']}] {gt['text'][:150]}...")
    
    print(f"\n--- Baseline Top-5 ---")
    for i, doc in enumerate(baseline["retrieved"][:5], 1):
        marker = "✓" if doc["is_relevant"] else "✗"
        print(f"  {i}. [{marker}] Doc {doc['doc_id']} (score={doc['score']:.4f})")
        print(f"      {doc['text'][:120]}...")
    
    print(f"\n--- Fine-tuned Top-5 ---")
    for i, doc in enumerate(finetuned["retrieved"][:5], 1):
        marker = "✓" if doc["is_relevant"] else "✗"
        print(f"  {i}. [{marker}] Doc {doc['doc_id']} (score={doc['score']:.4f})")
        print(f"      {doc['text'][:120]}...")


def find_cases(baseline_results, finetuned_results):
    """Find baseline-wins and finetuned-wins cases."""
    baseline_wins = []  # baseline MRR > 0, finetuned MRR = 0
    finetuned_wins = []  # finetuned MRR > 0, baseline MRR = 0
    
    for qid in baseline_results:
        if qid not in finetuned_results:
            continue
        
        b = baseline_results[qid]
        f = finetuned_results[qid]
        
        b_mrr = b["mrr@10"]
        f_mrr = f["mrr@10"]
        
        if b_mrr > 0 and f_mrr == 0:
            baseline_wins.append((qid, b, f))
        elif f_mrr > 0 and b_mrr == 0:
            finetuned_wins.append((qid, b, f))
    
    return baseline_wins, finetuned_wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True, help="Directory containing case study results (from case_study_medical.py)")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples per category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)

    random.seed(args.seed)

    print("Loading results...")
    baseline_results = load_results(OUTPUT_DIR / "baseline_detailed_results.json")
    finetuned_results = load_results(OUTPUT_DIR / "finetuned_detailed_results.json")
    
    baseline_wins, finetuned_wins = find_cases(baseline_results, finetuned_results)
    
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"Total queries: {len(baseline_results)}")
    print(f"Baseline wins (baseline found, fine-tuned missed): {len(baseline_wins)}")
    print(f"Fine-tuned wins (fine-tuned found, baseline missed): {len(finetuned_wins)}")
    
    # Sample cases
    n = args.num_samples
    sampled_baseline_wins = random.sample(baseline_wins, min(n, len(baseline_wins)))
    sampled_finetuned_wins = random.sample(finetuned_wins, min(n, len(finetuned_wins)))
    
    # Print baseline wins
    print(f"\n\n{'#'*100}")
    print(f"# CATEGORY 1: BASELINE WINS (baseline retrieved correctly, fine-tuned failed)")
    print(f"# Sampled {len(sampled_baseline_wins)} out of {len(baseline_wins)} cases")
    print(f"{'#'*100}")
    
    for i, (qid, b, f) in enumerate(sampled_baseline_wins, 1):
        print_case(b, f, "BASELINE WINS", i)
    
    # Print finetuned wins
    print(f"\n\n{'#'*100}")
    print(f"# CATEGORY 2: FINE-TUNED WINS (fine-tuned retrieved correctly, baseline failed)")
    print(f"# Sampled {len(sampled_finetuned_wins)} out of {len(finetuned_wins)} cases")
    print(f"{'#'*100}")
    
    for i, (qid, b, f) in enumerate(sampled_finetuned_wins, 1):
        print_case(b, f, "FINE-TUNED WINS", i)
    
    # Save sampled cases to JSON
    output = {
        "summary": {
            "total_queries": len(baseline_results),
            "baseline_wins_count": len(baseline_wins),
            "finetuned_wins_count": len(finetuned_wins),
        },
        "baseline_wins_samples": [
            {
                "query_id": qid,
                "query_text": b["query_text"],
                "baseline_mrr": b["mrr@10"],
                "finetuned_mrr": f["mrr@10"],
                "ground_truth": b["ground_truth"],
                "baseline_top10": b["retrieved"][:10],
                "finetuned_top10": f["retrieved"][:10],
            }
            for qid, b, f in sampled_baseline_wins
        ],
        "finetuned_wins_samples": [
            {
                "query_id": qid,
                "query_text": b["query_text"],
                "baseline_mrr": b["mrr@10"],
                "finetuned_mrr": f["mrr@10"],
                "ground_truth": b["ground_truth"],
                "baseline_top10": b["retrieved"][:10],
                "finetuned_top10": f["retrieved"][:10],
            }
            for qid, b, f in sampled_finetuned_wins
        ],
    }
    
    output_path = OUTPUT_DIR / "sampled_typical_cases.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n{'='*100}")
    print(f"Sampled cases saved to: {output_path}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
