"""
Benchmark quantization drift and latency for teacher_score.py.

Runs teacher_score.py across quantization modes and batch sizes,
then reports score drift (vs. FP16 baseline) and latency per pair.

Usage:
    python scripts/benchmark_quantization.py [--model minicpm] [--dry-run]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
TEACHER_SCRIPT = ROOT / "scripts" / "teacher_score.py"
INPUT_FILE = ROOT / "data" / "sample_128.jsonl"
OUTPUT_DIR = ROOT / "artifacts" / "benchmark"

QUANTIZE_MODES = ["none", "int8", "int4"]
BATCH_SIZES = [32, 64, 128, 256]
MAX_DOC = 20


def run_teacher(model, quantize, batch_size, output_path):
    cmd = [
        sys.executable, str(TEACHER_SCRIPT),
        "--input", str(INPUT_FILE),
        "--output", str(output_path),
        "--model", model,
        "--quantize", quantize,
        "--batch_size", str(batch_size),
        "--max_doc", str(MAX_DOC),
    ]
    print(f"  Running: quantize={quantize} batch_size={batch_size}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}")
        return False
    # Print compute_scores timing line from stdout
    for line in result.stdout.splitlines():
        if "wall clock" in line:
            print(f"    {line.strip()}")
    return True


def load_scores(path):
    with open(path) as f:
        data = json.load(f)
    return [s for sample in data for s in sample["scores"]]


def load_meta(score_path):
    meta_path = str(score_path).rsplit(".", 1)[0] + ".meta.json"
    with open(meta_path) as f:
        return json.load(f)


def compute_drift(baseline_scores, other_scores):
    base = np.array(baseline_scores)
    other = np.array(other_scores)
    diff = np.abs(base - other)
    # Per-query Spearman (assuming 21 docs per query)
    docs_per_query = MAX_DOC + 1
    n_queries = len(base) // docs_per_query
    spearman_vals = []
    rank_preserved = 0
    total = 0
    for i in range(n_queries):
        s = slice(i * docs_per_query, (i + 1) * docs_per_query)
        b_q, o_q = base[s], other[s]
        rho, _ = spearmanr(b_q, o_q)
        spearman_vals.append(rho)
        # Check if top-1 is preserved
        if np.argmax(b_q) == np.argmax(o_q):
            rank_preserved += 1
        total += 1
    return {
        "mean_abs_diff": float(np.mean(diff)),
        "max_abs_diff": float(np.max(diff)),
        "std_abs_diff": float(np.std(diff)),
        "mean_spearman": float(np.mean(spearman_vals)),
        "min_spearman": float(np.min(spearman_vals)),
        "top1_preserved": f"{rank_preserved}/{total}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="minicpm", choices=["minicpm", "gemma2"])
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Run all configurations ---
    print("=" * 60)
    print("Phase 1: Running teacher_score.py across configurations")
    print("=" * 60)

    for quantize in QUANTIZE_MODES:
        for batch_size in BATCH_SIZES:
            out_path = OUTPUT_DIR / f"{args.model}_{quantize}_bs{batch_size}.json"
            if args.dry_run:
                print(f"  [dry-run] quantize={quantize} batch_size={batch_size}")
                continue
            run_teacher(args.model, quantize, batch_size, out_path)

    if args.dry_run:
        return

    # --- Phase 2: Score drift analysis ---
    print()
    print("=" * 60)
    print("Phase 2: Score drift (vs. quantize=none, batch_size=64)")
    print("=" * 60)

    baseline_path = OUTPUT_DIR / f"{args.model}_none_bs64.json"
    baseline_scores = load_scores(baseline_path)

    drift_results = {}
    for quantize in QUANTIZE_MODES:
        score_path = OUTPUT_DIR / f"{args.model}_{quantize}_bs64.json"
        scores = load_scores(score_path)
        drift = compute_drift(baseline_scores, scores)
        drift_results[quantize] = drift

    # Print drift table
    print(f"\n{'Mode':<8} {'Mean|Δ|':>10} {'Max|Δ|':>10} {'Std|Δ|':>10} {'Spearman':>10} {'MinSpear':>10} {'Top-1':>10}")
    print("-" * 70)
    for mode, d in drift_results.items():
        print(f"{mode:<8} {d['mean_abs_diff']:>10.4f} {d['max_abs_diff']:>10.4f} "
              f"{d['std_abs_diff']:>10.4f} {d['mean_spearman']:>10.4f} "
              f"{d['min_spearman']:>10.4f} {d['top1_preserved']:>10}")

    # --- Phase 3: Latency analysis ---
    print()
    print("=" * 60)
    print("Phase 3: Latency (ms/pair) vs. batch size")
    print("=" * 60)

    # Header
    header = f"{'bs':<8}" + "".join(f"{q:>12}" for q in QUANTIZE_MODES)
    print(f"\n{header}")
    print("-" * (8 + 12 * len(QUANTIZE_MODES)))

    latency_data = {}
    for batch_size in BATCH_SIZES:
        row = f"{batch_size:<8}"
        for quantize in QUANTIZE_MODES:
            meta_path = OUTPUT_DIR / f"{args.model}_{quantize}_bs{batch_size}.meta.json"
            meta = load_meta(meta_path)
            ms_per_pair = meta["ms_per_pair"]
            row += f"{ms_per_pair:>12.2f}"
            latency_data.setdefault(quantize, {})[batch_size] = ms_per_pair
        print(row)

    # --- Write report ---
    report = {
        "model": args.model,
        "baseline": "quantize=none, batch_size=64",
        "drift": drift_results,
        "latency_ms_per_pair": latency_data,
    }
    report_path = OUTPUT_DIR / f"{args.model}_benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
