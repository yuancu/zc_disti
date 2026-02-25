#!/usr/bin/env python3
"""Generate evaluation report comparing retrieval methods against a base model.

All columns (except Base) show percentage change relative to the base model's
standalone dense retrieval: (new - base) / base * 100.

Columns are included dynamically based on which inputs are provided:
- Base: absolute score (always shown, reference point)
- BM25: standalone BM25 (if --bm25-results provided)
- Base+BM25 (Arith/Geome/Harmo): base model combined with BM25
- Distilled: tuned model standalone (if --tuned-results provided)
- Dist+BM25 (Arith/Geome/Harmo): tuned model combined with BM25

Expected input:
- --baseline-results: Path to baseline_results.json (standalone BAAI/bge-m3, no BM25)
- --bm25-results: (optional) Path to BM25-only results JSON
- --tuned-results: (optional) Path to tuned/distilled model standalone results JSON
- --combined-dir: Directory with combined evaluation results

Usage:
    python generate_combined_report.py \\
        --baseline-results results/tuned/baseline_results.json \\
        --bm25-results results/bm25_baseline/bm25_results.json \\
        --tuned-results results/distilled_gemma2_500steps/tuned_results.json \\
        --combined-dir results/combined-distilled
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_baseline_results(path: Path) -> dict:
    """Load standalone baseline results (no BM25 combination).

    Returns:
        dict mapping dataset_name -> {metric: value}
    """
    if not path.exists():
        print(f"Warning: Baseline results not found: {path}")
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("datasets", {})


def load_combined_results(combined_dir: Path) -> dict:
    """Load all combined evaluation results.

    Returns:
        {
            "baseline": {combination_method: {dataset: {metric: value}}},
            "distilled": {combination_method: {dataset: {metric: value}}},
            "meta": {combination_method: {"normalization": ..., "alpha": ...}},
        }
    """
    results = {
        "baseline": defaultdict(dict),
        "distilled": defaultdict(dict),
        "meta": {},
    }

    # New layout: combined_dir/combined_{identifier}_{norm}_{combo}.json
    # (alpha is encoded in the parent directory, e.g. alpha0.5/)
    pattern = re.compile(
        r"^combined_(.+?)_(min_max|l2)_(arithmetic|geometric|harmonic)\.json$"
    )

    for path in sorted(combined_dir.glob("combined_*.json")):
        m = pattern.match(path.name)
        if not m:
            continue

        identifier, norm, combo = m.groups()

        with open(path) as f:
            data = json.load(f)

        datasets = data.get("datasets", {})

        # Store metadata per combination method (alpha read from JSON)
        if combo not in results["meta"]:
            results["meta"][combo] = {
                "normalization": norm,
                "alpha": data.get("alpha", 0.0),
            }

        if identifier == "baseline":
            for dataset, metrics in datasets.items():
                results["baseline"][combo][dataset] = metrics
        else:
            # identifier is the dataset name, file has one dataset
            for dataset, metrics in datasets.items():
                results["distilled"][combo][dataset] = metrics

    return results


DATASET_DISPLAY = {
    "AILA_casedocs": "AILA Casedocs",
    "CUREv1_en": "CUREv1 (en)",
    "financebench": "FinanceBench",
    "finqa": "FinQA",
    "HC3Finance": "HC3 Finance",
    "legal_summarization": "Legal Summ.",
    "LegalQuAD": "LegalQuAD",
    "multi-cpr-video": "Multi-CPR Video",
}


def fmt(val, width=7):
    """Format a float value, or return '-' if None."""
    if val is None:
        return "-".center(width)
    return f"{val:.4f}"


def pct_delta(new_val, base_val):
    """Compute percentage delta: (new - base) / base * 100. Returns None if not computable."""
    if new_val is None or base_val is None or base_val == 0:
        return None
    return (new_val - base_val) / base_val * 100


def fmt_pct(pct):
    """Format a percentage value with sign, or '-' if None."""
    if pct is None:
        return "-"
    return f"{pct:+.1f}%"


def generate_report(baseline, bm25_results, tuned_results, combined, output_path):
    """Generate the markdown comparison report.

    All columns except Base show percentage delta relative to Base standalone.
    """
    combo_methods = sorted(combined["meta"].keys())
    base_combined = combined["baseline"]
    dist_combined = combined["distilled"]
    has_bm25 = bool(bm25_results)
    has_tuned = bool(tuned_results)
    has_dist_combined = bool(dist_combined)

    # Collect all datasets across all sources
    all_datasets = set(baseline.keys())
    if bm25_results:
        all_datasets.update(bm25_results.keys())
    if tuned_results:
        all_datasets.update(tuned_results.keys())
    for combo in combo_methods:
        all_datasets.update(base_combined.get(combo, {}).keys())
        all_datasets.update(dist_combined.get(combo, {}).keys())
    all_datasets = sorted(all_datasets)

    lines = []
    lines.append("# Combined BM25 + Semantic Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("**Baseline Model:** BAAI/bge-m3")
    if combo_methods and combo_methods[0] in combined["meta"]:
        meta = combined["meta"][combo_methods[0]]
        lines.append(f"**Normalization:** {meta['normalization']}")
        lines.append(f"**Alpha (BM25 weight):** {meta['alpha']}")
    lines.append("")
    lines.append("All values (except Base) are percentage change relative to Base.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for metric_key, metric_label in [("ndcg@10", "NDCG@10"), ("recall@10", "Recall@10")]:
        lines.append(f"## {metric_label}")
        lines.append("")

        # Build header dynamically
        header = "| Dataset | Base |"
        sep = "|---------|------|"

        if has_bm25:
            header += " BM25 |"
            sep += "------|"

        for combo in combo_methods:
            label = combo.capitalize()[:5]
            header += f" Base+BM25 ({label}) |"
            sep += "------|"

        if has_tuned:
            header += " Distilled |"
            sep += "------|"

        if has_dist_combined:
            for combo in combo_methods:
                label = combo.capitalize()[:5]
                header += f" Dist+BM25 ({label}) |"
                sep += "------|"

        lines.append(header)
        lines.append(sep)

        # Accumulators for averages
        pct_accum = defaultdict(list)
        base_vals = []

        for dataset in all_datasets:
            display = DATASET_DISPLAY.get(dataset, dataset)
            base_val = baseline.get(dataset, {}).get(metric_key)

            row = f"| {display} | {fmt(base_val)} |"

            if base_val is not None:
                base_vals.append(base_val)

            if has_bm25:
                bm25_val = bm25_results.get(dataset, {}).get(metric_key)
                pct = pct_delta(bm25_val, base_val)
                row += f" {fmt_pct(pct)} |"
                if pct is not None:
                    pct_accum["bm25"].append(pct)

            for combo in combo_methods:
                bc_val = base_combined.get(combo, {}).get(dataset, {}).get(metric_key)
                pct = pct_delta(bc_val, base_val)
                row += f" {fmt_pct(pct)} |"
                if pct is not None:
                    pct_accum[f"base_{combo}"].append(pct)

            if has_tuned:
                tuned_val = tuned_results.get(dataset, {}).get(metric_key)
                pct = pct_delta(tuned_val, base_val)
                row += f" {fmt_pct(pct)} |"
                if pct is not None:
                    pct_accum["tuned"].append(pct)

            if has_dist_combined:
                for combo in combo_methods:
                    dc_val = dist_combined.get(combo, {}).get(dataset, {}).get(metric_key)
                    pct = pct_delta(dc_val, base_val)
                    row += f" {fmt_pct(pct)} |"
                    if pct is not None:
                        pct_accum[f"dist_{combo}"].append(pct)

            lines.append(row)

        # Average row
        def _avg_pct(key):
            vals = pct_accum[key]
            return sum(vals) / len(vals) if vals else None

        avg_base = sum(base_vals) / len(base_vals) if base_vals else None
        avg_row = f"| **Average** | {fmt(avg_base)} |"

        if has_bm25:
            avg_row += f" {fmt_pct(_avg_pct('bm25'))} |"

        for combo in combo_methods:
            avg_row += f" {fmt_pct(_avg_pct(f'base_{combo}'))} |"

        if has_tuned:
            avg_row += f" {fmt_pct(_avg_pct('tuned'))} |"

        if has_dist_combined:
            for combo in combo_methods:
                avg_row += f" {fmt_pct(_avg_pct(f'dist_{combo}'))} |"

        lines.append(avg_row)
        lines.append("")

    report = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    return report


def print_console_report(baseline, bm25_results, tuned_results, combined):
    """Print a console-friendly summary."""
    combo_methods = sorted(combined["meta"].keys())
    base_combined = combined["baseline"]
    dist_combined = combined["distilled"]
    has_bm25 = bool(bm25_results)
    has_tuned = bool(tuned_results)
    has_dist_combined = bool(dist_combined)

    all_datasets = set(baseline.keys())
    if bm25_results:
        all_datasets.update(bm25_results.keys())
    if tuned_results:
        all_datasets.update(tuned_results.keys())
    for combo in combo_methods:
        all_datasets.update(base_combined.get(combo, {}).keys())
        all_datasets.update(dist_combined.get(combo, {}).keys())
    all_datasets = sorted(all_datasets)

    print("\n" + "=" * 120)
    print("COMBINED BM25 + SEMANTIC EVALUATION REPORT")
    print("All values (except Base) are % change relative to Base.")
    print("=" * 120)

    for metric_key, metric_label in [("ndcg@10", "NDCG@10"), ("recall@10", "Recall@10")]:
        print(f"\n--- {metric_label} ---\n")

        # Build header
        header = f"{'Dataset':<20} {'Base':>8}"
        if has_bm25:
            header += f" {'BM25':>8}"
        for combo in combo_methods:
            label = combo[:5].capitalize()
            header += f" {'B+BM25('+label+')':>16}"
        if has_tuned:
            header += f" {'Distilled':>10}"
        if has_dist_combined:
            for combo in combo_methods:
                label = combo[:5].capitalize()
                header += f" {'D+BM25('+label+')':>16}"
        print(header)
        print("-" * len(header))

        for dataset in all_datasets:
            base_val = baseline.get(dataset, {}).get(metric_key)
            row = f"{dataset:<20} {fmt(base_val):>8}"

            if has_bm25:
                bm25_val = bm25_results.get(dataset, {}).get(metric_key)
                row += f" {fmt_pct(pct_delta(bm25_val, base_val)):>8}"

            for combo in combo_methods:
                bc_val = base_combined.get(combo, {}).get(dataset, {}).get(metric_key)
                row += f" {fmt_pct(pct_delta(bc_val, base_val)):>16}"

            if has_tuned:
                tuned_val = tuned_results.get(dataset, {}).get(metric_key)
                row += f" {fmt_pct(pct_delta(tuned_val, base_val)):>10}"

            if has_dist_combined:
                for combo in combo_methods:
                    dc_val = dist_combined.get(combo, {}).get(dataset, {}).get(metric_key)
                    row += f" {fmt_pct(pct_delta(dc_val, base_val)):>16}"

            print(row)

        print("-" * len(header))

    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(
        description="Generate report comparing base, base+BM25, and distilled+BM25",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_combined_report.py \\
      --baseline-results results/tuned/baseline_results.json \\
      --combined-dir results/combined-distilled

  python generate_combined_report.py \\
      --baseline-results results/tuned/baseline_results.json \\
      --bm25-results results/bm25_baseline/bm25_results.json \\
      --tuned-results results/distilled_gemma2_500steps/tuned_results.json \\
      --combined-dir results/combined-distilled \\
      --output results/combined-distilled/report.md
"""
    )

    parser.add_argument(
        "--baseline-results",
        type=Path,
        required=True,
        help="Path to baseline_results.json (standalone dense model, no BM25)"
    )
    parser.add_argument(
        "--bm25-results",
        type=Path,
        default=None,
        help="Path to BM25-only results JSON (optional, adds BM25 column)"
    )
    parser.add_argument(
        "--tuned-results",
        type=Path,
        default=None,
        help="Path to tuned/distilled model standalone results JSON (optional, adds Distilled column)"
    )
    parser.add_argument(
        "--combined-dir",
        type=Path,
        required=True,
        help="Directory containing combined evaluation result JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown file path (default: combined-dir/report.md)"
    )

    args = parser.parse_args()

    if not args.combined_dir.exists():
        print(f"Error: Combined results directory not found: {args.combined_dir}")
        return

    output_path = args.output or (args.combined_dir / "report.md")

    # Load data
    baseline = load_baseline_results(args.baseline_results)
    bm25 = load_baseline_results(args.bm25_results) if args.bm25_results else {}
    tuned = load_baseline_results(args.tuned_results) if args.tuned_results else {}
    combined = load_combined_results(args.combined_dir)

    if not combined["baseline"] and not combined["distilled"]:
        print("Error: No combined results found")
        return

    # Print console report
    print_console_report(baseline, bm25, tuned, combined)

    # Generate markdown report
    generate_report(baseline, bm25, tuned, combined, output_path)
    print(f"\nMarkdown report saved to: {output_path}")


if __name__ == "__main__":
    main()
