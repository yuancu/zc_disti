#!/usr/bin/env python3
"""Generate evaluation report comparing base, base+BM25, and distilled+BM25 results.

Produces a single markdown report with two key comparisons:
1. Base vs Base+BM25 - does combining with BM25 help?
2. Base+BM25 vs Distilled+BM25 - does distillation help in the combined setting?

Expected input:
- --baseline-results: Path to baseline_results.json (standalone BAAI/bge-m3, no BM25)
- --combined-dir: Directory with combined evaluation results, containing:
    - combined_baseline_*.json (base model + BM25)
    - combined_{dataset}_*.json (distilled model + BM25)

Usage:
    python generate_combined_report.py \\
        --baseline-results results/tuned/baseline_results.json \\
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

    # Pattern: combined_{identifier}_{norm}_{combo}_alpha{alpha}.json
    pattern = re.compile(
        r"^combined_(.+?)_(min_max|l2)_(arithmetic|geometric|harmonic)_alpha([\d.]+)\.json$"
    )

    for path in sorted(combined_dir.glob("combined_*.json")):
        m = pattern.match(path.name)
        if not m:
            continue

        identifier, norm, combo, alpha = m.groups()

        with open(path) as f:
            data = json.load(f)

        datasets = data.get("datasets", {})

        # Store metadata per combination method
        if combo not in results["meta"]:
            results["meta"][combo] = {
                "normalization": norm,
                "alpha": float(alpha),
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


def fmt_delta(val, width=7):
    """Format a delta value with sign, or '-' if None."""
    if val is None:
        return "-".center(width)
    return f"{val:+.4f}"


def generate_report(baseline, combined, output_path):
    """Generate the markdown comparison report."""
    combo_methods = sorted(combined["meta"].keys())
    base_combined = combined["baseline"]
    dist_combined = combined["distilled"]

    # Collect all datasets across all sources
    all_datasets = set()
    all_datasets.update(baseline.keys())
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
    lines.append("**Legend:**")
    lines.append("- **Base** = BAAI/bge-m3 dense retrieval only (no BM25)")
    lines.append("- **Base+BM25** = BAAI/bge-m3 combined with BM25")
    lines.append("- **Dist+BM25** = Distilled model combined with BM25")
    lines.append("")
    lines.append("---")
    lines.append("")

    for metric_key, metric_label in [("ndcg@10", "NDCG@10"), ("recall@10", "Recall@10")]:
        lines.append(f"## {metric_label}")
        lines.append("")

        # Build header
        header = "| Dataset | Base |"
        sep = "|---------|------|"
        for combo in combo_methods:
            label = combo.capitalize()[:5]
            header += f" Base+BM25 ({label}) | {chr(916)} |"
            sep += "------------|------|"
            header += f" Dist+BM25 ({label}) | {chr(916)} |"
            sep += "------------|------|"

        lines.append(header)
        lines.append(sep)

        # Accumulators for averages
        base_vals = []
        base_combo_vals = {c: [] for c in combo_methods}
        dist_combo_vals = {c: [] for c in combo_methods}
        base_combo_deltas = {c: [] for c in combo_methods}
        dist_combo_deltas = {c: [] for c in combo_methods}

        for dataset in all_datasets:
            display = DATASET_DISPLAY.get(dataset, dataset)
            base_val = baseline.get(dataset, {}).get(metric_key)

            row = f"| {display} | {fmt(base_val)} |"

            if base_val is not None:
                base_vals.append(base_val)

            for combo in combo_methods:
                bc_val = base_combined.get(combo, {}).get(dataset, {}).get(metric_key)
                dc_val = dist_combined.get(combo, {}).get(dataset, {}).get(metric_key)

                # Delta: base-combined vs base
                bc_delta = (bc_val - base_val) if (bc_val is not None and base_val is not None) else None
                # Delta: distilled-combined vs base-combined
                dc_delta = (dc_val - bc_val) if (dc_val is not None and bc_val is not None) else None

                row += f" {fmt(bc_val)} | {fmt_delta(bc_delta)} |"
                row += f" {fmt(dc_val)} | {fmt_delta(dc_delta)} |"

                if bc_val is not None:
                    base_combo_vals[combo].append(bc_val)
                if dc_val is not None:
                    dist_combo_vals[combo].append(dc_val)
                if bc_delta is not None:
                    base_combo_deltas[combo].append(bc_delta)
                if dc_delta is not None:
                    dist_combo_deltas[combo].append(dc_delta)

            lines.append(row)

        # Average row
        avg_base = sum(base_vals) / len(base_vals) if base_vals else None
        avg_row = f"| **Average** | {fmt(avg_base)} |"
        for combo in combo_methods:
            avg_bc = sum(base_combo_vals[combo]) / len(base_combo_vals[combo]) if base_combo_vals[combo] else None
            avg_dc = sum(dist_combo_vals[combo]) / len(dist_combo_vals[combo]) if dist_combo_vals[combo] else None
            avg_bc_d = sum(base_combo_deltas[combo]) / len(base_combo_deltas[combo]) if base_combo_deltas[combo] else None
            avg_dc_d = sum(dist_combo_deltas[combo]) / len(dist_combo_deltas[combo]) if dist_combo_deltas[combo] else None
            avg_row += f" {fmt(avg_bc)} | {fmt_delta(avg_bc_d)} |"
            avg_row += f" {fmt(avg_dc)} | {fmt_delta(avg_dc_d)} |"
        lines.append(avg_row)

        lines.append("")

    report = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    return report


def print_console_report(baseline, combined):
    """Print a console-friendly summary."""
    combo_methods = sorted(combined["meta"].keys())
    base_combined = combined["baseline"]
    dist_combined = combined["distilled"]

    all_datasets = set()
    all_datasets.update(baseline.keys())
    for combo in combo_methods:
        all_datasets.update(base_combined.get(combo, {}).keys())
        all_datasets.update(dist_combined.get(combo, {}).keys())
    all_datasets = sorted(all_datasets)

    print("\n" + "=" * 120)
    print("COMBINED BM25 + SEMANTIC EVALUATION REPORT")
    print("=" * 120)

    for metric_key, metric_label in [("ndcg@10", "NDCG@10"), ("recall@10", "Recall@10")]:
        print(f"\n--- {metric_label} ---\n")

        # Header
        header = f"{'Dataset':<20} {'Base':>8}"
        for combo in combo_methods:
            label = combo[:5].capitalize()
            header += f" {'B+BM25('+label+')':>16} {'Delta':>8}"
            header += f" {'D+BM25('+label+')':>16} {'Delta':>8}"
        print(header)
        print("-" * len(header))

        for dataset in all_datasets:
            base_val = baseline.get(dataset, {}).get(metric_key)
            row = f"{dataset:<20} {fmt(base_val):>8}"

            for combo in combo_methods:
                bc_val = base_combined.get(combo, {}).get(dataset, {}).get(metric_key)
                dc_val = dist_combined.get(combo, {}).get(dataset, {}).get(metric_key)
                bc_delta = (bc_val - base_val) if (bc_val is not None and base_val is not None) else None
                dc_delta = (dc_val - bc_val) if (dc_val is not None and bc_val is not None) else None

                row += f" {fmt(bc_val):>16} {fmt_delta(bc_delta):>8}"
                row += f" {fmt(dc_val):>16} {fmt_delta(dc_delta):>8}"

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
    combined = load_combined_results(args.combined_dir)

    if not combined["baseline"] and not combined["distilled"]:
        print("Error: No combined results found")
        return

    # Print console report
    print_console_report(baseline, combined)

    # Generate markdown report
    generate_report(baseline, combined, output_path)
    print(f"\nMarkdown report saved to: {output_path}")


if __name__ == "__main__":
    main()
