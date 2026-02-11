#!/usr/bin/env python3
"""Generate evaluation report comparing baseline vs fine-tuned models."""

import argparse
import json
from pathlib import Path
from datetime import datetime


def load_results(results_dir: Path) -> dict:
    """Load all result files from the results directory."""
    results = {
        "baseline": {},
        "finetuned": {}
    }

    # Load baseline results
    baseline_path = results_dir / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            data = json.load(f)
            results["baseline"] = data.get("datasets", {})

    # Load fine-tuned results
    for path in results_dir.glob("finetuned_*.json"):
        with open(path) as f:
            data = json.load(f)
            for dataset, metrics in data.get("datasets", {}).items():
                results["finetuned"][dataset] = metrics

    return results


def generate_report(results: dict, output_path: Path) -> str:
    """Generate a markdown report comparing baseline vs fine-tuned models."""

    report_lines = []
    report_lines.append("# Fine-Tuning Evaluation Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("**Baseline Model:** BAAI/bge-m3")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Summary table
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append("| Dataset | Baseline NDCG@10 | Fine-tuned NDCG@10 | Δ NDCG@10 | Baseline Recall@10 | Fine-tuned Recall@10 | Δ Recall@10 |")
    report_lines.append("|---------|------------------|--------------------|-----------|--------------------|----------------------|-------------|")

    total_ndcg_delta = 0
    total_recall_delta = 0
    count = 0

    # Map dataset names for display
    dataset_display_names = {
        "AILA_casedocs": "AILA Casedocs",
        "financebench": "FinanceBench",
        "finqa": "FinQA",
        "HC3Finance": "HC3 Finance",
        "LeCaRDv2": "LeCaRD v2",
        "legal_summarization": "Legal Summarization",
        "LegalQuAD": "LegalQuAD",
        "AILA_statutes": "AILA Statutes",
        "cuad": "CUAD",
    }

    for dataset in sorted(results["finetuned"].keys()):
        baseline = results["baseline"].get(dataset, {})
        finetuned = results["finetuned"].get(dataset, {})

        if not baseline or not finetuned:
            continue

        baseline_ndcg = baseline.get("ndcg@10", 0)
        finetuned_ndcg = finetuned.get("ndcg@10", 0)
        delta_ndcg = finetuned_ndcg - baseline_ndcg

        baseline_recall = baseline.get("recall@10", 0)
        finetuned_recall = finetuned.get("recall@10", 0)
        delta_recall = finetuned_recall - baseline_recall

        total_ndcg_delta += delta_ndcg
        total_recall_delta += delta_recall
        count += 1

        # Format delta with color indicator
        ndcg_indicator = "✅" if delta_ndcg > 0 else "❌" if delta_ndcg < 0 else "➖"
        recall_indicator = "✅" if delta_recall > 0 else "❌" if delta_recall < 0 else "➖"

        display_name = dataset_display_names.get(dataset, dataset)

        report_lines.append(
            f"| {display_name} | {baseline_ndcg:.4f} | {finetuned_ndcg:.4f} | {delta_ndcg:+.4f} {ndcg_indicator} | "
            f"{baseline_recall:.4f} | {finetuned_recall:.4f} | {delta_recall:+.4f} {recall_indicator} |"
        )

    # Add average row
    if count > 0:
        avg_ndcg_delta = total_ndcg_delta / count
        avg_recall_delta = total_recall_delta / count
        avg_ndcg_indicator = "✅" if avg_ndcg_delta > 0 else "❌" if avg_ndcg_delta < 0 else "➖"
        avg_recall_indicator = "✅" if avg_recall_delta > 0 else "❌" if avg_recall_delta < 0 else "➖"

        report_lines.append("|---------|------------------|--------------------|-----------|--------------------|----------------------|-------------|")
        report_lines.append(
            f"| **Average** | - | - | **{avg_ndcg_delta:+.4f}** {avg_ndcg_indicator} | "
            f"- | - | **{avg_recall_delta:+.4f}** {avg_recall_indicator} |"
        )

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Detailed results per dataset
    report_lines.append("## Detailed Results")
    report_lines.append("")

    for dataset in sorted(results["finetuned"].keys()):
        baseline = results["baseline"].get(dataset, {})
        finetuned = results["finetuned"].get(dataset, {})

        if not baseline or not finetuned:
            continue

        display_name = dataset_display_names.get(dataset, dataset)
        report_lines.append(f"### {display_name}")
        report_lines.append("")

        baseline_ndcg = baseline.get("ndcg@10", 0)
        finetuned_ndcg = finetuned.get("ndcg@10", 0)
        delta_ndcg = finetuned_ndcg - baseline_ndcg
        pct_ndcg = (delta_ndcg / baseline_ndcg * 100) if baseline_ndcg > 0 else 0

        baseline_recall = baseline.get("recall@10", 0)
        finetuned_recall = finetuned.get("recall@10", 0)
        delta_recall = finetuned_recall - baseline_recall
        pct_recall = (delta_recall / baseline_recall * 100) if baseline_recall > 0 else 0

        report_lines.append(f"- **Queries:** {finetuned.get('num_queries', 'N/A')}")
        report_lines.append(f"- **Corpus size:** {finetuned.get('num_corpus', 'N/A')}")
        report_lines.append("")
        report_lines.append("| Metric | Baseline | Fine-tuned | Delta | % Change |")
        report_lines.append("|--------|----------|------------|-------|----------|")
        report_lines.append(f"| NDCG@10 | {baseline_ndcg:.4f} | {finetuned_ndcg:.4f} | {delta_ndcg:+.4f} | {pct_ndcg:+.2f}% |")
        report_lines.append(f"| Recall@10 | {baseline_recall:.4f} | {finetuned_recall:.4f} | {delta_recall:+.4f} | {pct_recall:+.2f}% |")
        report_lines.append("")

    # Write report
    report_text = "\n".join(report_lines)

    with open(output_path, "w") as f:
        f.write(report_text)

    return report_text


def print_console_report(results: dict):
    """Print a console-friendly report."""
    print("\n" + "=" * 100)
    print("FINE-TUNING EVALUATION REPORT")
    print("=" * 100)
    print(f"\nBaseline Model: BAAI/bge-m3")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "-" * 100)
    print(f"{'Dataset':<25} {'Base NDCG@10':>14} {'FT NDCG@10':>14} {'Δ NDCG@10':>12} {'Base Recall@10':>16} {'FT Recall@10':>14} {'Δ Recall@10':>14}")
    print("-" * 100)

    total_ndcg_delta = 0
    total_recall_delta = 0
    count = 0

    for dataset in sorted(results["finetuned"].keys()):
        baseline = results["baseline"].get(dataset, {})
        finetuned = results["finetuned"].get(dataset, {})

        if not baseline or not finetuned:
            continue

        baseline_ndcg = baseline.get("ndcg@10", 0)
        finetuned_ndcg = finetuned.get("ndcg@10", 0)
        delta_ndcg = finetuned_ndcg - baseline_ndcg

        baseline_recall = baseline.get("recall@10", 0)
        finetuned_recall = finetuned.get("recall@10", 0)
        delta_recall = finetuned_recall - baseline_recall

        total_ndcg_delta += delta_ndcg
        total_recall_delta += delta_recall
        count += 1

        print(f"{dataset:<25} {baseline_ndcg:>14.4f} {finetuned_ndcg:>14.4f} {delta_ndcg:>+12.4f} {baseline_recall:>16.4f} {finetuned_recall:>14.4f} {delta_recall:>+14.4f}")

    print("-" * 100)

    if count > 0:
        avg_ndcg_delta = total_ndcg_delta / count
        avg_recall_delta = total_recall_delta / count
        print(f"{'AVERAGE':<25} {'-':>14} {'-':>14} {avg_ndcg_delta:>+12.4f} {'-':>16} {'-':>14} {avg_recall_delta:>+14.4f}")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing evaluation result JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown file path (default: results_dir/report.md)"
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return

    output_path = args.output or (args.results_dir / "report.md")

    # Load results
    results = load_results(args.results_dir)

    if not results["finetuned"]:
        print("Error: No fine-tuned results found")
        return

    # Print console report
    print_console_report(results)

    # Generate markdown report
    report = generate_report(results, output_path)
    print(f"\nMarkdown report saved to: {output_path}")


if __name__ == "__main__":
    main()
