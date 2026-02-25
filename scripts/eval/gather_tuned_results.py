#!/usr/bin/env python3
"""Gather per-dataset result files into a single tuned_results.json.

Scans a results directory for per-dataset JSON files (any file with a
"datasets" key that is not baseline_results.json) and merges them into
one consolidated file matching the baseline_results.json format.

Usage:
    python gather_tuned_results.py results/distilled_gemma2_500steps
    python gather_tuned_results.py results/distilled_gemma2_500steps -o tuned_results.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Gather per-dataset result files into a single tuned_results.json",
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing per-dataset result JSON files",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="tuned_results.json",
        help="Output filename (written inside results_dir, default: tuned_results.json)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory")
        return

    output_path = results_dir / args.output
    merged_datasets = {}

    for path in sorted(results_dir.glob("*.json")):
        if path.name in ("baseline_results.json", args.output):
            continue

        with open(path) as f:
            data = json.load(f)

        datasets = data.get("datasets", {})
        if not datasets:
            continue

        for ds_name, metrics in datasets.items():
            merged_datasets[ds_name] = metrics
            print(f"  {ds_name} <- {path.name}")

    if not merged_datasets:
        print("No per-dataset result files found.")
        return

    output_data = {"datasets": merged_datasets}

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nGathered {len(merged_datasets)} datasets -> {output_path}")


if __name__ == "__main__":
    main()
