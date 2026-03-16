"""SageMaker entry point for sparse model evaluation.

Thin wrapper around scripts/eval/evaluate_model.py — all evaluation logic
lives there. This script only handles SageMaker-specific concerns:
  - Extracting model from tar.gz
  - Finding model root directory
  - Adapting SageMaker channel layout to evaluate_model's expected data layout
  - Saving results to SM_MODEL_DIR

Channels:
- /opt/ml/input/data/eval/     -- eval dataset (corpus.jsonl, queries.jsonl, relevance.jsonl)
- /opt/ml/input/data/model/    -- trained model (extracted from model.tar.gz)

Output:
- /opt/ml/model/results.json   -- evaluation metrics
"""

import argparse
import json
import os
import tarfile
from pathlib import Path

# evaluate_model.py is packaged alongside this script in the SM source tarball.
from evaluate_model import evaluate_model, load_model


def find_model_root(model_path: Path) -> Path:
    """Find the SparseEncoder model root directory."""
    st_configs = list(model_path.rglob("config_sentence_transformers.json"))
    if st_configs:
        return st_configs[0].parent
    modules = list(model_path.rglob("modules.json"))
    if modules:
        return modules[0].parent
    if (model_path / "config.json").exists():
        return model_path
    config_files = list(model_path.rglob("config.json"))
    if config_files:
        return config_files[0].parent
    return model_path


def extract_if_tarball(model_path: Path) -> Path:
    """Extract tar.gz if present, return usable model path."""
    tar_files = list(model_path.glob("*.tar.gz"))
    if tar_files:
        tar_file = tar_files[0]
        print(f"Extracting {tar_file}...")
        extract_dir = Path("/tmp/trained_model")
        if extract_dir.exists():
            import shutil
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(extract_dir)
        return extract_dir
    return model_path


def setup_data_dir(eval_channel: Path, dataset_name: str) -> Path:
    """Create directory layout expected by evaluate_model.load_eval_data.

    evaluate_model expects: data_dir/dataset_name/{corpus,queries,relevance}.jsonl
    SageMaker provides:     /opt/ml/input/data/eval/{corpus,queries,relevance}.jsonl

    Create a symlink so evaluate_model can find the data.
    """
    data_dir = Path("/tmp/eval_data")
    dataset_dir = data_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("corpus.jsonl", "queries.jsonl", "relevance.jsonl"):
        src = (eval_channel / fname).resolve()
        dst = dataset_dir / fname
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    return data_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_baseline", type=str, default="false")
    parser.add_argument("--baseline_model_name", type=str,
                        default="opensearch-project/opensearch-neural-sparse-encoding-v2-distill")
    parser.add_argument("--model_dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--eval_dir", type=str,
                        default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/input/data/eval"))
    parser.add_argument("--trained_model_dir", type=str,
                        default=os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model"))
    return parser.parse_args()


def main():
    args = parse_args()

    # Set up data directory layout for evaluate_model
    eval_channel = Path(args.eval_dir)
    data_dir = setup_data_dir(eval_channel, args.dataset_name)

    all_results = {}

    # Evaluate baseline
    if args.eval_baseline.lower() in ("true", "1", "yes"):
        print(f"\n=== Evaluating baseline: {args.baseline_model_name} ===")
        baseline_model = load_model(args.baseline_model_name, max_seq_length=4096, sparse=True)
        result = evaluate_model(
            baseline_model, args.dataset_name, data_dir, args.batch_size
        )
        all_results["baseline"] = {
            "ndcg@10": result.ndcg_at_10, "mrr@10": result.mrr_at_10,
            "recall@10": result.recall_at_10,
            "num_queries": result.num_queries, "num_corpus": result.num_corpus,
        }
        del baseline_model
        import torch
        torch.cuda.empty_cache()

    # Load and evaluate trained model
    model_path = extract_if_tarball(Path(args.trained_model_dir))
    model_path = find_model_root(model_path)
    print(f"\n=== Evaluating trained model: {model_path} ===")
    print(f"  Contents: {sorted(f.name for f in model_path.iterdir())}")

    trained_model = load_model(str(model_path), max_seq_length=4096, sparse=True)
    result = evaluate_model(
        trained_model, args.dataset_name, data_dir, args.batch_size
    )
    all_results["trained"] = {
        "ndcg@10": result.ndcg_at_10, "mrr@10": result.mrr_at_10,
        "recall@10": result.recall_at_10,
        "num_queries": result.num_queries, "num_corpus": result.num_corpus,
    }

    # Save results
    output_path = Path(args.model_dir) / "results.json"
    with open(output_path, "w") as f:
        json.dump({"dataset": args.dataset_name, "results": all_results}, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if "baseline" in all_results and "trained" in all_results:
        b = all_results["baseline"]["ndcg@10"]
        t = all_results["trained"]["ndcg@10"]
        pct = ((t - b) / b * 100) if b > 0 else 0
        print(f"\nSummary: baseline={b:.4f}, trained={t:.4f}, delta={pct:+.1f}%")


if __name__ == "__main__":
    main()
