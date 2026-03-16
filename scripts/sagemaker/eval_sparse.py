"""SageMaker entry point for sparse model evaluation.

Channels:
- /opt/ml/input/data/eval/     — eval dataset (corpus.jsonl, queries.jsonl, relevance.jsonl)
- /opt/ml/input/data/model/    — trained model (extracted from model.tar.gz)
- /opt/ml/input/data/baseline/ — baseline model (extracted from model.tar.gz), optional

Output:
- /opt/ml/model/results.json   — evaluation metrics
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.evaluation import InformationRetrievalEvaluator


def load_eval_data(data_dir: Path):
    corpus = {}
    with open(data_dir / "corpus.jsonl", "r") as f:
        for line in f:
            doc = json.loads(line.strip())
            doc_id = doc.get("_id") or doc.get("id")
            text = doc.get("text", "")
            title = doc.get("title", "")
            corpus[doc_id] = f"{title} {text}".strip() if title else text

    queries = {}
    with open(data_dir / "queries.jsonl", "r") as f:
        for line in f:
            q = json.loads(line.strip())
            qid = q.get("_id") or q.get("id")
            queries[qid] = q["text"]

    qrels = {}
    with open(data_dir / "relevance.jsonl", "r") as f:
        for line in f:
            rel = json.loads(line.strip())
            qid = str(rel["query-id"])
            did = str(rel["corpus-id"])
            score = int(rel["score"])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = score

    return queries, corpus, qrels


def evaluate_model(model, dataset_name, queries, corpus, qrels, batch_size):
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    print(f"Evaluating {dataset_name}: {len(queries)} queries, {len(corpus)} docs")

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=qrels,
        name=f"{dataset_name}-test",
        show_progress_bar=True,
        batch_size=batch_size,
        corpus_chunk_size=10_000,
    )

    # Patch for sparse models: strip truncate_dim, densify embeddings
    def embed_inputs_sparse(model, sentences, *, encode_fn_name=None, is_query=False, convert_to_tensor=True, **kwargs):
        use_query = encode_fn_name == "query" or is_query
        encode_fn = model.encode_query if use_query else model.encode
        encode_chunk = 500
        if isinstance(sentences, dict):
            embs = encode_fn(sentences, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=convert_to_tensor)
            if embs.is_sparse:
                embs = embs.to_dense()
            return embs.cpu()
        all_parts = []
        for i in range(0, len(sentences), encode_chunk):
            chunk = sentences[i:i + encode_chunk]
            embs = encode_fn(chunk, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=convert_to_tensor)
            if embs.is_sparse:
                embs = embs.to_dense()
            all_parts.append(embs.cpu())
        return torch.cat(all_parts, dim=0)

    evaluator.embed_inputs = embed_inputs_sparse

    results = evaluator(model)

    sim_name = model.similarity_fn_name.value if hasattr(model.similarity_fn_name, "value") else str(model.similarity_fn_name)
    ndcg = results.get(f"{dataset_name}-test_{sim_name}_ndcg@10", 0.0)
    mrr = results.get(f"{dataset_name}-test_{sim_name}_mrr@10", 0.0)
    recall = results.get(f"{dataset_name}-test_{sim_name}_recall@10", 0.0)

    print(f"  NDCG@10: {ndcg:.4f}, MRR@10: {mrr:.4f}, Recall@10: {recall:.4f}")
    return {"ndcg@10": ndcg, "mrr@10": mrr, "recall@10": recall, "num_queries": len(queries), "num_corpus": len(corpus)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_baseline", type=str, default="false",
                        help="Also evaluate baseline model (true/false, SageMaker passes as string)")
    parser.add_argument("--baseline_model_name", type=str,
                        default="opensearch-project/opensearch-neural-sparse-encoding-v2-distill")
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--eval_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/input/data/eval"))
    parser.add_argument("--trained_model_dir", type=str, default=os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model"))
    return parser.parse_args()


def main():
    args = parse_args()

    eval_dir = Path(args.eval_dir)
    queries, corpus, qrels = load_eval_data(eval_dir)

    all_results = {}

    # Evaluate baseline (download from HuggingFace)
    if args.eval_baseline.lower() in ("true", "1", "yes"):
        print(f"\n=== Evaluating baseline: {args.baseline_model_name} ===")
        baseline_model = SparseEncoder(args.baseline_model_name, trust_remote_code=True)
        all_results["baseline"] = evaluate_model(
            baseline_model, args.dataset_name, queries, corpus, qrels, args.batch_size
        )
        del baseline_model
        torch.cuda.empty_cache()

    # Evaluate trained model
    # SageMaker may not auto-extract model.tar.gz — do it manually if needed.
    import tarfile
    model_path = Path(args.trained_model_dir)
    tar_file = model_path / "model.tar.gz"
    if tar_file.exists():
        print(f"Extracting {tar_file}...")
        extract_dir = Path("/tmp/trained_model")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(extract_dir)
        model_path = extract_dir
    # Find model root: prefer config_sentence_transformers.json (SparseEncoder models),
    # fall back to modules.json, then config.json (but only at top level to avoid
    # picking up sub-module configs like SpladePooling/config.json).
    st_configs = list(model_path.rglob("config_sentence_transformers.json"))
    if st_configs:
        model_path = st_configs[0].parent
    elif list(model_path.rglob("modules.json")):
        model_path = list(model_path.rglob("modules.json"))[0].parent
    elif (model_path / "config.json").exists():
        pass  # already at the right level
    else:
        config_files = list(model_path.rglob("config.json"))
        if config_files:
            model_path = config_files[0].parent
    print(f"\n=== Evaluating trained model: {model_path} ===")
    print(f"  Contents: {[f.name for f in model_path.iterdir()]}")
    trained_model = SparseEncoder(str(model_path), trust_remote_code=True)
    all_results["trained"] = evaluate_model(
        trained_model, args.dataset_name, queries, corpus, qrels, args.batch_size
    )

    # Save results to model output dir
    output_path = Path(args.model_dir) / "results.json"
    with open(output_path, "w") as f:
        json.dump({"dataset": args.dataset_name, "results": all_results}, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    if "baseline" in all_results and "trained" in all_results:
        b = all_results["baseline"]["ndcg@10"]
        t = all_results["trained"]["ndcg@10"]
        pct = ((t - b) / b * 100) if b > 0 else 0
        print(f"\nSummary: baseline={b:.4f}, trained={t:.4f}, delta={pct:+.1f}%")


if __name__ == "__main__":
    main()
