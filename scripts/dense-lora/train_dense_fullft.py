"""SageMaker entry point for dense model full fine-tuning.

Same pipeline as train_dense_lora.py but trains all model parameters
instead of LoRA adapters. Uses BestModelSaver to save the full model
checkpoint on dev metric improvement.

SageMaker conventions:
- Training data: /opt/ml/input/data/training/ (scored_training_data.json)
- Dev data:      /opt/ml/input/data/dev/ (dev_data.jsonl + dev_labels.jsonl)
- Model output:  /opt/ml/model/
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import DistillKLDivLoss
from transformers import EarlyStoppingCallback, TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments as _TA

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


class BestModelSaver(TrainerCallback):
    """Save the full model on eval improvement.

    Overwrites the previous best to keep disk usage bounded (~1.2GB per save).
    """

    def __init__(self, save_dir, metric_name="eval_dev_cosine_ndcg@10"):
        self.save_dir = save_dir
        self.metric_name = metric_name
        self.best_metric = -float("inf")
        self.best_step = None

    def on_evaluate(self, args: _TA, state: TrainerState, control: TrainerControl, model=None, metrics=None, **kwargs):
        if metrics is None:
            return
        val = metrics.get(self.metric_name)
        if val is not None and val > self.best_metric:
            self.best_metric = val
            self.best_step = state.global_step
            if model is not None:
                label = "baseline" if state.global_step == 0 else "new best"
                print(f"Saving {label} model at step {self.best_step}: {self.metric_name}={val:.4f}")
                # Clear previous best
                if os.path.isdir(self.save_dir):
                    shutil.rmtree(self.save_dir)
                model.save_pretrained(self.save_dir)


def load_and_convert(data_dir, num_negatives, score_scale=0.025, score_scale_mode="multiply"):
    """Load teacher-scored JSON data and convert to KLDiv training format."""
    queries, positives, labels = [], [], []
    negatives_cols = {f"negative{i+1}": [] for i in range(num_negatives)}
    n_skipped = 0

    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    all_data = []
    for json_path in json_files:
        print(f"Loading {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            all_data.extend(json.load(f))

    for ex in all_data:
        q = ex["query"]
        docs = ex["docs"]
        raw_scores = ex["scores"]
        if not docs or not raw_scores or len(docs) != len(raw_scores):
            n_skipped += 1
            continue
        if len(docs) < 1 + num_negatives:
            n_skipped += 1
            continue

        scores = [float(s) for s in raw_scores]

        if score_scale_mode == "multiply":
            scores = [s * score_scale for s in scores]

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        pos_doc = docs[best_idx]
        pos_score = float(scores[best_idx])
        rest = [(docs[i], float(scores[i])) for i in range(len(docs)) if i != best_idx]
        rest.sort(key=lambda x: x[1], reverse=True)

        offset = 0
        while offset + num_negatives <= len(rest):
            chunk = rest[offset:offset + num_negatives]
            offset += num_negatives
            queries.append(q)
            positives.append(pos_doc)
            labels.append([pos_score] + [s for (_, s) in chunk])
            for i, (neg_doc, _) in enumerate(chunk):
                negatives_cols[f"negative{i+1}"].append(neg_doc)

    data = {"query": queries, "positive": positives, **negatives_cols, "label": labels}
    ds = Dataset.from_dict(data)
    ds = ds.map(lambda b: {"label": torch.tensor(b["label"], dtype=torch.float32)}, batched=False)

    print(f"Loaded {len(ds)} training examples; skipped {n_skipped} queries")
    return ds


def load_dev_evaluator(dev_dir):
    """Load dev set and build an InformationRetrievalEvaluator."""
    dev_path = Path(dev_dir)
    dev_data_file = dev_path / "dev_data.jsonl"
    dev_labels_file = dev_path / "dev_labels.jsonl"

    if not dev_data_file.exists() or not dev_labels_file.exists():
        print(f"Dev files not found in {dev_dir}, skipping dev evaluation")
        return None

    queries = {}
    corpus = {}
    query_to_qid = {}

    with open(dev_data_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            entry = json.loads(line)
            qid = f"q{line_num}"
            queries[qid] = entry["query"]
            query_to_qid[entry["query"]] = qid
            for cand in entry["candidates"]:
                corpus[cand["id"]] = cand["text"]

    relevant_docs = defaultdict(set)
    with open(dev_labels_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            qid = query_to_qid.get(entry["query"])
            if qid and entry["llm_score"] >= 0.5:
                relevant_docs[qid].add(entry["candidate_id"])

    relevant_docs = dict(relevant_docs)
    queries = {qid: q for qid, q in queries.items() if qid in relevant_docs}

    if not queries:
        print("No queries with relevant docs found, skipping dev evaluation")
        return None

    print(f"Dev evaluator: {len(queries)} queries, {len(corpus)} corpus docs, "
          f"{sum(len(v) for v in relevant_docs.values())} relevant pairs")

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="dev",
        ndcg_at_k=[10],
        mrr_at_k=[10],
        accuracy_at_k=[10],
        precision_recall_at_k=[10],
        map_at_k=[100],
        batch_size=32,
    )
    return evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Alibaba-NLP/gte-multilingual-base")
    parser.add_argument("--score_scale", type=float, default=0.025)
    parser.add_argument("--score_scale_mode", type=str, default="multiply")
    parser.add_argument("--num_negatives", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--eval_on_start", type=str, default="true")
    # SageMaker environment
    parser.add_argument("--model_dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--training_dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    parser.add_argument("--dev_dir", type=str,
                        default=os.environ.get("SM_CHANNEL_DEV", "/opt/ml/input/data/dev"))
    return parser.parse_args()


def main():
    args = parse_args()

    eval_on_start = args.eval_on_start.lower() in ("true", "1", "yes")

    print(f"Config: model={args.model_name}, FULL FINE-TUNE (no LoRA)")
    print(f"  batch_size={args.train_batch_size}, grad_accum={args.gradient_accumulation_steps}, "
          f"effective_batch={args.train_batch_size * args.gradient_accumulation_steps}")
    print(f"  lr={args.learning_rate}, max_steps={args.max_steps}")
    print(f"  score_scale={args.score_scale}, score_scale_mode={args.score_scale_mode}")
    print(f"  eval_steps={args.eval_steps}, early_stopping_patience={args.early_stopping_patience}")

    # Load data
    train_dataset = load_and_convert(
        args.training_dir, args.num_negatives,
        score_scale=args.score_scale, score_scale_mode=args.score_scale_mode,
    )

    # Load dev evaluator
    dev_evaluator = load_dev_evaluator(args.dev_dir)

    # Load model (no LoRA — train all parameters)
    model = SentenceTransformer(args.model_name, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} (100%)")

    # Build loss
    loss = DistillKLDivLoss(model=model)
    print(f"Using DistillKLDivLoss (scale={args.score_scale}, mode={args.score_scale_mode})")

    output_dir = "/opt/ml/output/intermediate"

    # Configure eval strategy
    eval_kwargs = {}
    callbacks = []
    best_model_dir = "/opt/ml/output/best_model"
    model_saver = None
    if dev_evaluator is not None:
        eval_kwargs["eval_strategy"] = "steps"
        eval_kwargs["eval_steps"] = args.eval_steps
        eval_kwargs["eval_on_start"] = eval_on_start
        eval_kwargs["metric_for_best_model"] = "dev_cosine_ndcg@10"
        eval_kwargs["greater_is_better"] = True
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        model_saver = BestModelSaver(save_dir=best_model_dir)
        callbacks.append(model_saver)
        print(f"Dev evaluation enabled: eval every {args.eval_steps} steps, "
              f"early stopping patience={args.early_stopping_patience}, "
              f"eval_on_start={eval_on_start}")

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=100,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_strategy="no",
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        **eval_kwargs,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=dev_evaluator,
        callbacks=callbacks,
    )

    trainer.train()

    # Save to SM_MODEL_DIR
    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main:
        if model_saver is not None and model_saver.best_step is not None and os.path.isdir(best_model_dir):
            print(f"Using best model from step {model_saver.best_step} "
                  f"(NDCG@10={model_saver.best_metric:.4f})")
            # Copy best model to output
            shutil.copytree(best_model_dir, args.model_dir, dirs_exist_ok=True)
        else:
            print("No best model saved, using final model")
            model.save_pretrained(args.model_dir)

        config = {
            "model_name": args.model_name,
            "fine_tune_mode": "full",
            "num_negatives": args.num_negatives,
            "loss_type": "kldiv",
            "score_scale": args.score_scale,
            "score_scale_mode": args.score_scale_mode,
            "eval_on_start": eval_on_start,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "eval_steps": args.eval_steps,
            "early_stopping_patience": args.early_stopping_patience,
            "best_step": model_saver.best_step if model_saver else None,
            "best_dev_ndcg10": model_saver.best_metric if model_saver else None,
        }
        with open(os.path.join(args.model_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved model to: {args.model_dir}")


if __name__ == "__main__":
    main()
