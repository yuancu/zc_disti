"""SageMaker entry point for MNRL sparse model finetuning.

SageMaker conventions:
- Training data: /opt/ml/input/data/training/
- Model output:  /opt/ml/model/
- Hyperparameters: passed as CLI args via HuggingFace container
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset

from sentence_transformers.sparse_encoder import (
    SparseEncoder,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
    losses,
)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


def load_triplets(data_dir: str, num_negatives: int = 1):
    """Load all JSONL files from data_dir into (anchor, positive, negative_1, ...) format."""
    anchors, positives = [], []
    neg_cols = {f"negative_{i+1}": [] for i in range(num_negatives)}

    n_skipped = 0
    data_path = Path(data_dir)
    jsonl_files = list(data_path.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")

    for jsonl_path in jsonl_files:
        print(f"Loading {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line.strip())
                negs = ex["negatives"]
                if len(negs) < num_negatives:
                    n_skipped += 1
                    continue
                sampled = random.sample(negs, num_negatives) if len(negs) > num_negatives else negs[:num_negatives]
                anchors.append(ex["anchor"])
                positives.append(ex["positive"])
                for i, neg in enumerate(sampled):
                    neg_cols[f"negative_{i+1}"].append(neg)

    ds = Dataset.from_dict({"anchor": anchors, "positive": positives, **neg_cols})
    print(f"Loaded {len(ds)} examples ({num_negatives} hard neg each); skipped {n_skipped}")
    return ds


def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker passes these environment variables
    parser.add_argument("--model_name", type=str,
                        default="opensearch-project/opensearch-neural-sparse-encoding-v2-distill")
    parser.add_argument("--num_negatives", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--document_regularizer_weight", type=float, default=0.002)
    parser.add_argument("--query_regularizer_weight", type=float, default=0.002)

    # SageMaker environment
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--training_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))

    return parser.parse_args()


def cap_max_steps(max_steps, dataset_size, batch_size, gradient_accumulation_steps):
    if max_steps <= 0:
        return max_steps
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    steps_per_epoch = max(dataset_size // (batch_size * world_size * gradient_accumulation_steps), 1)
    effective = min(max_steps, steps_per_epoch)
    print(f"1 epoch = {steps_per_epoch} steps, max_steps = {max_steps} -> effective_max_steps = {effective}")
    return effective


def main():
    args = parse_args()

    print(f"Training data dir: {args.training_dir}")
    print(f"Model output dir: {args.model_dir}")

    train_dataset = load_triplets(args.training_dir, args.num_negatives)

    model = SparseEncoder(args.model_name, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length

    inner_loss = losses.SparseMultipleNegativesRankingLoss(model)
    train_loss = losses.SpladeLoss(
        model=model,
        loss=inner_loss,
        document_regularizer_weight=args.document_regularizer_weight,
        query_regularizer_weight=args.query_regularizer_weight,
    )

    effective_max_steps = cap_max_steps(
        args.max_steps, len(train_dataset), args.train_batch_size, args.gradient_accumulation_steps
    )

    output_dir = "/opt/ml/output/intermediate"
    training_args = SparseEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=effective_max_steps,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_total_limit=1,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
    )

    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    trainer.train()

    # Save to SM_MODEL_DIR so SageMaker uploads it as model artifact
    model.save_pretrained(args.model_dir)
    print(f"Saved model to: {args.model_dir}")


if __name__ == "__main__":
    main()
