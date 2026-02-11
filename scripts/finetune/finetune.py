#!/usr/bin/env python3
"""Fine-tune BGE-M3 (full) using generated queries."""

import argparse
import json
import os
import random
import warnings
from datetime import datetime
from pathlib import Path

# Suppress PyTorch DDP FutureWarning (internal torch issue, not actionable)
warnings.filterwarnings("ignore", message="functools.partial will be a method descriptor")

import torch
from accelerate import PartialState
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

DEFAULT_MODEL = "BAAI/bge-m3"


def load_training_data(data_path: Path) -> Dataset:
    """Load training data from JSONL file."""
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line.strip())
                examples.append({"anchor": ex["anchor"], "positive": ex["positive"]})

    random.seed(42)
    random.shuffle(examples)
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BGE-M3 (full)")
    project_dir = Path(__file__).parent.parent

    # Required
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")

    # Paths
    parser.add_argument("--data-dir", type=Path, default=project_dir / "synthetic_data" / "training-data-20k")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts" / "models" / "finetuned")
    parser.add_argument("--log-dir", type=Path, default=project_dir / "output")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=10)

    args = parser.parse_args()
    state = PartialState()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup paths
    data_path = args.data_dir / f"{args.dataset}.jsonl"
    model_output_path = args.output_dir / args.dataset
    log_path = args.log_dir / args.dataset / timestamp
    if state.is_main_process:
        model_output_path.mkdir(parents=True, exist_ok=True)
        log_path.mkdir(parents=True, exist_ok=True)
        print(f"Dataset: {args.dataset}")
        print(f"Training data: {data_path}")
        print(f"Model output: {model_output_path}")
        print(f"Logs: {log_path}")

    # Load data
    if not data_path.exists():
        if state.is_main_process:
            print(f"Error: Training data not found at {data_path}")
        return

    train_dataset = load_training_data(data_path)
    if state.is_main_process:
        print(f"Training samples: {len(train_dataset)}")

    # Load model
    model = SentenceTransformer(args.model_name, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length

    # Train
    total_steps = (len(train_dataset) // args.batch_size) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir=str(log_path),
            logging_dir=str(log_path / "tensorboard"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=warmup_steps,
            fp16=torch.cuda.is_available(),
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy="no",
            save_strategy="no",
            logging_steps=args.logging_steps,
            report_to=["tensorboard"],
            dataloader_drop_last=True,
            ddp_find_unused_parameters=False,
        ),
        train_dataset=train_dataset,
        loss=MultipleNegativesRankingLoss(model),
    )
    trainer.train()

    # Wait for all processes to finish training
    state.wait_for_everyone()

    # Save only on main process
    if state.is_main_process:
        model.save_pretrained(str(model_output_path))
        print(f"Model saved: {model_output_path}")

        with open(model_output_path / "result.json", "w") as f:
            json.dump({
                "dataset": args.dataset,
                "training_samples": len(train_dataset),
                "model_path": str(model_output_path),
                "config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                }
            }, f, indent=2)

    # Cleanup distributed process group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
