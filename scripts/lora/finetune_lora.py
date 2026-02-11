#!/usr/bin/env python3
"""Fine-tune BGE-M3 with LoRA using generated queries."""

import argparse
import json
import os
import random
import warnings
from pathlib import Path

# Suppress PyTorch DDP FutureWarning (internal torch issue, not actionable)
warnings.filterwarnings("ignore", message="functools.partial will be a method descriptor")

import torch
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
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


def setup_lora(model: SentenceTransformer, r: int, alpha: int, dropout: float, target_modules: list, print_params: bool = True):
    """Apply LoRA to the model."""
    transformer = model[0].auto_model
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    model[0].auto_model = get_peft_model(transformer, config)
    if print_params:
        model[0].auto_model.print_trainable_parameters()
    return model


def merge_and_save(model: SentenceTransformer, output_path: Path):
    """Merge LoRA weights and save full model."""
    model[0].auto_model = model[0].auto_model.merge_and_unload()
    model.save_pretrained(str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BGE-M3 with LoRA")
    project_dir = Path(__file__).resolve().parent.parent.parent

    # Required
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")

    # Paths
    parser.add_argument("--data-dir", type=Path, default=project_dir / "synthetic_data" / "training-data-20k")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts" / "models" / "lora")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=10)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-target-modules", nargs="+", default=["query", "value"])

    args = parser.parse_args()
    state = PartialState()

    # Setup paths
    data_path = args.data_dir / f"{args.dataset}.jsonl"
    output_path = args.output_dir / args.dataset
    if state.is_main_process:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Dataset: {args.dataset}")
        print(f"Training data: {data_path}")
        print(f"Output: {output_path}")
        print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

    # Load data
    if not data_path.exists():
        if state.is_main_process:
            print(f"Error: Training data not found at {data_path}")
        return

    train_dataset = load_training_data(data_path)
    if state.is_main_process:
        print(f"Training samples: {len(train_dataset)}")

    # Load model and apply LoRA
    model = SentenceTransformer(args.model_name, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length
    model = setup_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules, print_params=state.is_main_process)

    # Train
    # Set tensorboard logging dir via environment variable (logging_dir is deprecated)
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(output_path / "tensorboard")

    # Calculate warmup steps (warmup_ratio is deprecated)
    total_steps = (len(train_dataset) // args.batch_size) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir=str(output_path),
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
            dataloader_drop_last=True,  # Recommended for DDP
            ddp_find_unused_parameters=False,  # LoRA uses all trainable params
        ),
        train_dataset=train_dataset,
        loss=MultipleNegativesRankingLoss(model),
    )
    trainer.train()

    # Wait for all processes to finish training
    state.wait_for_everyone()

    # Save only on main process
    if state.is_main_process:
        merge_and_save(model, output_path)
        print(f"Model saved: {output_path}")

        with open(output_path / "result.json", "w") as f:
            json.dump({
                "dataset": args.dataset,
                "training_samples": len(train_dataset),
                "model_path": str(output_path),
                "config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                }
            }, f, indent=2)

    # Cleanup distributed process group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
