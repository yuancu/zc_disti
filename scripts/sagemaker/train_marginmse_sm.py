"""SageMaker entry point for MarginMSE/RankNet/Hinge sparse model finetuning.

Supports all loss types from finetune_sparse_marginmse.py:
  - marginmse: SparseMarginMSELoss (score-magnitude sensitive)
  - ranknet: RankNet pairwise loss (rank-based)
  - hinge: Hinge pairwise loss with configurable margin

Also supports conditional FLOPS regularization (disables FLOPS when doc L0 < threshold).

SageMaker conventions:
- Training data: /opt/ml/input/data/training/ (JSON files with query/docs/scores)
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

from sentence_transformers import util
from sentence_transformers.sparse_encoder import (
    SparseEncoder,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
    losses,
)
from sentence_transformers.sparse_encoder.losses import FlopsLoss

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


class SparseRankNetLoss(torch.nn.Module):
    def __init__(self, model, similarity_fct=None):
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct or util.pairwise_dot_score

    def compute_loss_from_embeddings(self, embeddings, labels=None):
        query_emb = embeddings[0]
        pos_emb = embeddings[1]
        pos_sim = self.similarity_fct(query_emb, pos_emb)
        total_loss = torch.tensor(0.0, device=query_emb.device)
        num_pairs = 0
        for neg_emb in embeddings[2:]:
            neg_sim = self.similarity_fct(query_emb, neg_emb)
            total_loss = total_loss + torch.nn.functional.softplus(neg_sim - pos_sim).mean()
            num_pairs += 1
        if num_pairs > 0:
            total_loss = total_loss / num_pairs
        return total_loss

    def forward(self, sentence_features, labels=None):
        embeddings = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
        return self.compute_loss_from_embeddings(embeddings, labels)


class SparseHingeLoss(torch.nn.Module):
    def __init__(self, model, similarity_fct=None, margin=0.0):
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct or util.pairwise_dot_score
        self.margin = margin

    def compute_loss_from_embeddings(self, embeddings, labels=None):
        query_emb = embeddings[0]
        pos_emb = embeddings[1]
        pos_sim = self.similarity_fct(query_emb, pos_emb)
        total_loss = torch.tensor(0.0, device=query_emb.device)
        num_pairs = 0
        for neg_emb in embeddings[2:]:
            neg_sim = self.similarity_fct(query_emb, neg_emb)
            total_loss = total_loss + torch.relu(self.margin + neg_sim - pos_sim).mean()
            num_pairs += 1
        if num_pairs > 0:
            total_loss = total_loss / num_pairs
        return total_loss

    def forward(self, sentence_features, labels=None):
        embeddings = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
        return self.compute_loss_from_embeddings(embeddings, labels)


class ConditionalSpladeLoss(torch.nn.Module):
    def __init__(self, model, loss, document_regularizer_weight, query_regularizer_weight=None,
                 flops_disable_threshold=None):
        super().__init__()
        self.model = model
        self.loss = loss
        self.document_regularizer_weight = document_regularizer_weight
        self.query_regularizer_weight = query_regularizer_weight
        self.flops_disable_threshold = flops_disable_threshold
        self.document_regularizer = FlopsLoss(model)
        if query_regularizer_weight is not None:
            self.query_regularizer = FlopsLoss(model)

    def forward(self, sentence_features, labels=None):
        embeddings = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
        result = {}
        base_loss = self.loss.compute_loss_from_embeddings(embeddings, labels)
        if isinstance(base_loss, dict):
            result.update(base_loss)
        else:
            result["base_loss"] = base_loss

        doc_embeddings = torch.cat(embeddings[1:])
        enable_flops = True
        doc_dense = doc_embeddings.to_dense() if doc_embeddings.is_sparse else doc_embeddings
        avg_doc_len = (doc_dense > 0).float().sum(dim=1).mean().item()
        if self.flops_disable_threshold is not None and avg_doc_len < self.flops_disable_threshold:
            enable_flops = False

        if enable_flops:
            doc_flops = self.document_regularizer.compute_loss_from_embeddings(doc_embeddings)
            result["document_regularizer_loss"] = doc_flops * self.document_regularizer_weight
        else:
            result["document_regularizer_loss"] = torch.tensor(0.0, device=doc_embeddings.device)

        if self.query_regularizer_weight is not None and enable_flops:
            q_flops = self.query_regularizer.compute_loss_from_embeddings(embeddings[0])
            result["query_regularizer_loss"] = q_flops * self.query_regularizer_weight

        return result


def load_and_convert(data_dir, num_negatives, adaptive_temperature=False):
    """Load teacher-scored JSON data and convert to training format."""
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

    # Compute per-query statistics for adaptive temperature
    query_stats = []
    for ex in all_data:
        raw_scores = ex["scores"]
        if raw_scores and len(raw_scores) > 1:
            q_mu = sum(raw_scores) / len(raw_scores)
            q_std = (sum((s - q_mu) ** 2 for s in raw_scores) / len(raw_scores)) ** 0.5
            q_std = max(q_std, 1e-8)
        else:
            q_std = 1.0
        query_stats.append(q_std)

    reference_std = None
    min_std_threshold = None
    global_mu = None
    global_std = None
    if adaptive_temperature:
        sorted_stds = sorted(query_stats)
        reference_std = sorted_stds[len(sorted_stds) // 2]
        min_std_threshold = sorted_stds[len(sorted_stds) // 10]
        all_scores = [s for ex in all_data for s in ex["scores"]]
        global_mu = sum(all_scores) / len(all_scores)
        global_std = (sum((s - global_mu) ** 2 for s in all_scores) / len(all_scores)) ** 0.5
        global_std = max(global_std, 1e-8)
        print(f"Adaptive temperature: reference_std={reference_std:.4f}")

    max_temp_factor = 5.0
    min_temp_factor = 0.2

    for idx, ex in enumerate(all_data):
        q = ex["query"]
        docs = ex["docs"]
        raw_scores = ex["scores"]
        if not docs or not raw_scores or len(docs) != len(raw_scores):
            n_skipped += 1
            continue
        if len(docs) < 1 + num_negatives:
            n_skipped += 1
            continue

        if adaptive_temperature:
            q_std = query_stats[idx]
            if q_std < min_std_threshold:
                scores = [(s - global_mu) / global_std for s in raw_scores]
            else:
                temp_factor = reference_std / q_std
                temp_factor = max(min_temp_factor, min(temp_factor, max_temp_factor))
                scores = [s * temp_factor for s in raw_scores]
        else:
            scores = [float(s) for s in raw_scores]

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
    print(f"Loaded {len(ds)} examples; skipped {n_skipped} queries")
    return ds


def build_router_mapping(num_negatives):
    mapping = {"query": "query", "positive": "document"}
    for i in range(num_negatives):
        mapping[f"negative{i+1}"] = "document"
    return mapping


def cap_max_steps(max_steps, dataset_size, batch_size, gradient_accumulation_steps):
    if max_steps <= 0:
        return max_steps
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    steps_per_epoch = max(dataset_size // (batch_size * world_size * gradient_accumulation_steps), 1)
    effective = min(max_steps, steps_per_epoch)
    print(f"1 epoch = {steps_per_epoch} steps, max_steps = {max_steps} -> effective = {effective}")
    return effective


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte")
    parser.add_argument("--loss_type", type=str, default="marginmse",
                        choices=["marginmse", "ranknet", "hinge"])
    parser.add_argument("--num_negatives", type=int, default=2)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--adaptive_temperature", type=str, default="false")
    parser.add_argument("--document_regularizer_weight", type=float, default=3e-5)
    parser.add_argument("--flops_disable_threshold", type=float, default=-1,
                        help="Disable FLOPS when doc L0 < threshold. -1 = disabled.")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)

    # SageMaker environment
    parser.add_argument("--model_dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--training_dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Config: loss_type={args.loss_type}, num_negatives={args.num_negatives}, "
          f"adaptive_temperature={args.adaptive_temperature}, "
          f"flops_disable_threshold={args.flops_disable_threshold}, "
          f"document_regularizer_weight={args.document_regularizer_weight}")

    adaptive_temp = args.adaptive_temperature.lower() in ("true", "1", "yes")
    use_conditional_flops = args.flops_disable_threshold > 0

    train_dataset = load_and_convert(args.training_dir, args.num_negatives, adaptive_temp)

    model = SparseEncoder(args.model_name, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length

    # Freeze query encoder (inference-free IDF weights)
    n_frozen = 0
    for name, param in model.named_parameters():
        if "sub_modules" in name and "query" in name:
            param.requires_grad = False
            n_frozen += 1
    print(f"Froze {n_frozen} query-encoder parameters")

    # Build loss
    if args.loss_type == "ranknet":
        print("Using RankNet loss")
        inner_loss = SparseRankNetLoss(model=model, similarity_fct=util.pairwise_dot_score)
    elif args.loss_type == "hinge":
        print(f"Using Hinge loss (margin={args.margin})")
        inner_loss = SparseHingeLoss(model=model, similarity_fct=util.pairwise_dot_score, margin=args.margin)
    else:
        print("Using MarginMSE loss")
        inner_loss = losses.SparseMarginMSELoss(model=model, similarity_fct=util.pairwise_dot_score)

    if use_conditional_flops:
        print(f"Using ConditionalSpladeLoss (threshold={args.flops_disable_threshold})")
        train_loss = ConditionalSpladeLoss(
            model=model, loss=inner_loss,
            document_regularizer_weight=args.document_regularizer_weight,
            flops_disable_threshold=args.flops_disable_threshold,
        )
    else:
        train_loss = losses.SpladeLoss(
            model=model, loss=inner_loss,
            document_regularizer_weight=args.document_regularizer_weight,
        )

    effective_max_steps = cap_max_steps(
        args.max_steps, len(train_dataset), args.train_batch_size, args.gradient_accumulation_steps
    )

    router_mapping = build_router_mapping(args.num_negatives)

    output_dir = "/opt/ml/output/intermediate"
    training_args = SparseEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
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
        router_mapping=router_mapping,
    )

    trainer = SparseEncoderTrainer(
        model=model, args=training_args, train_dataset=train_dataset, loss=train_loss,
    )

    trainer.train()

    # Save to SM_MODEL_DIR
    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main:
        import glob
        import shutil
        for ckpt_dir in glob.glob(os.path.join(output_dir, "checkpoint-*")):
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        model.save_pretrained(args.model_dir)
        # Remove custom code files (configuration.py, modeling.py) saved by HuggingFace
        # models with trust_remote_code. sentence-transformers>=5.0 has native support
        # for these models, and leaving the .py files causes loading failures.
        for py_file in Path(args.model_dir).rglob("configuration.py"):
            py_file.unlink()
        for py_file in Path(args.model_dir).rglob("modeling.py"):
            py_file.unlink()
        # Save config for provenance
        config = {
            "experiment": "exp10",
            "loss_type": args.loss_type,
            "num_negatives": args.num_negatives,
            "margin": args.margin,
            "adaptive_temperature": adaptive_temp,
            "document_regularizer_weight": args.document_regularizer_weight,
            "flops_disable_threshold": args.flops_disable_threshold if use_conditional_flops else None,
            "max_steps": args.max_steps,
            "effective_max_steps": effective_max_steps,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "model_name": args.model_name,
        }
        with open(os.path.join(args.model_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved model to: {args.model_dir}")


if __name__ == "__main__":
    main()
