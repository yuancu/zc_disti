# pip install -U sentence-transformers datasets accelerate torch

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

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DATASET_TO_MTEB_TASK = {
    "arguana": "ArguAna",
    "dbpedia-entity": "DBPedia",
    "fiqa": "FiQA2018",
    "nfcorpus": "NFCorpus",
    "quora": "QuoraRetrieval",
    "scidocs": "SCIDOCS",
    "scifact": "SciFact",
    "trec-covid": "TRECCOVID",
}


def load_and_convert(jsonl_path: str, num_negatives: int, teacher_score_scale_factor: float, topN: int = None, normalize_scores: bool = True):
    """Load teacher-scored data and convert to (query, positive, negative1, ..., negativeK) format.

    Identical to the dense version -- the data format is model-agnostic.
    teacher_score_scale_factor controls the peakedness of the teacher softmax distribution.
    For sparse models with dot-product similarity (larger magnitude student scores),
    this should be much larger than for dense cosine models (default 1.0 vs 0.025).

    When normalize_scores is True, teacher scores are standardized (zero mean, unit
    variance) per-dataset before applying teacher_score_scale_factor.  This makes
    the scale factor portable across datasets with different raw score ranges.
    """
    queries, positives, labels = [], [], []
    negatives_cols = {f"negative{i+1}": [] for i in range(num_negatives)}

    n_skipped = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data_array = json.load(f)

    # Compute per-dataset score statistics for normalization
    if normalize_scores:
        all_scores = [s for ex in data_array for s in ex["scores"]]
        mu = sum(all_scores) / len(all_scores)
        std = (sum((s - mu) ** 2 for s in all_scores) / len(all_scores)) ** 0.5
        std = max(std, 1e-8)
        print(f"Teacher score normalization: mean={mu:.4f}, std={std:.4f}")
    else:
        mu, std = 0.0, 1.0

    def scale(raw_score: float) -> float:
        return ((raw_score - mu) / std) * teacher_score_scale_factor

    for ex in data_array:
        q = ex["query"]
        docs = ex["docs"]
        scores = ex["scores"]

        if not docs or not scores or len(docs) != len(scores):
            n_skipped += 1
            print(f"Skipping example because it has no docs or scores")
            assert False

        if len(docs) < 1 + num_negatives:
            n_skipped += 1
            print(f"Skipping example because it has less than 1 + {num_negatives} docs")
            assert False

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        pos_doc = docs[best_idx]
        pos_score = float(scores[best_idx])

        rest = [(docs[i], float(scores[i])) for i in range(len(docs)) if i != best_idx]
        rest.sort(key=lambda x: x[1], reverse=True)

        if topN is not None:
            rest = rest[:topN]

        offset = 0
        while offset + num_negatives <= len(rest):
            chunk = rest[offset : offset + num_negatives]
            offset += num_negatives

            queries.append(q)
            positives.append(pos_doc)
            labels.append(
                [scale(pos_score)]
                + [scale(s) for (_, s) in chunk]
            )
            for i, (neg_doc, _) in enumerate(chunk):
                negatives_cols[f"negative{i+1}"].append(neg_doc)

    data = {"query": queries, "positive": positives, **negatives_cols, "label": labels}
    ds = Dataset.from_dict(data)
    ds = ds.map(lambda b: {"label": torch.tensor(b["label"], dtype=torch.float32)}, batched=False)
    print(f"Loaded {len(ds)} examples; skipped {n_skipped}")
    return ds


def build_router_mapping(num_negatives: int) -> dict[str, str]:
    """Build router_mapping to route queries through SparseStaticEmbedding
    and documents through MLMTransformer+SpladePooling."""
    mapping = {"query": "query", "positive": "document"}
    for i in range(num_negatives):
        mapping[f"negative{i+1}"] = "document"
    return mapping


class AdaptiveSpladeLoss(losses.SpladeLoss):
    """SpladeLoss with adaptive regularization weight.

    Instead of a fixed lambda, scales the FLOPS regularization weight proportionally
    to the base (KL) loss magnitude.  This keeps the ratio between ranking signal and
    regularization constant regardless of dataset characteristics.

    At each step:
        effective_lambda = document_regularizer_weight * EMA(base_loss / flops_loss)

    ``document_regularizer_weight`` becomes a dimensionless ratio (e.g. 0.1 means
    "FLOPS contributes 10% of the KL gradient magnitude") rather than an absolute
    weight that needs per-dataset tuning.
    """

    def __init__(self, *args, ema_decay: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_decay = ema_decay
        self.ema_ratio: float | None = None

    def forward(self, sentence_features, labels=None):
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]

        loss_dict = {}
        base_loss = self.loss.compute_loss_from_embeddings(embeddings, labels)
        if isinstance(base_loss, dict):
            loss_dict.update(base_loss)
            base_loss_value = sum(base_loss.values())
        else:
            loss_dict["base_loss"] = base_loss
            base_loss_value = base_loss

        if self.use_document_regularizer_only:
            corpus_loss = self.document_regularizer.compute_loss_from_embeddings(
                torch.cat(embeddings)
            )
        else:
            corpus_loss = self.document_regularizer.compute_loss_from_embeddings(
                torch.cat(embeddings[1:])
            )

        # Compute per-step ratio and smooth with EMA
        ratio = base_loss_value.detach().item() / max(corpus_loss.detach().item(), 1e-8)
        if self.ema_ratio is None:
            self.ema_ratio = ratio
        else:
            self.ema_ratio = self.ema_decay * self.ema_ratio + (1 - self.ema_decay) * ratio

        loss_dict["document_regularizer_loss"] = (
            corpus_loss * self.document_regularizer_weight * self.ema_ratio
        )

        if self.query_regularizer_weight is not None:
            query_loss = self.query_regularizer.compute_loss_from_embeddings(embeddings[0])
            loss_dict["query_regularizer_loss"] = (
                query_loss * self.query_regularizer_weight * self.ema_ratio
            )

        return loss_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a SPLADE sparse retrieval model using KL divergence distillation"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="nfcorpus",
        help="Dataset name, used to construct data_path and output_dir if not explicitly provided (default: nfcorpus)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the JSON data file containing queries, docs, and scores (default: kd_samples/{dataset}.jsonl)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
        help="Name or path of the sparse model to finetune (default: opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for saving the trained model (default: sparse-kldiv-{dataset})"
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=2,
        help="Number of negative examples per query (default: 2)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Temperature for KL divergence loss. Higher values produce softer distributions "
             "and help prevent the student from collapsing to zero active dimensions. (default: 2.0)"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Per-device training batch size (default: 16)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (default: 1)"
    )
    parser.add_argument(
        "--teacher_score_scale_factor",
        type=float,
        default=2.0,
        help="Scaling factor for teacher scores, applied after optional normalization. "
             "(default: 2.0)"
    )
    parser.add_argument(
        "--normalize_scores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize teacher scores (zero mean, unit variance) per-dataset before "
             "applying teacher_score_scale_factor. Makes the scale factor portable across "
             "datasets with different raw score ranges. (default: True)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 5e-6)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of training steps. If set to positive value, overrides num_epochs (default: -1)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (default: 0.1)"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Logging steps (default: 50)"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep (default: 2)"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)"
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation after training"
    )
    parser.add_argument(
        "--topN",
        type=int,
        default=None,
        help="If set, only keep top N negatives (by teacher score) before generating training samples (default: None)"
    )
    # Sparse-specific: FLOPS regularization
    parser.add_argument(
        "--document_regularizer_weight",
        type=float,
        default=3e-5,
        help="Weight for FLOPS regularizer on document embeddings (lambda_d). "
             "Encourages sparsity in document representations. (default: 3e-5)"
    )
    parser.add_argument(
        "--query_regularizer_weight",
        type=float,
        default=None,
        help="Weight for FLOPS regularizer on query embeddings (lambda_q). "
             "Not needed for inference-free query models like the OpenSearch sparse "
             "encoders since their query path has no trainable parameters. (default: None)"
    )
    parser.add_argument(
        "--adaptive_regularizer",
        action="store_true",
        help="Scale FLOPS regularization weight proportionally to the base (KL) loss "
             "magnitude so that document_regularizer_weight acts as a dimensionless "
             "ratio rather than an absolute weight. Makes regularization strength "
             "dataset-agnostic. (default: False)"
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Symmetric (bi-encoder) sparse model where both query and document use "
             "the same neural MLM encoder (e.g., opensearch-neural-sparse-encoding-v2-distill). "
             "Disables router_mapping and query-encoder freezing. (default: False)"
    )

    return parser.parse_args()


def cap_max_steps(max_steps: int, dataset_size: int, batch_size: int, gradient_accumulation_steps: int) -> int:
    """Return effective max_steps capped at 1 epoch, or -1 if max_steps is not set."""
    if max_steps <= 0:
        return max_steps
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    steps_per_epoch = max(dataset_size // (batch_size * world_size * gradient_accumulation_steps), 1)
    effective = min(max_steps, steps_per_epoch)
    print(f"1 epoch = {steps_per_epoch} steps, max_steps = {max_steps} → effective_max_steps = {effective}")
    return effective


def main():
    args = parse_args()

    if args.data_path is None:
        args.data_path = f"kd_samples/{args.dataset}.jsonl"
    if args.output_dir is None:
        args.output_dir = f"sparse-kldiv-{args.dataset}"

    train_dataset = load_and_convert(
        args.data_path,
        args.num_negatives,
        args.teacher_score_scale_factor,
        args.topN,
        normalize_scores=args.normalize_scores,
    )

    model = SparseEncoder(args.model_name, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length

    # For asymmetric models (doc-v3-gte), freeze the inference-free query encoder
    # (SparseStaticEmbedding / IDF weights). These are not trainable and MUST be
    # excluded from DDP tracking: SpladeLoss calls model() multiple times per step
    # (once for query, once per document), and with find_unused_parameters=True DDP
    # marks query params as "ready" after the first call then errors when the
    # subsequent document calls trigger the hooks again.
    # For symmetric models (v2-distill), both encoders are the same neural MLM —
    # nothing to freeze.
    if not args.symmetric:
        n_frozen = 0
        for name, param in model.named_parameters():
            if "sub_modules" in name and "query" in name:
                param.requires_grad = False
                n_frozen += 1
        print(f"Froze {n_frozen} query-encoder parameters (inference-free IDF weights)")
    else:
        print("Symmetric mode: all parameters are trainable")

    # SparseDistillKLDivLoss uses dot-product similarity by default.
    # It MUST be wrapped in SpladeLoss which handles the forward pass and FLOPS regularization.
    inner_loss = losses.SparseDistillKLDivLoss(
        model=model,
        similarity_fct=util.pairwise_dot_score,
        temperature=args.temperature,
    )
    loss_cls = AdaptiveSpladeLoss if args.adaptive_regularizer else losses.SpladeLoss
    train_loss = loss_cls(
        model=model,
        loss=inner_loss,
        document_regularizer_weight=args.document_regularizer_weight,
        # For inference-free query models (SparseStaticEmbedding), query regularization
        # is not needed since the query encoder has no trainable parameters.
        query_regularizer_weight=args.query_regularizer_weight,
    )
    if args.adaptive_regularizer:
        print(f"Using adaptive FLOPS regularization (ratio={args.document_regularizer_weight})")

    effective_max_steps = cap_max_steps(
        args.max_steps, len(train_dataset), args.train_batch_size, args.gradient_accumulation_steps
    )

    # For asymmetric models, build router_mapping so the trainer routes "query"
    # columns through SparseStaticEmbedding and "positive"/"negative*" columns
    # through MLMTransformer + SpladePooling.
    # For symmetric models, no router_mapping is needed — all columns use the
    # same encoder.
    router_mapping = None if args.symmetric else build_router_mapping(args.num_negatives)

    training_args_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=effective_max_steps,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
    )
    if router_mapping is not None:
        training_args_kwargs["router_mapping"] = router_mapping

    training_args = SparseEncoderTrainingArguments(**training_args_kwargs)

    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    trainer.train()

    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main:
        model.save_pretrained(args.output_dir)
        print(f"Saved to: {args.output_dir}")

    if not args.skip_eval and is_main:
        try:
            import mteb

            mteb_task = DATASET_TO_MTEB_TASK.get(args.dataset)
            if mteb_task is None:
                print(f"Warning: No MTEB task mapping found for dataset '{args.dataset}'. Skipping evaluation.")
            else:
                tasks = mteb.get_tasks(tasks=[mteb_task])
                evaluation = mteb.MTEB(tasks=tasks)

                results = evaluation.run(
                    model,
                    eval_splits=["test"],
                    output_folder=None,
                    show_progress_bar=True,
                    encode_kwargs={
                        "batch_size": args.eval_batch_size,
                    },
                )

                print("\n====== Evaluation Results ======")
                for result in results:
                    print(f"Task Name: {result.task_name}")
                    test_scores = result.scores.get("test", {})
                    print(test_scores)
        except ImportError:
            print("Warning: mteb not installed. Skipping evaluation.")
            print("Install with: pip install mteb")


if __name__ == "__main__":
    main()
