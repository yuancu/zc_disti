"""
Parse Bedrock batch output and generate training samples with semantic hard negatives.

This script uses a dense retriever model to mine hard negatives:
1. Encode all corpus documents using the retriever model
2. Build a FAISS index for fast nearest neighbor search
3. Encode queries and retrieve top-k most similar documents
4. Exclude the positive document and keep hard negatives

Output format (same as BM25 version):
{"anchor": query, "positive": document, "negatives": [neg1, neg2, ...]}
"""

import json
import re
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_raw_corpus(corpus_path: str) -> tuple[dict[str, str], list[str], list[str]]:
    """
    Load raw corpus and return:
    - doc_id_to_text: mapping from doc_id to text
    - doc_ids: list of doc_ids (for indexing)
    - doc_texts: list of document texts
    """
    doc_id_to_text = {}
    doc_ids = []
    doc_texts = []

    with open(corpus_path, "r") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                doc_id = doc.get("id") or doc["_id"]
                text = doc["text"]
                doc_id_to_text[doc_id] = text
                doc_ids.append(doc_id)
                doc_texts.append(text)

    return doc_id_to_text, doc_ids, doc_texts


def extract_queries_from_model_output(model_output: dict) -> list[str]:
    """Extract queries from the model output content."""
    try:
        content = model_output.get("content", [])
        if not content:
            return []

        text = content[0].get("text", "")

        # Remove markdown code block markers if present
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        parsed = json.loads(text)

        if isinstance(parsed, dict) and "queries" in parsed:
            return parsed["queries"]
        elif isinstance(parsed, list):
            return parsed
        else:
            return []

    except (json.JSONDecodeError, KeyError, IndexError):
        return []


def find_output_file(dataset_dir: Path) -> Optional[Path]:
    """Find the .jsonl.out file in the dataset directory."""
    for subdir in dataset_dir.iterdir():
        if subdir.is_dir():
            for file in subdir.iterdir():
                if file.name.endswith("_batch_input.jsonl.out"):
                    return file
    return None


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 256,
) -> np.ndarray:
    """Encode texts using the sentence transformer model with L2 normalization."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from normalized embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def get_hard_negatives_batch(
    index: faiss.IndexFlatIP,
    query_embeddings: np.ndarray,
    positive_doc_ids: list[str],
    doc_ids: list[str],
    doc_texts: list[str],
    num_negatives: int = 20,
) -> list[list[str]]:
    """Get hard negatives for queries using FAISS nearest neighbor search."""
    top_k = min(num_negatives + 5, index.ntotal)
    scores, indices = index.search(query_embeddings, top_k)

    all_hard_negatives = []
    for i in range(len(query_embeddings)):
        hard_negatives = []
        for j in range(top_k):
            idx = indices[i][j]
            if idx == -1:
                continue
            if doc_ids[idx] != positive_doc_ids[i]:
                hard_negatives.append(doc_texts[idx])
                if len(hard_negatives) >= num_negatives:
                    break
        all_hard_negatives.append(hard_negatives)

    return all_hard_negatives


def parse_bedrock_output_with_negatives(
    output_dir: str,
    corpus_dir: str,
    result_dir: str,
    model_name: str = "BAAI/bge-m3",
    datasets: Optional[list[str]] = None,
    num_negatives: int = 20,
    batch_size: int = 256,
):
    """
    Parse Bedrock batch output and generate training samples with semantic hard negatives.
    """
    output_path = Path(output_dir)
    corpus_path = Path(corpus_dir)
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [d.name for d in output_path.iterdir() if d.is_dir()]

    print(f"Processing datasets: {datasets}")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Corpus directory: {corpus_dir}")
    print(f"Result directory: {result_dir}")
    print(f"Number of negatives per sample: {num_negatives}")
    print(f"Encoding batch size: {batch_size}")
    print()

    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print()

    total_samples = 0

    for dataset_name in datasets:
        print(f"{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")

        # Find the output file
        dataset_output_dir = output_path / dataset_name
        if not dataset_output_dir.exists():
            print(f"Warning: Dataset directory not found: {dataset_output_dir}")
            continue

        output_file = find_output_file(dataset_output_dir)
        if output_file is None:
            print(f"Warning: No output file found in {dataset_output_dir}")
            continue

        print(f"Found output file: {output_file}")

        # Load corpus
        corpus_file = corpus_path / f"{dataset_name}.jsonl"
        if not corpus_file.exists():
            print(f"Warning: Corpus file not found: {corpus_file}")
            continue

        doc_id_to_text, doc_ids, doc_texts = load_raw_corpus(str(corpus_file))
        print(f"Loaded {len(doc_ids)} documents from corpus")

        # Encode corpus
        print("Encoding corpus documents...")
        doc_embeddings = encode_texts(model, doc_texts, batch_size=batch_size)
        print(f"Corpus encoded: {doc_embeddings.shape}")

        # Build FAISS index
        print("Building FAISS index...")
        index = build_faiss_index(doc_embeddings)
        print(f"FAISS index built with {index.ntotal} vectors")

        # Parse Bedrock output to get query-document pairs
        print("Parsing Bedrock output...")
        query_doc_pairs = []
        error_count = 0
        missing_doc_count = 0

        with open(output_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    record_id = record.get("recordId")
                    model_output = record.get("modelOutput", {})

                    if record_id not in doc_id_to_text:
                        missing_doc_count += 1
                        continue

                    document_text = doc_id_to_text[record_id]
                    queries = extract_queries_from_model_output(model_output)

                    if not queries:
                        error_count += 1
                        continue

                    for query in queries:
                        if query and isinstance(query, str):
                            query_doc_pairs.append({
                                "query": query,
                                "doc_id": record_id,
                                "doc_text": document_text,
                            })

                except json.JSONDecodeError:
                    error_count += 1
                    continue

        print(f"Found {len(query_doc_pairs)} query-document pairs")
        print(f"Errors: {error_count}, Missing docs: {missing_doc_count}")

        if not query_doc_pairs:
            print("No query-document pairs found, skipping.")
            print()
            continue

        # Encode queries
        queries = [pair["query"] for pair in query_doc_pairs]
        positive_doc_ids = [pair["doc_id"] for pair in query_doc_pairs]
        positive_doc_texts = [pair["doc_text"] for pair in query_doc_pairs]

        print("Encoding queries...")
        query_embeddings = encode_texts(model, queries, batch_size=batch_size)
        print(f"Queries encoded: {query_embeddings.shape}")

        # Get hard negatives
        print("Retrieving semantic hard negatives...")
        all_hard_negatives = get_hard_negatives_batch(
            index=index,
            query_embeddings=query_embeddings,
            positive_doc_ids=positive_doc_ids,
            doc_ids=doc_ids,
            doc_texts=doc_texts,
            num_negatives=num_negatives,
        )

        # Build training samples
        training_samples = []
        for query, positive_text, hard_negatives in zip(
            queries, positive_doc_texts, all_hard_negatives
        ):
            training_samples.append({
                "anchor": query,
                "positive": positive_text,
                "negatives": hard_negatives,
            })

        # Save
        result_file = result_path / f"{dataset_name}.jsonl"
        with open(result_file, "w") as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"Generated {len(training_samples)} training samples with hard negatives")
        print(f"Saved to: {result_file}")
        print()

        total_samples += len(training_samples)

        # Free memory
        del doc_embeddings, query_embeddings, index

    print(f"{'='*60}")
    print(f"Total training samples across all datasets: {total_samples}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse Bedrock batch output and generate training samples with semantic hard negatives."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory containing Bedrock batch output",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="raw-corpus",
        help="Directory containing raw corpus files (default: raw-corpus)",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Directory to save the parsed training samples",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-m3",
        help="Sentence transformer model for encoding (default: BAAI/bge-m3)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of datasets to process. If not specified, process all.",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=20,
        help="Number of hard negatives per sample (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for encoding (default: 256)",
    )

    args = parser.parse_args()

    parse_bedrock_output_with_negatives(
        output_dir=args.output_dir,
        corpus_dir=args.corpus_dir,
        result_dir=args.result_dir,
        model_name=args.model_name,
        datasets=args.datasets,
        num_negatives=args.num_negatives,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
