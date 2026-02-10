"""
Parse Bedrock batch output and generate training samples with hard negatives.

This script extends the basic parsing by:
1. Building a BM25 index for each dataset's corpus using bm25s
2. Using BGE-M3 tokenizer for multilingual support
3. Using BM25 to retrieve top 21 documents for each query
4. Excluding the positive document and keeping 20 hard negatives

Output format:
{"anchor": query, "positive": document, "negatives": [neg1, neg2, ...]}
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import Optional

import bm25s
from transformers import AutoTokenizer
from tqdm import tqdm


# Global tokenizer (loaded once)
_tokenizer = None


def get_tokenizer():
    """Get or initialize the BGE-M3 tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        print("Loading BGE-M3 tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return _tokenizer


def tokenize_text(text: str) -> list[str]:
    """Tokenize a single text using BGE-M3 tokenizer."""
    tokenizer = get_tokenizer()
    # Tokenize and convert ids back to tokens (excluding special tokens)
    tokens = tokenizer.tokenize(text)
    return tokens


def tokenize_texts_batch(texts: list[str], batch_size: int = 1000) -> list[list[str]]:
    """
    Tokenize a list of texts in batches using BGE-M3 tokenizer.
    
    Args:
        texts: List of texts to tokenize
        batch_size: Number of texts per batch (default: 1000)
    
    Returns:
        List of tokenized texts (each is a list of tokens)
    """
    tokenizer = get_tokenizer()
    all_tokenized = []
    
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Tokenizing batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        # Use batch tokenization
        batch_encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=8192,  # BGE-M3 supports up to 8192 tokens
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        
        # Convert input_ids back to tokens for each text
        for input_ids in batch_encoded["input_ids"]:
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            all_tokenized.append(tokens)
    
    return all_tokenized


def load_raw_corpus(corpus_path: str) -> tuple[dict[str, str], list[str], list[str]]:
    """
    Load raw corpus and return:
    - doc_id_to_text: mapping from doc_id to text
    - doc_ids: list of doc_ids (for indexing)
    - doc_texts: list of document texts (for BM25)
    """
    doc_id_to_text = {}
    doc_ids = []
    doc_texts = []
    
    with open(corpus_path, "r") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                doc_id = doc["id"]
                text = doc["text"]
                doc_id_to_text[doc_id] = text
                doc_ids.append(doc_id)
                doc_texts.append(text)
    
    return doc_id_to_text, doc_ids, doc_texts


def build_bm25_index(doc_texts: list[str], batch_size: int = 1000) -> tuple[bm25s.BM25, list[list[str]]]:
    """Build BM25 index from document texts using bm25s with batch tokenization."""
    print(f"Tokenizing {len(doc_texts)} documents with BGE-M3 tokenizer (batch_size={batch_size})...")
    tokenized_docs = tokenize_texts_batch(doc_texts, batch_size=batch_size)
    
    print("Building BM25 index with bm25s...")
    retriever = bm25s.BM25()
    retriever.index(tokenized_docs)
    
    return retriever, tokenized_docs


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
            # Remove ```json or ``` at the start
            text = re.sub(r"^```\w*\n?", "", text)
            # Remove ``` at the end
            text = re.sub(r"\n?```$", "", text)
        
        # Parse JSON
        parsed = json.loads(text)
        
        # Handle both {"queries": [...]} and direct list format
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


def get_hard_negatives_batch(
    retriever: bm25s.BM25,
    queries: list[str],
    positive_doc_ids: list[str],
    doc_ids: list[str],
    doc_texts: list[str],
    num_negatives: int = 20,
    batch_size: int = 1000
) -> list[list[str]]:
    """
    Get hard negatives for a batch of queries using BM25.
    
    Args:
        retriever: BM25 retriever
        queries: List of query strings
        positive_doc_ids: List of positive document IDs (one per query)
        doc_ids: List of all document IDs in corpus
        doc_texts: List of all document texts in corpus
        num_negatives: Number of negatives to return per query
        batch_size: Batch size for tokenization and retrieval
    
    Returns:
        List of hard negative lists (one list per query)
    """
    all_hard_negatives = []
    num_batches = (len(queries) + batch_size - 1) // batch_size
    top_k = num_negatives + 5  # Extra buffer to account for filtering positive
    
    for i in tqdm(range(num_batches), desc="Retrieving hard negatives"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(queries))
        
        batch_queries = queries[start_idx:end_idx]
        batch_positive_ids = positive_doc_ids[start_idx:end_idx]
        
        # Batch tokenize queries
        tokenized_queries = tokenize_texts_batch(batch_queries, batch_size=len(batch_queries))
        
        # Batch retrieve
        results, scores = retriever.retrieve(tokenized_queries, k=min(top_k, len(doc_ids)))
        
        # Process results for each query in batch
        for j, (top_indices, pos_doc_id) in enumerate(zip(results, batch_positive_ids)):
            hard_negatives = []
            for idx in top_indices:
                if doc_ids[idx] != pos_doc_id:
                    hard_negatives.append(doc_texts[idx])
                    if len(hard_negatives) >= num_negatives:
                        break
            all_hard_negatives.append(hard_negatives)
    
    return all_hard_negatives


def parse_bedrock_output_with_negatives(
    output_dir: str,
    corpus_dir: str,
    result_dir: str,
    datasets: Optional[list[str]] = None,
    num_negatives: int = 20
):
    """
    Parse Bedrock batch output and generate training samples with hard negatives.
    
    Args:
        output_dir: Directory containing Bedrock batch output
        corpus_dir: Directory containing raw corpus files
        result_dir: Directory to save the parsed training samples
        datasets: Optional list of datasets to process
        num_negatives: Number of hard negatives to include (default: 20)
    """
    output_path = Path(output_dir)
    corpus_path = Path(corpus_dir)
    result_path = Path(result_dir)
    
    # Create result directory
    result_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of datasets to process
    if datasets is None:
        datasets = [d.name for d in output_path.iterdir() if d.is_dir()]
    
    print(f"Processing datasets: {datasets}")
    print(f"Output directory: {output_dir}")
    print(f"Corpus directory: {corpus_dir}")
    print(f"Result directory: {result_dir}")
    print(f"Number of negatives per sample: {num_negatives}")
    print()
    
    # Pre-load tokenizer
    get_tokenizer()
    
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
        
        # Build BM25 index
        retriever, tokenized_docs = build_bm25_index(doc_texts)
        print("BM25 index built successfully")
        
        # First pass: collect all query-document pairs
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
                    
                    # Get document text
                    if record_id not in doc_id_to_text:
                        missing_doc_count += 1
                        continue
                    
                    document_text = doc_id_to_text[record_id]
                    
                    # Extract queries
                    queries = extract_queries_from_model_output(model_output)
                    
                    if not queries:
                        error_count += 1
                        continue
                    
                    # Store query-document pairs
                    for query in queries:
                        if query and isinstance(query, str):
                            query_doc_pairs.append({
                                "query": query,
                                "doc_id": record_id,
                                "doc_text": document_text
                            })
                
                except json.JSONDecodeError:
                    error_count += 1
                    continue
        
        print(f"Found {len(query_doc_pairs)} query-document pairs")
        print(f"Errors: {error_count}, Missing docs: {missing_doc_count}")
        
        # Second pass: generate training samples with hard negatives (batch processing)
        print("Generating training samples with hard negatives (batch processing)...")
        
        # Extract queries and positive doc IDs for batch processing
        queries = [pair["query"] for pair in query_doc_pairs]
        positive_doc_ids = [pair["doc_id"] for pair in query_doc_pairs]
        positive_doc_texts = [pair["doc_text"] for pair in query_doc_pairs]
        
        # Get hard negatives in batches
        all_hard_negatives = get_hard_negatives_batch(
            retriever=retriever,
            queries=queries,
            positive_doc_ids=positive_doc_ids,
            doc_ids=doc_ids,
            doc_texts=doc_texts,
            num_negatives=num_negatives,
            batch_size=1000
        )
        
        # Build training samples
        training_samples = []
        for query, positive_text, hard_negatives in zip(queries, positive_doc_texts, all_hard_negatives):
            training_samples.append({
                "anchor": query,
                "positive": positive_text,
                "negatives": hard_negatives
            })
        
        # Save training samples
        result_file = result_path / f"{dataset_name}.jsonl"
        with open(result_file, "w") as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        print(f"Generated {len(training_samples)} training samples with hard negatives")
        print(f"Saved to: {result_file}")
        print()
        
        total_samples += len(training_samples)
    
    print(f"{'='*60}")
    print(f"Total training samples across all datasets: {total_samples}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse Bedrock batch output and generate training samples with hard negatives."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory containing Bedrock batch output (e.g., bedrock-output-20k)"
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="raw-corpus",
        help="Directory containing raw corpus files (default: raw-corpus)"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Directory to save the parsed training samples"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of datasets to process. If not specified, process all."
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=20,
        help="Number of hard negatives to include per sample (default: 20)"
    )
    
    args = parser.parse_args()
    
    parse_bedrock_output_with_negatives(
        output_dir=args.output_dir,
        corpus_dir=args.corpus_dir,
        result_dir=args.result_dir,
        datasets=args.datasets,
        num_negatives=args.num_negatives
    )


if __name__ == "__main__":
    main()
