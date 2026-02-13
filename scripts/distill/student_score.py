import argparse
import json
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_MAP = {
    "bge-m3": "BAAI/bge-m3",
}

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input jsonl file path")
parser.add_argument("--output", type=str, required=True, help="Output json file path")
parser.add_argument("--max_doc", type=int, default=4, help="Max number of negative docs per query")
parser.add_argument("--model", type=str, default="bge-m3", choices=list(MODEL_MAP.keys()), help="Embedding model to use")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
parser.add_argument("--max_length", type=int, default=512, help="Max input token length")
args = parser.parse_args()

model_name = MODEL_MAP[args.model]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
model.eval()


@torch.no_grad()
def encode(texts):
    all_embeddings = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(model.device)
        outputs = model(**inputs)
        # Use [CLS] token embedding
        embs = outputs.last_hidden_state[:, 0]
        embs = F.normalize(embs, p=2, dim=1)
        all_embeddings.append(embs.cpu())
        print(f"  Encoded {min(i + args.batch_size, len(texts))}/{len(texts)} texts")
    return torch.cat(all_embeddings, dim=0)


# Read input data
all_queries = []
all_docs_list = []
with open(args.input, "r") as f:
    for line in f:
        data = json.loads(line)
        query = data["anchor"]
        docs = [data["positive"]] + data["negatives"][: args.max_doc]
        all_queries.append(query)
        all_docs_list.append(docs)

# Flatten all docs
all_docs_flat = [doc for docs in all_docs_list for doc in docs]

print(f"Total queries: {len(all_queries)}, total docs: {len(all_docs_flat)}")

t0 = time.perf_counter()

# Encode queries and docs
print("Encoding queries...")
query_embs = encode(all_queries)
print("Encoding docs...")
doc_embs = encode(all_docs_flat)

# Compute scores per query
scores = []
idx = 0
for i, docs in enumerate(all_docs_list):
    q_emb = query_embs[i].unsqueeze(0).float()
    d_embs = doc_embs[idx : idx + len(docs)].float()
    sim = (q_emb @ d_embs.T).squeeze(0).tolist()
    scores.extend(sim)
    idx += len(docs)

torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
total_pairs = len(all_docs_flat)
print(f"Scoring wall clock: {elapsed:.3f}s ({elapsed / total_pairs * 1000:.2f} ms/pair)")

# Write timing metadata sidecar
meta = {
    "model": args.model,
    "batch_size": args.batch_size,
    "max_length": args.max_length,
    "num_pairs": total_pairs,
    "elapsed_s": round(elapsed, 4),
    "ms_per_pair": round(elapsed / total_pairs * 1000, 4),
}
meta_path = args.output.rsplit(".", 1)[0] + ".meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Metadata written to {meta_path}")

# Assemble output
all_samples = []
idx = 0
for query, docs in zip(all_queries, all_docs_list):
    all_samples.append({
        "query": query,
        "docs": docs,
        "scores": scores[idx : idx + len(docs)],
    })
    idx += len(docs)

with open(args.output, "w") as f:
    json.dump(all_samples, f)
