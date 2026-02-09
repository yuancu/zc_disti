import argparse
import json

from FlagEmbedding import LayerWiseFlagLLMReranker, LightWeightFlagLLMReranker

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input jsonl file path")
parser.add_argument("--output", type=str, required=True, help="Output json file path")
parser.add_argument("--max_doc", type=int, default=4, help="Max number of negative docs per query")
parser.add_argument("--model", type=str, default="minicpm", choices=["minicpm", "gemma2"], help="Reranker model to use")
args = parser.parse_args()

if args.model == "minicpm":
    reranker = LayerWiseFlagLLMReranker('BAAI/bge-reranker-v2-minicpm-layerwise', use_fp16=True)
else:
    reranker = LightWeightFlagLLMReranker('BAAI/bge-reranker-v2.5-gemma2-lightweight', use_fp16=True)

all_pairs = []
all_queries = []
all_docs_list = []
with open(args.input, "r") as f:
    for line in f:
        data = json.loads(line)
        query = data["anchor"]
        docs = [data["positive"]] + data["negatives"][:args.max_doc]
        all_queries.append(query)
        all_docs_list.append(docs)
        for doc in docs:
            all_pairs.append([query, doc])

print(f"Total pairs: {len(all_pairs)}")
if args.model == "minicpm":
    scores = reranker.compute_score(all_pairs, cutoff_layers=[28])
else:
    scores = reranker.compute_score(all_pairs, cutoff_layers=[28], compress_ratio=2, compress_layer=[24, 40])

# 后处理：将 scores 按 query 分组，组装成 all_samples
all_samples = []
idx = 0
for query, docs in zip(all_queries, all_docs_list):
    all_samples.append({
        "query": query,
        "docs": docs,
        "scores": scores[idx:idx+len(docs)]
    })
    idx += len(docs)

with open(args.output, "w") as f:
    json.dump(all_samples, f)
