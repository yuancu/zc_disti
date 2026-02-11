# requires transformers < 4.46
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_MAP = {
    "minicpm": "BAAI/bge-reranker-v2-minicpm-layerwise",
    "gemma2": "BAAI/bge-reranker-v2.5-gemma2-lightweight",
}

PROMPT_MAP = {
    "minicpm": "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'.",
    "gemma2": "Predict whether passage B contains an answer to query A.",
}

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input jsonl file path")
parser.add_argument("--output", type=str, required=True, help="Output json file path")
parser.add_argument("--max_doc", type=int, default=4, help="Max number of negative docs per query")
parser.add_argument("--model", type=str, default="minicpm", choices=["minicpm", "gemma2"], help="Reranker model to use")
parser.add_argument("--quantize", type=str, default="none", choices=["none", "int8", "int4"], help="Quantization mode")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
parser.add_argument("--max_length", type=int, default=1024, help="Max input token length")
args = parser.parse_args()

model_name = MODEL_MAP[args.model]
prompt = PROMPT_MAP[args.model]

# Build model loading kwargs
model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
if args.quantize == "int8":
    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
elif args.quantize == "int4":
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
else:
    model_kwargs["torch_dtype"] = torch.float16

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if args.model == "gemma2":
    tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
model.eval()


def get_inputs(pairs):
    sep = "\n"
    prompt_ids = tokenizer(prompt, return_tensors=None, add_special_tokens=False)["input_ids"]
    sep_ids = tokenizer(sep, return_tensors=None, add_special_tokens=False)["input_ids"]

    inputs = []
    query_lengths = []
    prompt_lengths = []
    for query, passage in pairs:
        query_inputs = tokenizer(
            f"A: {query}", return_tensors=None, add_special_tokens=False,
            max_length=args.max_length * 3 // 4, truncation=True,
        )
        passage_inputs = tokenizer(
            f"B: {passage}", return_tensors=None, add_special_tokens=False,
            max_length=args.max_length, truncation=True,
        )
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs["input_ids"],
            sep_ids + passage_inputs["input_ids"],
            truncation="only_second",
            max_length=args.max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        item["input_ids"] = item["input_ids"] + sep_ids + prompt_ids
        item["attention_mask"] = [1] * len(item["input_ids"])
        inputs.append(item)
        query_lengths.append(len([tokenizer.bos_token_id] + query_inputs["input_ids"] + sep_ids))
        prompt_lengths.append(len(sep_ids + prompt_ids))

    padded = tokenizer.pad(
        inputs,
        padding=True,
        max_length=args.max_length + len(sep_ids) + len(prompt_ids),
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    return padded, query_lengths, prompt_lengths


def last_logit_pool(logits, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return logits[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = logits.shape[0]
    return torch.stack([logits[i, sequence_lengths[i]] for i in range(batch_size)], dim=0)


@torch.no_grad()
def compute_scores(all_pairs):
    all_scores = []
    for i in range(0, len(all_pairs), args.batch_size):
        batch = all_pairs[i : i + args.batch_size]
        inputs, query_lengths, prompt_lengths = get_inputs(batch)
        inputs = inputs.to(model.device)

        if args.model == "minicpm":
            outputs = model(**inputs, return_dict=True, cutoff_layers=[28])
            layer_scores = [s[:, -1].view(-1).float() for s in outputs[0]]
            batch_scores = layer_scores[-1].cpu().tolist()
        else:
            outputs = model(
                **inputs,
                return_dict=True,
                cutoff_layers=[28],
                compress_ratio=2,
                compress_layer=[24, 40],
                query_lengths=query_lengths,
                prompt_lengths=prompt_lengths,
            )
            logits = last_logit_pool(outputs.logits[-1], outputs.attention_masks[-1])
            batch_scores = logits.cpu().float().tolist()

        all_scores.extend(batch_scores)
        print(f"  Scored {min(i + args.batch_size, len(all_pairs))}/{len(all_pairs)} pairs")

    return all_scores


# Read input data
all_pairs = []
all_queries = []
all_docs_list = []
with open(args.input, "r") as f:
    for line in f:
        data = json.loads(line)
        query = data["anchor"]
        docs = [data["positive"]] + data["negatives"][: args.max_doc]
        all_queries.append(query)
        all_docs_list.append(docs)
        for doc in docs:
            all_pairs.append([query, doc])

print(f"Total pairs: {len(all_pairs)}")

t0 = time.perf_counter()
scores = compute_scores(all_pairs)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"compute_scores wall clock: {elapsed:.3f}s ({elapsed / len(all_pairs) * 1000:.2f} ms/pair)")

# Write timing metadata sidecar
meta = {
    "model": args.model,
    "quantize": args.quantize,
    "batch_size": args.batch_size,
    "max_length": args.max_length,
    "num_pairs": len(all_pairs),
    "elapsed_s": round(elapsed, 4),
    "ms_per_pair": round(elapsed / len(all_pairs) * 1000, 4),
}
meta_path = args.output.rsplit(".", 1)[0] + ".meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Metadata written to {meta_path}")

# 后处理：将 scores 按 query 分组，组装成 all_samples
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
