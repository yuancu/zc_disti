#!/usr/bin/env bash
set -euo pipefail

# Dense Full Fine-Tune v1: LR=5e-6 (same as LoRA v4).
# 15 jobs = 15 datasets. All parameters trained (~305M).
# batch=16, grad_accum=2 (effective=32) to fit in 48GB VRAM.

REGION="us-east-2"
BUCKET="sagemaker-us-east-2-164763751923"
S3_PREFIX="dense-full-v1"
ROLE_ARN="arn:aws:iam::164763751923:role/service-role/AmazonSageMaker-ExecutionRole-20231227T111898"
IMAGE="763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04"
INSTANCE_TYPE="ml.g6e.xlarge"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATASETS=(
    "aila-casedocs|AILA_casedocs"
    "amazon-esci|amazon-esci"
    "chatdoctor-healthcaremagic|ChatDoctor_HealthCareMagic"
    "curev1-en|CUREv1_en"
    "ds-1000|DS-1000"
    "financebench|financebench"
    "finqa|finqa"
    "hc3finance|HC3Finance"
    "legal-summarization|legal_summarization"
    "legalquad|LegalQuAD"
    "medquad|MedQuAD"
    "multi-cpr-ecom|multi-cpr-ecom"
    "multi-cpr-medical|multi-cpr-medical"
    "multi-cpr-video|multi-cpr-video"
    "nfcorpus|NFCorpus"
)

# ============================================================
# Step 1: Package and upload training code
# ============================================================
echo "Packaging training code..."
CODE_DIR=$(mktemp -d)
cp "$SCRIPT_DIR/train_dense_fullft.py" "$CODE_DIR/train_dense_fullft.py"
cat > "$CODE_DIR/requirements.txt" << 'EOF'
sentence-transformers>=4.1.0
EOF
tar -czf /tmp/dense_fullft_v1_train_sourcedir.tar.gz -C "$CODE_DIR" .
aws s3 cp /tmp/dense_fullft_v1_train_sourcedir.tar.gz "s3://${BUCKET}/${S3_PREFIX}/code/train_sourcedir.tar.gz" --region "$REGION"
rm -rf "$CODE_DIR" /tmp/dense_fullft_v1_train_sourcedir.tar.gz
echo "Code uploaded."

# ============================================================
# Step 2: Upload training + dev data (reuse from v3/v4)
# ============================================================
echo ""
echo "Uploading training and dev data..."
for entry in "${DATASETS[@]}"; do
    IFS='|' read -r local_name eval_name <<< "$entry"

    S3_TRAIN="s3://${BUCKET}/${S3_PREFIX}/data/${local_name}/train/"
    S3_DEV="s3://${BUCKET}/${S3_PREFIX}/data/${local_name}/dev/"

    if aws s3 ls "${S3_TRAIN}scored_training_data.json" --region "$REGION" > /dev/null 2>&1; then
        echo "  $local_name train: already uploaded"
    elif aws s3 ls "s3://${BUCKET}/dense-lora-v4/data/${local_name}/train/scored_training_data.json" --region "$REGION" > /dev/null 2>&1; then
        echo "  $local_name train: copying from v4..."
        aws s3 cp "s3://${BUCKET}/dense-lora-v4/data/${local_name}/train/scored_training_data.json" "${S3_TRAIN}scored_training_data.json" --region "$REGION"
    elif aws s3 ls "s3://${BUCKET}/dense-lora-v3/data/${local_name}/train/scored_training_data.json" --region "$REGION" > /dev/null 2>&1; then
        echo "  $local_name train: copying from v3..."
        aws s3 cp "s3://${BUCKET}/dense-lora-v3/data/${local_name}/train/scored_training_data.json" "${S3_TRAIN}scored_training_data.json" --region "$REGION"
    else
        echo "  SKIP $local_name: no training data found"
        continue
    fi

    if aws s3 ls "${S3_DEV}dev_data.jsonl" --region "$REGION" > /dev/null 2>&1; then
        echo "  $local_name dev: already uploaded"
    elif aws s3 ls "s3://${BUCKET}/dense-lora-v4/data/${local_name}/dev/dev_data.jsonl" --region "$REGION" > /dev/null 2>&1; then
        echo "  $local_name dev: copying from v4..."
        aws s3 cp "s3://${BUCKET}/dense-lora-v4/data/${local_name}/dev/dev_data.jsonl" "${S3_DEV}dev_data.jsonl" --region "$REGION"
        aws s3 cp "s3://${BUCKET}/dense-lora-v4/data/${local_name}/dev/dev_labels.jsonl" "${S3_DEV}dev_labels.jsonl" --region "$REGION"
    elif aws s3 ls "s3://${BUCKET}/dense-lora-v3/data/${local_name}/dev/dev_data.jsonl" --region "$REGION" > /dev/null 2>&1; then
        echo "  $local_name dev: copying from v3..."
        aws s3 cp "s3://${BUCKET}/dense-lora-v3/data/${local_name}/dev/dev_data.jsonl" "${S3_DEV}dev_data.jsonl" --region "$REGION"
        aws s3 cp "s3://${BUCKET}/dense-lora-v3/data/${local_name}/dev/dev_labels.jsonl" "${S3_DEV}dev_labels.jsonl" --region "$REGION"
    else
        echo "  WARNING: no dev data for $local_name"
    fi
done

# ============================================================
# Step 3: Launch training jobs
# ============================================================
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TOTAL=0
LAUNCHED=0

MAX_RUNTIME=14400
MAX_WAIT=$((MAX_RUNTIME * 2))

echo ""
echo "========================================"
echo "Launching Dense Full FT v1 training jobs (LR=5e-6)"
echo "========================================"

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r local_name eval_name <<< "$entry"
    TOTAL=$((TOTAL + 1))

    SAFE_NAME=$(echo "$local_name" | tr '_' '-')
    JOB_NAME="df-v1-${SAFE_NAME}-${TIMESTAMP}"
    JOB_NAME=$(echo "$JOB_NAME" | cut -c1-63 | sed 's/-$//')

    S3_TRAIN="s3://${BUCKET}/${S3_PREFIX}/data/${local_name}/train/"
    S3_DEV="s3://${BUCKET}/${S3_PREFIX}/data/${local_name}/dev/"
    S3_OUTPUT="s3://${BUCKET}/${S3_PREFIX}/output"

    echo ""
    echo "[$local_name] $JOB_NAME"

    aws sagemaker create-training-job \
        --region "$REGION" \
        --training-job-name "$JOB_NAME" \
        --role-arn "$ROLE_ARN" \
        --algorithm-specification '{
            "TrainingImage": "'"$IMAGE"'",
            "TrainingInputMode": "File",
            "EnableSageMakerMetricsTimeSeries": false
        }' \
        --hyper-parameters '{
            "sagemaker_program": "train_dense_fullft.py",
            "sagemaker_submit_directory": "s3://'"$BUCKET"'/'"$S3_PREFIX"'/code/train_sourcedir.tar.gz",
            "model_name": "Alibaba-NLP/gte-multilingual-base",
            "score_scale": "0.025",
            "score_scale_mode": "multiply",
            "num_negatives": "4",
            "train_batch_size": "16",
            "gradient_accumulation_steps": "2",
            "learning_rate": "5e-6",
            "max_steps": "2000",
            "max_seq_length": "512",
            "warmup_ratio": "0.1",
            "logging_steps": "50",
            "eval_steps": "50",
            "early_stopping_patience": "5",
            "eval_on_start": "true"
        }' \
        --input-data-config '[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "'"$S3_TRAIN"'",
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "CompressionType": "None"
            },
            {
                "ChannelName": "dev",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "'"$S3_DEV"'",
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "CompressionType": "None"
            }
        ]' \
        --output-data-config '{"S3OutputPath": "'"$S3_OUTPUT"'"}' \
        --resource-config '{
            "InstanceType": "'"$INSTANCE_TYPE"'",
            "InstanceCount": 1,
            "VolumeSizeInGB": 100
        }' \
        --stopping-condition '{
            "MaxRuntimeInSeconds": '"$MAX_RUNTIME"',
            "MaxWaitTimeInSeconds": '"$MAX_WAIT"'
        }' \
        --enable-managed-spot-training \
        --query 'TrainingJobArn' --output text 2>&1 && LAUNCHED=$((LAUNCHED + 1))

    sleep 0.5
done

echo ""
echo "========================================"
echo "Launched $LAUNCHED / $TOTAL Dense Full FT v1 training jobs"
echo "========================================"
