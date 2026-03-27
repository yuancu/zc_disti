#!/usr/bin/env bash
set -euo pipefail

# Launch eval jobs for completed Dense Full FT v1 (LR=5e-6) training jobs.

REGION="us-east-2"
BUCKET="sagemaker-us-east-2-164763751923"
S3_PREFIX="dense-full-v1"
ROLE_ARN="arn:aws:iam::164763751923:role/service-role/AmazonSageMaker-ExecutionRole-20231227T111898"
IMAGE="763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04"
INSTANCE_TYPE="ml.g6e.xlarge"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_DATA_DIR="$PROJECT_DIR/data/eval"

declare -A NAME_TO_EVAL=(
    [aila-casedocs]="AILA_casedocs"
    [amazon-esci]="amazon-esci"
    [chatdoctor-healthcaremagic]="ChatDoctor_HealthCareMagic"
    [curev1-en]="CUREv1_en"
    [ds-1000]="DS-1000"
    [financebench]="financebench"
    [finqa]="finqa"
    [hc3finance]="HC3Finance"
    [legal-summarization]="legal_summarization"
    [legalquad]="LegalQuAD"
    [medquad]="MedQuAD"
    [multi-cpr-ecom]="multi-cpr-ecom"
    [multi-cpr-medical]="multi-cpr-medical"
    [multi-cpr-video]="multi-cpr-video"
    [nfcorpus]="NFCorpus"
)

# ============================================================
# Step 1: Package eval code
# ============================================================
echo "Packaging eval code..."
CODE_DIR=$(mktemp -d)
cp "$SCRIPT_DIR/eval_dense_lora.py" "$CODE_DIR/eval_dense_lora.py"
cp "$SCRIPT_DIR/../eval/evaluate_model.py" "$CODE_DIR/evaluate_model.py"
cat > "$CODE_DIR/requirements.txt" << 'EOF'
sentence-transformers>=4.1.0
EOF
tar -czf /tmp/dense_fullft_v1_eval_code.tar.gz -C "$CODE_DIR" .
aws s3 cp /tmp/dense_fullft_v1_eval_code.tar.gz "s3://${BUCKET}/${S3_PREFIX}/eval-code/sourcedir.tar.gz" --region "$REGION"
rm -rf "$CODE_DIR" /tmp/dense_fullft_v1_eval_code.tar.gz
echo "Eval code uploaded."

# ============================================================
# Step 2: Upload eval data (reuse from v4/v3)
# ============================================================
echo ""
echo "Uploading eval data..."
for EVAL_DATASET in "${NAME_TO_EVAL[@]}"; do
    S3_EVAL="s3://${BUCKET}/${S3_PREFIX}/eval-data/${EVAL_DATASET}/"
    if aws s3 ls "${S3_EVAL}corpus.jsonl" --region "$REGION" > /dev/null 2>&1; then
        echo "  $EVAL_DATASET: already uploaded"
    elif aws s3 ls "s3://${BUCKET}/dense-lora-v4/eval-data/${EVAL_DATASET}/corpus.jsonl" --region "$REGION" > /dev/null 2>&1; then
        echo "  $EVAL_DATASET: copying from v4..."
        aws s3 cp "s3://${BUCKET}/dense-lora-v4/eval-data/${EVAL_DATASET}/corpus.jsonl" "${S3_EVAL}corpus.jsonl" --region "$REGION"
        aws s3 cp "s3://${BUCKET}/dense-lora-v4/eval-data/${EVAL_DATASET}/queries.jsonl" "${S3_EVAL}queries.jsonl" --region "$REGION"
        aws s3 cp "s3://${BUCKET}/dense-lora-v4/eval-data/${EVAL_DATASET}/relevance.jsonl" "${S3_EVAL}relevance.jsonl" --region "$REGION"
    elif aws s3 ls "s3://${BUCKET}/dense-lora-v3/eval-data/${EVAL_DATASET}/corpus.jsonl" --region "$REGION" > /dev/null 2>&1; then
        echo "  $EVAL_DATASET: copying from v3..."
        aws s3 cp "s3://${BUCKET}/dense-lora-v3/eval-data/${EVAL_DATASET}/corpus.jsonl" "${S3_EVAL}corpus.jsonl" --region "$REGION"
        aws s3 cp "s3://${BUCKET}/dense-lora-v3/eval-data/${EVAL_DATASET}/queries.jsonl" "${S3_EVAL}queries.jsonl" --region "$REGION"
        aws s3 cp "s3://${BUCKET}/dense-lora-v3/eval-data/${EVAL_DATASET}/relevance.jsonl" "${S3_EVAL}relevance.jsonl" --region "$REGION"
    else
        LOCAL_EVAL="$EVAL_DATA_DIR/$EVAL_DATASET"
        if [ -d "$LOCAL_EVAL" ]; then
            echo "  $EVAL_DATASET: uploading..."
            aws s3 cp "$LOCAL_EVAL/corpus.jsonl" "${S3_EVAL}corpus.jsonl" --region "$REGION"
            aws s3 cp "$LOCAL_EVAL/queries.jsonl" "${S3_EVAL}queries.jsonl" --region "$REGION"
            aws s3 cp "$LOCAL_EVAL/relevance.jsonl" "${S3_EVAL}relevance.jsonl" --region "$REGION"
        else
            echo "  SKIP $EVAL_DATASET: no eval data"
        fi
    fi
done

# ============================================================
# Step 3: Launch eval jobs for completed v1 training jobs
# ============================================================
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LAUNCHED=0
SKIPPED=0

MAX_RUNTIME=7200
MAX_WAIT=$((MAX_RUNTIME * 2))

echo ""
echo "========================================"
echo "Launching Dense Full FT v1 eval jobs"
echo "========================================"

# Collect all completed v1 training jobs with pagination
ALL_TRAIN_JOBS=""
NEXT_TOKEN=""
while true; do
    CMD=(aws sagemaker list-training-jobs --region "$REGION" --name-contains "df-v1" --status-equals Completed --max-results 100 --output json)
    if [ -n "$NEXT_TOKEN" ]; then
        CMD+=(--next-token "$NEXT_TOKEN")
    fi
    RESULT=$("${CMD[@]}")
    JOBS=$(echo "$RESULT" | python3 -c "import sys,json; [print(j['TrainingJobName']) for j in json.load(sys.stdin).get('TrainingJobSummaries',[])]")
    ALL_TRAIN_JOBS="$ALL_TRAIN_JOBS $JOBS"
    NEXT_TOKEN=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('NextToken',''))" 2>/dev/null || echo "")
    if [ -z "$NEXT_TOKEN" ]; then
        break
    fi
done

for TRAIN_JOB in $ALL_TRAIN_JOBS; do
    # Parse: df-v1-{dataset}-{timestamp}
    SUFFIX="${TRAIN_JOB#df-v1-}"

    # Extract dataset (remove timestamp)
    DATASET=$(echo "$SUFFIX" | sed 's/-[0-9]\{8\}-[0-9]\{6\}.*$//')
    if [ -z "$DATASET" ] || [ "$DATASET" = "$SUFFIX" ]; then
        DATASET=$(echo "$SUFFIX" | sed 's/-[0-9]\{8\}-[0-9]*$//')
    fi
    if [ -z "$DATASET" ] || [ "$DATASET" = "$SUFFIX" ]; then
        DATASET=$(echo "$SUFFIX" | sed 's/-[0-9]\{7,\}$//')
    fi

    EVAL_DATASET="${NAME_TO_EVAL[$DATASET]:-}"
    if [ -z "$EVAL_DATASET" ]; then
        echo "  WARNING: Unknown dataset '$DATASET' from $TRAIN_JOB, skipping"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Check if eval already exists
    EVAL_CHECK="ef-v1-${DATASET}"
    EXISTING=$(aws sagemaker list-training-jobs \
        --region "$REGION" \
        --name-contains "$EVAL_CHECK" \
        --query 'length(TrainingJobSummaries)' \
        --output text 2>/dev/null || echo "0")
    if [ "$EXISTING" -gt 0 ] 2>/dev/null; then
        echo "  $EVAL_CHECK: eval already exists, skipping"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Get model artifact
    MODEL_S3=$(aws sagemaker describe-training-job \
        --training-job-name "$TRAIN_JOB" \
        --region "$REGION" \
        --query 'ModelArtifacts.S3ModelArtifacts' \
        --output text)

    EVAL_JOB_NAME="ef-v1-${DATASET}-${TIMESTAMP}"
    EVAL_JOB_NAME=$(echo "$EVAL_JOB_NAME" | cut -c1-63 | sed 's/-$//')

    S3_EVAL="s3://${BUCKET}/${S3_PREFIX}/eval-data/${EVAL_DATASET}/"
    S3_OUTPUT="s3://${BUCKET}/${S3_PREFIX}/eval-output"

    echo "  Launching: $EVAL_JOB_NAME"

    aws sagemaker create-training-job \
        --region "$REGION" \
        --training-job-name "$EVAL_JOB_NAME" \
        --role-arn "$ROLE_ARN" \
        --algorithm-specification '{"TrainingImage":"'"$IMAGE"'","TrainingInputMode":"File","EnableSageMakerMetricsTimeSeries":false}' \
        --hyper-parameters '{
            "sagemaker_program": "eval_dense_lora.py",
            "sagemaker_submit_directory": "s3://'"$BUCKET"'/'"$S3_PREFIX"'/eval-code/sourcedir.tar.gz",
            "dataset_name": "'"$EVAL_DATASET"'",
            "batch_size": "64",
            "eval_baseline": "false"
        }' \
        --input-data-config '[
            {
                "ChannelName": "eval",
                "DataSource": {"S3DataSource": {"S3DataType":"S3Prefix","S3Uri":"'"$S3_EVAL"'","S3DataDistributionType":"FullyReplicated"}},
                "CompressionType": "None"
            },
            {
                "ChannelName": "model",
                "DataSource": {"S3DataSource": {"S3DataType":"S3Prefix","S3Uri":"'"$MODEL_S3"'","S3DataDistributionType":"FullyReplicated"}},
                "ContentType": "application/x-sagemaker-model",
                "CompressionType": "None"
            }
        ]' \
        --output-data-config '{"S3OutputPath":"'"$S3_OUTPUT"'"}' \
        --resource-config '{"InstanceType":"'"$INSTANCE_TYPE"'","InstanceCount":1,"VolumeSizeInGB":50}' \
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
echo "Launched $LAUNCHED eval jobs (skipped $SKIPPED)"
echo "========================================"
