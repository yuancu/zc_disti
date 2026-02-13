#!/bin/bash
# Upload sample queries to S3
# aws s3 cp /home/ubuntu/src/data-prep/data/eval/legal-summ/sample_queries_10.jsonl \
#   s3://zirui-bucket/Finetune_datasets/sample_queries/legal_summarization_sample_queries.jsonl

# Launch CFN stack with sample query URI
aws cloudformation create-stack \
  --region us-west-2 \
  --stack-name training-data-gen-legal-summarization-v2 \
  --template-body file://cfn/training-data-generator-seed.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --disable-rollback \
  --parameters \
    ParameterKey=JobName,ParameterValue=legal-summ-gen-seed \
    ParameterKey=S3DocumentUri,ParameterValue="s3://zirui-bucket/Finetune_datasets/Sampled_docs_for_query_generation/legal_summarization.jsonl" \
    ParameterKey=SampleQueryUri,ParameterValue="s3://zirui-bucket/Finetune_datasets/sample_queries/legal_summarization_sample_queries.jsonl"
