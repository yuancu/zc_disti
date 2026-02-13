aws cloudformation create-stack \
  --region us-west-2 \
  --stack-name training-data-gen-legal-summarization \
  --template-body file:///home/ubuntu/src/ir-fine-tune-evaluation/cfn/training-data-generator.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --disable-rollback \
  --parameters \
    ParameterKey=JobName,ParameterValue=legal-summ-gen-query \
    ParameterKey=S3DocumentUri,ParameterValue="s3://zirui-bucket/Finetune_datasets/Sampled_docs_for_query_generation/legal_summarization.jsonl"
