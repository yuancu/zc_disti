## 1.prepare corpus file
Put corpus file in a jsonl. Each line is a document.

Each line has id and text 2 fields.

## 2. deploy CFN

### option A: deploy it at aws console. CloudFormation service.

### option B: via cli

replace the jsonl uri to your uri.

```
aws cloudformation create-stack \
  --region us-west-2 \
  --stack-name training-data-gen-v1 \
  --template-body file://training-data-generator.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --disable-rollback \
  --parameters \
    ParameterKey=JobName,ParameterValue=my-training-job-v1 \
    ParameterKey=S3DocumentUri,ParameterValue="s3://zhichao-splade/test.jsonl"
```

### option C: with seeding (sample queries)

Use `training-data-generator-seed.yaml` to guide synthetic query generation with sample queries. This includes example queries in the prompt so the generated queries match a desired style.

1. Prepare a JSONL file of sample queries. Each line should have a `text` or `query` field. Up to 10 samples will be loaded as examples.

2. Upload the sample queries to S3:
```
aws s3 cp sample_queries.jsonl \
  s3://my-bucket/sample_queries/sample_queries.jsonl
```

3. Deploy the stack with the `SampleQueryUri` parameter:
```
aws cloudformation create-stack \
  --region us-west-2 \
  --stack-name training-data-gen-v1-seed \
  --template-body file://training-data-generator-seed.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --disable-rollback \
  --parameters \
    ParameterKey=JobName,ParameterValue=my-training-job-v1-seed \
    ParameterKey=S3DocumentUri,ParameterValue="s3://my-bucket/docs.jsonl" \
    ParameterKey=SampleQueryUri,ParameterValue="s3://my-bucket/sample_queries/sample_queries.jsonl"
```

`SampleQueryUri` is optional. If omitted, the template falls back to generating queries without style guidance.

## 3. monitor the task

See CloudFormation stack to verify if resources are created. 

See StepFunctions to see the generation is finished and output path.