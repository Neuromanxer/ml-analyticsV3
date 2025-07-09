import boto3
def launch_job_on_ecs(
    user_id: str,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = "target",
    drop_columns: str = ""
):
    """
    Launch an isolated ECS task (Fargate) that runs do_classification inside its own container.
    """
    
    # Upload request parameters to S3 so the ECS container can pull them
    request_payload = {
        "user_id": user_id,
        "file_path": file_path,
        "train_path": train_path,
        "test_path": test_path,
        "target_column": target_column,
        "drop_columns": drop_columns
    }
    
    import json, uuid
    import boto3
    
    s3 = boto3.client('s3')
    request_key = f"requests/{user_id}/{uuid.uuid4()}.json"
    s3.put_object(
        Bucket="ml-insights-job-requests",
        Key=request_key,
        Body=json.dumps(request_payload)
    )
    
    request_s3_uri = f"s3://ml-insights-job-requests/{request_key}"
    
    # Launch ECS Fargate task
    ecs = boto3.client('ecs')
    response = ecs.run_task(
        cluster="ml-insights-cluster",
        launchType="FARGATE",
        taskDefinition="ml-insights-classification-job",
        overrides={
            'containerOverrides': [{
                'name': 'classification-container',
                'command': [
                    'python', 'run_job.py', request_s3_uri
                ]
            }]
        },
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': ['subnet-xxxxxxx'],  # Your subnet IDs
                'assignPublicIp': 'ENABLED'
            }
        }
    )
    
    return response
