"""
Simple AWS Bedrock Tool for watsonx Orchestrate
Calls Claude 3 Haiku on AWS Bedrock
"""

import boto3
import json
import os
from ibm_watsonx_orchestrate.agent_builder.tools import tool


def call_bedrock(prompt: str) -> str:
    """Call AWS Bedrock Claude model."""

    # Get credentials from environment variables
    # Set these before running: export AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')

    if not aws_access_key or not aws_secret_key:
        return "Error: Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"

    try:
        bedrock = boto3.client(
            'bedrock-runtime',
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )

        # Call Claude 3 Haiku (fast and cost-effective)
        response = bedrock.invoke_model(
            modelId='us.anthropic.claude-3-haiku-20240307-v1:0',
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 500,
                'messages': [{'role': 'user', 'content': prompt}]
            })
        )

        result = json.loads(response['body'].read())
        return result['content'][0]['text']

    except Exception as e:
        return f"Error calling Bedrock: {str(e)}"


@tool
def ask_bedrock(question: str) -> str:
    """
    Ask AWS Bedrock Claude to answer a question.
    Uses Claude 3 Haiku model for fast responses.

    Args:
        question: The question to ask Claude

    Returns:
        Claude's response from AWS Bedrock
    """
    prompt = f"""You are a helpful assistant. Answer this question concisely:

{question}

Keep your response brief and helpful."""

    return call_bedrock(prompt)
