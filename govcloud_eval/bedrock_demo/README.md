# AWS Bedrock Demo for IBM watsonx Orchestrate

Deploy an agent that uses AWS Bedrock Claude via AI Gateway.

## Quick Deploy (GovCloud or Commercial)

### 1. Set AWS Credentials

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

Also update `bedrock_model_config.yaml` with your AWS credentials.

### 2. Activate Environment

```bash
source adkvenv/bin/activate
orchestrate env activate <your-environment> --api-key <your-api-key>
```

### 2. Deploy All Components

```bash
# Import model to AI Gateway
orchestrate models import -f bedrock_demo/bedrock_model_config.yaml

# Import tool
orchestrate tools import -k python -f bedrock_demo/bedrock_simple_tool.py

# Import agent
orchestrate agents import -f bedrock_demo/bedrock_simple_agent.yaml
```

### 3. Test

```bash
orchestrate agents chat bedrock_demo_agent
```

## Files

| File | Description |
|------|-------------|
| `bedrock_model_config.yaml` | AI Gateway model registration |
| `bedrock_simple_tool.py` | Python tool calling Bedrock Claude |
| `bedrock_simple_agent.yaml` | Agent using `llm: virtual-model/bedrock/...` |
