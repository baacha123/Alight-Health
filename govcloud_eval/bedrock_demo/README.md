# AWS Bedrock Integration with IBM watsonx Orchestrate

End-to-end guide for integrating AWS Bedrock models into watsonx Orchestrate via AI Gateway.

---

## Prerequisites

- IBM watsonx Orchestrate CLI (`pip install ibm-watsonx-orchestrate`)
- An active Orchestrate environment
- AWS account with Bedrock access

---

## Step 1: Enable Bedrock Model Access in AWS

1. Log into **AWS Console** > Search for **Amazon Bedrock**
2. Go to **Model Access** (left sidebar under "Bedrock configurations")
3. Click **Modify model access**
4. Select the model(s) you want to enable
5. Click **Next** > **Submit**
6. Wait for status to show **Access granted**

> Note: Some models require you to accept terms before access is granted.

---

## Step 2: Create AWS IAM Credentials

1. Go to **AWS Console** > **IAM** > **Users**
2. Select or create a user for Bedrock access
3. Ensure the user has the following IAM permission:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "*"
    }
  ]
}
```

4. Go to **Security Credentials** tab > **Create access key**
5. Save the following (you will need them in Step 3):
   - `Access Key ID` (e.g. `AKIA...`)
   - `Secret Access Key` (shown only once)
6. Note your **AWS Region** where Bedrock is enabled (e.g. `us-east-1`)

---

## Step 3: Create the Model Configuration File

Create a file called `bedrock_model.yaml` with your credentials and model ID:

```yaml
spec_version: v1
kind: model
name: bedrock/YOUR_BEDROCK_MODEL_ID
display_name: AWS Bedrock Model
description: |
  AWS Bedrock model via AI Gateway.
tags:
  - aws
  - bedrock
model_type: chat
provider_config:
  aws_region: YOUR_AWS_REGION
  aws_access_key_id: YOUR_AWS_ACCESS_KEY_ID
  aws_secret_access_key: YOUR_AWS_SECRET_ACCESS_KEY
```

Replace the placeholders:
- `YOUR_BEDROCK_MODEL_ID` - The model ID from your enabled Bedrock models (see table below)
- `YOUR_AWS_REGION` - Region where Bedrock is enabled (e.g. `us-east-1`)
- `YOUR_AWS_ACCESS_KEY_ID` - From Step 2
- `YOUR_AWS_SECRET_ACCESS_KEY` - From Step 2

> **Common Bedrock model IDs** (use whichever you have enabled):
>
> | Model | ID |
> |-------|-----|
> | Claude 3.5 Sonnet | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` |
> | Claude 3 Haiku | `us.anthropic.claude-3-haiku-20240307-v1:0` |
> | Amazon Nova Pro | `us.amazon.nova-pro-v1:0` |
> | Amazon Nova Lite | `us.amazon.nova-lite-v1:0` |
>
> Full list: [AWS Bedrock Supported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)

---

## Step 4: Import Model to Orchestrate

```bash
# 1. Activate your Orchestrate environment
orchestrate env activate <your-environment> --api-key <your-orchestrate-api-key>

# 2. Import the model
orchestrate models import -f bedrock_model.yaml

# 3. Verify
orchestrate models list
```

You should see your Bedrock model listed with a `✨️` icon (custom provider).

---

## Step 5: Create and Import an Agent

Create a file called `bedrock_agent.yaml`:

```yaml
spec_version: v1
kind: native
name: my_bedrock_agent
description: Agent powered by AWS Bedrock via AI Gateway

llm: virtual-model/bedrock/YOUR_BEDROCK_MODEL_ID

instructions: |
  You are a helpful assistant.
  Answer questions clearly and concisely.
```

> **Important**: The `llm` field must use `virtual-model/bedrock/` prefix followed by the same model ID from Step 3.

Import the agent:

```bash
orchestrate agents import -f bedrock_agent.yaml
```

---

## Step 6: Test

Test via the Orchestrate UI:
1. Open your Orchestrate UI
2. Find the agent (e.g. `my_bedrock_agent`)
3. Ask: "What is the capital of France?"

---

## Quick Reference: All Commands

```bash
# Import model (credentials are in the yaml)
orchestrate models import -f bedrock_model.yaml

# Import agent
orchestrate agents import -f bedrock_agent.yaml

# Verify
orchestrate models list
orchestrate agents list

# Cleanup (if needed)
orchestrate models remove -n bedrock/YOUR_BEDROCK_MODEL_ID
orchestrate agents remove my_bedrock_agent
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `401 - Invalid security token` | Wrong or expired AWS credentials | Update credentials in yaml and re-import |
| `403 - Access denied` | IAM user lacks Bedrock permissions | Add `bedrock:InvokeModel` to IAM policy |
| `Model not available` | Model not enabled in Bedrock | Enable in AWS Console > Bedrock > Model Access |

---

## Reference

- [AI Gateway Documentation](https://developer.watson-orchestrate.ibm.com/manage-custom-llms)
- [AWS Bedrock Supported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- [AWS IAM Access Keys](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html)
