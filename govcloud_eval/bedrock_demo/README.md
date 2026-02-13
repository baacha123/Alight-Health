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
4. Select the model(s) you want to enable (e.g. Anthropic Claude, Amazon Nova)
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
5. Save the following (you will need them in Step 4):
   - `Access Key ID` (e.g. `AKIA...`)
   - `Secret Access Key` (shown only once)
6. Note your **AWS Region** where Bedrock is enabled (e.g. `us-east-1`)

---

## Step 3: Create the Model Configuration File

Create a file called `bedrock_model.yaml`:

```yaml
spec_version: v1
kind: model
name: bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0
display_name: AWS Bedrock Claude 3.5 Sonnet
description: |
  AWS Bedrock Claude 3.5 Sonnet model via AI Gateway.
tags:
  - aws
  - bedrock
  - claude
model_type: chat
provider_config:
  aws_region: us-east-1    # Replace with your region
```

> **Model name format**: `bedrock/<model-id>`
>
> Common Bedrock model IDs:
> | Model | ID |
> |-------|-----|
> | Claude 3.5 Sonnet | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` |
> | Claude 3 Haiku | `us.anthropic.claude-3-haiku-20240307-v1:0` |
> | Amazon Nova Pro | `us.amazon.nova-pro-v1:0` |
> | Amazon Nova Lite | `us.amazon.nova-lite-v1:0` |

---

## Step 4: Create Connection and Import Model

Run the following commands in your terminal (replace placeholders with your actual values):

```bash
# 1. Activate your Orchestrate environment
orchestrate env activate <your-environment> --api-key <your-orchestrate-api-key>

# 2. Create a connection to store AWS credentials
orchestrate connections add -a aws_bedrock_credentials

# 3. Configure the connection
orchestrate connections configure -a aws_bedrock_credentials --env draft -k key_value -t team

# 4. Set the AWS credentials (replace with your actual keys)
orchestrate connections set-credentials -a aws_bedrock_credentials --env draft -e "aws_access_key_id=YOUR_ACCESS_KEY_ID" -e "aws_secret_access_key=YOUR_SECRET_ACCESS_KEY"

# 5. Import the model with the connection
orchestrate models import -f bedrock_model.yaml --app-id aws_bedrock_credentials
```

### Verify

```bash
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
description: Agent powered by AWS Bedrock Claude

llm: virtual-model/bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0

instructions: |
  You are a helpful assistant powered by AWS Bedrock Claude.
  Answer questions clearly and concisely.
```

> **Important**: The `llm` field must match your model name with `virtual-model/` prefix.

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
# Setup connection
orchestrate connections add -a aws_bedrock_credentials
orchestrate connections configure -a aws_bedrock_credentials --env draft -k key_value -t team
orchestrate connections set-credentials -a aws_bedrock_credentials --env draft -e "aws_access_key_id=XXXX" -e "aws_secret_access_key=XXXX"

# Import model
orchestrate models import -f bedrock_model.yaml --app-id aws_bedrock_credentials

# Import agent
orchestrate agents import -f bedrock_agent.yaml

# Verify
orchestrate models list
orchestrate agents list

# Cleanup (if needed)
orchestrate models remove -n bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0
orchestrate agents remove my_bedrock_agent
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `401 - Invalid security token` | Wrong or expired AWS credentials | Re-run `set-credentials` with correct keys |
| `403 - Access denied` | IAM user lacks Bedrock permissions | Add `bedrock:InvokeModel` to IAM policy |
| `Model not available` | Model not enabled in Bedrock | Enable in AWS Console > Bedrock > Model Access |
| `Connection not found` | Connection not created | Run `orchestrate connections add` first |

---

## Reference

- [AI Gateway Documentation](https://developer.watson-orchestrate.ibm.com/manage-custom-llms)
- [AWS Bedrock Supported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- [AWS IAM Access Keys](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html)
