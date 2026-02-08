# GovCloud Agent Evaluation

Automated evaluation for IBM watsonx Orchestrate agents on GovCloud/FedRAMP.

---

## Quick Start

> **Already logged in?** If you have orchestrate activated and working, skip to [Step 10](#10-edit-config).

---

## FedRAMP Environment Setup

### 1. Install Python 3.13
Download from https://www.python.org/downloads/ and install.

### 2. Create Virtual Environment
```bash
python3.13 -m venv govcloud_venv
source govcloud_venv/bin/activate   # Mac/Linux
# OR
govcloud_venv\Scripts\activate      # Windows
```

### 3. Install ADK with Agent Ops
```bash
pip install "ibm-watsonx-orchestrate-adk[agentops]"
```

### 4. Fix Arize Version
The ADK has a dependency conflict with arize. Downgrade it:
```bash
pip install arize==7.26.1
```

### 5. Get Your API Key
1. Open the Orchestrate UI in your browser
2. Click **Settings** (gear icon)
3. Click **Generate API Key**
4. Copy the key and save it securely

### 6. Set Up SSL Certificate
Save your `.pem` certificate file to a known location.

### 7. Create FedRAMP Activation Script
Create a file called `fedramp_activate.py` in this directory:
```python
import subprocess
import os
import argparse

# UPDATE THIS with your certificate path
CERT_PATH = "/path/to/your/certificate.pem"

os.environ["SSL_CERT_FILE"] = CERT_PATH
os.environ["REQUESTS_CA_BUNDLE"] = CERT_PATH

parser = argparse.ArgumentParser()
parser.add_argument("env_name", help="Environment name")
parser.add_argument("--api-key", required=True, help="API key")
args = parser.parse_args()

subprocess.run([
    "orchestrate", "env", "activate", args.env_name, "--api-key", args.api_key
])
```

Update `CERT_PATH` on line 6 with your actual certificate path.

### 8. Add FedRAMP Environment
```bash
orchestrate env add -n <your-env-name> -u https://origin-api.us-gov-east-1.watson-orchestrate.ibmforusgov.com/instances/<your-instance-id> --type mcsp
```
Replace `<your-env-name>` with any name you want (e.g., `my-fedramp-env`) and `<your-instance-id>` with your instance ID.

### 9. Activate with FedRAMP Script
```bash
python fedramp_activate.py <your-env-name> --api-key <your-api-key>
```
Replace `<your-env-name>` and `<your-api-key>` with your values.

### 10. Edit Config
Open `govcloud_config.yaml` and set your test cases path:
```yaml
paths:
  test_cases: "./your_test_cases"
```

### 11. Run
```bash
python govcloud_eval.py --all
```

That's it!

---

## Commands

| Command | What it does |
|---------|--------------|
| `--setup` | Apply GovCloud patches to ADK |
| `--evaluate` | Run agent evaluation |
| `--judge` | Run LLM-as-Judge semantic evaluation |
| `--report` | Show results summary |
| `--all` | Run everything |
| `--limit N` | Only run N test cases |

### Examples
```bash
# Test with 1 case first
python govcloud_eval.py --evaluate --limit 1

# Full evaluation + LLM judge
python govcloud_eval.py --all

# Just see results
python govcloud_eval.py --report
```

---

## Test Case Format

Create JSON files in your test cases folder:

```json
{
  "agent": "your_agent_name",
  "starting_sentence": "User's question here",
  "story": "Brief description of user intent",
  "goals": {
    "your_tool-1": ["summarize"]
  },
  "goal_details": [
    {
      "type": "tool_call",
      "name": "your_tool-1",
      "tool_name": "your_tool",
      "args": {"query": "User's question here"}
    },
    {
      "type": "text",
      "name": "summarize",
      "response": "Expected answer from the agent",
      "keywords": ["key", "words", "to", "match"]
    }
  ]
}
```

---

## Output

Results are saved to `eval_results_govcloud/`:
```
eval_results_govcloud/
  2026-02-07_14-30-00/
    summary_metrics.csv     # All metrics
    messages/               # Conversation logs
    config.yml              # Config used
  llm_judge_results.json    # LLM-as-Judge results
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No active environment" | Run `orchestrate env activate <env> --api-key <key>` |
| "401 Unauthorized" | Re-run `orchestrate env activate` to refresh token |
| "Model not found" | Run `orchestrate models list` and update config |
| "Could not find agentops" | Run `pip install "ibm-watsonx-orchestrate-adk[agentops]"` |
| SSL certificate error | Set `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` env vars |
| "CERTIFICATE_VERIFY_FAILED" | Check your `.pem` file path in `fedramp_activate.py` |

---

## How It Works

This script:
1. **Patches ADK** - Fixes GovCloud compatibility issues (405b model not available, Langfuse disabled)
2. **Runs Evaluation** - Calls your agent with test questions
3. **LLM-as-Judge** - Semantically evaluates if responses are correct

Authentication is automatic - it reads your credentials from the orchestrate CLI.

---

## Files

```
govcloud_eval/
  govcloud_eval.py       # Main script
  govcloud_config.yaml   # Your configuration
  test_data/             # Your test cases go here
  eval_results_govcloud/ # Results output
  memory_tracker.md      # Development notes
```
