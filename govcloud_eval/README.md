# GovCloud Agent Evaluation

Automated evaluation for IBM watsonx Orchestrate agents on GovCloud/FedRAMP.

---

## Quick Start

> **Already logged in?** If you have orchestrate activated and working, skip to [Step 10](#10-edit-config).

> **Already have test cases?** Skip to [Run Evaluation](#run-evaluation).

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
#!/usr/bin/env python3
from typing import Annotated
import typer
import requests

from ibm_watsonx_orchestrate.cli.commands.tools.types import RegistryType
from ibm_watsonx_orchestrate.cli.config import Config, AUTH_CONFIG_FILE_FOLDER, AUTH_CONFIG_FILE, \
    AUTH_CONFIG_FILE_CONTENT, ENV_WXO_URL_OPT, AUTH_SECTION_HEADER, AUTH_MCSP_TOKEN_OPT, PYTHON_REGISTRY_HEADER, \
    PYTHON_REGISTRY_TYPE_OPT, PYTHON_REGISTRY_TEST_PACKAGE_VERSION_OVERRIDE_OPT, PYTHON_REGISTRY_SKIP_VERSION_CHECK_OPT, \
    CONTEXT_SECTION_HEADER, CONTEXT_ACTIVE_ENV_OPT, ENVIRONMENTS_SECTION_HEADER, DEFAULT_CONFIG_FILE_CONTENT
from ibm_watsonx_orchestrate.client.utils import check_token_validity, is_local_dev

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False
)

@app.command(name='activate', no_args_is_help=True)
def activate(
        name: Annotated[
            str,
            typer.Argument(),
        ],
        apikey: Annotated[
            str,
            typer.Option(
                "--api-key", "-a", help="WXO API Key."
            ),
        ],
        registry: Annotated[
            RegistryType,
            typer.Option("--registry", help="Which registry to use when importing python tools", hidden=True),
        ] = None,
        test_package_version_override: Annotated[
            str,
            typer.Option("--test-package-version-override", help="Which prereleased package version to reference when using --registry testpypi", hidden=True),
        ] = None,
        skip_version_check: Annotated[
            bool,
            typer.Option('--skip-version-check/--enable-version-check', help='Use this flag to skip validating that adk version in use exists in pypi (for clients who mirror the ADK to a local registry and do not have local access to pypi).')
        ] = None
):
    cfg = Config()
    auth_cfg = Config(AUTH_CONFIG_FILE_FOLDER, AUTH_CONFIG_FILE)
    env_cfg = cfg.read(ENVIRONMENTS_SECTION_HEADER, name)
    url = cfg.get(ENVIRONMENTS_SECTION_HEADER, name, ENV_WXO_URL_OPT)
    is_local = is_local_dev(url)

    if not env_cfg:
        print(f"Environment '{name}' does not exist. Please create it with `orchestrate env add`")
        return
    elif not env_cfg.get(ENV_WXO_URL_OPT):
        print(f"Environment '{name}' is misconfigured. Please re-create it with `orchestrate env add`")
        return

    existing_auth_config = auth_cfg.get(AUTH_SECTION_HEADER).get(name, {})
    existing_token = existing_auth_config.get(AUTH_MCSP_TOKEN_OPT) if existing_auth_config else None

    if not check_token_validity(existing_token) or is_local:
        try:
            resp = requests.post(
                "https://dai.ibmforusgov.com/api/rest/mcsp/apikeys/token",
                json={'apikey': apikey}, verify="/path/to/your/certificate.pem"
            )
            resp.raise_for_status()
            resp = resp.json()
            auth_cfg.save(
                {
                    AUTH_SECTION_HEADER: {
                        name: {
                            'wxo_mcsp_token': resp['token'],
                            'wxo_mcsp_token_expiry': resp['expiration']
                        }
                    },
                }
            )
        except requests.exceptions.HTTPError as e:
            print(e.response)
            exit(1)


    cfg.write(CONTEXT_SECTION_HEADER, CONTEXT_ACTIVE_ENV_OPT, name)
    if registry is not None:
        cfg.write(PYTHON_REGISTRY_HEADER, PYTHON_REGISTRY_TYPE_OPT, str(registry))
        cfg.write(PYTHON_REGISTRY_HEADER, PYTHON_REGISTRY_TEST_PACKAGE_VERSION_OVERRIDE_OPT, test_package_version_override)
    elif cfg.read(PYTHON_REGISTRY_HEADER, PYTHON_REGISTRY_TYPE_OPT) is None:
        cfg.write(PYTHON_REGISTRY_HEADER, PYTHON_REGISTRY_TYPE_OPT, DEFAULT_CONFIG_FILE_CONTENT[PYTHON_REGISTRY_HEADER][PYTHON_REGISTRY_TYPE_OPT])
        cfg.write(PYTHON_REGISTRY_HEADER, PYTHON_REGISTRY_TEST_PACKAGE_VERSION_OVERRIDE_OPT, test_package_version_override)
    if skip_version_check is not None:
        cfg.write(PYTHON_REGISTRY_HEADER, PYTHON_REGISTRY_SKIP_VERSION_CHECK_OPT, skip_version_check)

    print(f"Environment '{name}' is now active")

if __name__ == "__main__":
    app()
```

**Update line 63** with the path to your `.pem` certificate file.

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
Open `govcloud_config.yaml` and set your agent name:
```yaml
agent:
  name: "your_agent_name"
```

### 11. Generate or Record Test Cases
See [Generate Test Cases](#generate-test-cases-from-excel) or [Record Test Cases](#record-test-cases-from-chat) below.

### 12. Run Evaluation
```bash
python govcloud_eval.py --all
```

That's it!

---

## Generate Test Cases from Excel

If you have questions and expected answers in an Excel file, use this to generate test cases with LLM-powered keyword extraction.

### Step 1: Create Your Excel File

Create `questions.xlsx` with these columns:

| Question | Expected Answer |
|----------|-----------------|
| What is my deductible? | Your deductible is $500 for individual coverage and $1,000 for family coverage per calendar year. |
| How do I add a dependent? | You have 31 days from your life event to add dependents via the benefits portal. |

Optional columns: `Source Document`, `Source Page`

### Step 2: Run Generate

```bash
python govcloud_generate.py
```

This reads `questions.xlsx`, uses LLM to extract keywords, and outputs test cases to `./test_data/`.

### Step 3: Update Config

After generation, update `govcloud_config.yaml` with the test cases path:
```yaml
paths:
  test_cases: "./test_data"
```

### Generate Options

```bash
python govcloud_generate.py --excel myfile.xlsx   # Custom Excel file
python govcloud_generate.py --limit 5             # Only generate 5 cases
python govcloud_generate.py --skip-llm            # Skip LLM (faster, simpler keywords)
python govcloud_generate.py --sample              # Create sample Excel template
```

---

## Record Test Cases from Chat

Record live chat sessions with your agent and automatically create test cases.

### Step 1: Start Recording

```bash
python govcloud_record.py --record
```

### Step 2: Chat with Your Agent

1. Open Orchestrate Chat UI in your browser
2. Start a **NEW** chat session
3. Select your agent and ask questions
4. Wait for the agent to respond

### Step 3: Stop Recording

Press `Ctrl+C` in the terminal when done.

Test cases are saved to `./recordings/`.

### Step 4: Enhance Keywords (Optional)

Enhance the recorded test cases with LLM-extracted keywords:

```bash
python govcloud_record.py --enhance ./recordings --copy-to ./test_data
```

### Step 5: Update Config

Update `govcloud_config.yaml` with the test cases path:
```yaml
paths:
  test_cases: "./test_data"
```

### Manual Mode

If live recording doesn't work, create test cases manually by pasting the conversation:

```bash
python govcloud_record.py --manual
```

You'll be prompted to enter the question and agent response.

---

## Run Evaluation

Once you have test cases in your configured path:

```bash
python govcloud_eval.py --all
```

### Evaluation Commands

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

## Configuration

Edit `govcloud_config.yaml`:

```yaml
# Models (use 90b for GovCloud - 405b not available)
models:
  llm_judge: "meta-llama/llama-3-2-90b-vision-instruct"

# Paths
paths:
  test_cases: "./test_data"      # Update this after generating test cases
  excel_input: "./questions.xlsx"

# Agent settings
agent:
  name: "your_agent_name"
  # tool: "your_tool"  # Optional - only if testing specific tool calls
```

---

## Test Case Format

Each test case is a JSON file:

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
| Recording not capturing | Use `--manual` mode instead |

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
  govcloud_eval.py       # Main evaluation script
  govcloud_generate.py   # Generate test cases from Excel
  govcloud_record.py     # Record/create test cases from chat
  govcloud_config.yaml   # Configuration
  questions.xlsx         # Sample Excel input
  test_data/             # Generated test cases
  eval_results_govcloud/ # Results output
```
