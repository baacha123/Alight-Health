# AWS Commercial - Agent Evaluation Framework

End-to-end evaluation framework for IBM watsonx Orchestrate agents on AWS Commercial.

Includes three scripts that form a complete pipeline:
- **Generate** test cases from Excel with LLM-powered keyword extraction
- **Record** live chat sessions as ground truth
- **Evaluate** agent performance using ADK metrics + LLM-as-Judge

## Architecture

```
questions.xlsx
      |
      v
commcloud_generate.py  -->  test_data/*.json  (test cases)
                                  |
                                  v
commcloud_eval.py      -->  eval_results/     (ADK metrics + LLM-as-Judge)

commcloud_record.py    -->  recordings/       (live chat ground truth)
```

**Agent Flow:**
```
User Question
      |
      v
alight_supervisor_agent  (routes to sub-agents)
      |
      v
verint_agent  (HR & benefits specialist)
      |
      v
call_verint_studio  (tool - calls MCP server via API Gateway + Lambda)
      |
      v
Answer returned to user
```

## Prerequisites

- Python 3.12+
- IBM watsonx Orchestrate API key (MCSP auth)
- Access to the AWS Commercial WxO instance

## Setup

### macOS / Linux

```bash
# 1. Create virtual environment
python3 -m venv adkcomm
source adkcomm/bin/activate

# 2. Install dependencies
pip install ibm-watsonx-orchestrate pandas openpyxl requests pyyaml scikit-learn

# 3. Add and activate environment
orchestrate env add \
  --name aws-commercial \
  --url "https://api.dl.watson-orchestrate.ibm.com/instances/20260217-1534-5395-5004-d25963381fc9" \
  --activate

# When prompted, enter the WXO API key

# 4. Set API key for eval/record scripts
export WO_API_KEY="<your-api-key>"
```

### Windows (Command Prompt)

```cmd
:: 1. Create virtual environment
py -3.13 -m venv adkcomm
adkcomm\Scripts\activate

:: 2. Install dependencies
pip install ibm-watsonx-orchestrate pandas openpyxl requests pyyaml scikit-learn

:: 3. Set SSL certs (REQUIRED on VDI - must be done BEFORE orchestrate commands)
set REQUESTS_CA_BUNDLE=D:\Users\%USERNAME%\Downloads\alight-agent\allcerts2-new.pem
set SSL_CERT_FILE=D:\Users\%USERNAME%\Downloads\alight-agent\allcerts2-new.pem

:: 4. Add and activate environment (do NOT use --type mcsp, let it auto-detect)
orchestrate env add --name aws-commercial --url "https://api.dl.watson-orchestrate.ibm.com/instances/20260217-1534-5395-5004-d25963381fc9" --activate

:: When prompted, enter the WXO API key

:: 5. Set API key for eval/record scripts
set WO_API_KEY=<your-api-key>
```

> **Windows Note:** Do NOT use `--type mcsp` when adding the environment. It forces MCSP v1/v2 token exchange which fails with "Scope not found". Omitting `--type` lets it auto-detect and works correctly.

## Deploy Agents (First Time Only)

After activating the environment, deploy the agents and tool:

```bash
orchestrate tools import -f agents/tools/call_verint_studio.yaml -k openapi
orchestrate agents import -f agents/verint_agent.yaml
orchestrate agents import -f agents/supervisor_agent.yaml
```

Verify deployment:
```bash
orchestrate agents list
orchestrate tools list
```

## Running the Pipeline

### Step 1: Generate Test Cases

Creates ADK-compatible test case JSON files from the questions Excel file.

```bash
# Generate all test cases with LLM keywords
python commcloud_generate.py

# Generate with a limit (for quick testing)
python commcloud_generate.py --limit 2

# Skip LLM calls (faster, uses simple keyword extraction)
python commcloud_generate.py --skip-llm

# Create a sample Excel template
python commcloud_generate.py --sample
```

**Input:** `questions.xlsx` (columns: Question, Expected Answer)
**Output:** `test_data/test_001.json`, `test_data/test_002.json`, ...

### Step 2: Record Live Sessions (Optional)

Records real chat sessions from the Orchestrate UI as ground truth test cases.

```bash
python commcloud_record.py --record
```

Then:
1. Open the Orchestrate Chat UI in your browser
2. Start a new chat and ask your question
3. Press `Ctrl+C` in the terminal when done

**Output:** `recordings/<session-id>/<thread-id>_annotated_data.json`

Other record options:
```bash
# Create test case manually (paste question + response)
python commcloud_record.py --manual

# Enhance recorded files with LLM keywords
python commcloud_record.py --enhance ./recordings

# List all recordings
python commcloud_record.py --list ./recordings
```

### Step 3: Evaluate

Runs the ADK evaluation against the agent, then optionally runs LLM-as-Judge for semantic scoring.

```bash
# Run ADK evaluation only
python commcloud_eval.py --evaluate

# Run with a limit
python commcloud_eval.py --evaluate --limit 2

# Run full pipeline: evaluate + LLM-as-Judge + report
python commcloud_eval.py --all

# Run LLM-as-Judge on existing results
python commcloud_eval.py --judge

# Show results report
python commcloud_eval.py --report
```

**Output:** `eval_results/<timestamp>/` with metrics including:
- Text Match %
- Journey Completion %
- Tool Call Precision / Recall
- LLM-as-Judge semantic scores

## File Structure

```
commcloud_eval/
├── README.md                    # This file
├── commcloud_config.yaml        # Configuration (models, paths, agent settings)
├── commcloud_eval.py            # Evaluation script (ADK + LLM-as-Judge)
├── commcloud_generate.py        # Test case generator (Excel -> JSON)
├── commcloud_record.py          # Live recording & test case creator
├── questions.xlsx               # Input questions and expected answers
└── agents/
    ├── supervisor_agent.yaml    # Supervisor agent definition (405b)
    ├── verint_agent.yaml        # Verint sub-agent definition (405b)
    └── tools/
        └── call_verint_studio.yaml  # OpenAPI spec for Verint tool
```

## Configuration

Edit `commcloud_config.yaml` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `models.llm_user` | `meta-llama/llama-3-405b-instruct` | Model for simulated user |
| `models.llm_judge` | `meta-llama/llama-3-405b-instruct` | Model for LLM-as-Judge |
| `paths.test_cases` | `./test_data` | Where test cases are stored |
| `paths.output_dir` | `./eval_results` | Where results are saved |
| `paths.excel_input` | `./questions.xlsx` | Input Excel file |
| `agent.name` | `alight_supervisor_agent` | Agent to evaluate |
| `evaluation.n_runs` | `1` | Number of evaluation runs |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `SSL: CERTIFICATE_VERIFY_FAILED` (Windows) | Set `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE` env vars before running any orchestrate commands |
| `--type mcsp` fails with "Scope not found" | Omit the `--type` flag — let auto-detect handle it |
| `401 Unauthorized` on LLM/eval calls | Token expired. Re-run: `orchestrate env activate aws-commercial --api-key <key>` |
| `WO_API_KEY must be specified` during record/eval | Set env var: `export WO_API_KEY=<key>` (macOS) or `set WO_API_KEY=<key>` (Windows) |
| `orchestrate env list` crashes with TypeError | Known SDK v2.4.0 bug. Scripts handle this with a config file fallback |
