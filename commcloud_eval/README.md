# AWS Commercial - Agent Evaluation

Automated evaluation for IBM watsonx Orchestrate agents on AWS Commercial.

---

## Quick Start

> **Already logged in?** If you have orchestrate activated and working, skip to [Step 5: Edit Config](#5-edit-config).

> **Already have test cases?** Skip to [Run Evaluation](#run-evaluation).

---

## Environment Setup

### 1. Install Python 3.12+
Download from https://www.python.org/downloads/ and install.

### 2. Create Virtual Environment
```bash
python3 -m venv adkcomm
source adkcomm/bin/activate   # Mac/Linux
# OR
adkcomm\Scripts\activate      # Windows
```

### 3. Install Dependencies
```bash
pip install ibm-watsonx-orchestrate pandas openpyxl requests pyyaml scikit-learn
pip install ibm-watsonx-orchestrate-evaluation-framework
```

> **Note:** The evaluation framework is a separate package. Without it, `orchestrate evaluations evaluate` will fail with "AgentOps not found".

### 4. Set SSL Certificates (Windows VDI Only)

If you are on a corporate VDI with SSL inspection, set these **before** running any orchestrate commands:

```cmd
set REQUESTS_CA_BUNDLE=D:\Users\%USERNAME%\Downloads\alight-agent\allcerts2-new.pem
set SSL_CERT_FILE=D:\Users\%USERNAME%\Downloads\alight-agent\allcerts2-new.pem
```

### 5. Add and Activate Environment

```bash
orchestrate env add \
  --name aws-commercial \
  --url "https://api.dl.watson-orchestrate.ibm.com/instances/<your-instance-id>" \
  --activate
```

When prompted, enter your WXO API key.

**Windows:**
```cmd
orchestrate env add --name aws-commercial --url "https://api.dl.watson-orchestrate.ibm.com/instances/<your-instance-id>" --activate
```

> **Important:** Do NOT use `--type mcsp` on Windows. It forces MCSP v1/v2 token exchange which fails with "Scope not found". Omitting `--type` lets it auto-detect and works correctly.

### 6. Get Your API Key
1. Open the Orchestrate UI in your browser
2. Click **Settings** (gear icon)
3. Click **Generate API Key**
4. Copy the key and save it securely

### 7. Set API Key Environment Variable

This is required for evaluation and recording commands:

```bash
export WO_API_KEY="<your-api-key>"   # Mac/Linux
```
```cmd
set WO_API_KEY=<your-api-key>        :: Windows
```

### 8. Edit Config

Open `commcloud_config.yaml` and set your agent name:

```yaml
agent:
  name: "your_agent_name"        # The agent you want to evaluate
  # tool: "your_tool_name"       # Optional - only if testing specific tool calls
```

Run `orchestrate agents list` to see available agents in your instance.

### 9. Generate or Record Test Cases

See [Generate Test Cases](#generate-test-cases-from-excel) or [Record Test Cases](#record-test-cases-from-chat) below.

### 10. Run Evaluation

```bash
python commcloud_eval.py --all
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
python commcloud_generate.py
```

This reads `questions.xlsx`, uses LLM to extract keywords, and outputs test cases to `./test_data/`.

### Generate Options

```bash
python commcloud_generate.py --excel myfile.xlsx   # Custom Excel file
python commcloud_generate.py --limit 5             # Only generate 5 cases
python commcloud_generate.py --skip-llm            # Skip LLM (faster, simpler keywords)
python commcloud_generate.py --sample              # Create sample Excel template
```

---

## Record Test Cases from Chat

Record live chat sessions with your agent and automatically create test cases.

### Step 1: Start Recording

```bash
python commcloud_record.py --record
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
python commcloud_record.py --enhance ./recordings --copy-to ./test_data
```

### Manual Mode

If live recording doesn't work, create test cases manually by pasting the conversation:

```bash
python commcloud_record.py --manual
```

You'll be prompted to enter the question and agent response.

---

## Run Evaluation

Once you have test cases in your configured path:

```bash
python commcloud_eval.py --all
```

### Evaluation Commands

| Command | What it does |
|---------|--------------|
| `--evaluate` | Run agent evaluation |
| `--judge` | Run LLM-as-Judge semantic evaluation |
| `--report` | Show results summary |
| `--all` | Run everything (evaluate + judge + report) |
| `--limit N` | Only run N test cases |

### Examples

```bash
# Test with 1 case first
python commcloud_eval.py --evaluate --limit 1

# Full evaluation + LLM judge
python commcloud_eval.py --all

# Just see results
python commcloud_eval.py --report
```

---

## Configuration

Edit `commcloud_config.yaml`:

```yaml
# Models (405b available on AWS Commercial)
models:
  llm_user: "meta-llama/llama-3-405b-instruct"
  llm_judge: "meta-llama/llama-3-405b-instruct"

# Paths
paths:
  test_cases: "./test_data"
  output_dir: "./eval_results"
  excel_input: "./questions.xlsx"
  recordings: "./recordings"

# Agent settings - UPDATE THIS to match your agent
agent:
  name: "your_agent_name"
  # tool: "your_tool_name"  # Optional - only if testing specific tool calls
```

Run `orchestrate models list` to see available models.

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

Results are saved to `eval_results/`:
```
eval_results/
  2026-02-17_21-23-08/
    summary_metrics.csv     # All metrics
    messages/               # Conversation logs
    config.yml              # Config used
  llm_judge_results.json    # LLM-as-Judge results
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "AgentOps not found" | Run `pip install ibm-watsonx-orchestrate-evaluation-framework` |
| "No active environment" | Run `orchestrate env activate aws-commercial --api-key <key>` |
| "401 Unauthorized" | Token expired. Re-run `orchestrate env activate aws-commercial --api-key <key>` |
| `SSL: CERTIFICATE_VERIFY_FAILED` (Windows) | Set `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE` env vars before running orchestrate commands |
| `--type mcsp` fails with "Scope not found" | Omit the `--type` flag — let auto-detect handle it |
| `WO_API_KEY must be specified` | Set env var: `export WO_API_KEY=<key>` (Mac) or `set WO_API_KEY=<key>` (Windows) |
| "Model not found" | Run `orchestrate models list` and update config |
| `orchestrate env list` crashes with TypeError | Known SDK v2.4.0 bug. Scripts handle this with a config file fallback |
| Recording not capturing | Ensure you started a **NEW** chat after running `--record` |

---

## Files

```
commcloud_eval/
  commcloud_eval.py       # Main evaluation script
  commcloud_generate.py   # Generate test cases from Excel
  commcloud_record.py     # Record/create test cases from chat
  commcloud_config.yaml   # Configuration
  questions.xlsx          # Sample Excel input
  test_data/              # Generated test cases
  eval_results/           # Results output
  recordings/             # Recorded chat sessions
```
