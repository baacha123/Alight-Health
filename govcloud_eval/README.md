# GovCloud Agent Evaluation

Automated evaluation for IBM watsonx Orchestrate agents on GovCloud/FedRAMP.

---

## Quick Start

**Already have test cases?** Skip to [Run Evaluation](#run-evaluation).

---

## 1. Setup Environment

### Install Python 3.13
Download from https://www.python.org/downloads/

### Create Virtual Environment
```bash
python3.13 -m venv govcloud_venv
source govcloud_venv/bin/activate   # Mac/Linux
govcloud_venv\Scripts\activate      # Windows
```

### Install Dependencies
```bash
pip install "ibm-watsonx-orchestrate-adk[agentops]"
pip install arize==7.26.1
pip install pandas openpyxl
```

### Activate Orchestrate
```bash
orchestrate env activate <your-env> --api-key <your-key>
```

---

## 2. Generate Test Cases from Excel

If you have questions and expected answers in Excel, use this to generate test cases.

### Excel Format

Create `questions.xlsx` with these columns:

| Question | Expected Answer |
|----------|-----------------|
| What is my deductible? | Your deductible is $500 individual... |
| How do I enroll? | Go to benefits.company.com and... |

Optional columns: `Source Document`, `Source Page`

### Run Generate

```bash
python govcloud_generate.py
```

This reads `questions.xlsx` and outputs test cases to `./test_data/`.

**Options:**
```bash
python govcloud_generate.py --excel myfile.xlsx   # Custom Excel
python govcloud_generate.py --limit 5             # Only 5 cases
python govcloud_generate.py --skip-llm            # No LLM (faster)
python govcloud_generate.py --sample              # Create sample Excel
```

---

## 3. Record Test Cases from Chat

Record live chat sessions and create test cases automatically.

### Start Recording

```bash
python govcloud_record.py --record
```

Then:
1. Open Orchestrate Chat UI in browser
2. Start a **NEW** chat session
3. Chat with your agent
4. Press Ctrl+C when done

Test cases are saved to `./recordings/`.

### Enhance Keywords (Optional)

After recording, enhance keywords with LLM:

```bash
python govcloud_record.py --enhance ./recordings --copy-to ./test_data
```

### Manual Mode

If recording doesn't work, create test cases manually:

```bash
python govcloud_record.py --manual
```

You'll be prompted to paste the question and response.

---

## 4. Run Evaluation

Once you have test cases in `./test_data/`:

```bash
python govcloud_eval.py --all
```

**Individual steps:**
```bash
python govcloud_eval.py --setup      # Apply GovCloud patches
python govcloud_eval.py --evaluate   # Run agent tests
python govcloud_eval.py --judge      # LLM-as-Judge evaluation
python govcloud_eval.py --report     # Show results
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
  test_cases: "./test_data"
  excel_input: "./questions.xlsx"

# Agent settings (for generate/record)
agent:
  name: "your_agent_name"
  tool: "your_tool_name"
```

---

## Test Case Format

Each test case is a JSON file:

```json
{
  "agent": "alight_supervisor_agent",
  "goals": {
    "call_tool-1": ["summarize"]
  },
  "goal_details": [
    {
      "type": "tool_call",
      "name": "call_tool-1",
      "tool_name": "call_tool",
      "args": {"query": "User question here"}
    },
    {
      "name": "summarize",
      "type": "text",
      "response": "Expected answer here",
      "keywords": ["key", "terms", "to", "match"]
    }
  ],
  "story": "Brief scenario description",
  "starting_sentence": "User question here"
}
```

---

## Output

Results are saved to `./eval_results_govcloud/`:

```
eval_results_govcloud/
  2026-02-08_14-30-00/
    summary_metrics.csv     # All metrics
    messages/               # Conversation logs
  llm_judge_results.json    # LLM-as-Judge scores
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No active environment" | `orchestrate env activate <env> --api-key <key>` |
| "401 Unauthorized" | Re-activate environment to refresh token |
| "Model not found" | Run `orchestrate models list` |
| Recording not capturing | Use `--manual` mode instead |

---

## Files

```
govcloud_eval/
  govcloud_eval.py       # Main evaluation script
  govcloud_generate.py   # Generate test cases from Excel
  govcloud_record.py     # Record/create test cases from chat
  govcloud_config.yaml   # Configuration
  questions.xlsx         # Sample Excel input
  test_data/             # Your test cases
  eval_results_govcloud/ # Results output
```
