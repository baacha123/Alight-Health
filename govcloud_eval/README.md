# GovCloud Agent Evaluation

Automated evaluation for IBM watsonx Orchestrate agents on GovCloud/FedRAMP.

---

## Quick Start

### 1. Install ADK
```bash
pip install ibm-watsonx-orchestrate-adk
```

### 2. Activate Your Environment
```bash
orchestrate env activate <your-env-name> --api-key <your-api-key>
```

### 3. Edit Config
Open `govcloud_config.yaml` and set your test cases path:
```yaml
paths:
  test_cases: "./your_test_cases"
```

### 4. Run
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
| "Could not find agentops" | Run `pip install ibm-watsonx-orchestrate-adk` |

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
