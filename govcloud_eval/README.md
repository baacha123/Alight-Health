# GovCloud Agent Evaluation Framework

Automated evaluation script for IBM watsonx Orchestrate agents on GovCloud/FedRAMP environments.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Creating Ground Truth Test Cases](#creating-ground-truth-test-cases)
6. [Running Evaluations](#running-evaluations)
7. [Understanding Results](#understanding-results)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What This Solves

The standard ADK evaluation framework has compatibility issues on GovCloud:

| Issue | Problem | Our Fix |
|-------|---------|---------|
| Hardcoded 405b model | Model not available on GovCloud | Auto-patches to use configurable model |
| Langfuse telemetry | Auth errors without credentials | Auto-disables Langfuse |
| Config parsing bug | `'dict' object has no attribute` errors | Auto-patches config handling |

### What's Included

```
govcloud_eval/
├── govcloud_eval.py        # Main script (patches + evaluates)
├── govcloud_config.yaml    # Your configuration (edit this)
├── README.md               # This file
├── sample_test_cases/      # Example test cases
└── templates/              # Templates for creating test cases
```

---

## Prerequisites

### 1. Python Environment

```bash
# Create virtual environment
python -m venv adkvenv

# Activate it
source adkvenv/bin/activate        # Linux/Mac
adkvenv\Scripts\activate           # Windows
```

### 2. Install ADK

```bash
pip install ibm-watsonx-orchestrate-adk
```

### 3. Verify Installation

```bash
orchestrate --version
```

### 4. Know Your Available Models

```bash
# Activate your environment first
orchestrate env activate <your-env-name> --api-key <your-api-key>

# List available models
orchestrate models list
```

Save this list - you'll need it for configuration.

---

## Quick Start

### Step 1: Edit Configuration

Open `govcloud_config.yaml` and update:

```yaml
# Use models from your `orchestrate models list` output
models:
  llm_user: "meta-llama/llama-3-2-90b-vision-instruct"    # For simulating user
  llm_judge: "meta-llama/llama-3-2-90b-vision-instruct"   # For evaluation
  embedding: "sentence-transformers/all-minilm-l6-v2"

# Point to your test cases
paths:
  test_cases: "./your_test_cases"      # Folder with your ground truth JSONs
  output_dir: "./eval_results"         # Where results are saved
```

### Step 2: Activate Orchestrate Environment

```bash
orchestrate env activate <your-env-name> --api-key <your-api-key>
```

### Step 3: Verify Patches

```bash
python govcloud_eval.py --setup
```

Expected output:
```
✅ main.py: Applied 2 patch(es)
✔️ clients.py: Already patched
✔️ runner.py: Already patched
✔️ evaluation_package.py: Already patched

Patches verified: 4/4
Status: READY for GovCloud evaluation
```

### Step 4: Run Evaluation

```bash
# Test with 1 case first
python govcloud_eval.py --evaluate --limit 1

# Run all cases
python govcloud_eval.py --evaluate

# View results
python govcloud_eval.py --report
```

---

## Configuration

### Full Configuration Reference

```yaml
# =============================================================================
# govcloud_config.yaml - GovCloud Evaluation Configuration
# =============================================================================

# === Authentication ===
# Leave empty to use environment variables (recommended)
auth:
  url: ""           # Or set WXO_INSTANCE_URL env var
  tenant_name: ""
  # api_key: ""     # Use WXO_API_KEY env var instead (more secure)

# === Models ===
# IMPORTANT: Run `orchestrate models list` to see available models
# Common GovCloud models:
#   - meta-llama/llama-3-2-90b-vision-instruct
#   - meta-llama/llama-3-3-70b-instruct
#   - ibm/granite-3-8b-instruct
models:
  llm_user: "meta-llama/llama-3-2-90b-vision-instruct"
  llm_judge: "meta-llama/llama-3-2-90b-vision-instruct"
  embedding: "sentence-transformers/all-minilm-l6-v2"

# === Paths ===
paths:
  test_cases: "./sample_test_cases"    # Your ground truth folder
  output_dir: "./eval_results"         # Results output folder
  excel_file: "./questions.xlsx"       # For --generate command (optional)

# === Provider Settings ===
provider:
  type: "gateway"      # "gateway" for WXO, "watsonx" for direct
  vendor: "ibm"
  project_id: "skills-flow"

# === Evaluation Settings ===
evaluation:
  num_workers: 1                  # Parallel workers
  n_runs: 1                       # Runs per test case
  similarity_threshold: 0.8       # Semantic matching threshold
  enable_fuzzy_matching: false
  is_strict: true

# === Test Case Generation (for --generate) ===
generation:
  agent_name: "your_agent_name"
  tool_name: "your_tool_name"
```

---

## Creating Ground Truth Test Cases

### Option 1: Manual Creation (Recommended for Accuracy)

Create JSON files in your test cases folder. Use this template:

**Template: `templates/test_case_template.json`**

```json
{
  "agent": "your_agent_name",
  "goals": {
    "your_tool_name-1": ["summarize"]
  },
  "goal_details": [
    {
      "type": "tool_call",
      "name": "your_tool_name-1",
      "tool_name": "your_tool_name",
      "args": {
        "query": "The user's question goes here"
      }
    },
    {
      "type": "text",
      "name": "summarize",
      "response": "The expected answer the agent should provide",
      "keywords": ["keyword1", "keyword2", "keyword3"]
    }
  ],
  "story": "Brief description of user's intent",
  "starting_sentence": "The user's question goes here"
}
```

### Field Explanations

| Field | Description | Example |
|-------|-------------|---------|
| `agent` | Your agent's name in WXO | `"benefits_agent"` |
| `goals` | Maps tool calls to outcomes | `{"tool-1": ["summarize"]}` |
| `goal_details[0]` | Expected tool call | Tool name and arguments |
| `goal_details[1]` | Expected response | Keywords and/or full response |
| `story` | User scenario description | `"User asking about benefits"` |
| `starting_sentence` | The actual user question | `"What is my deductible?"` |

### Example: Benefits Question

```json
{
  "agent": "benefits_supervisor_agent",
  "goals": {
    "call_benefits_api-1": ["summarize"]
  },
  "goal_details": [
    {
      "type": "tool_call",
      "name": "call_benefits_api-1",
      "tool_name": "call_benefits_api",
      "args": {
        "query": "What is my annual deductible?"
      }
    },
    {
      "type": "text",
      "name": "summarize",
      "response": "Your annual deductible is $500 for in-network services and $1,000 for out-of-network services.",
      "keywords": ["deductible", "$500", "in-network", "out-of-network"]
    }
  ],
  "story": "Employee wants to know their health plan deductible amount",
  "starting_sentence": "What is my annual deductible?"
}
```

### Option 2: Generate from Excel (Batch Creation)

If you have questions in an Excel file:

1. Create Excel with columns:
   - Column A: Category
   - Column B: Subcategory
   - Column C: Question
   - Column D: Expected Answer (optional)
   - Column E: MVP (yes/no for filtering)

2. Update config:
```yaml
paths:
  excel_file: "./your_questions.xlsx"

generation:
  agent_name: "your_agent_name"
  tool_name: "your_tool_name"
```

3. Generate:
```bash
python govcloud_eval.py --generate
python govcloud_eval.py --generate --limit 10  # Generate only 10
```

### Option 3: Copy and Modify Samples

```bash
# Copy a sample
cp sample_test_cases/allstate_gt_001.json my_test_cases/my_test_001.json

# Edit with your agent/tool names and questions
```

---

## Running Evaluations

### All Commands

| Command | Description |
|---------|-------------|
| `python govcloud_eval.py --setup` | Verify/apply patches |
| `python govcloud_eval.py --setup --dry-run` | Preview patches without applying |
| `python govcloud_eval.py --generate` | Generate test cases from Excel |
| `python govcloud_eval.py --evaluate` | Run ADK evaluation |
| `python govcloud_eval.py --evaluate --limit N` | Run N test cases only |
| `python govcloud_eval.py --judge` | Run LLM-as-Judge evaluation |
| `python govcloud_eval.py --report` | Display results summary |
| `python govcloud_eval.py --all` | Full pipeline |

### Typical Workflow

```bash
# 1. First time setup
orchestrate env activate <env> --api-key <key>
python govcloud_eval.py --setup

# 2. Test with one case
python govcloud_eval.py --evaluate --limit 1

# 3. Run full evaluation
python govcloud_eval.py --evaluate

# 4. View results
python govcloud_eval.py --report

# 5. (Optional) LLM-as-Judge for semantic evaluation
python govcloud_eval.py --judge
```

---

## Understanding Results

### Console Output

```
                                 Agent Metrics
╭──────┬──────┬──────┬──────┬──────┬──────┬──────┬───────┬──────┬───────┬──────╮
│ Name │ Runs │Steps │ LLM  │ Tool │ Prec │ Rec  │ Route │ Text │ Pass  │ Time │
├──────┼──────┼──────┼──────┼──────┼──────┼──────┼───────┼──────┼───────┼──────┤
│ tc1  │ 1.0  │ 5.0  │ 3.0  │ 1.0  │ 1.0  │ 1.0  │ 1.0   │ 100% │ 100%  │ 6.0  │
╰──────┴──────┴──────┴──────┴──────┴──────┴──────┴───────┴──────┴───────┴──────╯
```

### Key Metrics

| Metric | Meaning | Good Value |
|--------|---------|------------|
| Tool Call Precision | Correct tools called / Total tools called | 1.0 |
| Tool Call Recall | Correct tools called / Expected tools | 1.0 |
| Agent Routing Accuracy | Correct agent routing | 1.0 |
| Text Match | Response matched expected | 100% |
| Journey Success | Full conversation succeeded | 100% |

### Output Files

```
eval_results/
└── 2024-02-06_14-30-00/
    ├── summary_metrics.csv      # All metrics in CSV
    ├── config.yml               # Config used for this run
    ├── messages/                # Detailed conversation logs
    │   ├── test_001.messages.json
    │   └── test_001.metrics.json
    └── knowledge_base_metrics/  # RAG metrics (if applicable)
```

---

## Troubleshooting

### "No active orchestrate environment"

```bash
# Solution: Activate your environment
orchestrate env activate <env-name> --api-key <your-key>
```

### "Could not find agentops package"

```bash
# Solution: Install ADK
pip install ibm-watsonx-orchestrate-adk
```

### "401 Unauthorized" or "Token expired"

```bash
# Solution: Re-authenticate
orchestrate env activate <env-name> --api-key <your-key>
```

### Patches not applying / "Pattern not found"

The ADK version may have changed. Check:
```bash
python govcloud_eval.py --setup --dry-run
```

If patches show "no_match", the underlying files may have been updated. Contact support.

### "Model not found"

Your configured model isn't available. Check available models:
```bash
orchestrate models list
```

Update `govcloud_config.yaml` with a model from that list.

### Clear Python Cache

If you're seeing stale behavior:
```bash
# Linux/Mac
find . -name __pycache__ -exec rm -rf {} +

# Windows
Get-ChildItem -Recurse -Directory -Name __pycache__ | Remove-Item -Recurse -Force
```

---

## Support

For issues with this script, contact the Alight development team.

For ADK issues, open a ticket with IBM watsonx Orchestrate support.
