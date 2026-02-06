# Alight Health - Agent Evaluation Tools

Tools for evaluating IBM watsonx Orchestrate agents on GovCloud/FedRAMP environments.

## Contents

| Folder | Description |
|--------|-------------|
| `govcloud_eval/` | GovCloud-compatible evaluation framework with auto-patching |

## Quick Start

```bash
# Clone this repo
git clone https://github.com/baacha123/Alight-Health.git
cd Alight-Health/govcloud_eval

# Follow the README in govcloud_eval/
```

## GovCloud Evaluation

See [govcloud_eval/README.md](govcloud_eval/README.md) for full documentation.

### TL;DR

```bash
# 1. Install ADK
pip install ibm-watsonx-orchestrate-adk

# 2. Edit config with your models
nano govcloud_config.yaml

# 3. Activate orchestrate
orchestrate env activate <env> --api-key <key>

# 4. Run evaluation
python govcloud_eval.py --setup      # Apply patches
python govcloud_eval.py --evaluate   # Run evaluation
python govcloud_eval.py --report     # View results
```
