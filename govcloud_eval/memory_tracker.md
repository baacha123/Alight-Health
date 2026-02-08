# GovCloud Evaluation - Development Memory Tracker

> **Purpose**: This file tracks development progress, technical decisions, and lessons learned for the GovCloud evaluation script. Use this as context when resuming work.

---

## Current Status: WORKING

**Last Updated**: 2026-02-07
**Last Test**: GovCloud LLM-as-Judge - 100% success

---

## Quick Reference

### Commands
```bash
# Activate environment (REQUIRED before any commands)
orchestrate env activate <env-name> --api-key <your-key>

# Run evaluation
python govcloud_eval.py --setup      # Apply patches
python govcloud_eval.py --evaluate   # Run ADK evaluation
python govcloud_eval.py --judge      # Run LLM-as-Judge
python govcloud_eval.py --report     # Show results
python govcloud_eval.py --all        # Everything
```

### Key Files
| File | Purpose |
|------|---------|
| `govcloud_eval.py` | Main script - auto-patches ADK + runs evaluation + LLM-as-Judge |
| `govcloud_config.yaml` | Customer configuration (models, paths) |
| `patched_files/` | Known-good patched versions (fallback reference) |
| `README.md` | Customer-facing documentation |

---

## What's Different: GovCloud vs Commercial

| Aspect | Commercial (IBM Cloud) | GovCloud (FedRAMP) |
|--------|------------------------|-------------------|
| **LLM Model** | llama-3-405b-instruct | llama-3-2-90b-vision-instruct |
| **Auth Endpoint** | iam.cloud.ibm.com (token exchange) | Uses cached token from orchestrate CLI |
| **Langfuse** | Enabled | Disabled (not available) |
| **URL Pattern** | `*.cloud.ibm.com` | `*.ibmforusgov.com` |

### Why These Differences
1. **Model**: 405b not deployed on GovCloud infrastructure - use 90b instead
2. **Auth**: GovCloud uses different IAM - solved by reading token from orchestrate CLI cache
3. **Langfuse**: Telemetry platform not available on GovCloud - auto-disabled

---

## Authentication Details (IMPORTANT)

### How It Works
The script reads the token from the **same place the orchestrate CLI stores it**:

1. **Config file**: `~/.config/orchestrate/config.yaml`
   - Contains: `context.active_environment` (which env is active)

2. **Credentials cache**: `~/.cache/orchestrate/credentials.yaml`
   - Contains: `auth.{env_name}.wxo_mcsp_token` (the Bearer token)

### Auth Flow in Code
```python
def get_orchestrate_cached_token():
    # 1. Read active env from ~/.config/orchestrate/config.yaml
    # 2. Read token from ~/.cache/orchestrate/credentials.yaml
    # 3. Return auth.{active_env}.wxo_mcsp_token
```

### Fallback Auth (if cache not found)
- **IBM Cloud URLs** (`cloud.ibm.com`): Token exchange via `iam.cloud.ibm.com`
- **SaaS/AWS URLs**: Token exchange via `iam.platform.saas.ibm.com`
- **GovCloud URLs** (`ibmforusgov.com`): Use API key as static token

---

## ADK Patches Applied

The script patches 4 files in the `agentops` package:

### 1. `main.py`
- Disables `TELEMETRY_PLATFORM = "langfuse"`
- Replaces `exporters=[LangfusePersistence()]` with `exporters=[]`

### 2. `clients.py`
- Fixes `config.custom_metrics_config` access (dict vs dataclass)

### 3. `runner.py`
- Fixes `config.extractors_config.paths` access (dict vs dataclass)
- Adds `custom_llmaaj_client=llmaaj_provider` to EvaluationPackage

### 4. `evaluation_package.py`
- Replaces hardcoded `llama-3-405b-instruct` with configurable model
- Uses `custom_llmaaj_client` if provided

---

## LLM-as-Judge Feature

### What It Does
Semantically evaluates agent responses against expected answers using an LLM.

### Evaluation Criteria
1. **Correctness**: Factually correct vs expected answer
2. **Completeness**: Covers key points
3. **Relevance**: Addresses the question
4. **No Hallucination**: Doesn't make up false info

### Output Format
```json
{
  "verdict": "CORRECT|PARTIALLY_CORRECT|INCORRECT",
  "score": 0.0-1.0,
  "reasoning": "Explanation..."
}
```

### Gateway Call Structure
```python
# Headers
{
  "Authorization": "Bearer {token}",
  "x-gateway-config": {"strategy": {"mode": "single"}, "targets": [...]},
  "x-request-id": "{uuid}"
}

# Payload
{
  "model": "watsonx/{model_id}",
  "messages": [...],
  "temperature": 0.0
}
```

---

## Development History

### 2026-02-07: LLM-as-Judge Working on GovCloud
- **Problem**: 401 Unauthorized errors on GovCloud
- **Tried**:
  - Token exchange (failed - wrong endpoint for GovCloud)
  - Static token with Bearer auth (failed)
  - Static token with Basic auth (failed)
  - ADK's tenant_setup (failed)
- **Solution**: Read token from `~/.cache/orchestrate/credentials.yaml`
- **Result**: 100% success on GovCloud

### 2026-02-06: ADK Evaluation Working
- Programmatic patching implemented
- Config generation for `--config` parameter
- All 4 patch files verified exact match with known-good versions

### Earlier: Initial Development
- Created patched_files/ with known-good versions
- Built govcloud_config.yaml simplified format
- Implemented --setup, --evaluate, --report commands

---

## Lessons Learned

### 1. GovCloud Auth is Different
Don't try to exchange tokens - read from orchestrate CLI cache instead.

### 2. ADK Requires --config
The `orchestrate evaluations evaluate` command needs `--config eval_config.yaml` to use custom models.

### 3. Patching Must Be Exact
String replacement patching requires exact character matching including whitespace.

### 4. Token Location
- Config: `~/.config/orchestrate/config.yaml`
- Credentials: `~/.cache/orchestrate/credentials.yaml`
- Token key: `auth.{env}.wxo_mcsp_token`

### 5. Gateway Headers Matter
Must include `x-gateway-config` with provider/targets structure.

---

## Test Environments

### Local/Commercial (IBM Cloud)
```
URL: https://api.us-south.watson-orchestrate.cloud.ibm.com/instances/{id}
Auth: Token exchange via iam.cloud.ibm.com
Status: WORKING
```

### GovCloud (FedRAMP)
```
URL: https://origin-api.us-gov-east-1.watson-orchestrate.ibmforusgov.com/instances/{id}
Auth: Cached token from orchestrate CLI
Status: WORKING
```

---

## Future Work / TODO

- [ ] Add batch processing for large test sets
- [ ] Add progress bar for --judge
- [ ] Add retry logic for transient failures
- [ ] Add --compare to compare two evaluation runs
- [ ] Support for multi-turn conversations in judge
- [ ] Export judge results to Excel

---

## Troubleshooting Quick Reference

| Error | Cause | Fix |
|-------|-------|-----|
| 401 Unauthorized | Token expired/invalid | `orchestrate env activate <env> --api-key <key>` |
| "No active environment" | Orchestrate not activated | `orchestrate env activate <env> --api-key <key>` |
| "Model not found" | Wrong model ID | Run `orchestrate models list` |
| "Pattern not found" | ADK version changed | Check patched_files/ for reference |
| 400 Bad Request | Wrong auth endpoint | Check is_govcloud_url() detection |

---

## Related Files Outside This Directory

- `/Users/bacha/Downloads/AlightWxO/verint-mvp/tests/run_eval_at_scale.py` - Reference LLM-as-Judge implementation
- `/Users/bacha/Downloads/AlightWxO/verint-mvp/config/.env` - Local IBM Cloud credentials
- `/Users/bacha/Downloads/AlightWxO/verint-mvp/adkvenv/` - ADK virtual environment with agentops package

---

## Contact

For issues with this script, contact the Alight development team.
