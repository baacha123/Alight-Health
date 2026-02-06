#!/usr/bin/env python3
"""
GovCloud Agent Evaluation (with Auto-Patching + LLM-as-Judge)
=============================================================
All-in-one script for running IBM watsonx Orchestrate evaluations on GovCloud/FedRAMP
environments where certain models (like 405b) are not available.

This script automatically:
1. Patches agentops files programmatically for GovCloud compatibility
2. Runs the ADK evaluation
3. Runs LLM-as-Judge semantic evaluation
4. Provides results summary

Usage:
    python govcloud_eval.py --setup             # Apply patches
    python govcloud_eval.py --evaluate          # Run ADK evaluation
    python govcloud_eval.py --judge             # Run LLM-as-Judge evaluation
    python govcloud_eval.py --report            # Show results from last run
    python govcloud_eval.py --all               # Run everything
"""

import json
import os
import sys
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import importlib.util

# For LLM-as-Judge via Orchestrate Gateway
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CONFIG_FILE = SCRIPT_DIR / "govcloud_config.yaml"

FILES_TO_PATCH = ["main.py", "clients.py", "runner.py", "evaluation_package.py"]


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required. Run: pip install pyyaml")
        sys.exit(1)

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Create govcloud_config.yaml or specify with --config")
        sys.exit(1)

    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_env_file():
    """Load environment variables from .env file."""
    env_paths = [
        Path(__file__).parent.parent / "config" / ".env",
        Path(__file__).parent / "eval.env",
        Path(__file__).parent / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value and key not in os.environ:
                            os.environ[key] = value
            break


load_env_file()


# =============================================================================
# PATCHING FUNCTIONS
# =============================================================================

def patch_main_py(content: str) -> str:
    """Patch main.py: disable Langfuse telemetry."""

    # Patch 1: Comment out the TELEMETRY_PLATFORM line
    content = content.replace(
        'os.environ["TELEMETRY_PLATFORM"] = "langfuse"',
        '''# ============ ORIGINAL CODE (COMMENTED OUT FOR GOVCLOUD) ============
# os.environ["TELEMETRY_PLATFORM"] = "langfuse"
# ============ END ORIGINAL CODE ============'''
    )

    # Patch 2: Replace exporters=[LangfusePersistence()] with exporters=[]
    content = content.replace(
        '        exporters=[LangfusePersistence()],',
        '''        # ============ ORIGINAL CODE (COMMENTED OUT FOR GOVCLOUD) ============
        # exporters=[LangfusePersistence()],
        # ============ END ORIGINAL CODE ============
        # ============ GOVCLOUD FIX: Disable Langfuse export ============
        exporters=[],
        # ============ END GOVCLOUD FIX ============'''
    )

    return content


def patch_clients_py(content: str) -> str:
    """Patch clients.py: handle dict vs dataclass config."""

    # Find and replace the config access pattern
    old_pattern = '''    llamaj_config_dict["model_id"] = (
        config.custom_metrics_config.llmaaj_config.model_id
    )
    llamaj_config_dict["embedding_model_id"] = (
        config.custom_metrics_config.llmaaj_config.embedding_model_id
    )'''

    new_pattern = '''    # ============ GOVCLOUD FIX: Handle both dict and dataclass ============
    cmc = config.custom_metrics_config
    if isinstance(cmc, dict):
        llmaaj_cfg = cmc.get("llmaaj_config", {})
        llamaj_config_dict["model_id"] = llmaaj_cfg.get("model_id", llamaj_config_dict["model_id"])
        llamaj_config_dict["embedding_model_id"] = llmaaj_cfg.get("embedding_model_id", llamaj_config_dict["embedding_model_id"])
    else:
        llamaj_config_dict["model_id"] = cmc.llmaaj_config.model_id
        llamaj_config_dict["embedding_model_id"] = cmc.llmaaj_config.embedding_model_id
    # ============ END GOVCLOUD FIX ============
'''

    content = content.replace(old_pattern, new_pattern)

    return content


def patch_runner_py(content: str) -> str:
    """Patch runner.py: handle dict vs dataclass config and add custom_llmaaj_client."""

    # Patch 1: Replace the extractors/custom_evals loading section
    old_extractors = '''    # Load custom extractors
    if config.extractors_config.paths is not None:
        for path in config.extractors_config.paths:
            extractors = find_evaluation_subclasses(
                directory=path, base_class_name="Extractor"
            )
            for extractor_class in extractors:
                all_extractors.append(extractor_class())

    # Load custom evaluations
    if config.custom_metrics_config.paths is not None:
        for path in config.custom_metrics_config.paths:
            custom_eval_classes = find_evaluation_subclasses(path)
            for _class in custom_eval_classes:
                all_custom_evals.append(_class(llm_client=llmaaj_provider))

    # Create evaluation package and generate summary
    evaluation_package = EvaluationPackage('''

    new_extractors = '''    # ============ ORIGINAL CODE (COMMENTED OUT FOR GOVCLOUD) ============
    # # Load custom extractors
    # if config.extractors_config.paths is not None:
    #     for path in config.extractors_config.paths:
    #         extractors = find_evaluation_subclasses(
    #             directory=path, base_class_name="Extractor"
    #         )
    #         for extractor_class in extractors:
    #             all_extractors.append(extractor_class())
    #
    # # Load custom evaluations
    # if config.custom_metrics_config.paths is not None:
    #     for path in config.custom_metrics_config.paths:
    #         custom_eval_classes = find_evaluation_subclasses(path)
    #         for _class in custom_eval_classes:
    #             all_custom_evals.append(_class(llm_client=llmaaj_provider))
    # ============ END ORIGINAL CODE ============

    # ============ GOVCLOUD FIX: Handle both dict and dataclass ============
    # Load custom extractors
    ext_cfg = config.extractors_config
    ext_paths = ext_cfg.get("paths") if isinstance(ext_cfg, dict) else ext_cfg.paths
    if ext_paths is not None:
        for path in ext_paths:
            extractors = find_evaluation_subclasses(
                directory=path, base_class_name="Extractor"
            )
            for extractor_class in extractors:
                all_extractors.append(extractor_class())

    # Load custom evaluations
    cmc = config.custom_metrics_config
    cmc_paths = cmc.get("paths") if isinstance(cmc, dict) else cmc.paths
    if cmc_paths is not None:
        for path in cmc_paths:
            custom_eval_classes = find_evaluation_subclasses(path)
            for _class in custom_eval_classes:
                all_custom_evals.append(_class(llm_client=llmaaj_provider))
    # ============ END GOVCLOUD FIX ============

    # Create evaluation package and generate summary
    # GOVCLOUD FIX: Pass llmaaj_provider as custom_llmaaj_client to avoid hardcoded 405b
    evaluation_package = EvaluationPackage('''

    content = content.replace(old_extractors, new_extractors)

    # Patch 2: Add custom_llmaaj_client parameter to EvaluationPackage
    old_eval_pkg = '''        custom_evals=all_custom_evals,
        extractors=all_extractors,'''

    new_eval_pkg = '''        custom_evals=all_custom_evals,
        custom_llmaaj_client=llmaaj_provider,
        extractors=all_extractors,'''

    content = content.replace(old_eval_pkg, new_eval_pkg)

    return content


def patch_evaluation_package_py(content: str) -> str:
    """Patch evaluation_package.py: use custom_llmaaj_client instead of hardcoded 405b."""

    # Find the section where 405b is used and wrap it with our custom client logic
    # Original pattern - the matcher creation with hardcoded 405b
    old_matcher = '''        # output response matching
        self.matcher = LLMMatcher(
            llm_client=get_provider(
                model_id="meta-llama/llama-3-405b-instruct",
                params={
                    "min_new_tokens": 0,
                    "decoding_method": "greedy",
                    "max_new_tokens": 10,
                },
                embedding_model_id="sentence-transformers/all-minilm-l6-v2",
                **extra_kwargs,
            ),'''

    new_matcher = '''        # ============ GOVCLOUD FIX: Use custom_llmaaj_client if provided ============
        # Use custom client if provided, otherwise fall back to default (405b)
        if custom_llmaaj_client:
            llm_client_for_eval = custom_llmaaj_client
        else:
            llm_client_for_eval = get_provider(
                model_id="meta-llama/llama-3-405b-instruct",
                params={
                    "min_new_tokens": 0,
                    "decoding_method": "greedy",
                    "max_new_tokens": 4096,
                },
                embedding_model_id="sentence-transformers/all-minilm-l6-v2",
                **extra_kwargs,
            )
        # ============ END GOVCLOUD FIX ============

        # output response matching
        self.matcher = LLMMatcher(
            llm_client=llm_client_for_eval,'''

    content = content.replace(old_matcher, new_matcher)

    # Replace other 405b occurrences with llm_client_for_eval
    # For rag_llm_as_a_judge
    old_rag = '''        # only used for RAG evaluation
        self.rag_llm_as_a_judge = LLMJudge(
            llm_client=get_provider(
                model_id="meta-llama/llama-3-405b-instruct",
                params={
                    "min_new_tokens": 0,
                    "decoding_method": "greedy",
                    "max_new_tokens": 4096,
                },
                **extra_kwargs,
            ),'''

    new_rag = '''        # only used for RAG evaluation
        self.rag_llm_as_a_judge = LLMJudge(
            llm_client=llm_client_for_eval,'''

    content = content.replace(old_rag, new_rag)

    # For safety_llm_as_a_judge
    old_safety = '''        self.safety_llm_as_a_judge = LLMSafetyJudge(
            llm_client=get_provider(
                model_id="meta-llama/llama-3-405b-instruct",
                params={
                    "min_new_tokens": 0,
                    "decoding_method": "greedy",
                    "max_new_tokens": 4096,
                },
                **extra_kwargs,
            ),'''

    new_safety = '''        self.safety_llm_as_a_judge = LLMSafetyJudge(
            llm_client=llm_client_for_eval,'''

    content = content.replace(old_safety, new_safety)

    return content


# =============================================================================
# AGENTOPS PATCHING SYSTEM
# =============================================================================

def find_agentops_package() -> Optional[Path]:
    """Find the agentops package in the current environment."""
    try:
        spec = importlib.util.find_spec("agentops")
        if spec and spec.origin:
            return Path(spec.origin).parent
    except (ImportError, AttributeError):
        pass

    for path in sys.path:
        agentops_path = Path(path) / "agentops"
        if agentops_path.exists() and (agentops_path / "__init__.py").exists():
            return agentops_path

    return None


def check_file_patched(file_path: Path) -> bool:
    """Check if a file has already been patched."""
    if not file_path.exists():
        return False

    try:
        content = file_path.read_text(encoding='utf-8')
        # Check for our patch markers
        if "GOVCLOUD FIX" in content:
            # Verify key patches are present based on file
            if file_path.name == "evaluation_package.py":
                return "llm_client_for_eval" in content
            elif file_path.name == "runner.py":
                return "custom_llmaaj_client=llmaaj_provider" in content
            elif file_path.name == "clients.py":
                return "isinstance(cmc, dict)" in content
            elif file_path.name == "main.py":
                return "exporters=[]" in content
            return True
        return False
    except Exception:
        return False


def apply_patch_to_file(file_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Apply patch to a single agentops file."""
    result = {
        "file": file_path.name,
        "status": "unknown",
        "message": "",
    }

    if not file_path.exists():
        result["status"] = "error"
        result["message"] = "File not found"
        return result

    # Check if already patched
    if check_file_patched(file_path):
        result["status"] = "already_patched"
        result["message"] = "Already patched"
        return result

    if dry_run:
        result["status"] = "would_patch"
        result["message"] = "Would apply patch"
        return result

    try:
        # Read original content
        content = file_path.read_text(encoding='utf-8')
        original_content = content

        # Apply appropriate patch
        if file_path.name == "main.py":
            content = patch_main_py(content)
        elif file_path.name == "clients.py":
            content = patch_clients_py(content)
        elif file_path.name == "runner.py":
            content = patch_runner_py(content)
        elif file_path.name == "evaluation_package.py":
            content = patch_evaluation_package_py(content)
        else:
            result["status"] = "skipped"
            result["message"] = "No patch defined for this file"
            return result

        # Check if patch actually changed anything
        if content == original_content:
            result["status"] = "no_match"
            result["message"] = "Pattern not found (ADK version may differ)"
            return result

        # Create backup
        backup_path = file_path.with_suffix(file_path.suffix + ".original.bak")
        if not backup_path.exists():
            shutil.copy2(file_path, backup_path)

        # Write patched content
        file_path.write_text(content, encoding='utf-8')

        # Clear pycache
        pycache_dir = file_path.parent / "__pycache__"
        if pycache_dir.exists():
            shutil.rmtree(pycache_dir, ignore_errors=True)

        result["status"] = "patched"
        result["message"] = "Patch applied successfully"

    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Error: {str(e)}"

    return result


def apply_all_patches(config: Dict[str, Any], dry_run: bool = False) -> bool:
    """Apply all GovCloud patches to agentops files."""
    print(f"\n{'='*60}")
    print("  GOVCLOUD PATCH SYSTEM")
    print(f"{'='*60}")

    agentops_path = find_agentops_package()
    if not agentops_path:
        print("\nERROR: Could not find agentops package.")
        print("Make sure ibm-watsonx-orchestrate-adk is installed.")
        return False

    print(f"\nFound agentops at: {agentops_path}")

    if dry_run:
        print("(DRY RUN - no changes will be made)")

    all_success = True
    results = []

    for filename in FILES_TO_PATCH:
        file_path = agentops_path / filename
        result = apply_patch_to_file(file_path, dry_run)
        results.append(result)

        status_symbol = {
            "patched": "\u2705",
            "already_patched": "\u2714\ufe0f",
            "would_patch": "\U0001f504",
            "no_match": "\u26a0\ufe0f",
            "error": "\u274c",
            "skipped": "\u23ed\ufe0f",
        }.get(result["status"], "?")

        print(f"\n  {status_symbol} {filename}: {result['message']}")

        if result["status"] in ["error", "no_match"]:
            all_success = False

    print(f"\n{'─'*60}")

    success_count = sum(1 for r in results if r["status"] in ["patched", "already_patched"])
    print(f"  Files ready: {success_count}/{len(FILES_TO_PATCH)}")

    if all_success:
        print("  Status: READY for GovCloud evaluation")
    else:
        print("  Status: Some patches could not be applied")
        print("\n  This may happen if the ADK version differs from expected.")
        print("  Try updating ibm-watsonx-orchestrate-adk to the latest version.")

    print(f"{'─'*60}\n")

    return all_success


def verify_patches(config: Dict[str, Any]) -> bool:
    """Verify all patches are applied."""
    agentops_path = find_agentops_package()
    if not agentops_path:
        return False

    for filename in FILES_TO_PATCH:
        file_path = agentops_path / filename
        if not check_file_patched(file_path):
            return False

    return True


# =============================================================================
# EVALUATION COMMANDS
# =============================================================================

def ensure_orchestrate_env(config: Dict[str, Any]) -> bool:
    """Ensure orchestrate environment is activated."""
    result = subprocess.run(
        ["orchestrate", "env", "list"],
        capture_output=True,
        text=True
    )

    if "(active)" not in result.stdout:
        print("\nNo active orchestrate environment. Please run:")
        print("  orchestrate env activate <env-name> --api-key <your-key>")
        return False

    return True


def cmd_setup(config: Dict[str, Any], args) -> int:
    """Apply patches programmatically."""
    success = apply_all_patches(config, dry_run=args.dry_run)
    return 0 if success else 1


def generate_adk_config(config: Dict[str, Any], output_path: Path) -> Path:
    """Generate ADK-compatible config YAML from our simplified config."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required. Run: pip install pyyaml")
        sys.exit(1)

    models = config.get("models", {})
    provider = config.get("provider", {})
    evaluation = config.get("evaluation", {})

    # Get model IDs with defaults
    llm_user_model = models.get("llm_user", "meta-llama/llama-3-2-90b-vision-instruct")
    llm_judge_model = models.get("llm_judge", "meta-llama/llama-3-2-90b-vision-instruct")
    embedding_model = models.get("embedding", "sentence-transformers/all-minilm-l6-v2")

    adk_config = {
        "llm_user_config": {
            "model_id": llm_user_model,
        },
        "provider_config": {
            "model_id": llm_user_model,
            "provider": provider.get("type", "gateway"),
            "embedding_model_id": embedding_model,
            "vendor": provider.get("vendor", "ibm"),
            "referenceless_eval": False,
        },
        "custom_metrics_config": {
            "llmaaj_config": {
                "model_id": llm_judge_model,
                "provider": "watsonx",
                "embedding_model_id": embedding_model,
                "vendor": provider.get("vendor", "ibm"),
                "referenceless_eval": False,
            },
        },
        "metrics": [
            "JourneySuccessMetric",
            "ToolCalling",
        ],
        "similarity_threshold": evaluation.get("similarity_threshold", 0.8),
        "enable_fuzzy_matching": evaluation.get("enable_fuzzy_matching", False),
        "is_strict": evaluation.get("is_strict", True),
        "num_workers": evaluation.get("num_workers", 1),
        "n_runs": evaluation.get("n_runs", 1),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(adk_config, f, default_flow_style=False)

    return output_path


def cmd_evaluate(config: Dict[str, Any], args) -> int:
    """Run ADK evaluation."""
    print(f"\n{'='*60}")
    print("  RUNNING GOVCLOUD EVALUATION")
    print(f"{'='*60}")

    # Check orchestrate environment
    if not ensure_orchestrate_env(config):
        print("\nTIP: Run this first:")
        print("  orchestrate env activate <env-name> --api-key <your-key>")
        return 1

    # Verify patches are applied
    if not verify_patches(config):
        print("\nPatches not applied. Running setup first...")
        if not apply_all_patches(config):
            print("ERROR: Could not apply patches")
            return 1

    paths = config.get("paths", {})
    test_dir = Path(paths.get("test_cases", "./sample_test_cases"))
    output_dir = Path(paths.get("output_dir", "./eval_results_govcloud"))

    # Get test files
    test_files = sorted(test_dir.glob("*.json"))
    if args.limit:
        test_files = test_files[:args.limit]

    if not test_files:
        print(f"\nERROR: No test files found in {test_dir}")
        return 1

    print(f"\nRunning {len(test_files)} test case(s)")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate ADK-compatible config
    adk_config_path = SCRIPT_DIR / "eval_config_generated.yaml"
    generate_adk_config(config, adk_config_path)
    print(f"Generated ADK config: {adk_config_path}")

    # Build test paths string
    test_paths_str = ",".join(str(f) for f in test_files)

    cmd = [
        "orchestrate", "evaluations", "evaluate",
        "--test-paths", test_paths_str,
        "--output-dir", str(output_dir),
        "--config", str(adk_config_path),
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print(f"{'─'*60}")

    start_time = datetime.now()
    result = subprocess.run(cmd)
    duration = (datetime.now() - start_time).total_seconds()

    print(f"{'─'*60}")
    print(f"Completed in {duration:.1f}s (exit code: {result.returncode})")

    return result.returncode


def cmd_report(config: Dict[str, Any], args) -> int:
    """Show evaluation report."""
    print(f"\n{'='*60}")
    print("  EVALUATION REPORT")
    print(f"{'='*60}")

    paths = config.get("paths", {})
    results_dir = Path(paths.get("output_dir", "./eval_results_govcloud"))

    import csv

    csv_files = list(results_dir.rglob("summary_metrics.csv"))
    if not csv_files:
        print("ERROR: No results found. Run --evaluate first.")
        return 1

    csv_file = sorted(csv_files)[-1]
    print(f"\nResults: {csv_file}")

    with open(csv_file, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    passed = sum(1 for r in rows if r.get("is_success", "").lower() == "true")
    pass_rate = (passed / total * 100) if total > 0 else 0

    print(f"\n  RESULTS")
    print(f"  {'─'*40}")
    print(f"  Total:  {total}")
    print(f"  Passed: {passed} ({pass_rate:.1f}%)")
    print(f"  Failed: {total - passed}")
    print(f"  {'─'*40}")
    print(f"\n  PASS RATE: {pass_rate:.0f}%")

    return 0


# =============================================================================
# LLM AS JUDGE (via Orchestrate Gateway)
# =============================================================================

def get_orchestrate_auth() -> Optional[Dict[str, str]]:
    """Get authentication info from orchestrate CLI."""
    try:
        # Get active environment info
        result = subprocess.run(
            ["orchestrate", "env", "list"],
            capture_output=True,
            text=True
        )

        if "(active)" not in result.stdout:
            return None

        # Parse the active environment to get the URL
        # The orchestrate CLI stores auth in ~/.orchestrate or similar
        # We'll use the gateway endpoint directly

        # Try to get from environment variables set by orchestrate
        api_key = os.getenv("WXO_API_KEY") or os.getenv("WATSONX_API_KEY")
        instance_url = os.getenv("WXO_INSTANCE_URL") or os.getenv("WATSONX_URL")

        if api_key and instance_url:
            return {"api_key": api_key, "instance_url": instance_url}

        return None
    except Exception:
        return None


def call_gateway_llm(config: Dict[str, Any], prompt: str, model_id: str) -> str:
    """Call LLM via Orchestrate Gateway (same as ADK uses)."""

    # Get auth from config or environment
    watsonx_config = config.get("watsonx", {})
    api_key = watsonx_config.get("api_key") or os.getenv("WXO_API_KEY") or os.getenv("WATSONX_API_KEY")
    instance_url = watsonx_config.get("url") or os.getenv("WXO_INSTANCE_URL")

    if not api_key:
        raise ValueError("No API key found. Set watsonx.api_key in config or WXO_API_KEY env var")

    if not instance_url:
        raise ValueError("No instance URL found. Set watsonx.url in config or WXO_INSTANCE_URL env var")

    # Build gateway URL
    gateway_url = f"{instance_url}/v1/orchestrate/gateway/model/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are an evaluation judge. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 500
    }

    response = requests.post(gateway_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def llm_judge_evaluate(
    config: Dict[str, Any],
    question: str,
    expected_answer: str,
    agent_response: str,
    model_id: str
) -> Dict[str, Any]:
    """
    Use LLM as a Judge to evaluate if agent response is correct.

    Uses the Orchestrate Gateway (same auth as ADK) to call the LLM.
    Returns verdict (CORRECT/PARTIALLY_CORRECT/INCORRECT) and score (0-1).
    """
    judge_prompt = f"""You are an expert evaluation judge for a health benefits Q&A system.
Your task is to evaluate if an AI agent's response correctly answers a question about employee benefits.

EVALUATION CRITERIA:
1. CORRECTNESS: Does the response contain factually correct information matching the expected answer?
2. COMPLETENESS: Does it cover the key points from the expected answer?
3. RELEVANCE: Does it directly address the question asked?
4. NO HALLUCINATION: Does it avoid making up false information not in the expected answer?

QUESTION:
{question}

EXPECTED ANSWER (Ground Truth):
{expected_answer}

AGENT'S ACTUAL RESPONSE:
{agent_response}

INSTRUCTIONS:
- Compare the AGENT'S RESPONSE against the EXPECTED ANSWER
- The agent doesn't need to match word-for-word, but must convey the same key information
- If the agent provides additional helpful context beyond the expected answer, that's OK
- If the agent contradicts or omits key information from the expected answer, mark as incorrect

Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{{
    "verdict": "CORRECT" or "PARTIALLY_CORRECT" or "INCORRECT",
    "score": <float between 0.0 and 1.0>,
    "correctness": <float 0-1>,
    "completeness": <float 0-1>,
    "relevance": <float 0-1>,
    "reasoning": "<brief explanation of your evaluation>"
}}
"""

    result_text = ""
    try:
        result_text = call_gateway_llm(config, judge_prompt, model_id)

        # Handle potential markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]

        # Find JSON in response
        json_start = result_text.find("{")
        json_end = result_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            result_text = result_text[json_start:json_end]

        result = json.loads(result_text)

        # Ensure required fields
        result.setdefault("verdict", "INCORRECT")
        result.setdefault("score", 0.0)
        result.setdefault("reasoning", "No reasoning provided")

        return {
            "success": True,
            "verdict": result["verdict"],
            "score": float(result["score"]),
            "correctness": float(result.get("correctness", result["score"])),
            "completeness": float(result.get("completeness", result["score"])),
            "relevance": float(result.get("relevance", result["score"])),
            "reasoning": result["reasoning"],
            "model": model_id,
            "provider": "orchestrate-gateway"
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "verdict": "ERROR",
            "score": 0.0,
            "reasoning": f"Failed to parse LLM response as JSON: {str(e)}",
            "raw_response": result_text if result_text else None
        }
    except Exception as e:
        return {
            "success": False,
            "verdict": "ERROR",
            "score": 0.0,
            "reasoning": f"LLM Judge error: {str(e)}"
        }


def load_ground_truth(ground_truth_dir: Path) -> Dict[str, Dict]:
    """Load ground truth files that have expected answers."""
    ground_truth = {}

    if not ground_truth_dir.exists():
        return ground_truth

    for gt_file in ground_truth_dir.glob("*.json"):
        try:
            with open(gt_file, encoding='utf-8') as f:
                data = json.load(f)

            question = data.get("starting_sentence", "")
            if not question:
                continue

            # Extract expected response and keywords from goal_details
            expected_response = ""
            keywords = []
            for detail in data.get("goal_details", []):
                if detail.get("type") == "text" or detail.get("name") == "summarize":
                    expected_response = detail.get("response", "")
                    keywords = detail.get("keywords", [])
                    break

            ground_truth[question.lower()] = {
                "question": question,
                "expected_response": expected_response,
                "keywords": keywords,
                "source_file": gt_file.name
            }
        except Exception as e:
            print(f"  Warning: Could not load {gt_file.name}: {e}")

    return ground_truth


def cmd_judge(config: Dict[str, Any], args) -> int:
    """Run LLM-as-Judge evaluation on existing results."""
    print(f"\n{'='*60}")
    print("  LLM AS JUDGE EVALUATION")
    print(f"{'='*60}")

    # Check for required config
    watsonx_config = config.get("watsonx", {})
    api_key = watsonx_config.get("api_key") or os.getenv("WXO_API_KEY") or os.getenv("WATSONX_API_KEY")
    instance_url = watsonx_config.get("url") or os.getenv("WXO_INSTANCE_URL")

    if not api_key:
        print("\nERROR: No API key found for LLM-as-Judge.")
        print("Set one of the following:")
        print("  1. watsonx.api_key in govcloud_config.yaml")
        print("  2. WXO_API_KEY environment variable")
        print("  3. WATSONX_API_KEY environment variable")
        return 1

    if not instance_url:
        print("\nERROR: No instance URL found.")
        print("Set one of the following:")
        print("  1. watsonx.url in govcloud_config.yaml")
        print("  2. WXO_INSTANCE_URL environment variable")
        return 1

    models = config.get("models", {})
    model_id = models.get("llm_judge", "meta-llama/llama-3-2-90b-vision-instruct")
    print(f"\nLLM Judge Model: {model_id}")
    print(f"Using Gateway: {instance_url}")

    paths = config.get("paths", {})
    test_dir = Path(paths.get("test_cases", "./test_data"))
    results_dir = Path(paths.get("output_dir", "./eval_results_govcloud"))

    # Load ground truth with expected answers
    print(f"Loading ground truth from: {test_dir}")
    ground_truth = load_ground_truth(test_dir)
    print(f"Loaded {len(ground_truth)} ground truth entries")

    if not ground_truth:
        print("\nWARNING: No ground truth files found with expected answers.")
        print("Make sure your test case JSON files have a 'response' field in goal_details.")
        return 1

    # Find evaluation results (messages directory)
    messages_dirs = list(results_dir.rglob("messages"))
    if not messages_dirs:
        print("\nERROR: No evaluation results found. Run --evaluate first.")
        return 1

    messages_dir = sorted(messages_dirs)[-1]  # Latest
    print(f"Reading results from: {messages_dir}")

    # Process each result
    judge_results = []
    processed = 0

    for result_file in sorted(messages_dir.glob("*.json")):
        # Skip metrics and analyze files
        if "metrics" in result_file.name or "analyze" in result_file.name:
            continue

        try:
            with open(result_file, encoding='utf-8') as f:
                result_data = json.load(f)

            # Extract question and agent response
            if isinstance(result_data, list):
                messages = result_data
            else:
                messages = result_data.get("messages", [])

            question = ""
            agent_response = ""

            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "user" and not question:
                    question = msg.get("content", "")
                elif msg.get("role") == "assistant" and msg.get("type") == "text" and not agent_response:
                    content = msg.get("content", "")
                    if content and "unfortunately" not in content.lower():
                        agent_response = content

            if not question or not agent_response:
                continue

            # Find matching ground truth
            gt = ground_truth.get(question.lower())

            if gt and gt.get("expected_response"):
                print(f"\n  Evaluating: {question[:50]}...")

                judge_result = llm_judge_evaluate(
                    config=config,
                    question=question,
                    expected_answer=gt["expected_response"],
                    agent_response=agent_response,
                    model_id=model_id
                )

                result_entry = {
                    "question": question,
                    "expected_answer": gt["expected_response"][:200] + "..." if len(gt["expected_response"]) > 200 else gt["expected_response"],
                    "agent_response": agent_response[:200] + "..." if len(agent_response) > 200 else agent_response,
                    "llm_judge": judge_result,
                    "source_file": result_file.name,
                    "ground_truth_file": gt.get("source_file", "")
                }

                judge_results.append(result_entry)
                processed += 1

                # Print result
                verdict = judge_result.get("verdict", "ERROR")
                score = judge_result.get("score", 0)
                symbol = "+" if verdict == "CORRECT" else "~" if verdict == "PARTIALLY_CORRECT" else "-"
                print(f"    [{symbol}] {verdict} (score: {score:.2f})")

                if args.verbose:
                    print(f"    Reasoning: {judge_result.get('reasoning', 'N/A')[:100]}...")

        except Exception as e:
            print(f"  Error processing {result_file.name}: {e}")
            continue

        if args.limit and processed >= args.limit:
            break

    # Save judge results
    judge_results_file = results_dir / "llm_judge_results.json"
    with open(judge_results_file, "w", encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": model_id,
            "total_evaluated": len(judge_results),
            "results": judge_results
        }, f, indent=2)

    print(f"\n{'─'*60}")
    print(f"Processed {processed} results")
    print(f"Judge results saved to: {judge_results_file}")

    # Summary
    if judge_results:
        correct = sum(1 for r in judge_results if r["llm_judge"].get("verdict") == "CORRECT")
        partial = sum(1 for r in judge_results if r["llm_judge"].get("verdict") == "PARTIALLY_CORRECT")
        incorrect = sum(1 for r in judge_results if r["llm_judge"].get("verdict") == "INCORRECT")
        avg_score = sum(r["llm_judge"].get("score", 0) for r in judge_results) / len(judge_results)

        print(f"\n  LLM JUDGE SUMMARY")
        print(f"  {'─'*40}")
        print(f"  Total evaluated:    {len(judge_results)}")
        print(f"  [+] Correct:        {correct} ({correct/len(judge_results)*100:.1f}%)")
        print(f"  [~] Partial:        {partial} ({partial/len(judge_results)*100:.1f}%)")
        print(f"  [-] Incorrect:      {incorrect} ({incorrect/len(judge_results)*100:.1f}%)")
        print(f"  Average Score:      {avg_score:.2f}")
        print(f"  {'─'*40}")
        print(f"\n  LLM JUDGE SCORE: {avg_score*100:.0f}%")

    return 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GovCloud Agent Evaluation (with Auto-Patching + LLM-as-Judge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python govcloud_eval.py --setup              # Apply patches
  python govcloud_eval.py --evaluate           # Run ADK evaluation
  python govcloud_eval.py --judge              # Run LLM-as-Judge evaluation
  python govcloud_eval.py --report             # Show results
  python govcloud_eval.py --all                # Setup + Evaluate + Judge + Report

LLM-as-Judge requires:
  - pip install ibm-watsonx-ai
  - Set watsonx.api_key in govcloud_config.yaml OR set WATSONX_API_KEY env var
        """
    )

    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_FILE),
                        help="Path to govcloud_config.yaml")
    parser.add_argument("--setup", action="store_true",
                        help="Apply patches to agentops files")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run ADK evaluation")
    parser.add_argument("--judge", action="store_true",
                        help="Run LLM-as-Judge semantic evaluation")
    parser.add_argument("--report", action="store_true",
                        help="Show results report")
    parser.add_argument("--all", action="store_true",
                        help="Run full pipeline: setup + evaluate + judge + report")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of test cases")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be patched without making changes")

    args = parser.parse_args()

    # Load config
    config = load_yaml_config(Path(args.config))

    if not any([args.setup, args.evaluate, args.judge, args.report, args.all]):
        parser.print_help()
        sys.exit(0)

    exit_code = 0

    if args.all or args.setup:
        exit_code = cmd_setup(config, args)
        if exit_code != 0 and not args.all:
            sys.exit(exit_code)

    if args.all or args.evaluate:
        exit_code = cmd_evaluate(config, args)
        if exit_code != 0 and not args.all:
            sys.exit(exit_code)

    if args.all or args.judge:
        exit_code = cmd_judge(config, args)
        if exit_code != 0 and not args.all:
            sys.exit(exit_code)

    if args.all or args.report:
        exit_code = cmd_report(config, args)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
