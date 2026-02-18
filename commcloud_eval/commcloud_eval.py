#!/usr/bin/env python3
"""
Agent Evaluation (ADK + LLM-as-Judge)
======================================
All-in-one script for running IBM watsonx Orchestrate agent evaluations.

This script:
1. Runs the ADK evaluation
2. Runs LLM-as-Judge semantic evaluation
3. Provides results summary

Usage:
    python commcloud_eval.py --evaluate          # Run ADK evaluation
    python commcloud_eval.py --judge             # Run LLM-as-Judge evaluation
    python commcloud_eval.py --report            # Show results from last run
    python commcloud_eval.py --all               # Run everything
"""

import json
import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

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
DEFAULT_CONFIG_FILE = SCRIPT_DIR / "commcloud_config.yaml"


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required. Run: pip install pyyaml")
        sys.exit(1)

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Create commcloud_config.yaml or specify with --config")
        sys.exit(1)

    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_env_file():
    """Load environment variables from .env file."""
    env_paths = [
        Path(__file__).parent / ".env",
        Path(__file__).parent / "eval.env",
        Path(__file__).parent.parent / "config" / ".env",
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
# EVALUATION COMMANDS
# =============================================================================

def ensure_orchestrate_env(config: Dict[str, Any]) -> bool:
    """Ensure orchestrate environment is activated."""
    # Try CLI first
    result = subprocess.run(
        ["orchestrate", "env", "list"],
        capture_output=True,
        text=True
    )

    if "(active)" in result.stdout:
        return True

    # Fallback: check config file directly (CLI may crash on some SDK versions)
    try:
        import yaml
        config_path = Path.home() / ".config" / "orchestrate" / "config.yaml"
        if config_path.exists():
            with open(config_path, encoding='utf-8') as f:
                orch_config = yaml.safe_load(f) or {}
            active_env = orch_config.get("context", {}).get("active_environment")
            if active_env:
                print(f"Active environment: {active_env} (from config file)")
                return True
    except Exception:
        pass

    print("\nNo active orchestrate environment. Please run:")
    print("  orchestrate env activate <env-name> --api-key <your-key>")
    return False


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

    llm_user_model = models.get("llm_user", "meta-llama/llama-3-405b-instruct")
    llm_judge_model = models.get("llm_judge", "meta-llama/llama-3-405b-instruct")
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
    print("  RUNNING EVALUATION")
    print(f"{'='*60}")

    # Check orchestrate environment
    if not ensure_orchestrate_env(config):
        print("\nTIP: Run this first:")
        print("  orchestrate env activate <env-name> --api-key <your-key>")
        return 1

    # Set WO_TOKEN env var so ADK providers use cached token
    cached_token = get_orchestrate_cached_token()
    if cached_token:
        os.environ["WO_TOKEN"] = cached_token
        print("Using cached orchestrate token for evaluation")

    paths = config.get("paths", {})
    test_dir = Path(paths.get("test_cases", "./test_data"))
    output_dir = Path(paths.get("output_dir", "./eval_results"))

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

    # Pass environment with WO_TOKEN and WO_API_KEY
    env = os.environ.copy()
    if cached_token:
        env["WO_TOKEN"] = cached_token
        env["USE_GATEWAY_MODEL_PROVIDER"] = "TRUE"
        # Set WO_API_KEY from env if available (needed by ADK)
        if not env.get("WO_API_KEY"):
            env["WO_API_KEY"] = os.getenv("WXO_API_KEY", "")

    start_time = datetime.now()
    result = subprocess.run(cmd, env=env)
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
    results_dir = Path(paths.get("output_dir", "./eval_results"))

    # First, try to read LLM Judge results
    llm_judge_file = results_dir / "llm_judge_results.json"
    if llm_judge_file.exists():
        print(f"\nSource: {llm_judge_file}")
        with open(llm_judge_file, encoding='utf-8') as f:
            llm_results = json.load(f)

        total = llm_results.get("total_evaluated", 0)
        passed = llm_results.get("total_passed", 0)
        pass_rate = llm_results.get("pass_rate", "0%")

        print(f"\n  RESULTS (LLM Judge)")
        print(f"  {'─'*40}")
        print(f"  Total:  {total}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {total - passed}")
        print(f"  {'─'*40}")
        print(f"\n  PASS RATE: {pass_rate}")
        return 0

    # Fallback: read ADK's summary_metrics.csv
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

    print(f"\n  RESULTS (ADK Evaluation)")
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

# Auth endpoints
AUTH_ENDPOINT_IBM_CLOUD = "https://iam.cloud.ibm.com/identity/token"
AUTH_ENDPOINT_SAAS = "https://iam.platform.saas.ibm.com/siusermgr/api/1.0/apikeys/token"

# Token cache
_cached_token = None
_token_refresh_time = 0


def is_ibm_cloud_url(url: str) -> bool:
    """Check if URL is IBM Cloud (vs SaaS/AWS)."""
    return "cloud.ibm.com" in url if url else False


def get_orchestrate_cached_token() -> Optional[str]:
    """Read token from orchestrate CLI's cached credentials file."""
    try:
        import yaml

        config_path = Path.home() / ".config" / "orchestrate" / "config.yaml"
        if not config_path.exists():
            return None

        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

        active_env = config.get("context", {}).get("active_environment")
        if not active_env:
            return None

        creds_path = Path.home() / ".cache" / "orchestrate" / "credentials.yaml"
        if not creds_path.exists():
            return None

        with open(creds_path, encoding='utf-8') as f:
            creds = yaml.safe_load(f) or {}

        auth = creds.get("auth", {})
        env_auth = auth.get(active_env, {})
        token = env_auth.get("wxo_mcsp_token")

        if token:
            return token

    except Exception:
        pass

    return None


def get_access_token(api_key: str, instance_url: str) -> str:
    """Get access token - tries orchestrate cache first, then token exchange."""
    global _cached_token, _token_refresh_time
    import time

    if _cached_token and time.time() < _token_refresh_time:
        return _cached_token

    # Try orchestrate CLI cached credentials
    cached_token = get_orchestrate_cached_token()
    if cached_token:
        _cached_token = cached_token
        _token_refresh_time = time.time() + 3000
        return cached_token

    # Try ADK's tenant_setup
    try:
        from agentops.service_instance import tenant_setup
        token, resolved_url, _ = tenant_setup(service_url=instance_url, tenant_name=None)
        if token:
            _cached_token = token
            _token_refresh_time = time.time() + 3000
            return token
    except Exception:
        pass

    # IBM Cloud IAM token exchange
    if is_ibm_cloud_url(instance_url):
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        }
        response = requests.post(AUTH_ENDPOINT_IBM_CLOUD, headers=headers, data=data, timeout=30)
    else:
        # SaaS/AWS token exchange
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        response = requests.post(AUTH_ENDPOINT_SAAS, headers=headers, json={"apikey": api_key}, timeout=30)

    response.raise_for_status()
    result = response.json()

    token = result.get("access_token") or result.get("token")
    if not token:
        raise RuntimeError(f"No token in response: {result}")

    expires_in = result.get("expires_in", 3600)
    _cached_token = token
    _token_refresh_time = time.time() + int(0.8 * expires_in)

    return token


def call_gateway_llm(config: Dict[str, Any], prompt: str, model_id: str) -> str:
    """Call LLM via Orchestrate Gateway."""
    import json
    import uuid

    watsonx_config = config.get("watsonx", {})
    api_key = watsonx_config.get("api_key") or os.getenv("WXO_API_KEY") or os.getenv("WO_API_KEY")
    instance_url = watsonx_config.get("url") or os.getenv("WXO_INSTANCE_URL") or os.getenv("WO_INSTANCE")

    if not api_key:
        raise ValueError("No API key found. Set WXO_API_KEY or WO_API_KEY env var")
    if not instance_url:
        raise ValueError("No instance URL found. Set WXO_INSTANCE_URL or WO_INSTANCE env var")

    instance_url = instance_url.rstrip("/")
    access_token = get_access_token(api_key, instance_url)
    chat_url = f"{instance_url}/v1/orchestrate/gateway/model/chat/completions"

    x_gateway_config = {
        "strategy": {"mode": "single"},
        "targets": [{
            "provider": "watsonx",
            "api_key": "gateway",
            "override_params": {"model": model_id}
        }]
    }

    request_id = str(uuid.uuid4())

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
        "x-request-id": request_id,
        "x-gateway-config": json.dumps(x_gateway_config, separators=(",", ":")),
    }

    payload = {
        "model": f"watsonx/{model_id}",
        "messages": [
            {"role": "system", "content": "You are an evaluation judge. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
    }

    response = requests.post(chat_url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    result = response.json()
    choices = result.get("choices", [])
    if choices and len(choices) > 0:
        message = choices[0].get("message", {})
        return message.get("content", "")

    raise RuntimeError(f"Unexpected response format: {result}")


def llm_judge_evaluate(
    config: Dict[str, Any],
    question: str,
    expected_answer: str,
    agent_response: str,
    model_id: str
) -> Dict[str, Any]:
    """Use LLM as a Judge to evaluate if agent response is correct."""
    judge_prompt = f"""You are an expert evaluation judge for a health benefits Q&A system.
Your task is to evaluate if an AI agent's response correctly answers a question about employee benefits.

EVALUATION CRITERIA:
1. CORRECTNESS: Does the response contain factually correct information matching the expected answer?
2. COMPLETENESS: Does it cover the key points from the expected answer?
3. RELEVANCE: Does it directly address the question asked?
4. NO HALLUCINATION: Does it avoid making up false information not in the expected answer?

IMPORTANT - NUMERIC EQUIVALENCES:
When comparing numbers and time periods, recognize these as EQUIVALENT:
- "15 days" = "2 weeks and 1 day" (7+7+1=15)
- "31 days" = "about one month" = "approximately 1 month"
- "10 days" = "2 weeks" (business days) OR "10 calendar days"
- Different numeric expressions of the SAME value are CORRECT, not partially correct

QUESTION:
{question}

EXPECTED ANSWER (Ground Truth):
{expected_answer}

AGENT'S ACTUAL RESPONSE:
{agent_response}

INSTRUCTIONS:
- Compare the AGENT'S RESPONSE against the EXPECTED ANSWER
- The agent doesn't need to match word-for-word, but must convey the same key information
- NUMERIC VALUES: If the numbers are mathematically equivalent (e.g., "15 days" vs "2 weeks + 1 day"), mark as CORRECT
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

        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]

        json_start = result_text.find("{")
        json_end = result_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            result_text = result_text[json_start:json_end]

        result = json.loads(result_text)

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

    watsonx_config = config.get("watsonx", {})
    api_key = watsonx_config.get("api_key") or os.getenv("WXO_API_KEY") or os.getenv("WO_API_KEY")
    instance_url = watsonx_config.get("url") or os.getenv("WXO_INSTANCE_URL") or os.getenv("WO_INSTANCE")

    if not api_key:
        print("\nERROR: No API key found.")
        print("Set one of: WXO_API_KEY, WO_API_KEY env var, or watsonx.api_key in config")
        return 1

    if not instance_url:
        print("\nERROR: No instance URL found.")
        print("Set one of: WXO_INSTANCE_URL, WO_INSTANCE env var, or watsonx.url in config")
        return 1

    models = config.get("models", {})
    model_id = models.get("llm_judge", "meta-llama/llama-3-405b-instruct")
    print(f"\nLLM Judge Model: {model_id}")
    print(f"Gateway: {instance_url}")

    paths = config.get("paths", {})
    test_dir = Path(paths.get("test_cases", "./test_data"))
    results_dir = Path(paths.get("output_dir", "./eval_results"))

    print(f"Loading ground truth from: {test_dir}")
    ground_truth = load_ground_truth(test_dir)
    print(f"Loaded {len(ground_truth)} ground truth entries")

    if not ground_truth:
        print("\nWARNING: No ground truth files found with expected answers.")
        print("Make sure your test case JSON files have a 'response' field in goal_details.")
        return 1

    messages_dirs = list(results_dir.rglob("messages"))
    if not messages_dirs:
        print("\nERROR: No evaluation results found. Run --evaluate first.")
        return 1

    messages_dir = sorted(messages_dirs)[-1]
    print(f"Reading results from: {messages_dir}")

    judge_results = []
    processed = 0

    for result_file in sorted(messages_dir.glob("*.json")):
        if "metrics" in result_file.name or "analyze" in result_file.name:
            continue

        try:
            with open(result_file, encoding='utf-8') as f:
                result_data = json.load(f)

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

    for r in judge_results:
        verdict = r["llm_judge"].get("verdict", "")
        score = r["llm_judge"].get("score", 0)
        r["llm_judge"]["passed"] = (verdict == "CORRECT") or (verdict == "PARTIALLY_CORRECT" and score >= 0.7)

    total_passed = sum(1 for r in judge_results if r["llm_judge"].get("passed", False))

    judge_results_file = results_dir / "llm_judge_results.json"
    with open(judge_results_file, "w", encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": model_id,
            "total_evaluated": len(judge_results),
            "total_passed": total_passed,
            "pass_rate": f"{total_passed/len(judge_results)*100:.0f}%" if judge_results else "0%",
            "results": judge_results
        }, f, indent=2)

    print(f"\n{'─'*60}")
    print(f"Processed {processed} results")
    print(f"Judge results saved to: {judge_results_file}")

    if judge_results:
        correct = sum(1 for r in judge_results if r["llm_judge"].get("verdict") == "CORRECT")
        partial = sum(1 for r in judge_results if r["llm_judge"].get("verdict") == "PARTIALLY_CORRECT")
        incorrect = sum(1 for r in judge_results if r["llm_judge"].get("verdict") == "INCORRECT")
        avg_score = sum(r["llm_judge"].get("score", 0) for r in judge_results) / len(judge_results)

        partial_passed = sum(1 for r in judge_results
                            if r["llm_judge"].get("verdict") == "PARTIALLY_CORRECT"
                            and r["llm_judge"].get("score", 0) >= 0.7)
        total_passed = correct + partial_passed
        total_failed = incorrect + (partial - partial_passed)

        print(f"\n  LLM JUDGE SUMMARY")
        print(f"  {'─'*40}")
        print(f"  Total evaluated:    {len(judge_results)}")
        print(f"  [+] Correct:        {correct} ({correct/len(judge_results)*100:.1f}%)")
        print(f"  [~] Partial:        {partial} ({partial/len(judge_results)*100:.1f}%)")
        print(f"  [-] Incorrect:      {incorrect} ({incorrect/len(judge_results)*100:.1f}%)")
        print(f"  {'─'*40}")
        print(f"  PASSED (Correct + Partial>=0.7): {total_passed}/{len(judge_results)} ({total_passed/len(judge_results)*100:.0f}%)")
        print(f"  Average Score:      {avg_score:.2f}")
        print(f"  {'─'*40}")
        print(f"\n  LLM JUDGE SCORE: {avg_score*100:.0f}%")

    return 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Agent Evaluation (ADK + LLM-as-Judge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python commcloud_eval.py --evaluate           # Run ADK evaluation
  python commcloud_eval.py --judge              # Run LLM-as-Judge evaluation
  python commcloud_eval.py --report             # Show results
  python commcloud_eval.py --all                # Evaluate + Judge + Report

Prerequisites:
  - Active orchestrate environment (orchestrate env activate <env> --api-key <key>)
  - WXO_API_KEY and WXO_INSTANCE_URL env vars (or in .env file)
        """
    )

    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_FILE),
                        help="Path to config yaml")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run ADK evaluation")
    parser.add_argument("--judge", action="store_true",
                        help="Run LLM-as-Judge semantic evaluation")
    parser.add_argument("--report", action="store_true",
                        help="Show results report")
    parser.add_argument("--all", action="store_true",
                        help="Run full pipeline: evaluate + judge + report")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of test cases")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")

    args = parser.parse_args()

    config = load_yaml_config(Path(args.config))

    if not any([args.evaluate, args.judge, args.report, args.all]):
        parser.print_help()
        sys.exit(0)

    exit_code = 0

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
