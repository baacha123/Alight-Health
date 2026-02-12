#!/usr/bin/env python3
"""
GovCloud Test Case Generator
=============================
Generates ADK-compatible test case JSON files from Excel input.
Uses LLM to automatically extract keywords from expected answers.

Usage:
    python govcloud_generate.py                    # Uses config defaults
    python govcloud_generate.py --excel data.xlsx  # Custom Excel file
    python govcloud_generate.py --limit 5          # Generate only 5 cases
    python govcloud_generate.py --skip-llm         # Skip LLM (simple keywords)
    python govcloud_generate.py --sample           # Create sample Excel template
"""

import json
import os
import sys
import argparse
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CONFIG_FILE = SCRIPT_DIR / "govcloud_config.yaml"

# Token caching
_cached_token = None
_token_refresh_time = 0

# Auth endpoints
AUTH_ENDPOINT_IBM_CLOUD = "https://iam.cloud.ibm.com/identity/token"
AUTH_ENDPOINT_SAAS = "https://iam.platform.saas.ibm.com/siusermgr/api/1.0/apikeys/token"


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required. Run: pip install pyyaml")
        sys.exit(1)

    if not config_path.exists():
        return {}

    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


# =============================================================================
# URL DETECTION
# =============================================================================

def is_govcloud_url(url: str) -> bool:
    """Check if URL is for GovCloud/FedRAMP environment."""
    return "ibmforusgov.com" in url.lower()


def is_ibm_cloud_url(url: str) -> bool:
    """Check if URL is for IBM Cloud (vs SaaS/AWS)."""
    return "cloud.ibm.com" in url.lower()


# =============================================================================
# AUTHENTICATION (same as govcloud_eval.py)
# =============================================================================

def get_orchestrate_cached_credentials() -> tuple[Optional[str], Optional[str]]:
    """Read token AND instance URL from orchestrate CLI's cached config."""
    try:
        import yaml

        config_path = Path.home() / ".config" / "orchestrate" / "config.yaml"
        if not config_path.exists():
            return None, None

        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

        active_env = config.get("context", {}).get("active_environment")
        if not active_env:
            return None, None

        environments = config.get("environments", {})
        env_config = environments.get(active_env, {})
        instance_url = env_config.get("wxo_url")

        creds_path = Path.home() / ".cache" / "orchestrate" / "credentials.yaml"
        if not creds_path.exists():
            return None, instance_url

        with open(creds_path, encoding='utf-8') as f:
            creds = yaml.safe_load(f) or {}

        auth = creds.get("auth", {})
        env_auth = auth.get(active_env, {})
        token = env_auth.get("wxo_mcsp_token")

        if token and instance_url:
            return token, instance_url

    except Exception:
        pass

    return None, None


def get_access_token(api_key: str, instance_url: str) -> str:
    """Get access token via token exchange (IBM Cloud IAM)."""
    global _cached_token, _token_refresh_time
    import time
    import requests

    if _cached_token and time.time() < _token_refresh_time:
        return _cached_token

    if is_govcloud_url(instance_url):
        _cached_token = api_key
        _token_refresh_time = float("inf")
        return api_key

    if is_ibm_cloud_url(instance_url):
        headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
        data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}
        response = requests.post(AUTH_ENDPOINT_IBM_CLOUD, headers=headers, data=data, timeout=30)
    else:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
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


# =============================================================================
# LLM FUNCTIONS
# =============================================================================

def call_gateway_llm(prompt: str, model_id: str, system_prompt: str = None) -> str:
    """Call LLM via Orchestrate Gateway (uses cached CLI credentials)."""
    import requests

    # Get credentials from orchestrate CLI cache
    cached_token, cached_url = get_orchestrate_cached_credentials()

    if not cached_token or not cached_url:
        raise ValueError("No credentials found. Run: orchestrate env activate <env> --api-key <key>")

    instance_url = cached_url.rstrip("/")
    access_token = cached_token

    chat_url = f"{instance_url}/v1/orchestrate/gateway/model/chat/completions"

    x_gateway_config = {
        "strategy": {"mode": "single"},
        "targets": [{"provider": "watsonx", "api_key": "gateway", "override_params": {"model": model_id}}]
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
        "x-request-id": str(uuid.uuid4()),
        "x-gateway-config": json.dumps(x_gateway_config, separators=(",", ":")),
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": f"watsonx/{model_id}", "messages": messages, "temperature": 0.0}

    response = requests.post(chat_url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    result = response.json()
    choices = result.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")

    raise RuntimeError(f"Unexpected response: {result}")


def extract_keywords(expected_answer: str, model_id: str) -> List[str]:
    """Use LLM to extract key terms from an expected answer."""
    prompt = f"""Extract 3-6 key terms or short phrases from this answer that are essential for evaluating correctness.
Focus on: specific numbers, dates, names, actions, or unique terms that MUST appear in a correct response.

ANSWER:
{expected_answer}

Return ONLY a JSON array of strings. Example: ["31 days", "enrollment", "portal.com"]

JSON array:"""

    try:
        response = call_gateway_llm(prompt, model_id, "You extract keywords. Respond only with a JSON array.")
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        keywords = json.loads(response)
        if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
            return keywords[:6]
    except Exception as e:
        print(f"  Warning: LLM keyword extraction failed: {e}")

    # Fallback
    words = expected_answer.split()
    return list(set(w.strip('.,!?()[]"\'') for w in words if len(w) > 4))[:4]


def generate_story(question: str, model_id: str) -> str:
    """Use LLM to generate a brief scenario description."""
    prompt = f"""Write a very brief (1 sentence, max 20 words) scenario description for this question.
Describe WHO is asking and WHY.

QUESTION: {question}

Brief scenario:"""

    try:
        response = call_gateway_llm(prompt, model_id, "Write brief scenarios. No quotes.")
        story = response.strip().strip('"').strip("'")
        return story[:150] if len(story) > 150 else story
    except Exception:
        return f"User asking: {question[:50]}..."


# =============================================================================
# TEST CASE GENERATION
# =============================================================================

def create_test_case(question: str, expected_answer: str, agent_name: str,
                     keywords: List[str], story: str, tool_name: str = None,
                     source_doc: str = None, source_page: str = None) -> Dict[str, Any]:
    """Create an ADK-compatible test case.

    If tool_name is provided, includes tool_call expectation (for Tool Precision/Recall).
    If tool_name is None, only evaluates final response (for complex multi-tool agents).
    """
    test_case = {
        "agent": agent_name,
        "story": story,  # Required by ADK
        "starting_sentence": question
    }

    if tool_name:
        # Include tool call expectation
        tool_call_name = f"{tool_name}-1"
        test_case["goals"] = {tool_call_name: ["summarize"]}
        test_case["goal_details"] = [
            {"type": "tool_call", "name": tool_call_name, "tool_name": tool_name, "args": {"query": question}},
            {"name": "summarize", "type": "text", "response": expected_answer, "keywords": keywords}
        ]
    else:
        # Response-only evaluation (no tool call expectation)
        test_case["goals"] = {"summarize": []}
        test_case["goal_details"] = [
            {"name": "summarize", "type": "text", "response": expected_answer, "keywords": keywords}
        ]

    if source_doc:
        test_case["source_document"] = source_doc
    if source_page:
        test_case["source_page"] = str(source_page)
    return test_case


def generate_test_cases(excel_path: Path, agent_name: str, output_dir: Path,
                        model_id: str, tool_name: str = None, limit: int = None,
                        skip_llm: bool = False) -> List[Path]:
    """Generate test case JSON files from Excel."""
    import pandas as pd

    print(f"\n{'='*60}")
    print("GOVCLOUD TEST CASE GENERATOR")
    print(f"{'='*60}")
    print(f"\nExcel: {excel_path}")
    print(f"Agent: {agent_name}")
    print(f"Tool: {tool_name or '(auto - response-only evaluation)'}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_id}")

    df = pd.read_excel(excel_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Check required columns
    if "question" not in df.columns or "expected_answer" not in df.columns:
        alt_map = {"query": "question", "answer": "expected_answer", "expected": "expected_answer", "response": "expected_answer"}
        for old, new in alt_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})

    if "question" not in df.columns or "expected_answer" not in df.columns:
        print(f"\nERROR: Excel must have 'Question' and 'Expected Answer' columns")
        print(f"Found: {df.columns.tolist()}")
        sys.exit(1)

    df = df.dropna(subset=["question", "expected_answer"])
    if limit:
        df = df.head(limit)

    print(f"Found {len(df)} valid rows")
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    print(f"\n{'-'*60}")

    for idx, row in df.iterrows():
        num = len(generated) + 1
        question = str(row["question"]).strip()
        expected = str(row["expected_answer"]).strip()
        print(f"\n[{num}/{len(df)}] {question[:50]}...")

        if skip_llm:
            # Simple keyword extraction, minimal story (faster, no follow-ups)
            keywords = list(set(w.strip('.,!?()[]"\'') for w in expected.split() if len(w) > 4))[:4]
            story = "User asks a single question."  # Minimal story = no follow-ups
            print("  (skipping LLM - using minimal story)")
        else:
            # LLM generates smart keywords AND detailed story
            print("  Generating story + keywords with LLM...")
            story = generate_story(question, model_id)
            keywords = extract_keywords(expected, model_id)

        print(f"  Keywords: {keywords}")

        test_case = create_test_case(
            question=question, expected_answer=expected, agent_name=agent_name,
            keywords=keywords, story=story, tool_name=tool_name,
            source_doc=row.get("source_document") if pd.notna(row.get("source_document")) else None,
            source_page=row.get("source_page") if pd.notna(row.get("source_page")) else None
        )

        filepath = output_dir / f"test_{num:03d}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(test_case, f, indent=2, ensure_ascii=False)
        print(f"  -> {filepath.name}")
        generated.append(filepath)

    print(f"\n{'='*60}")
    print(f"COMPLETE: Generated {len(generated)} test cases in {output_dir}")
    print(f"{'='*60}")
    return generated


def create_sample_excel(output_path: Path):
    """Create a sample Excel template."""
    import pandas as pd

    sample = [
        {"Question": "What is my medical deductible?",
         "Expected Answer": "Your deductible is $500 individual / $1,000 family per year.",
         "Source Document": "Benefits Guide", "Source Page": "12"},
        {"Question": "How do I add a dependent after marriage?",
         "Expected Answer": "You have 31 days from your marriage date to add dependents via the benefits portal.",
         "Source Document": "Life Events FAQ", "Source Page": "5"}
    ]
    df = pd.DataFrame(sample)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"\nCreated sample Excel: {output_path}")
    print("\nColumns: Question (required), Expected Answer (required), Source Document (optional), Source Page (optional)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate test cases from Excel with LLM keywords")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE, help="Config file")
    parser.add_argument("--excel", type=Path, help="Excel file with questions/answers")
    parser.add_argument("--agent", type=str, help="Agent name")
    parser.add_argument("--tool", type=str, help="Tool name")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit number of test cases")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM (simple keywords)")
    parser.add_argument("--sample", action="store_true", help="Create sample Excel template")
    args = parser.parse_args()

    if args.sample:
        create_sample_excel(SCRIPT_DIR / "sample_questions.xlsx")
        return

    # Load config
    config = load_yaml_config(args.config)
    paths = config.get("paths", {})
    agent_config = config.get("agent", {})
    models = config.get("models", {})

    excel_path = args.excel or Path(paths.get("excel_input", "./questions.xlsx"))
    agent_name = args.agent or agent_config.get("name", "alight_supervisor_agent")
    tool_name = args.tool or agent_config.get("tool")  # None if not specified
    output_dir = args.output or Path(paths.get("test_cases", "./test_data"))
    model_id = models.get("llm_judge", "meta-llama/llama-3-2-90b-vision-instruct")

    if not excel_path.exists():
        print(f"ERROR: Excel file not found: {excel_path}")
        print("Create it or run: python govcloud_generate.py --sample")
        sys.exit(1)

    try:
        import pandas
        import requests
        import yaml
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Run: pip install pandas openpyxl requests pyyaml")
        sys.exit(1)

    generate_test_cases(excel_path, agent_name, output_dir, model_id, tool_name, args.limit, args.skip_llm)


if __name__ == "__main__":
    main()
