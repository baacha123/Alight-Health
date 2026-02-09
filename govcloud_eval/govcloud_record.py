#!/usr/bin/env python3
"""
GovCloud Recording & Test Case Creator
========================================
Create test cases from chat conversations with LLM keyword extraction.

Works on both commercial and GovCloud environments by setting ADK-native
environment variables (WO_TOKEN, WO_INSTANCE, MODEL_OVERRIDE).

Usage:
    python govcloud_record.py --record              # Start live recording via ADK
    python govcloud_record.py --manual              # Create from pasted conversation
    python govcloud_record.py --enhance ./recordings  # Enhance existing files with LLM keywords
    python govcloud_record.py --list ./recordings   # List recorded files
"""

import json
import os
import sys
import argparse
import subprocess
import shutil
import uuid
import importlib.util
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


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    try:
        import yaml
    except ImportError:
        return {}
    if not config_path.exists():
        return {}
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


# =============================================================================
# URL DETECTION
# =============================================================================

def is_govcloud_url(url: str) -> bool:
    return "ibmforusgov.com" in url.lower() if url else False


def is_ibm_cloud_url(url: str) -> bool:
    return "cloud.ibm.com" in url.lower() if url else False


# =============================================================================
# AUTHENTICATION
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

        token = creds.get("auth", {}).get(active_env, {}).get("wxo_mcsp_token")
        if token and instance_url:
            return token, instance_url
    except Exception:
        pass
    return None, None


# =============================================================================
# ADK PATCHING (OPTIONAL - for debugging)
# =============================================================================

def find_agentops_package() -> Optional[Path]:
    """Find the agentops package in the current environment."""
    try:
        spec = importlib.util.find_spec("agentops")
        if spec and spec.origin:
            return Path(spec.origin).parent
    except (ImportError, AttributeError):
        pass
    return None


def show_adk_info():
    """Show ADK package info for debugging."""
    agentops_path = find_agentops_package()
    if agentops_path:
        print(f"ADK agentops path: {agentops_path}")
    else:
        print("ADK agentops: NOT FOUND")


# =============================================================================
# LLM FUNCTIONS
# =============================================================================

def call_gateway_llm(prompt: str, model_id: str, system_prompt: str = None) -> str:
    """Call LLM via Orchestrate Gateway (uses cached CLI credentials)."""
    import requests

    cached_token, cached_url = get_orchestrate_cached_credentials()
    if not cached_token or not cached_url:
        raise ValueError("No credentials found. Run: orchestrate env activate <env> --api-key <key>")

    instance_url = cached_url.rstrip("/")
    chat_url = f"{instance_url}/v1/orchestrate/gateway/model/chat/completions"

    x_gateway_config = {
        "strategy": {"mode": "single"},
        "targets": [{"provider": "watsonx", "api_key": "gateway", "override_params": {"model": model_id}}]
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cached_token}",
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
    prompt = f"""Extract 3-6 key terms from this answer that are essential for correctness.
Focus on: numbers, dates, names, URLs, or unique terms.

ANSWER:
{expected_answer}

Return ONLY a JSON array. Example: ["31 days", "portal.com"]

JSON array:"""

    try:
        response = call_gateway_llm(prompt, model_id, "Extract keywords. JSON array only.")
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1].lstrip("json")
        keywords = json.loads(response)
        if isinstance(keywords, list):
            return [str(k) for k in keywords[:6]]
    except Exception as e:
        print(f"    Warning: LLM failed: {e}")

    words = expected_answer.split()
    return list(set(w.strip('.,!?()[]"\'') for w in words if len(w) > 4))[:4]


# =============================================================================
# RECORDING FUNCTIONS
# =============================================================================

def run_record_command(output_dir: Path):
    """Run the ADK record command with GovCloud-compatible environment."""
    print(f"\n{'='*60}")
    print("GOVCLOUD RECORDING SESSION")
    print(f"{'='*60}")

    # Get cached credentials from orchestrate CLI
    cached_token, cached_url = get_orchestrate_cached_credentials()

    if not cached_token or not cached_url:
        print("\nERROR: No credentials found.")
        print("Run: orchestrate env activate <env> --api-key <key>")
        sys.exit(1)

    # Set environment variables for static token auth
    # The ADK's GatewayProvider already supports these natively:
    # - WO_TOKEN: Static bearer token (bypasses auth endpoint)
    # - WO_INSTANCE: Instance URL (bypasses tenant_setup)
    # - MODEL_OVERRIDE: Override the model ID
    os.environ["WO_TOKEN"] = cached_token
    os.environ["WO_INSTANCE"] = cached_url
    print(f"\nUsing cached token for authentication")
    print(f"Instance: {cached_url}")

    # For GovCloud, override the model (405b is not available)
    if is_govcloud_url(cached_url):
        config = load_yaml_config(DEFAULT_CONFIG_FILE)
        model_id = config.get("models", {}).get("llm_judge", "meta-llama/llama-3-2-90b-vision-instruct")
        os.environ["MODEL_OVERRIDE"] = model_id
        os.environ["GOVCLOUD_MODEL"] = model_id
        print(f"GovCloud detected, using model: {model_id}")

    print(f"Output: {output_dir}")
    print("\n1. Open Orchestrate Chat UI in browser")
    print("2. Start a NEW chat session")
    print("3. Chat with your agent")
    print("4. Press Ctrl+C here when done")
    print(f"\n{'-'*60}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["orchestrate", "evaluations", "record", "--output-dir", str(output_dir)]

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\nRecording stopped.")

    files = list(output_dir.glob("*_annotated_data.json"))
    if files:
        print(f"\nRecorded {len(files)} file(s). Enhance with:")
        print(f"  python govcloud_record.py --enhance {output_dir}")


def manual_create_test_case(agent_name: str, question: str, response: str,
                            output_dir: Path, model_id: str, tool_name: str = None,
                            skip_llm: bool = False) -> Path:
    """Manually create a test case from a conversation."""
    print(f"\n{'='*60}")
    print("MANUAL TEST CASE CREATION")
    print(f"{'='*60}")
    print(f"\nAgent: {agent_name}")
    print(f"Tool: {tool_name or '(auto - response-only evaluation)'}")
    print(f"Question: {question[:60]}...")

    if skip_llm:
        keywords = list(set(w.strip('.,!?()[]"\'') for w in response.split() if len(w) > 4))[:4]
    else:
        print("Extracting keywords with LLM...")
        keywords = extract_keywords(response, model_id)

    print(f"Keywords: {keywords}")

    test_case = {
        "agent": agent_name,
        "story": f"User asking: {question[:50]}...",
        "starting_sentence": question
    }

    if tool_name:
        tool_call_name = f"{tool_name}-1"
        test_case["goals"] = {tool_call_name: ["summarize"]}
        test_case["goal_details"] = [
            {"type": "tool_call", "name": tool_call_name, "tool_name": tool_name, "args": {"query": question}},
            {"name": "summarize", "type": "text", "response": response, "keywords": keywords}
        ]
    else:
        test_case["goals"] = {"summarize": []}
        test_case["goal_details"] = [
            {"name": "summarize", "type": "text", "response": response, "keywords": keywords}
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{agent_name}_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(test_case, f, indent=2, ensure_ascii=False)

    print(f"\nCreated: {filepath}")
    return filepath


def enhance_recordings(input_path: Path, model_id: str, copy_to: Path = None, skip_llm: bool = False) -> List[Path]:
    """Enhance recorded files with LLM keywords."""
    print(f"\n{'='*60}")
    print("ENHANCE RECORDED TEST CASES")
    print(f"{'='*60}")

    files = [input_path] if input_path.is_file() else list(input_path.glob("*_annotated_data.json"))
    if not files:
        files = list(input_path.glob("*.json"))
    if not files:
        print(f"\nNo JSON files found in: {input_path}")
        return []

    print(f"\nFound {len(files)} file(s)")
    enhanced = []

    for i, fpath in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {fpath.name}")
        try:
            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)

            for detail in data.get("goal_details", []):
                if detail.get("type") == "text" or detail.get("name") == "summarize":
                    response = detail.get("response", "")
                    original = detail.get("keywords", [])

                    if skip_llm:
                        keywords = list(set(w.strip('.,!?()[]"\'') for w in response.split() if len(w) > 4))[:4]
                    else:
                        keywords = extract_keywords(response, model_id)

                    print(f"    Original: {original}")
                    print(f"    Enhanced: {keywords}")
                    detail["keywords"] = keywords
                    detail["original_keywords"] = original
                    break

            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            enhanced.append(fpath)

            if copy_to:
                copy_to.mkdir(parents=True, exist_ok=True)
                dest = copy_to / f"{data.get('agent', 'agent')}_recorded_{i:03d}.json"
                with open(dest, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"    Copied to: {dest.name}")

        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"COMPLETE: Enhanced {len(enhanced)} file(s)")
    return enhanced


def list_recordings(recordings_dir: Path):
    """List recorded files."""
    print(f"\n{'='*60}")
    print("RECORDED TEST CASES")
    print(f"{'='*60}")

    if not recordings_dir.exists():
        print(f"\nDirectory not found: {recordings_dir}")
        return

    files = list(recordings_dir.glob("*_annotated_data.json")) + list(recordings_dir.glob("*.json"))
    files = list(set(files))
    if not files:
        print(f"\nNo recordings found in: {recordings_dir}")
        return

    print(f"\nFound {len(files)} file(s):\n")
    for f in sorted(files):
        try:
            with open(f, encoding='utf-8') as fp:
                data = json.load(fp)
            print(f"  {f.name}")
            print(f"    Agent: {data.get('agent', 'unknown')}")
            print(f"    Question: {data.get('starting_sentence', '')[:50]}...")
        except:
            print(f"  {f.name} - Error reading")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Record and create test cases with LLM keywords")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE, help="Config file")
    parser.add_argument("--record", action="store_true", help="Start live recording (ADK)")
    parser.add_argument("--manual", action="store_true", help="Create from pasted conversation")
    parser.add_argument("--enhance", type=Path, help="Enhance recorded files")
    parser.add_argument("--list", type=Path, help="List recordings in directory")
    parser.add_argument("--output-dir", "-o", type=Path, help="Output directory")
    parser.add_argument("--copy-to", type=Path, help="Copy enhanced files here")
    parser.add_argument("--agent", type=str, help="Agent name")
    parser.add_argument("--tool", type=str, help="Tool name")
    parser.add_argument("--question", type=str, help="Question (for --manual)")
    parser.add_argument("--response", type=str, help="Response (for --manual)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM calls")
    parser.add_argument("--debug", action="store_true", help="Show debug info")
    args = parser.parse_args()

    if args.debug:
        print("=== DEBUG INFO ===")
        show_adk_info()
        token, url = get_orchestrate_cached_credentials()
        print(f"Cached token: {'Yes' if token else 'No'}")
        print(f"Instance URL: {url or 'Not found'}")
        print(f"Is GovCloud: {is_govcloud_url(url) if url else 'N/A'}")
        print("==================")
        if not (args.record or args.manual or args.enhance or args.list):
            return

    config = load_yaml_config(args.config)
    paths = config.get("paths", {})
    agent_config = config.get("agent", {})
    models = config.get("models", {})

    output_dir = args.output_dir or Path(paths.get("recordings", "./recordings"))
    agent_name = args.agent or agent_config.get("name", "alight_supervisor_agent")
    tool_name = args.tool or agent_config.get("tool")  # None if not specified
    model_id = models.get("llm_judge", "meta-llama/llama-3-2-90b-vision-instruct")

    if args.record:
        run_record_command(output_dir)

    elif args.manual:
        question = args.question
        response = args.response

        if not question:
            print("Enter the question:")
            question = input("> ").strip()
        if not response:
            print("\nEnter the agent's response (paste and press Enter twice):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            response = "\n".join(lines)

        test_dir = Path(paths.get("test_cases", "./test_data"))
        manual_create_test_case(agent_name, question, response, test_dir, model_id, tool_name, args.skip_llm)

    elif args.enhance:
        if not args.enhance.exists():
            print(f"ERROR: Path not found: {args.enhance}")
            sys.exit(1)
        enhance_recordings(args.enhance, model_id, args.copy_to, args.skip_llm)

    elif args.list:
        list_recordings(args.list)

    else:
        print("Specify an action: --record, --manual, --enhance, or --list")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
