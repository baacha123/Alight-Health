#!/usr/bin/env python3
"""
GovCloud Recording & Test Case Creator
========================================
Record chat sessions and create test cases with LLM keyword extraction.

Prerequisites:
  1. Activate environment first:
     - IBM Cloud: orchestrate env activate <env> --api-key <key>
     - GovCloud:  python fedramp_activate.py <env> --api-key <key>

  2. Then run recording:
     python govcloud_record.py --record

Usage:
    python govcloud_record.py --record              # Start live recording
    python govcloud_record.py --manual              # Create from pasted conversation
    python govcloud_record.py --enhance ./recordings  # Enhance with LLM keywords
    python govcloud_record.py --list ./recordings   # List recorded files
"""

import json
import os
import sys
import argparse
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CONFIG_FILE = SCRIPT_DIR / "govcloud_config.yaml"


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
# ORCHESTRATE CLI HELPERS
# =============================================================================

def get_active_environment() -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Get active environment name, URL, and token from orchestrate CLI config."""
    try:
        import yaml
        config_path = Path.home() / ".config" / "orchestrate" / "config.yaml"
        if not config_path.exists():
            return None, None, None

        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

        active_env = config.get("context", {}).get("active_environment")
        if not active_env:
            return None, None, None

        environments = config.get("environments", {})
        env_config = environments.get(active_env, {})
        instance_url = env_config.get("wxo_url")

        # Get token from credentials
        creds_path = Path.home() / ".cache" / "orchestrate" / "credentials.yaml"
        token = None
        if creds_path.exists():
            with open(creds_path, encoding='utf-8') as f:
                creds = yaml.safe_load(f) or {}
            token = creds.get("auth", {}).get(active_env, {}).get("wxo_mcsp_token")

        return active_env, instance_url, token
    except Exception:
        pass
    return None, None, None


def is_govcloud_url(url: str) -> bool:
    """Check if URL is for GovCloud/FedRAMP environment."""
    return "ibmforusgov.com" in url.lower() if url else False


# =============================================================================
# LLM FUNCTIONS
# =============================================================================

def call_gateway_llm(prompt: str, model_id: str, system_prompt: str = None) -> str:
    """Call LLM via Orchestrate Gateway."""
    import requests

    env_name, instance_url, token = get_active_environment()
    if not token or not instance_url:
        raise ValueError("No credentials. Run activation first.")

    instance_url = instance_url.rstrip("/")
    chat_url = f"{instance_url}/v1/orchestrate/gateway/model/chat/completions"

    x_gateway_config = {
        "strategy": {"mode": "single"},
        "targets": [{"provider": "watsonx", "api_key": "gateway", "override_params": {"model": model_id}}]
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
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
        print(f"    Warning: LLM keyword extraction failed: {e}")

    # Fallback: simple word extraction
    words = expected_answer.split()
    return list(set(w.strip('.,!?()[]"\'') for w in words if len(w) > 4))[:4]


# =============================================================================
# RECORDING FUNCTIONS
# =============================================================================

def run_record_command(output_dir: Path):
    """Run ADK recording command."""
    print(f"\n{'='*60}")
    print("RECORDING SESSION")
    print(f"{'='*60}")

    # Check if environment is activated
    env_name, instance_url, token = get_active_environment()

    if not env_name or not instance_url:
        print("\nERROR: No active environment found.")
        print("\nFor IBM Cloud:")
        print("  orchestrate env activate <env> --api-key <key>")
        print("\nFor GovCloud:")
        print("  python fedramp_activate.py <env> --api-key <key>")
        sys.exit(1)

    if not token:
        print(f"\nERROR: No token found for environment '{env_name}'.")
        print("Please re-activate your environment.")
        sys.exit(1)

    print(f"\nEnvironment: {env_name}")
    print(f"Instance: {instance_url}")
    print(f"Token: {'Yes' if token else 'No'}")

    # For GovCloud, set MODEL_OVERRIDE (405b not available)
    if is_govcloud_url(instance_url):
        config = load_yaml_config(DEFAULT_CONFIG_FILE)
        model_id = config.get("models", {}).get("llm_judge", "meta-llama/llama-3-2-90b-vision-instruct")
        os.environ["MODEL_OVERRIDE"] = model_id
        print(f"GovCloud: Using model {model_id}")

    print(f"\nOutput: {output_dir}")
    print(f"\n{'-'*60}")
    print("Instructions:")
    print("  1. Open Orchestrate Chat UI in browser")
    print("  2. Start a NEW chat session")
    print("  3. Chat with your agent")
    print("  4. Press Ctrl+C here when done")
    print(f"{'-'*60}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run ADK recording command
    cmd = ["orchestrate", "evaluations", "record", "--output-dir", str(output_dir)]

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    except FileNotFoundError:
        print("\nERROR: 'orchestrate' command not found.")
        print("Make sure ibm-watsonx-orchestrate-adk is installed.")
        sys.exit(1)

    # Check for recorded files
    files = list(output_dir.glob("**/*_annotated_data.json"))
    if files:
        print(f"\n{'='*60}")
        print(f"SUCCESS: Recorded {len(files)} test case(s)")
        for f in files:
            print(f"  - {f}")
        print(f"\nTo enhance keywords: python govcloud_record.py --enhance {output_dir}")
    else:
        print(f"\nNo recordings found in {output_dir}")


def manual_create_test_case(agent_name: str, question: str, response: str,
                            output_dir: Path, model_id: str, tool_name: str = None,
                            skip_llm: bool = False) -> Path:
    """Manually create a test case from a conversation."""
    print(f"\n{'='*60}")
    print("MANUAL TEST CASE CREATION")
    print(f"{'='*60}")
    print(f"\nAgent: {agent_name}")
    print(f"Tool: {tool_name or '(response-only evaluation)'}")
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

    # Find JSON files
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob("**/*_annotated_data.json"))
        if not files:
            files = list(input_path.glob("**/*.json"))

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

    files = list(recordings_dir.glob("**/*_annotated_data.json"))
    if not files:
        files = list(recordings_dir.glob("**/*.json"))

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
        except Exception:
            print(f"  {f.name} - Error reading")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Record chat sessions and create test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First activate your environment:
  orchestrate env activate alight-mvp --api-key <key>        # IBM Cloud
  python fedramp_activate.py alight-dev --api-key <key>      # GovCloud

  # Then record:
  python govcloud_record.py --record

  # Or create manually:
  python govcloud_record.py --manual
"""
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE, help="Config file")
    parser.add_argument("--record", action="store_true", help="Start live recording")
    parser.add_argument("--manual", action="store_true", help="Create from pasted conversation")
    parser.add_argument("--enhance", type=Path, metavar="PATH", help="Enhance recorded files with LLM keywords")
    parser.add_argument("--list", type=Path, metavar="PATH", help="List recordings in directory")
    parser.add_argument("--output-dir", "-o", type=Path, help="Output directory for recordings")
    parser.add_argument("--copy-to", type=Path, help="Copy enhanced files to this directory")
    parser.add_argument("--agent", type=str, help="Agent name")
    parser.add_argument("--tool", type=str, help="Tool name (for manual creation)")
    parser.add_argument("--question", type=str, help="Question (for --manual)")
    parser.add_argument("--response", type=str, help="Response (for --manual)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM calls for keywords")
    parser.add_argument("--debug", action="store_true", help="Show debug info")
    args = parser.parse_args()

    # Debug mode
    if args.debug:
        print("=== DEBUG INFO ===")
        env_name, url, token = get_active_environment()
        print(f"Active env: {env_name or 'None'}")
        print(f"Instance URL: {url or 'None'}")
        print(f"Token: {'Yes' if token else 'No'}")
        print(f"Is GovCloud: {is_govcloud_url(url) if url else 'N/A'}")
        print("==================")
        if not (args.record or args.manual or args.enhance or args.list):
            return

    # Load config
    config = load_yaml_config(args.config)
    paths = config.get("paths", {})
    agent_config = config.get("agent", {})
    models = config.get("models", {})

    output_dir = args.output_dir or Path(paths.get("recordings", "./recordings"))
    agent_name = args.agent or agent_config.get("name", "alight_supervisor_agent")
    tool_name = args.tool or agent_config.get("tool")
    model_id = models.get("llm_judge", "meta-llama/llama-3-2-90b-vision-instruct")

    # Execute command
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
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
