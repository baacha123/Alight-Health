#!/usr/bin/env python3
"""
Quick test: Chat with an agent via API to verify it responds.
"""
import json
import yaml
import requests
from pathlib import Path

def get_credentials():
    """Get token and URL from orchestrate cache."""
    config_path = Path.home() / ".config" / "orchestrate" / "config.yaml"
    creds_path = Path.home() / ".cache" / "orchestrate" / "credentials.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    active_env = config["context"]["active_environment"]
    instance_url = config["environments"][active_env]["wxo_url"]

    with open(creds_path) as f:
        creds = yaml.safe_load(f)

    token = creds["auth"][active_env]["wxo_mcsp_token"]

    return token, instance_url, active_env

def chat_with_agent(agent_name: str, message: str):
    """Send a message to an agent and get response."""
    token, instance_url, env = get_credentials()

    print(f"Environment: {env}")
    print(f"Instance: {instance_url}")
    print(f"Agent: {agent_name}")
    print(f"Message: {message}")
    print("-" * 50)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Step 1: Create a thread
    print("\n1. Creating thread...")
    threads_url = f"{instance_url}/v1/orchestrate/threads"
    resp = requests.post(threads_url, headers=headers, json={})
    print(f"   Status: {resp.status_code}")

    if resp.status_code != 201 and resp.status_code != 200:
        print(f"   Error: {resp.text}")
        return

    thread = resp.json()
    thread_id = thread.get("id")
    print(f"   Thread ID: {thread_id}")

    # Step 2: Send message to agent
    print(f"\n2. Sending message to {agent_name}...")
    runs_url = f"{instance_url}/v1/orchestrate/threads/{thread_id}/runs"

    payload = {
        "agent_name": agent_name,
        "input": {
            "content": message
        }
    }

    resp = requests.post(runs_url, headers=headers, json=payload)
    print(f"   Status: {resp.status_code}")

    if resp.status_code != 201 and resp.status_code != 200:
        print(f"   Error: {resp.text}")
        return

    run = resp.json()
    run_id = run.get("id")
    run_status = run.get("status")
    print(f"   Run ID: {run_id}")
    print(f"   Status: {run_status}")

    # Step 3: Poll for completion
    print("\n3. Waiting for response...")
    import time
    run_url = f"{instance_url}/v1/orchestrate/threads/{thread_id}/runs/{run_id}"

    for i in range(30):  # Max 30 seconds
        time.sleep(1)
        resp = requests.get(run_url, headers=headers)
        run = resp.json()
        status = run.get("status")
        print(f"   [{i+1}s] Status: {status}")

        if status in ["completed", "failed", "cancelled"]:
            break

    # Step 4: Get messages
    print("\n4. Getting messages...")
    messages_url = f"{instance_url}/v1/orchestrate/threads/{thread_id}/messages"
    resp = requests.get(messages_url, headers=headers)
    print(f"   Status: {resp.status_code}")

    if resp.status_code == 200:
        messages = resp.json()
        print(f"\n   MESSAGES ({len(messages)} total):")
        print("   " + "=" * 50)
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", [])
            print(f"\n   [{role.upper()}]")
            for c in content:
                if isinstance(c, dict):
                    text = c.get("text", {})
                    if isinstance(text, dict):
                        print(f"   {text.get('value', str(c))}")
                    else:
                        print(f"   {text}")
                else:
                    print(f"   {c}")
    else:
        print(f"   Error: {resp.text}")

if __name__ == "__main__":
    import sys

    agent = sys.argv[1] if len(sys.argv) > 1 else "test_faq_agent"
    message = sys.argv[2] if len(sys.argv) > 2 else "How many vacation days do I have?"

    chat_with_agent(agent, message)
