#!/usr/bin/env python3
"""
GovCloud Agent Evaluation (with Auto-Patching)
===============================================
All-in-one script for running IBM watsonx Orchestrate evaluations on GovCloud/FedRAMP
environments where certain models (like 405b) are not available.

This script automatically:
1. Replaces agentops files with GovCloud-compatible versions
2. Runs the evaluation
3. Provides LLM-as-Judge evaluation

Usage:
    python govcloud_eval.py --setup             # Apply patches (replace files)
    python govcloud_eval.py --evaluate          # Run ADK evaluation
    python govcloud_eval.py --report            # Show results from last run
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

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CONFIG_FILE = SCRIPT_DIR / "govcloud_config.yaml"
PATCHED_FILES_DIR = SCRIPT_DIR / "patched_files"


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
# AGENTOPS FILE REPLACEMENT SYSTEM
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


def get_files_to_replace() -> List[str]:
    """List of files that need to be replaced for GovCloud compatibility."""
    return ["clients.py", "main.py", "runner.py", "evaluation_package.py"]


def check_file_needs_replacement(agentops_file: Path, patched_file: Path) -> bool:
    """Check if a file needs to be replaced by comparing content."""
    if not agentops_file.exists():
        return False
    if not patched_file.exists():
        return False

    try:
        agentops_content = agentops_file.read_text(encoding='utf-8')
        patched_content = patched_file.read_text(encoding='utf-8')

        # If they're the same, no replacement needed
        if agentops_content == patched_content:
            return False

        # Check if already has our patches
        if "GOVCLOUD FIX" in agentops_content:
            # Has some patches but might be incomplete - check key fixes
            if "custom_llmaaj_client=llmaaj_provider" in agentops_content or "llm_client_for_eval" in agentops_content:
                return False  # Already properly patched

        return True
    except Exception:
        return True


def replace_file(agentops_file: Path, patched_file: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Replace an agentops file with the patched version."""
    result = {
        "file": agentops_file.name,
        "status": "unknown",
        "message": "",
    }

    if not agentops_file.exists():
        result["status"] = "error"
        result["message"] = "Original file not found"
        return result

    if not patched_file.exists():
        result["status"] = "error"
        result["message"] = "Patched file not found in patched_files/"
        return result

    # Check if replacement is needed
    if not check_file_needs_replacement(agentops_file, patched_file):
        result["status"] = "already_patched"
        result["message"] = "Already using patched version"
        return result

    if dry_run:
        result["status"] = "would_replace"
        result["message"] = "Would replace with patched version"
        return result

    try:
        # Backup original
        backup_path = agentops_file.with_suffix(agentops_file.suffix + ".original.bak")
        if not backup_path.exists():
            shutil.copy2(agentops_file, backup_path)

        # Copy patched file over
        shutil.copy2(patched_file, agentops_file)

        # Clear pycache
        pycache_dir = agentops_file.parent / "__pycache__"
        if pycache_dir.exists():
            shutil.rmtree(pycache_dir, ignore_errors=True)

        result["status"] = "replaced"
        result["message"] = "Replaced with patched version"
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Error: {str(e)}"

    return result


def apply_all_patches(config: Dict[str, Any], dry_run: bool = False) -> bool:
    """Replace all agentops files with patched versions."""
    print(f"\n{'='*60}")
    print("  GOVCLOUD PATCH SYSTEM")
    print(f"{'='*60}")

    agentops_path = find_agentops_package()
    if not agentops_path:
        print("\nERROR: Could not find agentops package.")
        print("Make sure ibm-watsonx-orchestrate-adk is installed.")
        return False

    print(f"\nFound agentops at: {agentops_path}")

    if not PATCHED_FILES_DIR.exists():
        print(f"\nERROR: Patched files directory not found: {PATCHED_FILES_DIR}")
        print("Make sure the 'patched_files' folder exists with the patched .py files.")
        return False

    print(f"Patched files from: {PATCHED_FILES_DIR}")

    if dry_run:
        print("(DRY RUN - no changes will be made)")

    files_to_replace = get_files_to_replace()
    all_success = True
    results = []

    for filename in files_to_replace:
        agentops_file = agentops_path / filename
        patched_file = PATCHED_FILES_DIR / filename

        result = replace_file(agentops_file, patched_file, dry_run)
        results.append(result)

        status_symbol = {
            "replaced": "\u2705",
            "already_patched": "\u2714\ufe0f",
            "would_replace": "\U0001f504",
            "error": "\u274c",
        }.get(result["status"], "?")

        print(f"\n  {status_symbol} {filename}: {result['message']}")

        if result["status"] == "error":
            all_success = False

    print(f"\n{'─'*60}")

    success_count = sum(1 for r in results if r["status"] in ["replaced", "already_patched"])
    print(f"  Files ready: {success_count}/{len(files_to_replace)}")

    if all_success:
        print("  Status: READY for GovCloud evaluation")
    else:
        print("  Status: Some files could not be replaced")
        print("\n  Make sure 'patched_files/' folder contains:")
        for f in files_to_replace:
            print(f"    - {f}")

    print(f"{'─'*60}\n")

    return all_success


def verify_patches(config: Dict[str, Any]) -> bool:
    """Verify all patches are applied."""
    agentops_path = find_agentops_package()
    if not agentops_path:
        return False

    for filename in get_files_to_replace():
        agentops_file = agentops_path / filename
        patched_file = PATCHED_FILES_DIR / filename

        if check_file_needs_replacement(agentops_file, patched_file):
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
    """Apply patches by replacing files."""
    success = apply_all_patches(config, dry_run=args.dry_run)
    return 0 if success else 1


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

    # Build test paths string
    test_paths_str = ",".join(str(f) for f in test_files)

    cmd = [
        "orchestrate", "evaluations", "evaluate",
        "--test-paths", test_paths_str,
        "--output-dir", str(output_dir),
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
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GovCloud Agent Evaluation (with Auto-Patching)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python govcloud_eval.py --setup              # Apply patches (replace files)
  python govcloud_eval.py --evaluate           # Run evaluation
  python govcloud_eval.py --evaluate --limit 1 # Test with 1 case
  python govcloud_eval.py --report             # Show results
        """
    )

    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_FILE),
                        help="Path to govcloud_config.yaml")
    parser.add_argument("--setup", action="store_true",
                        help="Apply patches (replace agentops files)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run ADK evaluation")
    parser.add_argument("--report", action="store_true",
                        help="Show results report")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of test cases")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be replaced without making changes")

    args = parser.parse_args()

    # Load config
    config = load_yaml_config(Path(args.config))

    if not any([args.setup, args.evaluate, args.report]):
        parser.print_help()
        sys.exit(0)

    exit_code = 0

    if args.setup:
        exit_code = cmd_setup(config, args)
        if exit_code != 0:
            sys.exit(exit_code)

    if args.evaluate:
        exit_code = cmd_evaluate(config, args)
        if exit_code != 0:
            sys.exit(exit_code)

    if args.report:
        exit_code = cmd_report(config, args)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
