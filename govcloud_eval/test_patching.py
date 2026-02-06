#!/usr/bin/env python3
"""
Test script to verify patching logic produces exact output as known-good patched files.

This script:
1. Reads original (unpatched) ADK files from Archive
2. Applies programmatic patches
3. Compares results to known-good patched files
4. Reports any differences
"""

import re
import difflib
from pathlib import Path

# Paths
ORIGINAL_DIR = Path("/Users/bacha/Downloads/AlightWxO/Archive/_ARCHIVE_OLD_DEMOS/verint-mvp/adkvenv/lib/python3.12/site-packages/agentops")
KNOWN_GOOD_DIR = Path("/Users/bacha/Downloads/AlightWxO/verint-mvp/tests/govcloud_test/govcloud_patches")

FILES_TO_PATCH = ["main.py", "clients.py", "runner.py", "evaluation_package.py"]


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
# TEST LOGIC
# =============================================================================

def test_patching():
    """Test patching logic against known-good files."""
    print("=" * 60)
    print("  TESTING PATCHING LOGIC")
    print("=" * 60)

    all_passed = True

    for filename in FILES_TO_PATCH:
        original_path = ORIGINAL_DIR / filename
        known_good_path = KNOWN_GOOD_DIR / filename

        if not original_path.exists():
            print(f"\n❌ {filename}: Original file not found at {original_path}")
            all_passed = False
            continue

        if not known_good_path.exists():
            print(f"\n❌ {filename}: Known-good file not found at {known_good_path}")
            all_passed = False
            continue

        # Read original
        original_content = original_path.read_text(encoding='utf-8')

        # Apply patch
        if filename == "main.py":
            patched_content = patch_main_py(original_content)
        elif filename == "clients.py":
            patched_content = patch_clients_py(original_content)
        elif filename == "runner.py":
            patched_content = patch_runner_py(original_content)
        elif filename == "evaluation_package.py":
            patched_content = patch_evaluation_package_py(original_content)
        else:
            print(f"\n⚠️  {filename}: No patch function defined")
            continue

        # Read known-good
        known_good_content = known_good_path.read_text(encoding='utf-8')

        # Compare
        if patched_content == known_good_content:
            print(f"\n✅ {filename}: EXACT MATCH")
        else:
            print(f"\n❌ {filename}: DIFFERS from known-good")
            all_passed = False

            # Show diff
            patched_lines = patched_content.splitlines(keepends=True)
            known_good_lines = known_good_content.splitlines(keepends=True)

            diff = list(difflib.unified_diff(
                patched_lines, known_good_lines,
                fromfile=f"patched_{filename}",
                tofile=f"known_good_{filename}",
                lineterm=""
            ))

            if diff:
                print("   First 50 lines of diff:")
                for line in diff[:50]:
                    print(f"   {line.rstrip()}")
                if len(diff) > 50:
                    print(f"   ... ({len(diff) - 50} more lines)")

    print("\n" + "=" * 60)
    if all_passed:
        print("  ✅ ALL PATCHES PRODUCE EXACT MATCHES")
    else:
        print("  ❌ SOME PATCHES DIFFER - NEED ADJUSTMENT")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    test_patching()
