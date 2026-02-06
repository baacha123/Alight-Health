import json
import os
import pathlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import rich
import yaml
from jsonargparse import CLI

from agentops.arg_configs import TestConfig
from agentops.clients import bootstrap_clients
from agentops.collection import LangfuseCollection
from agentops.langfuse_evaluation_package import (
    EvaluationRunner,
    default_aggregator,
    load_metrics,
)
from agentops.metrics.metrics import (
    extract_error_cases,
    extract_metrics,
    format_metrics_for_display,
)
from agentops.metrics.orchestrate_agent_routing_accuracy import (
    OrchestrateAgentRoutingAccuracy,
)
from agentops.metrics.text_match import TextMatchMetric
from agentops.metrics.tool_calling import ToolCalling
from agentops.persistence.langfuse_persistence import LangfusePersistence
from agentops.runner import RichLogger, process_test_case
from agentops.scheduler import (
    discover_tests,
    enumerate_jobs,
    filter_tests,
    run_jobs,
)
from agentops.service_provider import get_provider
from agentops.simulation_runner import ExperimentResultsWrapper
from agentops.utils.utils import SummaryPanel, create_table, csv_dump

# ============ ORIGINAL CODE (COMMENTED OUT FOR GOVCLOUD) ============
# os.environ["TELEMETRY_PLATFORM"] = "langfuse"
# ============ END ORIGINAL CODE ============

################## Common functions ##################


def _setup_output_directory(config: TestConfig) -> None:
    """Setup output directory with timestamp if not skipping available results."""
    if not getattr(config, "skip_available_results", False):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config.output_dir = os.path.join(config.output_dir, ts)


def _discover_and_run_tests(config: TestConfig, clients) -> tuple[list, list]:
    """Discover, filter, schedule and execute test cases.

    Returns:
        tuple: (test_cases, results) - discovered test cases and execution results
    """
    test_cases = discover_tests(
        config.test_paths, config.enable_recursive_search
    )
    filtered_test_cases = filter_tests(test_cases, config.tags)

    jobs = enumerate_jobs(
        filtered_test_cases,
        config.n_runs,
        config.skip_available_results,
        config.output_dir,
    )

    results = run_jobs(
        jobs, config, clients, process_test_case, config.num_workers
    )
    return test_cases, results


def _save_config(config: TestConfig, output_dir: str) -> None:
    """Save configuration to YAML file."""
    config_path = pathlib.Path(output_dir) / "config.yml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f)


################## Langfuse functions ##################


def _run_langfuse_evaluation(
    test_cases: list, config: TestConfig, clients
) -> tuple[str, list[str]]:
    """Process results using Langfuse evaluation."""
    collection_name = config.collection_name
    collection = LangfuseCollection(
        name=collection_name,
        description="",
    )

    # Read session IDs (thread_ids) from metadata files created during test execution
    session_ids = []
    for test_case in test_cases:
        name = os.path.basename(test_case).replace(".json", "")
        metadata_path = os.path.join(config.output_dir, f"{name}.metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        session_id = metadata["thread_id"]
        session_ids.append(session_id)

    # Upload to collection and run evaluation
    collection.delete_all_items()
    collection.upload(paths=test_cases)

    evaluators = load_metrics(config, config.metrics)
    experiment_results = ExperimentResultsWrapper(session_ids)
    run = EvaluationRunner(
        evaluation_name=f"{os.path.basename(config.output_dir)}_evaluation",
        run_name=f"{os.path.basename(config.output_dir)}_run",
        experiment_results=experiment_results,
        collection=collection,
        metrics=evaluators,
        aggregator=default_aggregator,
        operator_configs=config.operator_configs,
        resource_map=clients.resource_map,
        # ============ ORIGINAL CODE (COMMENTED OUT FOR GOVCLOUD) ============
        # exporters=[LangfusePersistence()],
        # ============ END ORIGINAL CODE ============
        # ============ GOVCLOUD FIX: Disable Langfuse export ============
        exporters=[],
        # ============ END GOVCLOUD FIX ============
        llm_client=clients.llmaaj_provider,
    )

    evaluation_results = run.evaluate()
    collection.delete_all_items()

    return collection_name, session_ids


def _print_langfuse_results(
    output_dir: str, collection_name: str, session_ids: list[str]
) -> None:
    """Print Langfuse evaluation results."""
    print(f"Config and metadata saved to {output_dir}")
    print(
        f"Langfuse Evaluation run completed for collection {collection_name}:"
    )
    for session_id in session_ids:
        print(
            f" - http://localhost:3010/project/orchestrate-lite/sessions/{session_id}"
        )


################## Legacy functions ##################


def _setup_legacy_paths(config: TestConfig) -> dict[str, Path]:
    """Create and return paths needed for legacy evaluation mode."""
    base_dir = Path(config.output_dir)
    kb_folder = base_dir / "knowledge_base_metrics"
    kb_folder.mkdir(exist_ok=True, parents=True)

    messages_dir = base_dir / "messages"
    messages_dir.mkdir(exist_ok=True, parents=True)

    return {
        "kb_detailed": kb_folder / "knowledge_base_detailed_metrics.json",
        "kb_summary": base_dir / "knowledge_base_summary_metrics.json",
        "debug": base_dir / "debug.json",
        "summary_metrics": base_dir / "summary_metrics.csv",
    }


def _get_legacy_evaluation_results(
    results: list,
    config: TestConfig,
    paths: dict[str, Path],
    logger: RichLogger,
) -> None:
    """Process results using legacy evaluation metrics."""
    filtered_results, error_cases = extract_error_cases(results)
    logger.info(f"{len(filtered_results)} test case(s) completed successfully")

    if error_cases:
        _handle_error_cases(error_cases, paths["debug"], logger)

    tool_metrics, kb_summary, custom_metrics = extract_metrics(filtered_results)
    _save_legacy_metrics(tool_metrics, kb_summary, paths)
    _display_legacy_results(tool_metrics, kb_summary, custom_metrics)


def _handle_error_cases(
    error_cases: list, debug_path: Path, logger: RichLogger
) -> None:
    """Log and save error cases to debug file."""
    logger.warn(
        f"{len(error_cases)} test case(s) failed during execution. "
        f"See debug.json for failure reasons\n"
        + "\n".join(f" - {error.test_case_path}" for error in error_cases)
    )
    with open(debug_path, "w") as f:
        json.dump([error.model_dump() for error in error_cases], f, indent=4)


def _save_legacy_metrics(
    tool_metrics: list, kb_summary, paths: dict[str, Path]
) -> None:
    """Save legacy evaluation metrics to files."""
    csv_dump(
        paths["summary_metrics"],
        rows=[metric.model_dump() for metric in tool_metrics],
    )

    for file_path, key in [
        (paths["kb_detailed"], "detailed"),
        (paths["kb_summary"], "summary"),
    ]:
        with open(file_path, "w+", encoding="utf-8") as f:
            json.dump(kb_summary.model_dump(by_alias=True)[key], f, indent=4)


def _display_legacy_results(
    tool_metrics: list, kb_summary, custom_metrics: list
) -> None:
    """Display legacy evaluation results in console."""
    SummaryPanel(kb_summary).print()

    tool_table = create_table(
        format_metrics_for_display(tool_metrics), title="Agent Metrics"
    )
    if tool_table:
        tool_table.print()

    if any(cm.custom_metrics for cm in custom_metrics):
        rows = []
        for cm in custom_metrics:
            row = {"dataset_name": cm.dataset_name}
            for m in cm.custom_metrics:
                row[m.eval_name] = str(m.value)
            rows.append(row)
        custom_metrics_table = create_table(rows, title="Custom Metrics")
        if custom_metrics_table:
            custom_metrics_table.print()


def main(config: TestConfig):
    """Main entry point for agent evaluation."""
    logger = RichLogger(__name__)

    # Setup
    clients = bootstrap_clients(config)
    _setup_output_directory(config)

    # Discover and execute tests
    test_cases, results = _discover_and_run_tests(config, clients)

    # Process results based on evaluation mode
    if config.skip_legacy_evaluation:
        collection_name, session_ids = _run_langfuse_evaluation(
            test_cases, config, clients
        )
        _print_langfuse_results(config.output_dir, collection_name, session_ids)
    else:
        # Setup paths and run legacy evaluation
        paths = _setup_legacy_paths(config)
        _get_legacy_evaluation_results(results, config, paths, logger)
        print(f"Results saved to {config.output_dir}")

    # Persist configuration
    _save_config(config, config.output_dir)


if __name__ == "__main__":
    main(CLI(TestConfig, as_positional=False))
