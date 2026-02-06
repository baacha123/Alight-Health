import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from agentops.arg_configs import TestConfig
from agentops.evaluation_controller.evaluation_controller import (
    EvaluationController,
)
from agentops.evaluation_package import EvaluationPackage
from agentops.llm_user.llm_user_v1 import LLMUser
from agentops.metrics.metrics import (
    CustomEvalMetrics,
    KnowledgeBaseMetricSummary,
    ToolCallAndRoutingMetrics,
)
from agentops.resource_map import ResourceMap
from agentops.runtime_adapter.wxo_runtime_adapter import WXORuntimeAdapter
from agentops.service_provider.provider import Provider
from agentops.type import ErrorLog, OrchestrateDataset
from agentops.utils import json_dump
from agentops.utils.evaluation_discovery import find_evaluation_subclasses
from agentops.utils.rich_utils import RichLogger


def _save_data(
    config: TestConfig,
    test_case_name: str,
    run_tag: str,
    data,
    file_path: str | None = None,
    file_suffix: str | None = None,
) -> None:
    """
    Save data to a JSON file.

    Args:
        config: Test configuration
        test_case_name: Test case name
        run_tag: Run tag
        data: Data to save
        file_path: Complete file path (optional)
        file_suffix: File suffix for messages directory (optional)
    """
    if file_path:
        json_dump(str(file_path), data)
    elif file_suffix:
        json_dump(
            os.path.join(
                config.output_dir,
                "messages",
                f"{test_case_name}{run_tag}{file_suffix}",
            ),
            data,
        )

    # Handle conversational search data
    if (
        isinstance(data, list)
        and data
        and hasattr(data[0], "model_dump")
        and file_suffix == ".retrieval_context.json"
    ):
        out_folder = Path(config.output_dir) / "knowledge_base_metrics"
        out_folder.mkdir(exist_ok=True)
        retrieval_context = [context.model_dump() for context in data]
        json_dump(
            str(out_folder / f"{test_case_name}{run_tag}{file_suffix}"),
            retrieval_context,
        )


def process_test_case(
    task_n: int,
    test_case: str,
    config: TestConfig,
    runtime_adapter: WXORuntimeAdapter,
    resource_map: ResourceMap,
    llm_user: LLMUser,
    llmaaj_provider: Provider,
    run_idx: int = 0,
) -> List[
    Tuple[
        ToolCallAndRoutingMetrics,
        KnowledgeBaseMetricSummary,
        CustomEvalMetrics,
        Optional[ErrorLog],
    ]
]:
    """
    Process a single test case.

    Args:
        task_n: Task number
        test_case: Path to the test case file
        config: Test configuration
        inference_backend: Inference backend
        resource_map: Resource map
        llm_user: LLM user
        llmaaj_provider: Provider for custom metrics
        run_idx: Run index

    Returns:
        List of tuples (metrics, knowledge_base_metrics, custom_metrics)
    """
    logger = RichLogger(__name__)
    summary_results_for_path = []
    test_case_name = os.path.basename(test_case).replace(".json", "")
    run_tag = f".run{run_idx+1}" if config.n_runs > 1 else ""

    try:
        with open(test_case, "r") as f:
            evaluation_data = OrchestrateDataset.model_validate(json.load(f))
    except Exception as e:
        err_msg = f"Error loading the test case: {test_case_name} {e}"
        logger.error(err_msg)
        summary_results_for_path.append(
            (
                None,
                None,
                None,
                ErrorLog(
                    test_case=test_case_name,
                    test_case_path=test_case,
                    reason=err_msg,
                ),
            )
        )
        return summary_results_for_path

    # Set up evaluation controller and run test
    evaluation_controller = EvaluationController(
        runtime=runtime_adapter,
        llm_user=llm_user,
        config=config,
    )

    logger.info(f"Running test case: {test_case_name}")

    # Run the evaluation
    history, call_tracker = [], None
    try:
        history, call_tracker, conversational_search_data, _ = (
            evaluation_controller.run(
                task_n,
                story=evaluation_data.story,
                agent_name=evaluation_data.agent,
                starting_user_input=evaluation_data.starting_sentence,
                max_user_turns=evaluation_data.max_user_turns,
                runtime_context=evaluation_data.runtime_context,
                file_upload=evaluation_data.file_upload,
            )
        )
    except Exception as e:
        err_msg = f"Error running the test case: {test_case_name} {e}"
        logger.error(err_msg)
        summary_results_for_path.append(
            (
                None,
                None,
                None,
                ErrorLog(
                    test_case=test_case_name,
                    test_case_path=test_case,
                    reason=err_msg,
                ),
            )
        )
        return summary_results_for_path

    # Save metadata (that contains thread_id)
    json_dump(
        os.path.join(
            config.output_dir,
            f"{test_case_name}{run_tag}.metadata.json",
        ),
        call_tracker.metadata if call_tracker else {},
    )

    if config.skip_legacy_evaluation:
        return summary_results_for_path  # empty result set, skip evaluation

    # Save message history
    result = [message.model_dump() for message in history]
    _save_data(
        config, test_case_name, run_tag, result, file_suffix=".messages.json"
    )

    # Save conversational search data if available
    if conversational_search_data:
        retrieval_context = [
            context.model_dump() for context in conversational_search_data
        ]
        out_folder = Path(config.output_dir) / "knowledge_base_metrics"
        out_folder.mkdir(exist_ok=True)
        file_path = str(
            out_folder / f"{test_case_name}{run_tag}.retrieval_context.json"
        )
        _save_data(
            config,
            test_case_name,
            run_tag,
            retrieval_context,
            file_path=file_path,
        )

    # If data annotation run, skip summary generation
    if config.data_annotation_run:
        return summary_results_for_path  # empty result set, skip summary

    # Load custom extractors and evaluations
    all_extractors = []
    all_custom_evals = []

    # ============ ORIGINAL CODE (COMMENTED OUT FOR GOVCLOUD) ============
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
    evaluation_package = EvaluationPackage(
        test_case_name=test_case_name,
        messages=history,
        ground_truth=evaluation_data,
        conversational_search_data=conversational_search_data,
        resource_map=resource_map,
        config=config,
        custom_evals=all_custom_evals,
        custom_llmaaj_client=llmaaj_provider,
        extractors=all_extractors,
        similarity_threshold=config.similarity_threshold,
        enable_fuzzy_matching=config.enable_fuzzy_matching,
        strict_topological_matching=config.is_strict,
        error_keywords=config.error_keywords,
    )

    # Generate summary

    (eval_res, knowledge_base_metrics, custom_metrics) = (
        evaluation_package.generate_summary()
    )

    eval_res.analysis_result.dump(
        os.path.join(
            config.output_dir,
            "messages",
            f"{test_case_name}{run_tag}.messages.analyze.json",
        )
    )

    metrics = eval_res.metrics
    _save_data(
        config,
        test_case_name,
        run_tag,
        metrics.model_dump(),
        file_suffix=".metrics.json",
    )

    # Update metrics
    metrics.dataset_name = test_case_name

    # Calculate average response time
    metrics.avg_resp_time = 0.0
    if hasattr(call_tracker, "generic") and hasattr(call_tracker, "tool_call"):
        generic_calls = getattr(call_tracker, "generic", [])
        tool_calls = getattr(call_tracker, "tool_call", [])

        if generic_calls or tool_calls:
            total_time = sum(generic_calls) + sum(tool_calls)
            total_calls = len(generic_calls) + len(tool_calls)
            if total_calls > 0:
                metrics.avg_resp_time = round(total_time / total_calls, 2)
    metrics.avg_resp_time = round(metrics.avg_resp_time, 2)

    # Add results to summary
    summary_results_for_path.append(
        (metrics, knowledge_base_metrics, custom_metrics, None)
    )

    return summary_results_for_path
