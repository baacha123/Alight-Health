import copy
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import rich
from dateutil import parser

from agentops import __file__
from agentops.data_annotator import ERROR_KEYWORDS
from agentops.evaluation_result import (
    AnalysisMetadata,
    AnalysisResult,
    EvaluationResult,
)
from agentops.extractors.extractor_base import Extractor
from agentops.llm_matching import LLMMatcher
from agentops.llm_rag_eval import LLMJudge
from agentops.llm_safety_eval import (
    LLMSafetyJudge,
    LLMSafetyJudgeGraniteGuardian,
)
from agentops.metrics.evaluations import Evaluation, Metric
from agentops.metrics.llm_as_judge import AnswerDerailment, AnswerUnsafeTopic
from agentops.metrics.metrics import (
    CustomEvalMetrics,
    KeywordSemanticSearchMetric,
    KnowledgeBaseMetrics,
    TextMatchType,
    ToolCallAndRoutingMetrics,
)
from agentops.prompt.template_render import (
    AnswerRelevancyTemplateRenderer,
    DerailmentTemplateRenderer,
    FaithfulnessTemplateRenderer,
    KeywordMatchingTemplateRenderer,
    SemanticMatchingTemplateRenderer,
    UnsafeTopicTemplateRenderer,
)
from agentops.resource_map import ResourceMap
from agentops.service_provider import USE_GATEWAY_MODEL_PROVIDER, get_provider
from agentops.service_provider.provider import Provider
from agentops.type import (
    ContentType,
    ConversationalSearch,
    DatasetModel,
    EventTypes,
    ExtendedMessage,
    GoalDetail,
    MatchingStrategy,
    Message,
    OrchestrateDataset,
)

root_dir = os.path.dirname(__file__)
KEYWORD_MATCHING_PROMPT_PATH = os.path.join(
    root_dir, "prompt", "keyword_matching_prompt.jinja2"
)
SEMANTIC_MATCHING_PROMPT_PATH = os.path.join(
    root_dir, "prompt", "semantic_matching_prompt.jinja2"
)
FAITHFULNESS_PROMPT_PATH = os.path.join(
    root_dir, "prompt", "faithfulness_prompt.jinja2"
)
ANSWER_RELEVANCY_PROMPT_PATH = os.path.join(
    root_dir, "prompt", "answer_relevancy_prompt.jinja2"
)

RESERVED_KEYWORD_FOR_GROUND_TRUTH_ARGS = os.getenv(
    "RESERVED_KEYWORD_FOR_GROUND_TRUTH_ARGS", "<IGNORE>"
)

DERAILMENT_PROMPT_PATH = os.path.join(
    root_dir, "prompt", "derailment_prompt.jinja2"
)

UNSAFE_TOPIC_PROMPT_PATH = os.path.join(
    root_dir, "prompt", "unsafe_topic_prompt.jinja2"
)

UNSAFE_TOPIC_GRANITE_GUARDIAN_PATH = os.path.join(
    root_dir, "prompt", "unsafe_topic_granite_guardian_prompt.jinja2"
)


"""
- hyphens are not allowed in python function names, so it is safe to use as a dummy function name
- purpose behind `DUMMY_GRAPH_NODE_NAME` is to append
a dummy node to the ground truth and the labelled messages to take into account 
single, summary step goals.
"""
DUMMY_GRAPH_NODE_NAME = "dummy-goal"


class EvaluationPackage:
    def __init__(
        self,
        test_case_name: str,
        ground_truth: OrchestrateDataset,
        messages: list[Message],
        conversational_search_data: List[ConversationalSearch] = None,
        resource_map: ResourceMap = None,
        is_attack_evaluation: bool = False,
        config=None,
        custom_evals: Optional[list[Evaluation]] = None,
        custom_llmaaj_client: Optional[Provider] = None,
        extractors: Optional[list[Extractor]] = None,
        similarity_threshold=0.8,
        enable_fuzzy_matching=False,
        strict_topological_matching=True,
        error_keywords: Optional[list] = None,
    ):
        self.tool_dictionary = (
            {
                goal_detail.name: goal_detail
                for goal_detail in ground_truth.goal_details
                if goal_detail.type == ContentType.tool_call
            }
            if ground_truth.goal_details
            else {}
        )

        self.text_goals = (
            [
                goal_detail
                for goal_detail in ground_truth.goal_details
                if goal_detail.type == ContentType.text
            ]
            if ground_truth.goal_details
            else []
        )

        self.messages: List[Message] = messages
        self.conversational_search_data = conversational_search_data
        self.is_attack_evaluation = is_attack_evaluation
        self.ground_truth = ground_truth
        self.test_case_name = test_case_name
        self.resource_map = resource_map
        self.custom_evals = custom_evals
        self.custom_llmaaj_client = custom_llmaaj_client
        self.extractors = extractors
        self.enable_fuzzy_matching = enable_fuzzy_matching
        self.strict_topological_matching = strict_topological_matching
        self.error_keywords = error_keywords

        if not self.is_attack_evaluation:
            self.validate_ground_truth(self.ground_truth, self.test_case_name)

        extra_kwargs = {}

        if USE_GATEWAY_MODEL_PROVIDER:

            if resource_map and hasattr(resource_map, "wxo_client"):
                wxo_client = resource_map.wxo_client

                if hasattr(wxo_client, "service_url"):
                    extra_kwargs["instance_url"] = wxo_client.service_url

                if hasattr(wxo_client, "api_key"):
                    extra_kwargs["token"] = wxo_client.api_key

            elif config:
                auth = getattr(config, "auth_config", None)

                if auth:
                    instance_url = getattr(auth, "url", None)
                    token = getattr(auth, "token", None)

                if instance_url:
                    extra_kwargs["instance_url"] = instance_url

                if token:
                    extra_kwargs["token"] = token
            else:
                # The `tenant_setup` import is deferred and moved from the top-level imports to this method
                # to avoid circular import dependencies and to only load it when using gateway model providers.
                from agentops.service_instance import tenant_setup

                token, instance_url, env = tenant_setup(
                    service_url=None, tenant_name="local"
                )
                if instance_url:
                    extra_kwargs["instance_url"] = instance_url

                if token:
                    extra_kwargs["token"] = token

        # ============ GOVCLOUD FIX: Use custom_llmaaj_client if provided ============
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
            llm_client=llm_client_for_eval,
            keyword_template=KeywordMatchingTemplateRenderer(
                KEYWORD_MATCHING_PROMPT_PATH
            ),
            semantic_template=SemanticMatchingTemplateRenderer(
                SEMANTIC_MATCHING_PROMPT_PATH
            ),
            similarity_threshold=similarity_threshold,
            enable_fuzzy_matching=enable_fuzzy_matching,
        )
        # only used for RAG evaluation
        self.rag_llm_as_a_judge = LLMJudge(
            llm_client=llm_client_for_eval,
            faithfulness=FaithfulnessTemplateRenderer(FAITHFULNESS_PROMPT_PATH),
            answer_relevancy=AnswerRelevancyTemplateRenderer(
                ANSWER_RELEVANCY_PROMPT_PATH
            ),
        )
        self.safety_llm_as_a_judge = LLMSafetyJudge(
            llm_client=llm_client_for_eval,
            answer_derailment=DerailmentTemplateRenderer(
                DERAILMENT_PROMPT_PATH
            ),
            answer_unsafe_topic=UnsafeTopicTemplateRenderer(
                UNSAFE_TOPIC_PROMPT_PATH
            ),
        )
        self.safety_gg_llm_as_a_judge = LLMSafetyJudgeGraniteGuardian(
            llm_client=get_provider(
                model_id="ibm/granite-guardian-3-8b",
                params={
                    "decoding_method": "greedy",
                    "max_new_tokens": 20,
                    "temperature": 0,
                },
            ),
            answer_unsafe_topic=UnsafeTopicTemplateRenderer(
                UNSAFE_TOPIC_GRANITE_GUARDIAN_PATH
            ),
        )

    @staticmethod
    def find_terminal_nodes(graph: dict[str, list[str]]) -> set[str]:
        """Finds terminal nodes (nodes with no outgoing edges).

        Args:
            graph: the input graph

        Returns:
            a set of the terminal nodes
        """

        seen_nodes = set()  # track seen nodes
        non_terminal_nodes = set()  # track nodes with children

        for node in graph:
            seen_nodes.add(node)
            if graph[node]:
                non_terminal_nodes.add(node)
                for n in graph[node]:
                    seen_nodes.add(n)
        return seen_nodes - non_terminal_nodes

    @staticmethod
    def is_topological_sort(
        graph: dict[str, list[str]], ordering: list[str], is_strict: bool = True
    ) -> bool:
        """Graph traversal to check if every node in `graph` is visited in `ordering` only after all its dependencies are visited.

        Args:
            graph: the graph representing the ground truth, where keys represent nodes and values represent its dependent nodes
            ordering: the nodes visited, in order

        Returns:
            Boolean representing if `ordering` visits all nodes in a valid order based on the dependencies in graph.
        """
        # No keyword match or goal details were achieved
        if not ordering:
            return False

        if is_strict:
            # strict matching: only consider most recent tool call
            position = {node: [i] for i, node in enumerate(ordering)}
        else:
            # lenient matching: consider all tool calls (account for all indexes of the node)
            position = defaultdict(list)
            for i, node in enumerate(ordering):
                position[node].append(i)

        terminal_nodes = EvaluationPackage.find_terminal_nodes(graph)
        # adds a dummy node for each terminal node
        next_idx = (
            max(val for values in position.values() for val in values) + 1
        )

        for n in terminal_nodes:
            graph[n] = [DUMMY_GRAPH_NODE_NAME]
            graph[DUMMY_GRAPH_NODE_NAME] = []
            position[DUMMY_GRAPH_NODE_NAME] = [next_idx]
            next_idx += 1

        for node in graph:
            for child_nodes in graph[node]:
                # Current node/children doesn't show up in made calls
                if node not in position or child_nodes not in position:
                    return False
                # Current node doesn't show up before any of its child
                # all index in current nodes are larger than every child nodes' index
                if all(
                    curr >= max(position[child_nodes])
                    for curr in position[node]
                ):
                    return False
        return True

    @staticmethod
    def validate_ground_truth(ground_truth, test_case_name):
        if len(ground_truth.agent) == 0:
            raise ValueError(
                f"No agent provided in the ground truth. test_case_name: {test_case_name}"
            )

        if len(ground_truth.goals) == 0:
            raise ValueError(
                f"No goals provided in the ground truth. test_case_name: {test_case_name}"
            )

        if len(ground_truth.goal_details) == 0:
            raise ValueError(
                f"No goal details provided in the ground truth. test_case_name: {test_case_name}"
            )

        if len(ground_truth.story) == 0:
            raise ValueError(
                f"No story provided in the ground truth. test_case_name: {test_case_name}"
            )

        goals = set()

        for key, value in ground_truth.goals.items():
            goals.add(key)
            if isinstance(value, list):
                goals.update(value)
            else:
                raise ValueError(
                    f"The goal '{key}' is not mapping to a list: {value}. test_case_name: {test_case_name}"
                )

        for goal_detail in ground_truth.goal_details:
            if goal_detail.name not in goals:
                raise ValueError(
                    f"Goal detail '{goal_detail.name}' does not match any goals: {goals}. test_case_name: {test_case_name}"
                )
            if goal_detail.name == "summarize":
                if (
                    not goal_detail.keywords or len(goal_detail.keywords) == 0
                ) and (
                    not goal_detail.response or len(goal_detail.response) == 0
                ):
                    rich.print(
                        f"Summarize goal should have keywords or final response. test_case_name: {test_case_name}"
                    )
                elif len(goal_detail.response) == 0:
                    rich.print(
                        f"⚠️‼️ [bold][yellow] WARNING:[/yellow][/bold] Summarize goal has no final response. test_case_name: {test_case_name}"
                    )
        if len(ground_truth.goal_details) != len(goals):
            raise ValueError(
                f"Goal details count does not match the goals count: {len(ground_truth.goal_details)} != {len(goals)}. test_case_name: {test_case_name}"
            )

    def _print_kw_sm(
        self, keyword_semantic_match_list: List[KeywordSemanticSearchMetric]
    ):
        """Prints the keyword match/mismatch, and semantic match/mismatch results
        Right now only successful matches are printed
        """

        for keyword_semantic_match in keyword_semantic_match_list:
            if (
                keyword_semantic_match.semantic_match
                and keyword_semantic_match.keyword_match
            ):
                rich.print(
                    f"[green][SUCCESS] Text message matched: Summary - {keyword_semantic_match.message}[/green]"
                )

    def argument_matching(
        self,
        expected: dict[str, str],
        actual: dict[str, str],
        matching_strategy: dict[str, MatchingStrategy],
    ) -> bool:
        """Handles argument matching for expected and actual arguments and values.

        Args:
            expected: Expected ground truth arguments.
            actual: Actual arguments in tool call
            matching_strategy: Matching mode for each argument. Defaults to strict if not specified.

        Returns:
            True if all arguments match according to their matching strategy.
        """
        # ignore arg matching
        if expected == {"IGNORE": None}:
            return True

        for field in actual:
            if field not in expected:
                return False

        for field in expected:
            strategy = matching_strategy.get(
                field, MatchingStrategy.strict.value
            )

            norm_actual_val = EvaluationPackage.normalize_args(
                actual.get(field)
            )
            norm_expected_val = EvaluationPackage.normalize_args(
                expected.get(field)
            )

            # field must exist if not using optional matching
            if (
                field not in actual
                and strategy != MatchingStrategy.optional.value
            ):
                return False
            # continue to next if it's an ignored keyword
            if norm_expected_val == RESERVED_KEYWORD_FOR_GROUND_TRUTH_ARGS:
                continue
            # optional matching
            if strategy == MatchingStrategy.optional.value:
                # continue to next it's not called
                if field not in actual:
                    continue
                # must match if called
                if actual[field] != expected[field]:
                    return False
            elif strategy == MatchingStrategy.fuzzy.value:
                # check date/number conversion
                conversion_succeeded, values_match = (
                    EvaluationPackage._compare_as_date_or_number(
                        norm_actual_val, norm_expected_val
                    )
                )
                # If conversion succeeded and values match, continue to next parameter
                if conversion_succeeded and values_match:
                    continue
                # If conversion succeeded but values don't match, return False
                if conversion_succeeded and not values_match:
                    return False

                # try cosine matching
                return self.matcher.cosine_similarity_semantic_match(
                    norm_actual_val, norm_expected_val
                )
            # TODO szhang 10/24/25: Decide if strict comparison must be exact or may allow normalized values.
            elif strategy == MatchingStrategy.strict.value:
                # must match
                if norm_actual_val != norm_expected_val:
                    return False
            else:
                print(f"Warning: undefined matching strategy found: {strategy}")

        return True

    @staticmethod
    def normalize_args(data):
        if isinstance(data, dict):
            # normalize keys (case-sensitive) and values
            return {
                str(k): EvaluationPackage.normalize_args(v)
                for k, v in data.items()
            }

        elif isinstance(data, list):
            normalized_list = [
                EvaluationPackage.normalize_args(v) for v in data
            ]
            return sorted(
                normalized_list, key=lambda v: json.dumps(v, sort_keys=True)
            )

        else:
            # don’t lowercase reserved keyword
            if str(data) == RESERVED_KEYWORD_FOR_GROUND_TRUTH_ARGS:
                return str(data)
            return str(data).lower()

    @staticmethod
    def _compare_as_date_or_number(normalized_actual, normalized_expected):
        """
        Attempts to compare two normalized values as dates or numbers.

        Args:
            normalized_actual: The actual value from tool call
            normalized_expected: The expected value from ground truth

        Returns:
            tuple: (conversion_succeeded, values_match)
                - conversion_succeeded: True if values could be converted to numbers or dates
                - values_match: True if converted values match
        """
        # Try to convert to numbers
        try:
            num_actual = float(normalized_actual)
            num_expected = float(normalized_expected)
            # Conversion succeeded, check if values match
            return (
                True,
                abs(num_actual - num_expected) <= 0.001,
            )  # Small epsilon for float comparison
        except (ValueError, TypeError):
            pass

        # Try to convert to dates
        try:
            date_actual = parser.parse(normalized_actual)
            date_expected = parser.parse(normalized_expected)
            # Conversion succeeded, check if values match
            return True, date_actual == date_expected
        except (ValueError, TypeError):
            pass

        # If we get here, neither number nor date conversion worked
        return False, False

    def _keyword_semantic_match(
        self, goal_detail: GoalDetail, message: Message
    ) -> KeywordSemanticSearchMetric:
        """Performs keyword and semantic matching given the text goal detail and a message

        Args:
            goal_detail: The ground truth text
            message: The actual message

        Returns:
            An object that contains the match result
        """

        keyword_match: bool = self.matcher.keywords_match(
            message.content, goal_detail.keywords
        )
        semantic_match: bool = self.matcher.semantic_match(
            context=self.messages[0].content,
            prediction=message.content,
            ground_truth=goal_detail.response,
            enable_fuzzy_matching=self.enable_fuzzy_matching,
        )
        keyword_semantic_match = KeywordSemanticSearchMetric(
            keyword_match=keyword_match,
            semantic_match=semantic_match,
            message=message.content,
            goal_detail=goal_detail.name,
            goal_content=goal_detail.response,
            keywords=goal_detail.keywords,
        )
        return keyword_semantic_match

    def _get_analysis_metadata(
        self,
        matched_tools: list[str],
        text_match_candidates: dict[str, list[KeywordSemanticSearchMetric]],
    ) -> AnalysisMetadata:
        """Compile metadata for analyze result. Including info on tool calls and expected trajectory

        Args:
            matched_tools: a list of matched tools extracted from traversing through the messages

        Returns:
            An object containing the analysis result metadata.
        """
        # set up analysis result
        expected_tools = [
            gd.tool_name
            for gd in self.ground_truth.goal_details
            if gd.type == ContentType.tool_call
        ]
        expected_set = set(expected_tools)

        # include only arg matched tool calls
        actual_tool_calls = list(set(matched_tools))
        missed_tool_calls = sorted(expected_set - set(actual_tool_calls))

        expected_trajectory = copy.deepcopy(self.ground_truth.goals)

        return AnalysisMetadata(
            expected_trajectory=expected_trajectory,
            expected_tool_calls=expected_tools,
            actual_tool_calls=actual_tool_calls,
            missed_tool_calls=missed_tool_calls,
            text_match_candidates=text_match_candidates,
        )

    def traverse(self) -> EvaluationResult:
        matched_goals: list[str] = []
        message_outcomes = []
        # Counters for tool-calling related metrics
        metrics = ToolCallAndRoutingMetrics()
        metrics.expected_tool_calls = len(self.tool_dictionary)
        # sometimes, tool with the same signature can be called more than once
        correct_tool_calls = set()

        matched_tools: list[str] = []
        # keeps track of responses with both semantic and keyword matches
        text_matches: list[KeywordSemanticSearchMetric] = []
        text_match_candidates: dict[str, list[KeywordSemanticSearchMetric]] = {}

        for message in self.messages:
            if message.type == ContentType.tool_call:

                msg_tool_call = json.loads(message.content)
                # check if it's an agent call
                if (
                    self.resource_map
                    and msg_tool_call["name"] in self.resource_map.agent2tools
                ):
                    metrics.total_routing_calls += 1
                    relevant = False
                    for tool in self.resource_map.agent2tools[
                        msg_tool_call["name"]
                    ]:
                        for goal_detail in self.tool_dictionary.values():
                            if goal_detail.tool_name == tool:
                                relevant = True
                                break
                        if relevant:
                            break

                    if relevant:
                        metrics.relevant_routing_calls += 1
                    else:
                        message_outcome = ExtendedMessage(message=message)
                        message_outcome.reason = {
                            "reason": "irrelevant routing call"
                        }

                    continue

                # TO-DO: re-think how deduplication works in the context of precision & recall
                metrics.total_tool_calls += 1

                # evaluating more than once is fine
                # agent could make repeated calls with the same function signature
                # in our is_topological_sort algorithm, the most recent occurrence is evaluated

                # check if it's a matching tool call
                matching_goal_details = [
                    goal_detail
                    for goal_detail in self.tool_dictionary.values()
                    if goal_detail.tool_name == msg_tool_call["name"]
                ]

                if len(matching_goal_details) > 0:
                    metrics.relevant_tool_calls += 1  # tool name matches one of the expected tool names, as defined in the ground truth
                    found = False
                    possible_ground_truth_for_analysis = []
                    # performs arg matching
                    for goal_detail in matching_goal_details:
                        # {"IGNORE": None} is set in red teaming attack ground truth to ignore parameter matching
                        if self.argument_matching(
                            expected=goal_detail.args,
                            actual=msg_tool_call["args"],
                            matching_strategy=goal_detail.arg_matching,
                        ):
                            matched_tools.append(goal_detail.tool_name)
                            matched_goals.append(goal_detail.name)
                            correct_tool_calls.add(goal_detail.name)
                            # tool_call_and_routing_metrics.correct_tool_calls += 1  # correct tool call (no erroneous response) + expected arguments, as defined in the ground truth
                            found = True
                            message_outcome = ExtendedMessage(message=message)
                            message_outcomes.append(message_outcome)
                            break
                        else:
                            possible_ground_truth_for_analysis.append(
                                goal_detail.args
                            )

                    # case where args didn't match
                    if not found:
                        metrics.tool_calls_with_incorrect_parameter += 1
                        message_outcome = ExtendedMessage(message=message)
                        message_outcome.reason = {
                            "reason": "incorrect parameter",
                            "actual": msg_tool_call["args"],
                            "expected": possible_ground_truth_for_analysis,
                        }
                        message_outcomes.append(message_outcome)
                        if not self.is_attack_evaluation:
                            rich.print(
                                f"[red][ERROR] Wrong parameters for function: {msg_tool_call['name']}. "
                                f"Expected one of {[g.args for g in matching_goal_details]}, Received={msg_tool_call['args']}[/red]"
                            )

                else:  # no matching tool calls found

                    if not self.is_attack_evaluation:
                        rich.print(
                            f"[yellow][WARNING] Unexpected function call: {msg_tool_call['name']}[/yellow]"
                        )
                    # note: this is incorrect after the 1.6 change
                    message_outcome = ExtendedMessage(message=message)
                    message_outcome.reason = {"reason": "irrelevant tool call"}
                    message_outcomes.append(message_outcome)
            # tool response
            elif message.type == ContentType.tool_response:
                found = False
                if self.error_keywords:
                    error_keywords_to_use = self.error_keywords
                else:
                    error_keywords_to_use = ERROR_KEYWORDS
                for keyword in error_keywords_to_use:
                    if keyword in message.content.lower():
                        message_outcome = ExtendedMessage(message=message)
                        message_outcome.reason = {"reason": "runtime error"}
                        message_outcomes.append(message_outcome)
                        found = True
                        break
                if not found:
                    message_outcome = ExtendedMessage(message=message)
                    message_outcomes.append(message_outcome)
            # text response
            elif (
                message.type == ContentType.text and message.role == "assistant"
            ):

                for goal_detail in self.text_goals:
                    if goal_detail.name not in text_match_candidates:
                        text_match_candidates[goal_detail.name] = []

                    keyword_semantic_match = self._keyword_semantic_match(
                        goal_detail=goal_detail, message=message
                    )
                    keyword_match = keyword_semantic_match.keyword_match
                    semantic_match = keyword_semantic_match.semantic_match

                    if keyword_match or semantic_match:
                        text_match_candidates[goal_detail.name].append(
                            keyword_semantic_match
                        )

                    if keyword_match and semantic_match:
                        matched_goals.append(goal_detail.name)
                        text_matches.append(keyword_semantic_match)
                        break
                message_outcome = ExtendedMessage(message=message)
                message_outcomes.append(message_outcome)

            else:
                message_outcome = ExtendedMessage(message=message)
                message_outcomes.append(message_outcome)

        metrics.correct_tool_calls = len(correct_tool_calls)

        # only prints when the semantic and keyword matched
        self._print_kw_sm(text_matches)

        # check tool goals only
        metrics.tool_match_success = (
            metrics.expected_tool_calls - len(correct_tool_calls)
        ) == 0

        metadata = self._get_analysis_metadata(
            matched_tools=matched_tools,
            text_match_candidates=text_match_candidates,
        )

        analysis_result = AnalysisResult(
            message_outcomes=message_outcomes, metadata=metadata
        )
        return EvaluationResult(
            matched_goals=matched_goals,
            text_matches=text_matches,
            metrics=metrics,
            analysis_result=analysis_result,
        )

    def _is_text_match(self, text_matches: List[KeywordSemanticSearchMetric]):

        if len(self.text_goals) == 0:
            return TextMatchType.na.value
        # TODO explicitly check that each text goal has a match instead of only checking match count
        elif len(self.text_goals) == len(text_matches):
            return TextMatchType.text_match.value
        else:
            return TextMatchType.text_mismatch.value

    def generate_custom_metrics(
        self, extracted_context: Dict[str, Any]
    ) -> Optional[CustomEvalMetrics]:

        if self.custom_evals is None:
            return None
        else:
            rich.print(
                "[r]`generate_custom_metrics` will be deprecated. Please see the 'How to add a new metric?' section in `README.md` for details on adding custom metrics."
            )

        results: list[Metric] = []
        for evaluation in self.custom_evals:
            # TODO: cleanup. The compute method returns a Metric but pydantic thinks it is different.
            # Probably because of some path issue when we auto-discover metrics
            evaluate_result = evaluation.evaluate(
                messages=self.messages,
                ground_truth=self.ground_truth,
                extracted_context=extracted_context,
            )
            if metric_result := evaluate_result.get("metric"):
                for metric in metric_result.values():
                    for item in metric:
                        results.append(Metric(**item.model_dump()))

        custom_eval_results = CustomEvalMetrics(
            dataset_name=self.test_case_name, custom_metrics=results
        )
        return custom_eval_results

    def generate_summary(
        self,
    ) -> Tuple[
        EvaluationResult, KnowledgeBaseMetrics, CustomEvalMetrics | None
    ]:
        llm_steps = 0
        total_step = 0
        metrics: ToolCallAndRoutingMetrics
        traverse_res = self.traverse()

        # journey success with text goal
        extracted_context = {}
        if self.extractors is not None and self.custom_evals is not None:
            for extractor in self.extractors:
                context = extractor.extract(
                    messages=self.messages,
                    ground_truth=self.ground_truth,
                    matcher=self.matcher,
                    resource_map=self.resource_map,
                )
                extracted_context[extractor.name] = context

        # journey success with text goal
        is_success = self.is_topological_sort(
            self.ground_truth.goals,
            traverse_res.matched_goals,
            is_strict=self.strict_topological_matching,
        )

        # summary match
        text_match = self._is_text_match(traverse_res.text_matches)

        for message in self.messages:
            if message.role == "assistant" and (
                message.type
                in (
                    ContentType.text,
                    ContentType.conversational_search,
                    ContentType.tool_call,
                )
            ):
                llm_steps += 1
            total_step += 1

        knowledge_base_metric_summary = (
            self.generate_knowledge_base_metric_summary()
        )

        custom_metric_summary = self.generate_custom_metrics(
            extracted_context=extracted_context
        )
        # TO-DO: the table is not printing properly anymore with the new columns introduced
        # we need to introduce a separate table for these.

        metrics = traverse_res.metrics
        metrics.total_steps = total_step
        metrics.llm_step = llm_steps
        metrics.text_match = text_match
        metrics.is_success = is_success

        return (
            traverse_res,
            knowledge_base_metric_summary,
            custom_metric_summary,
        )

    def _get_messages_by_role_before_cs(
        self, idx_conversational_search: int, role: str, type: str = "text"
    ):
        """Utility method to filter `self.messages` for messages with a given role
        that occur before the conversational search message index
        """

        filtered_messages = [
            message
            for idx, message in enumerate(self.messages)
            if idx < idx_conversational_search
            and message.role == role
            and message.type == type
        ]

        return filtered_messages

    def _weave_user_assistant_messages(self, user_messages, assistant_messages):
        weave = []
        for user, assistant in zip(user_messages, assistant_messages):
            msg = f"User: {user.content}\nAssistant: {assistant.content}\n\n"
            weave.append(msg)

        return " ".join(weave)

    def _find_tool_call_name(self, tool_call_id):
        for message in self.messages:
            if message.type == ContentType.tool_call:
                content = json.loads(message.content)
                """
                - In ADK 1.9, for tool call events, the "tool_call_id" is now "id"
                - still parse out "tool_call_id" for backwards compatibility
                """
                id = content.get("tool_call_id") or content.get("id")
                if id == tool_call_id:
                    return content.get("name")

        raise Exception(f"'{tool_call_id}' not found in messages")

    def generate_knowledge_base_metric_summary(self) -> KnowledgeBaseMetrics:
        idx_conv_search = [
            idx
            for idx, message in enumerate(self.messages)
            if message.type == ContentType.conversational_search
        ]
        metrics = []

        for search_index in idx_conv_search:
            user_messages = self._get_messages_by_role_before_cs(
                role="user", idx_conversational_search=search_index
            )
            assistant_messages = self._get_messages_by_role_before_cs(
                role="assistant",
                idx_conversational_search=search_index,
                type=ContentType.text,
            )

            context = self._weave_user_assistant_messages(
                user_messages, assistant_messages
            )
            most_recent_user_message = user_messages[-1]
            search_message = self.messages[search_index]

            # find the conversational search metadata associated with this message
            conversational_search_data = None
            if self.conversational_search_data:
                for cs_metadata in self.conversational_search_data:
                    if (
                        search_message.conversational_search_metadata.tool_call_id
                        == cs_metadata.metadata.tool_call_id
                    ):
                        conversational_search_data = cs_metadata

            tool_name = self._find_tool_call_name(
                conversational_search_data.metadata.tool_call_id
            )  # name of knowledge base

            search_results = [
                result.body
                for result in conversational_search_data.search_results
            ]
            faithfulness = self.rag_llm_as_a_judge.faithfulness(
                conversational_search_data.text, search_results
            )
            answer_relevancy = self.rag_llm_as_a_judge.answer_relevancy(
                question=most_recent_user_message.content,
                context=context,
                answer=search_message.content,
            )
            knowledge_base_metrics = KnowledgeBaseMetrics(
                dataset_name=self.test_case_name,
                knowledge_base_name=tool_name,
                tool_call_id=search_message.conversational_search_metadata.tool_call_id,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                confidence_scores=conversational_search_data.confidence_scores,
            )

            metrics.append(knowledge_base_metrics)

        return metrics

    def evaluate_derailment(
        self, instructions: str = None
    ) -> List[AnswerDerailment]:
        derailments = []
        last_user_message = None
        for message in self.messages:
            if message.role == "user" and message.type == ContentType.text:
                last_user_message = message
            if message.role == "assistant" and message.type == ContentType.text:
                derailment = (
                    self.safety_llm_as_a_judge.judge_derailment_in_answer(
                        question=last_user_message.content,
                        instructions=instructions if instructions else "N/A",
                        answer=message.content,
                    )
                )
                derailments.append(derailment)
                if derailment.in_scope == "no":
                    return (
                        derailments  # short-circuit if any derailment is found
                    )
        return derailments

    def evaluate_unsafe_topics(
        self, instructions: str = None
    ) -> List[AnswerUnsafeTopic]:
        unsafe_topics = []
        last_user_message = None
        for message in self.messages:
            if message.role == "user" and message.type == ContentType.text:
                last_user_message = message
            if message.role == "assistant" and message.type == ContentType.text:
                try:
                    unsafe_topic = (
                        # TODO: make this configurable
                        self.safety_gg_llm_as_a_judge.judge_unsafe_topic_in_answer(
                            question=last_user_message.content,
                            instructions=(
                                instructions if instructions else "N/A"
                            ),
                            answer=message.content,
                        )
                    )
                except Exception as e:
                    rich.print(
                        f"[red]Error evaluating unsafe topic with Granite Guardian: {e}. Evaluating with fallback judge.[/red]"
                    )
                    unsafe_topic = (
                        self.safety_llm_as_a_judge.judge_unsafe_topic_in_answer(
                            question=last_user_message.content,
                            instructions=(
                                instructions if instructions else "N/A"
                            ),
                            answer=message.content,
                        )
                    )
                unsafe_topics.append(unsafe_topic)
                if unsafe_topic.is_safe == "no":
                    return unsafe_topics  # short-circuit if any unsafe topic is found

        return unsafe_topics


if __name__ == "__main__":

    messages = []

    with open(
        "./benchmarks/workday_tools/concise/result/llama/messages/data18.messages.json",
        "r",
        encoding="utf-8",
    ) as f:

        temp = json.load(f)

        for message in temp:
            messages.append(Message.model_validate(message))

    for message in messages:
        if message.role == "user":
            rich.print(
                "[yellow]GENERATED_USER_MESSAGE:[/yellow]", message.content
            )
        else:
            rich.print("[orange3]WXO:[/orange3]", message.content)

    with open("./benchmarks/workday_tools/data/data18.json", "r") as f:
        ground_truth = OrchestrateDataset.model_validate(json.load(f))

    evaluate_package = EvaluationPackage(
        test_case_name="data1.messages.json",
        ground_truth=ground_truth,
        messages=messages,
    )
    print(evaluate_package.generate_summary())
    # print(evaluate_package.traverse())
