"""Shared tool policy and validation for robustness attack composition."""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from med_red_team.models.utils.response_utils import (
    complement_answer_label_set,
    parse_answer_label_set,
)
from med_red_team.robustness.data import OrchestratorQuestionData


POLICY_VERSION = "1"

ORCHESTRATOR_TOOL_NAMES = [
    "generate_distractor_options_tool",
    "replace_correct_answer_to_none_of_the_options_are_correct_tool",
    "invert_final_question_and_answer_tool",
    "add_distraction_sentence_tool",
    "adjust_impossible_measurement_tool",
    "introduce_cognitive_bias_tool",
]

TOOL_NAME_TO_STRATEGY_NAME = {
    "generate_distractor_options_tool": "generate_distractor_options",
    "replace_correct_answer_to_none_of_the_options_are_correct_tool": "replace_correct_answer_with_none",
    "invert_final_question_and_answer_tool": "invert_question_answer",
    "add_distraction_sentence_tool": "add_distraction_sentence",
    "adjust_impossible_measurement_tool": "adjust_impossible_measurement",
    "introduce_cognitive_bias_tool": "bias_manipulation",
    "add_none_of_the_options_are_correct_tool": "add_none_of_the_above",
}
STRATEGY_NAME_TO_TOOL_NAME = {
    strategy_name: tool_name
    for tool_name, strategy_name in TOOL_NAME_TO_STRATEGY_NAME.items()
}

ABSOLUTE_CONFLICTS = [
    frozenset({
        "invert_final_question_and_answer_tool",
        "replace_correct_answer_to_none_of_the_options_are_correct_tool",
    }),
    frozenset({
        "replace_correct_answer_to_none_of_the_options_are_correct_tool",
        "adjust_impossible_measurement_tool",
    }),
    frozenset({
        "adjust_impossible_measurement_tool",
        "invert_final_question_and_answer_tool",
    }),
    frozenset({
        "add_none_of_the_options_are_correct_tool",
        "replace_correct_answer_to_none_of_the_options_are_correct_tool",
    }),
]

ORDERING_DEPENDENCIES = [
    ("generate_distractor_options_tool", "invert_final_question_and_answer_tool"),
    (
        "generate_distractor_options_tool",
        "replace_correct_answer_to_none_of_the_options_are_correct_tool",
    ),
    ("generate_distractor_options_tool", "adjust_impossible_measurement_tool"),
    ("invert_final_question_and_answer_tool", "introduce_cognitive_bias_tool"),
    ("invert_final_question_and_answer_tool", "add_distraction_sentence_tool"),
]


@dataclass(frozen=True)
class ToolPolicyIssue:
    """One validation issue in a proposed tool sequence."""

    issue_type: str
    tools: Tuple[str, ...]
    message: str
    recoverable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type,
            "tools": list(self.tools),
            "message": self.message,
            "recoverable": self.recoverable,
        }


@dataclass
class ToolPolicyDecision:
    """Validation and normalization result for a proposed tool sequence."""

    valid: bool
    original_sequence: List[str]
    normalized_sequence: List[str] = field(default_factory=list)
    issues: List[ToolPolicyIssue] = field(default_factory=list)
    reordered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "original_sequence": list(self.original_sequence),
            "normalized_sequence": list(self.normalized_sequence),
            "issues": [issue.to_dict() for issue in self.issues],
            "reordered": self.reordered,
        }


@dataclass(frozen=True)
class ToolRuntimeCheck:
    """Precondition or postcondition result for a tool execution."""

    valid: bool
    status: str = "success"
    reason: Optional[str] = None


def _duplicates(sequence: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    duplicates: List[str] = []
    for item in sequence:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    return duplicates


def _stable_topological_sort(
    sequence: Sequence[str],
    dependencies: Sequence[Tuple[str, str]],
) -> Tuple[List[str], Optional[ToolPolicyIssue]]:
    """Apply selected dependency edges while preserving unconstrained order."""
    selected = list(sequence)
    selected_set = set(selected)
    original_index = {name: index for index, name in enumerate(selected)}
    adjacency = {name: set() for name in selected}
    indegree = {name: 0 for name in selected}

    for before, after in dependencies:
        if before not in selected_set or after not in selected_set:
            continue
        if after not in adjacency[before]:
            adjacency[before].add(after)
            indegree[after] += 1

    ready = sorted(
        (name for name, degree in indegree.items() if degree == 0),
        key=original_index.__getitem__,
    )
    ordered: List[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for dependent in sorted(adjacency[current], key=original_index.__getitem__):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort(key=original_index.__getitem__)

    if len(ordered) != len(selected):
        return [], ToolPolicyIssue(
            issue_type="ordering_cycle",
            tools=tuple(selected),
            message="Tool ordering dependencies contain a cycle.",
            recoverable=False,
        )
    return ordered, None


def validate_normalized_sequence(sequence: Sequence[str]) -> List[ToolPolicyIssue]:
    """Validate that all selected dependency edges are satisfied."""
    positions = {tool: index for index, tool in enumerate(sequence)}
    issues: List[ToolPolicyIssue] = []
    for before, after in ORDERING_DEPENDENCIES:
        if before in positions and after in positions and positions[before] > positions[after]:
            issues.append(ToolPolicyIssue(
                issue_type="invalid_order",
                tools=(before, after),
                message=f"{before} must execute before {after}.",
            ))
    return issues


def normalize_tool_sequence(
    tool_sequence: Sequence[str],
    allowed_tools: Optional[Iterable[str]] = None,
) -> ToolPolicyDecision:
    """Validate a sequence and normalize dependency ordering."""
    original = list(tool_sequence)
    allowed = set(
        ORCHESTRATOR_TOOL_NAMES if allowed_tools is None else allowed_tools
    )
    issues: List[ToolPolicyIssue] = []

    if not original:
        issues.append(ToolPolicyIssue(
            issue_type="empty_sequence",
            tools=(),
            message="At least one manipulation tool must be selected.",
        ))

    unknown = [tool for tool in original if tool not in allowed]
    if unknown:
        issues.append(ToolPolicyIssue(
            issue_type="unknown_tool",
            tools=tuple(unknown),
            message=f"Unknown tools: {', '.join(unknown)}.",
        ))

    duplicates = _duplicates(original)
    if duplicates:
        issues.append(ToolPolicyIssue(
            issue_type="duplicate_tool",
            tools=tuple(duplicates),
            message=f"Tools may be selected at most once: {', '.join(duplicates)}.",
        ))

    selected = set(original)
    for conflict in ABSOLUTE_CONFLICTS:
        if conflict.issubset(selected):
            tools = tuple(sorted(conflict))
            issues.append(ToolPolicyIssue(
                issue_type="absolute_conflict",
                tools=tools,
                message=f"Conflicting tools cannot be combined: {', '.join(tools)}.",
            ))

    if issues:
        return ToolPolicyDecision(
            valid=False,
            original_sequence=original,
            issues=issues,
        )

    normalized, cycle_issue = _stable_topological_sort(
        original,
        ORDERING_DEPENDENCIES,
    )
    if cycle_issue:
        return ToolPolicyDecision(
            valid=False,
            original_sequence=original,
            issues=[cycle_issue],
        )

    order_issues = validate_normalized_sequence(normalized)
    return ToolPolicyDecision(
        valid=not order_issues,
        original_sequence=original,
        normalized_sequence=normalized,
        issues=order_issues,
        reordered=normalized != original,
    )


def normalize_strategy_sequence(strategy_names: Sequence[str]) -> ToolPolicyDecision:
    """Validate and normalize a fixed strategy chain through the tool policy."""
    tool_names: List[str] = []
    issues: List[ToolPolicyIssue] = []
    for strategy_name in strategy_names:
        tool_name = STRATEGY_NAME_TO_TOOL_NAME.get(strategy_name)
        if not tool_name:
            issues.append(ToolPolicyIssue(
                issue_type="unknown_strategy",
                tools=(strategy_name,),
                message=f"Unknown robustness strategy: {strategy_name}.",
            ))
        else:
            tool_names.append(tool_name)

    if issues:
        return ToolPolicyDecision(
            valid=False,
            original_sequence=list(strategy_names),
            issues=issues,
        )

    decision = normalize_tool_sequence(
        tool_names,
        allowed_tools=TOOL_NAME_TO_STRATEGY_NAME.keys(),
    )
    if not decision.valid:
        return decision

    normalized_strategies = [
        TOOL_NAME_TO_STRATEGY_NAME[tool_name]
        for tool_name in decision.normalized_sequence
    ]
    return ToolPolicyDecision(
        valid=True,
        original_sequence=list(strategy_names),
        normalized_sequence=normalized_strategies,
        issues=[],
        reordered=normalized_strategies != list(strategy_names),
    )


def check_question_state(question: OrchestratorQuestionData) -> ToolRuntimeCheck:
    """Validate global MCQ invariants for an orchestrator question."""
    if not question.question or not question.question.strip():
        return ToolRuntimeCheck(False, "invalid_state", "Question text is empty.")
    if not question.options:
        return ToolRuntimeCheck(False, "invalid_state", "Question has no options.")

    labels = list(question.options)
    empty_option_labels = [
        label for label, text in question.options.items()
        if not str(text).strip()
    ]
    if empty_option_labels:
        return ToolRuntimeCheck(
            False,
            "invalid_state",
            f"Options have empty text: {empty_option_labels}.",
        )
    if len(labels) != len(set(labels)):
        return ToolRuntimeCheck(False, "invalid_state", "Option labels are not unique.")

    raw_answer_labels = parse_answer_label_set(
        question.answer_idx,
        valid_labels="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )
    invalid_answer_labels = raw_answer_labels - {
        str(label).strip().upper() for label in labels
    }
    if invalid_answer_labels:
        return ToolRuntimeCheck(
            False,
            "invalid_state",
            f"Answer contains labels missing from options: "
            f"{sorted(invalid_answer_labels)}.",
        )

    correct_labels = parse_answer_label_set(question.answer_idx, labels)
    if not correct_labels:
        return ToolRuntimeCheck(False, "invalid_state", "No valid correct-answer labels remain.")
    if not correct_labels.issubset(set(labels)):
        return ToolRuntimeCheck(False, "invalid_state", "Correct-answer labels are missing from options.")
    return ToolRuntimeCheck(True)


def check_tool_preconditions(
    tool_name: str,
    question: OrchestratorQuestionData,
) -> ToolRuntimeCheck:
    """Validate question-dependent requirements before running one tool."""
    state_check = check_question_state(question)
    if not state_check.valid:
        return state_check

    correct_labels = parse_answer_label_set(question.answer_idx, question.options.keys())
    incorrect_labels = set(question.options) - set(correct_labels)

    if tool_name == "replace_correct_answer_to_none_of_the_options_are_correct_tool":
        if len(correct_labels) != 1:
            return ToolRuntimeCheck(
                False,
                "not_applicable",
                "replace-none requires exactly one correct answer.",
            )
    elif tool_name in {
        "introduce_cognitive_bias_tool",
        "add_distraction_sentence_tool",
    }:
        if not incorrect_labels:
            return ToolRuntimeCheck(
                False,
                "not_applicable",
                "No incorrect option is available for this attack.",
            )
    elif tool_name == "invert_final_question_and_answer_tool":
        if not complement_answer_label_set(question.options, question.answer_idx):
            return ToolRuntimeCheck(
                False,
                "not_applicable",
                "Inversion would produce an empty answer complement.",
            )
    return ToolRuntimeCheck(True)


def check_tool_postconditions(
    tool_name: str,
    before: OrchestratorQuestionData,
    after: OrchestratorQuestionData,
) -> ToolRuntimeCheck:
    """Validate global invariants and expected changes after one tool."""
    state_check = check_question_state(after)
    if not state_check.valid:
        return state_check

    question_changed = before.question != after.question
    options_changed = before.options != after.options
    answer_changed = before.answer_idx != after.answer_idx
    if not (question_changed or options_changed or answer_changed):
        return ToolRuntimeCheck(False, "invalid_noop", "Tool made no changes.")

    if tool_name == "generate_distractor_options_tool":
        if len(after.options) <= len(before.options) or answer_changed:
            return ToolRuntimeCheck(
                False,
                "invalid_state",
                "Distractor generation must add options without changing the answer.",
            )
    elif tool_name == "replace_correct_answer_to_none_of_the_options_are_correct_tool":
        if not options_changed or question_changed or answer_changed:
            return ToolRuntimeCheck(
                False,
                "invalid_state",
                "Replace-none must only change the correct option text.",
            )
    elif tool_name == "invert_final_question_and_answer_tool":
        if not question_changed or not answer_changed:
            return ToolRuntimeCheck(
                False,
                "invalid_state",
                "Inversion must change both question text and answer labels.",
            )
    elif tool_name in {
        "introduce_cognitive_bias_tool",
        "add_distraction_sentence_tool",
    }:
        if not question_changed or options_changed or answer_changed:
            return ToolRuntimeCheck(
                False,
                "invalid_state",
                "Bias and distraction must only change question text.",
            )
    elif tool_name == "adjust_impossible_measurement_tool":
        if not question_changed or not options_changed or not answer_changed:
            return ToolRuntimeCheck(
                False,
                "invalid_state",
                "Impossible measurement must change question, options, and answer.",
            )
    return ToolRuntimeCheck(True)


def _bounded(text: str, limit: int = 240) -> Tuple[str, bool]:
    text = text or ""
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _question_edits(before: str, after: str, max_edits: int = 3) -> List[Dict[str, Any]]:
    edits: List[Dict[str, Any]] = []
    matcher = SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        before_text, before_truncated = _bounded(before[i1:i2])
        after_text, after_truncated = _bounded(after[j1:j2])
        edits.append({
            "type": tag,
            "before": before_text,
            "after": after_text,
            "truncated": before_truncated or after_truncated,
        })
        if len(edits) >= max_edits:
            break
    return edits


def compute_question_delta(
    before: OrchestratorQuestionData,
    after: OrchestratorQuestionData,
) -> Dict[str, Any]:
    """Create a bounded deterministic state delta for logs and planner feedback."""
    before_labels = set(before.options)
    after_labels = set(after.options)
    common_labels = before_labels & after_labels
    modified_labels = [
        label for label in before.options
        if label in common_labels and before.options[label] != after.options[label]
    ]
    modified_options: Dict[str, Dict[str, Any]] = {}
    for label in modified_labels[:3]:
        old_text, old_truncated = _bounded(before.options[label])
        new_text, new_truncated = _bounded(after.options[label])
        modified_options[label] = {
            "before": old_text,
            "after": new_text,
            "truncated": old_truncated or new_truncated,
        }

    added_options = {}
    for label in after.options:
        if label in after_labels - before_labels:
            value, truncated = _bounded(after.options[label])
            added_options[label] = {"text": value, "truncated": truncated}

    before_answer_labels = parse_answer_label_set(
        before.answer_idx,
        before.options.keys(),
    )
    after_answer_labels = parse_answer_label_set(
        after.answer_idx,
        after.options.keys(),
    )

    return {
        "question": {
            "changed": before.question != after.question,
            "length_before": len(before.question),
            "length_after": len(after.question),
            "edits": _question_edits(before.question, after.question),
        },
        "options": {
            "added": added_options,
            "removed": [label for label in before.options if label not in after.options],
            "modified": modified_options,
        },
        "answer": {
            "before": [
                label for label in before.options if label in before_answer_labels
            ],
            "after": [
                label for label in after.options if label in after_answer_labels
            ],
        },
    }


def build_tool_policy_prompt_section() -> str:
    """Render the runtime tool policy for the planner system prompt."""
    planner_tools = set(ORCHESTRATOR_TOOL_NAMES)
    conflict_lines = [
        " + ".join(sorted(conflict))
        for conflict in ABSOLUTE_CONFLICTS
        if conflict.issubset(planner_tools)
    ]
    dependency_lines = [f"{before} before {after}" for before, after in ORDERING_DEPENDENCIES]
    return (
        "Absolute conflicts (never select both):\n- "
        + "\n- ".join(conflict_lines)
        + "\nOrdering dependencies (runtime will normalize these):\n- "
        + "\n- ".join(dependency_lines)
    )
