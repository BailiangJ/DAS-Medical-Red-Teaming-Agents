"""
Bias-specific utility functions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from med_red_team.models.utils.response_utils import extract_answer_with_quality_check


VALID_VOTE_OUTCOME = "valid"
NO_WINNER_VOTE_OUTCOME = "no_winner"
INVALID_VOTE_OUTCOME = "invalid"


@dataclass(frozen=True)
class VoteSummary:
    """Normalized vote details for a single majority-vote round."""

    valid_labels: Tuple[str, ...]
    raw_responses: List[str] = field(default_factory=list)
    valid_votes: List[str] = field(default_factory=list)
    invalid_votes: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    majority_vote: str = ""
    outcome: str = INVALID_VOTE_OUTCOME
    configured_total_attempts: int = 0
    required_votes: int = 0


def is_mental_health_case(
    category: Optional[str],
    question: Optional[str],
) -> bool:
    """Return whether emotional manipulation should exclude this case."""
    normalized_category = (category or "").strip().casefold()
    normalized_question = (question or "").strip().casefold()

    return (
        normalized_category == "psychiatric"
        or "mental" in normalized_question
        or "psych" in normalized_question
    )


def clean_answer_response(response: str) -> str:
    """Normalize surrounding whitespace without inventing answer labels."""
    return re.sub(r"\s+", " ", str(response or "").strip())


def derive_valid_answer_labels(question_text: str) -> Tuple[str, ...]:
    """Derive the ordered answer-label set from the rendered question text."""
    if not question_text:
        return tuple()

    labels: List[str] = []
    for match in re.finditer(r"(?<![A-Z0-9])(?:\(([A-Z])\)|([A-Z])\s*[:.)])", question_text.upper()):
        label = match.group(1) or match.group(2)
        if label and label not in labels:
            labels.append(label)
    return tuple(labels)


_SINGLE_LABEL_PATTERNS = [
    re.compile(r"^\s*[\(\[]?([A-Za-z])[\)\].:]*\s*$"),
    re.compile(
        r"^\s*(?:the\s+answer|answer|option|choice|my\s+answer)\s*(?:is|:)?\s*"
        r"[\(\[]?([A-Za-z])[\)\].:]*\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^\s*i\s+(?:choose|pick|select|would\s+choose|would\s+pick)\s*"
        r"[\(\[]?([A-Za-z])[\)\].:]*\s*$",
        flags=re.IGNORECASE,
    ),
]


def normalize_single_vote(response: str, valid_labels: Sequence[str]) -> Optional[str]:
    """Canonicalize a single-choice answer or return None when malformed."""
    cleaned = clean_answer_response(response)
    if not cleaned:
        return None

    allowed = {str(label).strip().upper() for label in valid_labels}
    if not allowed:
        return None

    for pattern in _SINGLE_LABEL_PATTERNS:
        match = pattern.fullmatch(cleaned)
        if match:
            candidate = match.group(1).upper()
            return candidate if candidate in allowed else None

    normalized, is_clean, _warning = extract_answer_with_quality_check(
        cleaned,
        max_extra_words=3,
        valid_labels=valid_labels,
    )
    if not normalized or not is_clean:
        return None

    labels = [label for label in normalized.split(",") if label]
    if len(labels) != 1:
        return None
    return labels[0] if labels[0] in allowed else None


def summarize_votes(
    responses: List[str],
    *,
    question_text: str,
    configured_total_attempts: Optional[int] = None,
) -> VoteSummary:
    """Canonicalize responses and require a strict configured-denominator majority."""
    total_attempts = (
        len(responses)
        if configured_total_attempts is None
        else configured_total_attempts
    )
    if (
        isinstance(total_attempts, bool)
        or not isinstance(total_attempts, int)
        or total_attempts < 1
    ):
        raise ValueError("configured_total_attempts must be a positive integer")
    if len(responses) > total_attempts:
        raise ValueError(
            "responses cannot exceed configured_total_attempts "
            f"({len(responses)} > {total_attempts})"
        )

    required_votes = total_attempts // 2 + 1
    valid_labels = derive_valid_answer_labels(question_text)
    cleaned_responses = [clean_answer_response(response) for response in responses]

    valid_votes: List[str] = []
    invalid_votes: List[str] = []
    for cleaned in cleaned_responses:
        normalized = normalize_single_vote(cleaned, valid_labels)
        if normalized is None:
            invalid_votes.append(cleaned)
        else:
            valid_votes.append(normalized)

    counts = Counter(valid_votes)
    most_common = counts.most_common(1)
    if most_common and most_common[0][1] >= required_votes:
        majority_vote = most_common[0][0]
        outcome = VALID_VOTE_OUTCOME
    elif len(valid_votes) < required_votes:
        majority_vote = ""
        outcome = INVALID_VOTE_OUTCOME
    else:
        majority_vote = ""
        outcome = NO_WINNER_VOTE_OUTCOME

    return VoteSummary(
        valid_labels=valid_labels,
        raw_responses=cleaned_responses,
        valid_votes=valid_votes,
        invalid_votes=invalid_votes,
        counts=dict(counts),
        majority_vote=majority_vote,
        outcome=outcome,
        configured_total_attempts=total_attempts,
        required_votes=required_votes,
    )


def majority_vote(responses: List[str], question_text: str = "") -> str:
    """Backward-compatible wrapper returning the strict majority or sentinel."""
    summary = summarize_votes(
        responses,
        question_text=question_text,
        configured_total_attempts=len(responses),
    )
    if summary.outcome == VALID_VOTE_OUTCOME:
        return summary.majority_vote
    if summary.outcome == NO_WINNER_VOTE_OUTCOME:
        return "no winner"
    return "invalid"


def calculate_vote_entropy(responses: List[str]) -> float:
    """Calculate Shannon entropy over valid canonical votes only."""
    cleaned = [str(response).strip() for response in responses if str(response).strip()]
    if not cleaned:
        return 0.0

    count = Counter(cleaned)
    total = len(cleaned)
    entropy = 0.0
    for _response, freq in count.items():
        p = freq / total
        if p > 0:
            entropy += -p * math.log2(p)
    return entropy


def get_full_choice_text(letter_choice: str, question_text: str) -> str:
    """Extract the full option text for a canonical single-letter answer."""
    if not letter_choice or not question_text:
        return letter_choice

    letter = letter_choice.strip().upper()
    for line in question_text.splitlines():
        line = line.strip()
        match = re.match(r"^\s*([A-Z])\s*[:.)]\s*(.*)", line, re.IGNORECASE)
        if match and match.group(1).upper() == letter:
            separator = ":" if ":" in line else line[len(match.group(1)) :].lstrip()[0]
            return f"{match.group(1)}{separator} {match.group(2)}".strip()

    inline_match = re.search(
        rf"(?<![A-Z0-9]){re.escape(letter)}\s*[:.)]\s*([^\n]+?)(?=(?:\s+[A-Z]\s*[:.)])|$)",
        question_text,
        flags=re.IGNORECASE,
    )
    if inline_match:
        option_text = inline_match.group(1).strip()
        return f"{letter}: {option_text}"

    return letter


__all__ = [
    "INVALID_VOTE_OUTCOME",
    "NO_WINNER_VOTE_OUTCOME",
    "VALID_VOTE_OUTCOME",
    "VoteSummary",
    "calculate_vote_entropy",
    "clean_answer_response",
    "derive_valid_answer_labels",
    "get_full_choice_text",
    "is_mental_health_case",
    "majority_vote",
    "normalize_single_vote",
    "summarize_votes",
]
