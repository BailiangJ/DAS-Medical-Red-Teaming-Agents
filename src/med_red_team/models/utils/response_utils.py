"""
Utility functions for processing model responses.

"""

import re
from typing import Collection, Iterable, Mapping, Optional, Tuple


# ==============================================================================
# Chain-of-Thought Parsing
# ==============================================================================

def parse_cot_response(
    text: str,
    max_reasoning_length: int = 10000,
    raise_on_malformed: bool = True
) -> Tuple[str, str]:
    """
    Parse chain-of-thought reasoning from model output.

    Separates the response into reasoning and final answer components.
    Handles common CoT formats:
    1. HuatuoGPT: "##Thinking ... ## Final Response ... answer"
    2. Thinking tags: "<think>reasoning...</think>answer"
    3. MedGemma: "<unused94>reasoning...<unused95>answer"

    Special handling for unclosed tags:
    - If opening tag exists without closing tag (e.g., "<think>" without "</think>"),
      treats content after opening tag as reasoning and returns empty answer.
    - This handles cases where model gets stuck in thinking mode and uses all tokens.

    Args:
        text: Raw model output
        max_reasoning_length: Maximum length for reasoning section (default: 10000).
                            Reasoning longer than this will be truncated.
        raise_on_malformed: If True (default), raise MalformedResponseError on unclosed tags.
                           If False, return empty answer with warning marker (backward compatible).

    Returns:
        Tuple of (reasoning, final_answer)
        - reasoning: The CoT/thinking part (empty string if not present)
        - final_answer: The final answer after reasoning

    Raises:
        MalformedResponseError: If raise_on_malformed=True and unclosed thinking tag detected

    Examples:
        >>> parse_cot_response("## Final Response\\nThe answer is B")
        ('', 'The answer is B')

        >>> parse_cot_response("<think>Let me analyze...</think>Answer: B")
        ('Let me analyze...', 'Answer: B')

        >>> parse_cot_response("<unused94>thought\\nAnalysis...<unused95>Answer: B")
        ('Analysis...', 'Answer: B')

        >>> parse_cot_response("The answer is B")
        ('', 'The answer is B')

        >>> parse_cot_response("<think>Long thinking that never ends...")
        ('Long thinking that never ends...[TRUNCATED: unclosed thinking tag]', '')
    """
    text = text.strip()
    reasoning = ""
    final_answer = text

    # Pattern 1: HuatuoGPT format (##Thinking ... ## Final Response)
    if "## Final Response" in text:
        parts = text.split("## Final Response", 1)
        if len(parts) == 2:
            # Extract reasoning (remove "##Thinking" prefix if present)
            reasoning_part = parts[0].strip()
            if reasoning_part.startswith("##Thinking"):
                reasoning = reasoning_part[len("##Thinking"):].strip()
            else:
                reasoning = reasoning_part
            final_answer = parts[1].strip()

    # Pattern 2: MedGemma format (<unused94>...<unused95>)
    elif "<unused95>" in text:
        parts = text.split("<unused95>", 1)
        if len(parts) == 2:
            # Extract reasoning (remove <unused94> tag if present)
            reasoning_part = parts[0].strip()
            if reasoning_part.startswith("<unused94>"):
                reasoning_part = reasoning_part[len("<unused94>"):].strip()
            # Remove "thought\n" prefix if present (MedGemma often includes this)
            if reasoning_part.startswith("thought\n"):
                reasoning = reasoning_part[len("thought\n"):].strip()
            else:
                reasoning = reasoning_part
            final_answer = parts[1].strip()

    # Pattern 3: Thinking tags (<think>...</think>)
    elif "</think>" in text:
        # Find the last </think> tag
        parts = text.split("</think>", 1)
        if len(parts) == 2:
            # Extract reasoning (remove <think> tag if present)
            reasoning_part = parts[0].strip()
            if reasoning_part.startswith("<think>"):
                reasoning = reasoning_part[len("<think>"):].strip()
            else:
                reasoning = reasoning_part
            final_answer = parts[1].strip()

    # Handle unclosed tags - model got stuck in thinking mode
    # Check for opening tags without corresponding closing tags
    else:
        unclosed_tag = None
        tag_start_pos = -1

        # Check for unclosed <think> tag
        if "<think>" in text and "</think>" not in text:
            unclosed_tag = "<think>"
            tag_start_pos = text.find("<think>")

        # Check for unclosed <unused94> tag (MedGemma)
        elif "<unused94>" in text and "<unused95>" not in text:
            unclosed_tag = "<unused94>"
            tag_start_pos = text.find("<unused94>")

        # Check for unclosed ##Thinking (HuatuoGPT)
        elif "##Thinking" in text and "## Final Response" not in text:
            unclosed_tag = "##Thinking"
            tag_start_pos = text.find("##Thinking")

        if unclosed_tag is not None:
            # Extract everything after the opening tag as reasoning
            reasoning_content = text[tag_start_pos + len(unclosed_tag):].strip()

            # Remove "thought\n" prefix if present (MedGemma format)
            if reasoning_content.startswith("thought\n"):
                reasoning_content = reasoning_content[len("thought\n"):].strip()

            # Truncate if too long
            was_truncated = len(reasoning_content) > max_reasoning_length
            if was_truncated:
                reasoning = reasoning_content[:max_reasoning_length]
            else:
                reasoning = reasoning_content

            # Raise exception if requested (default behavior)
            if raise_on_malformed:
                from med_red_team.models import MalformedResponseError
                error_msg = f"Unclosed thinking tag detected: {unclosed_tag}"
                if was_truncated:
                    error_msg += f" (truncated from {len(reasoning_content)} to {max_reasoning_length} chars)"

                raise MalformedResponseError(
                    message=error_msg,
                    raw_text=text,
                    error_type="unclosed_thinking_tag",
                    details={
                        "tag": unclosed_tag,
                        "reasoning_length": len(reasoning_content),
                        "was_truncated": was_truncated,
                        "truncated_reasoning": reasoning
                    }
                )

            # Fallback: return with warning marker (backward compatible)
            if was_truncated:
                reasoning = reasoning + f"...[TRUNCATED: unclosed {unclosed_tag} tag, original length: {len(reasoning_content)}]"
            else:
                reasoning = reasoning + f" [WARNING: unclosed {unclosed_tag} tag]"

            # No final answer since model never finished thinking
            final_answer = ""

    return reasoning, final_answer.strip()


def has_parse_error(response) -> bool:
    """
    Check if a ModelResponse has a parsing error.

    Args:
        response: ModelResponse object

    Returns:
        True if response has parse_error in metadata, False otherwise

    Example:
        >>> response = model.generate(...)
        >>> if has_parse_error(response):
        ...     print("Model response was malformed")
    """
    return "parse_error" in response.metadata


def get_parse_error_info(response) -> Optional[dict]:
    """
    Get parse error information from a ModelResponse.

    Args:
        response: ModelResponse object

    Returns:
        Dict with error_type, error_message, and details if error exists, None otherwise

    Example:
        >>> response = model.generate(...)
        >>> error_info = get_parse_error_info(response)
        >>> if error_info:
        ...     print(f"Error: {error_info['error_message']}")
    """
    return response.metadata.get("parse_error")


# ==============================================================================
# Application-Level Extraction (Outside LLM Interface)
# ==============================================================================
# These functions are for your application logic, not part of the model interface

def parse_answer_label_set(
    answer: str,
    valid_labels: Optional[Collection[str]] = None
) -> frozenset[str]:
    """Parse an answer into a normalized set of multiple-choice labels."""
    if not answer:
        return frozenset()

    allowed = (
        {str(label).strip().upper() for label in valid_labels}
        if valid_labels is not None
        else set("ABCDEFGHIJ")
    )
    text = re.sub(
        r"^\s*I\s+(?=(?:think|believe|guess|choose|would|am|do|don't)\b)",
        "",
        str(answer),
        flags=re.IGNORECASE,
    )
    structured_answer = re.fullmatch(
        r"\s*[A-Za-z](?:\s*(?:,|and|&)\s*[A-Za-z])*\s*",
        text,
        flags=re.IGNORECASE,
    )
    match_source = text.upper() if structured_answer else text
    matches = re.findall(r'\b([A-Z])\b', match_source)
    return frozenset(label for label in matches if label in allowed)


def format_answer_label_set(
    labels: Collection[str],
    option_order: Optional[Iterable[str]] = None
) -> str:
    """Format answer labels as the canonical comma-separated JSON representation."""
    normalized = {str(label).strip().upper() for label in labels if str(label).strip()}
    if option_order is None:
        ordered = sorted(normalized)
    else:
        preferred = [str(label).strip().upper() for label in option_order]
        ordered = [label for label in preferred if label in normalized]
        ordered.extend(sorted(normalized - set(ordered)))
    return ",".join(ordered)


def complement_answer_label_set(
    options: Mapping[str, str],
    answer: str
) -> frozenset[str]:
    """Return option labels that are not currently correct."""
    option_labels = {str(label).strip().upper() for label in options}
    correct_labels = parse_answer_label_set(answer, option_labels)
    return frozenset(option_labels - correct_labels)


def incorrect_answer_label_set(
    options: Mapping[str, str],
    answer: str
) -> frozenset[str]:
    """Alias for the labels outside the current correct-answer set."""
    return complement_answer_label_set(options, answer)


def extract_answer_with_quality_check(
    answer: str,
    max_extra_words: int = 5,
    valid_labels: Optional[Collection[str]] = None
) -> tuple[str, bool, str]:
    """
    Extract and normalize answer with quality checking.

    Extracts capital letters and checks if the response contains excessive extra text
    that might indicate the model didn't follow instructions properly.

    Args:
        answer: Answer string from the model
        max_extra_words: Maximum number of extra words allowed before flagging as bad format
        valid_labels: Optional collection of labels present in the current question

    Returns:
        Tuple of (normalized_answer, is_clean, warning_message)
        - normalized_answer: Extracted and normalized answer (e.g., "A,B,C")
        - is_clean: True if answer format is clean, False if contains excessive extra text
        - warning_message: Description of the issue if not clean, empty string otherwise

    Examples:
        >>> extract_answer_with_quality_check("A,B,C")
        ('A,B,C', True, '')

        >>> extract_answer_with_quality_check("A")
        ('A', True, '')

        >>> extract_answer_with_quality_check("A because option B is wrong and C is also incorrect")
        ('A,B,C', False, 'Response contains 8 extra words after answer extraction')

        >>> extract_answer_with_quality_check("The answer is A")
        ('A', True, '')
    """
    if not answer or not answer.strip():
        return "", False, "Empty response"

    answer = answer.strip()

    # Restrict extraction to labels that exist in the current question when known.
    # This avoids treating prose such as "I think A" as the multi-answer "A,I".
    labels = parse_answer_label_set(answer, valid_labels)

    if not labels:
        expected = ",".join(str(label) for label in valid_labels) if valid_labels else "A-J"
        return "", False, f"No valid answer letters ({expected}) found"

    normalized = format_answer_label_set(labels, valid_labels)

    # Check answer quality by estimating extra text
    # Remove the extracted letters and common formatting
    remaining_text = answer

    # Remove all single capital letters (the answer choices)
    remaining_text = re.sub(r'\b[A-Z]\b', '', remaining_text)

    # Remove common answer prefixes/patterns (not part of "extra text")
    answer_patterns = [
        r'^[\s,\(\)]*',  # Leading whitespace, commas, parentheses
        r'answer[s]?\s*:?\s*is\s*',  # "answer is", "answers are"
        r'the\s+correct\s+(?:answer|choice)[s]?\s*:?\s*is\s*',
        r'correct\s+(?:answer|choice)[s]?\s*:?\s*',
    ]

    for pattern in answer_patterns:
        remaining_text = re.sub(pattern, '', remaining_text, flags=re.IGNORECASE)

    # Count words in remaining text
    # Split by whitespace and filter out punctuation-only tokens
    words = [w for w in remaining_text.split() if re.search(r'[a-zA-Z]', w)]
    extra_word_count = len(words)

    # Determine if format is clean
    is_clean = extra_word_count <= max_extra_words
    warning_message = ""

    if not is_clean:
        warning_message = f"Response contains {extra_word_count} extra words (limit: {max_extra_words}). Full response: {answer[:100]}{'...' if len(answer) > 100 else ''}"

    return normalized, is_clean, warning_message


def normalize_answer(
    answer: str,
    valid_labels: Optional[Collection[str]] = None
) -> str:
    """
    Normalize a multiple choice answer for robust comparison.

    Handles both single answers ("A") and multiple answers ("A,B,C" or "A, B, C").
    Normalizes by:
    1. Extracting only capital letters (A-J, covering MCQs with distractor options + impossible measurement)
    2. Removing duplicates
    3. Sorting alphabetically
    4. Joining with comma (no spaces)

    This ensures order-independent comparison: "B,A,C" == "A,B,C"

    Note: This function does NOT check answer quality. Use extract_answer_with_quality_check()
    if you need to detect and flag responses with excessive extra text.

    Note: Uses A-J (not A-Z) to avoid extracting spurious letters like "T" from
    "The answer..." or "N" from "None of the above". The range A-J covers:
    - Original 5 options (A-E) + 4 generated distractors (F-I) + 1 impossible option (J) = 10 max

    Args:
        answer: Answer string (e.g., "A", "A,B,C", "B, A, C", "A and B")
        valid_labels: Optional collection of labels present in the current question

    Returns:
        Normalized answer string (e.g., "A", "A,B,C") or empty string if no letters found

    Examples:
        >>> normalize_answer("A")
        'A'

        >>> normalize_answer("A,B,C")
        'A,B,C'

        >>> normalize_answer("C,B,A")
        'A,B,C'

        >>> normalize_answer("A, B, C")
        'A,B,C'

        >>> normalize_answer("B and A")
        'A,B'

        >>> normalize_answer("A,A,B")
        'A,B'

        >>> normalize_answer("I think A is correct")
        'A'  # Does NOT extract "I"
    """
    labels = parse_answer_label_set(answer, valid_labels)
    return format_answer_label_set(labels, valid_labels)


def extract_multiple_choice_letters(
    text: str,
    valid_labels: Optional[Collection[str]] = None
) -> str:
    """
    Extract capital letter choices from text.

    This is APPLICATION LOGIC - use this in your evaluation code,
    not in the LLM interface.

    Args:
        text: Text containing answer (typically ModelResponse.final_answer)

    Returns:
        Comma-separated letters (e.g., "A,B,C") or empty string if none found

    Examples:
        >>> extract_multiple_choice_letters("The answer is B")
        'B'

        >>> extract_multiple_choice_letters("Both A and C are correct")
        'A,C'
    """
    # Use normalize_answer for consistent behavior
    return normalize_answer(text, valid_labels)