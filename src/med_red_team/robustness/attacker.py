"""
Attack strategy concrete implementations.

This module contains both rule-based and LLM-based attack strategies
that manipulate test cases to try to make the testee fail.
"""

from typing import Dict, Any, Optional, List
import copy
import random
import re
import json

from med_red_team.actors import AttackStrategy
from med_red_team.utils import parse_json_response, update_attack_metadata
from med_red_team.data import TestCase
from med_red_team.model_pool import ModelPool
from med_red_team.models import GenerationConfig
from med_red_team.models.utils.response_utils import (
    complement_answer_label_set,
    format_answer_label_set,
    parse_answer_label_set,
)
from med_red_team.robustness.prompts import (
    DISTRACTION_SYSTEM_PROMPT,
    build_distraction_user_prompt,
    DISTRACTOR_OPTIONS_SYSTEM_PROMPT,
    build_distractor_options_user_prompt,
    BIAS_STYLES,
    build_bias_manipulation_system_prompt,
    build_bias_manipulation_user_prompt,
    INVERT_QUESTION_SYSTEM_PROMPT,
    build_invert_question_user_prompt,
    ADJUST_MEASUREMENT_SYSTEM_PROMPT,
    build_adjust_measurement_user_prompt
)


# ==============================================================================
# Rule-Based Attack Strategies (No LLM Required)
# ==============================================================================

class AddNoneOfTheAboveStrategy(AttackStrategy):
    """
    Adds "None of the options are correct" as an additional choice.

    This is a rule-based strategy that does not require an LLM.

    Attributes:
        name: Always "add_none_of_the_above"
        model_id: None (rule-based strategy)
        model: None (rule-based strategy)
        system_prompt: None (rule-based strategy)
        config: Default GenerationConfig (unused)

    Example:
        Original: A, B, C, D
        Modified: A, B, C, D, E) None of the options are correct
    """

    def __init__(self):
        """Initialize add none of the above strategy (rule-based, no LLM required)."""
        super().__init__(
            name="add_none_of_the_above",
            model_id=None,
            model_pool=None,
            system_prompt=None,
            config=None
        )
    
    def apply(self, test_case: TestCase, context: Dict[str, Any]) -> TestCase:
        """Add "None of the above" option."""
        # Copy test case to avoid modifying original
        modified = copy.deepcopy(test_case)

        if not modified.options:
            print(f"[WARNING] Cannot {self.name}: missing options")
            return modified

        # Find next letter
        existing_labels = sorted(modified.options.keys())
        if not existing_labels:
            next_label = "A"
        else:
            last_label = existing_labels[-1]
            next_label = chr(ord(last_label) + 1)

        # Add new option
        modified.options[next_label] = "None of the options are correct"

        # Update metadata to track manipulation
        update_attack_metadata(modified, self.name, {
            "options_added": [next_label]
        })

        return modified


class ReplaceCorrectAnswerStrategy(AttackStrategy):
    """
    Replaces the correct answer text with "None of the options are correct".

    This is a rule-based strategy that does not require an LLM.
    This is a stronger attack that directly manipulates the correct answer.

    Attributes:
        name: Always "replace_correct_answer_with_none"
        model_id: None (rule-based strategy)
        model: None (rule-based strategy)
        system_prompt: None (rule-based strategy)
        config: Default GenerationConfig (unused)

    Example:
        Original: A) Hypertension → B) Diabetes (correct)
        Modified: A) Hypertension → B) None of the options are correct (correct)
    """

    def __init__(self):
        """Initialize replace correct answer strategy (rule-based, no LLM required)."""
        super().__init__(
            name="replace_correct_answer_with_none",
            model_id=None,
            model_pool=None,
            system_prompt=None,
            config=None
        )
    
    def apply(self, test_case: TestCase, context: Dict[str, Any]) -> TestCase:
        """Replace correct answer with 'None of the options are correct'."""
        modified = copy.deepcopy(test_case)

        if not modified.options or not modified.correct_answer:
            print(f"[WARNING] Cannot {self.name}: missing options or correct_answer")
            return modified

        correct_labels = parse_answer_label_set(
            modified.correct_answer,
            valid_labels=modified.options.keys()
        )
        if len(correct_labels) != 1:
            update_attack_metadata(modified, self.name, {
                "manipulation_failed": True,
                "failure_category": "not_applicable",
                "reason": "replace-none requires exactly one correct answer",
                "correct_labels": sorted(correct_labels),
            })
            return modified

        correct_idx = next(iter(correct_labels))
        original_text = modified.options[correct_idx]
        modified.options[correct_idx] = "None of the options are correct"

        update_attack_metadata(modified, self.name, {
            "original_correct_answer_text": original_text,
            "replaced_answer_idx": correct_idx
        })

        return modified


# ==============================================================================
# LLM-Based Attack Strategies
# ==============================================================================

class AddDistractionSentenceStrategy(AttackStrategy):
    """
    Uses an LLM to add a distracting sentence that references one of the wrong choices.

    The distraction sentence is inserted into the question stem to mislead or confuse
    the testee by casually mentioning an incorrect concept from the answer choices.

    Key features:
    - Selects a random wrong answer to reference
    - Generates a casual, tangential distraction sentence using LLM
    - Inserts the distraction before the final question sentence
    - Preserves question structure and formatting

    Attributes:
        name: Always "add_distraction_sentence"
        model_id: Model ID for generating distractions
        model: Model instance from pool
        system_prompt: System instructions (set to DISTRACTION_SYSTEM_PROMPT)
        config: Generation configuration (default: temperature=0.2)

    Example:
        Original: "A 45-year-old presents with fever. What is the diagnosis?"
        Modified: "A 45-year-old presents with fever. The patient's dog has diabetes
                   and cannot see properly. What is the diagnosis?"
    """

    def __init__(
        self,
        model_id: str,
        model_pool: ModelPool,
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize the distraction sentence strategy.

        Args:
            model_id: Model to use for generating distractions (required)
            model_pool: Shared model pool (required)
            config: Optional generation config (default: temperature=0.2)
        """
        super().__init__(
            name="add_distraction_sentence",
            model_id=model_id,
            model_pool=model_pool,
            system_prompt=DISTRACTION_SYSTEM_PROMPT,
            config=config or GenerationConfig(temperature=0.2, max_tokens=2048)
        )
    
    def apply(self, test_case: TestCase, context: Dict[str, Any]) -> TestCase:
        """
        Add a distracting sentence to the question.
        
        Args:
            test_case: Original test case
            context: Additional context (unused)
            
        Returns:
            Modified test case with distraction sentence inserted
        """
        modified = copy.deepcopy(test_case)
        
        # Validate we have options and correct answer
        if not modified.options or not modified.correct_answer:
            print(f"[WARNING] Cannot {self.name}: missing options or correct_answer")
            return modified
        
        correct_labels = parse_answer_label_set(
            modified.correct_answer,
            valid_labels=modified.options.keys()
        )
        wrong_choices = {
            label: text for label, text in modified.options.items()
            if label not in correct_labels
        }
        
        if not wrong_choices:
            print(f"[WARNING] Cannot {self.name}: no wrong choices available")
            return modified
        
        # Randomly pick one wrong choice to reference
        distractor_label = random.choice(list(wrong_choices.keys()))
        distractor_text = wrong_choices[distractor_label]
        print(f"[INFO] Chosen distractor item: {distractor_label}: {distractor_text}")
        
        # Build prompts using TestCase object directly
        user_prompt = build_distraction_user_prompt(
            test_case=modified,
            distractor_label=distractor_label
        )
        
        # Generate distraction sentence using LLM
        response = self.model.generate(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
            config=self.config
        )
        
        distraction_sentence = response.final_answer.strip()
        print(f"[INFO] Generated distraction: {distraction_sentence}")
        
        # Insert distraction into question
        updated_question = self._insert_distraction(
            question_stem=modified.question,
            distraction=distraction_sentence
        )
        
        # Update the modified test case
        modified.question = updated_question

        # Update metadata
        update_attack_metadata(modified, self.name, {
            "distraction_added": distraction_sentence,
            "distractor_label": distractor_label,
            "distractor_text": distractor_text,
            "attacker_model": self.model_id
        })

        return modified
    
    _COMMON_ABBREVIATIONS = frozenset({
        "approx",
        "dept",
        "dr",
        "e.g",
        "etc",
        "fig",
        "i.e",
        "inc",
        "jr",
        "mg",
        "ml",
        "mr",
        "mrs",
        "ms",
        "no",
        "prof",
        "pt",
        "pts",
        "sr",
        "st",
        "vs",
    })

    @classmethod
    def _is_safe_sentence_boundary(
        cls,
        text: str,
        start: int,
        end: int,
    ) -> bool:
        """Return whether punctuation at ``start:end`` ends a sentence."""
        punctuation = text[start:end]
        if "." not in punctuation:
            return True

        previous = text[start - 1] if start else ""
        following = text[end] if end < len(text) else ""
        if previous.isdigit() or following.isdigit():
            return False

        prefix = text[:start].rstrip()
        token_match = re.search(r"([A-Za-z](?:[A-Za-z.]*)?)$", prefix)
        if not token_match:
            return True

        token = token_match.group(1)
        normalized = token.casefold().rstrip(".")
        if "." in token or normalized in cls._COMMON_ABBREVIATIONS:
            return False
        if len(token) == 1 and (
            token_match.start() == 0
            or text[token_match.start() - 1].isspace()
        ):
            return False
        return True

    @classmethod
    def _find_safe_boundary(
        cls,
        text: str,
        before_index: int,
    ) -> Optional[int]:
        """Find the last conservative sentence boundary before an index."""
        candidates = list(
            re.finditer(r"[.!?]+(?=\s+[A-Z])", text[:before_index])
        )
        for candidate in reversed(candidates):
            if cls._is_safe_sentence_boundary(
                text,
                candidate.start(),
                candidate.end(),
            ):
                return candidate.end()
        return None

    def _insert_distraction(self, question_stem: str, distraction: str) -> str:
        """Insert distraction without rewriting the original question text."""
        original = str(question_stem or "").strip()
        distraction = str(distraction or "").strip()
        if not original:
            return distraction
        if not distraction:
            return original

        final_question = original.rfind("?")
        search_end = final_question if final_question >= 0 else len(original)
        boundary = self._find_safe_boundary(original, search_end)
        if boundary is None:
            # Conservative fallback: add context before the question rather than
            # risking a modification to a decimal, abbreviation, or initial.
            return f"{distraction} {original}"

        suffix = original[boundary:]
        separator = "" if not suffix or suffix[0].isspace() else " "
        return original[:boundary] + " " + distraction + separator + suffix


class GenerateDistractorOptionsStrategy(AttackStrategy):
    """
    Uses an LLM to generate additional plausible but incorrect options.

    This increases the difficulty by adding more choices. The LLM generates
    medically plausible distractors that are closely related to the correct
    answer but still incorrect.

    Key features:
    - Generates specified number of new distractor options
    - Ensures distractors are plausible but incorrect
    - Avoids parent/child relationships with correct answer
    - Parses numbered list format from LLM response
    - Appends new options with sequential letter labels

    Attributes:
        name: Always "generate_distractor_options"
        model_id: Model ID for generating distractors
        model: Model instance from pool
        system_prompt: System instructions (set to DISTRACTOR_OPTIONS_SYSTEM_PROMPT)
        config: Generation configuration (default: temperature=0.2)
        num_distractors: Number of new distractor options to generate

    Example:
        Original: A, B, C, D (4 options)
        Modified: A, B, C, D, E, F, G (7 options with 3 new distractors)
    """

    def __init__(
        self,
        model_id: str,
        model_pool: ModelPool,
        num_distractors: int = 3,
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize the distractor options generation strategy.

        Args:
            model_id: Model to use for generating distractors (required)
            model_pool: Shared model pool (required)
            num_distractors: Number of new distractor options to generate
            config: Optional generation config (default: temperature=0.2)
        """
        super().__init__(
            name="generate_distractor_options",
            model_id=model_id,
            model_pool=model_pool,
            system_prompt=DISTRACTOR_OPTIONS_SYSTEM_PROMPT,
            config=config or GenerationConfig(temperature=0.2, max_tokens=2048)
        )
        self.num_distractors = num_distractors

    def to_dict(self) -> Dict[str, Any]:
        """Export strategy configuration for logging."""
        strategy_dict = super().to_dict()
        strategy_dict["num_distractors"] = self.num_distractors
        return strategy_dict

    def apply(self, test_case: TestCase, context: Dict[str, Any]) -> TestCase:
        """
        Generate and add distractor options to the test case.
        
        Args:
            test_case: Original test case
            context: Additional context (unused)
            
        Returns:
            Modified test case with additional distractor options
        """
        modified = copy.deepcopy(test_case)
        
        # Validate we have options
        if not modified.options:
            print(f"[WARNING] Cannot {self.name}: missing options")
            return modified
        
        # Build prompt using TestCase object directly
        user_prompt = build_distractor_options_user_prompt(
            test_case=modified,
            num_distractors=self.num_distractors
        )
        
        # Generate distractors using LLM
        response = self.model.generate(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
            config=self.config
        )
        
        print(f"[INFO] LLM response for distractors:\n{response.final_answer}")
        
        # Parse distractors from response
        distractors = self._parse_distractors(response.final_answer)
        
        if len(distractors) == 0:
            print(f"[WARNING] No distractors parsed from response")
            update_attack_metadata(modified, self.name, {
                "manipulation_failed": True,
                "failure_category": "parser_error",
                "reason": "No numbered distractors could be parsed from attacker output",
                "raw_response": (response.final_answer or "")[:500],
            })
            return modified
        
        print(f"[INFO] Parsed {len(distractors)} distractor(s)")
        
        # Add distractors as new options
        existing_labels = sorted(modified.options.keys())
        max_key_char = existing_labels[-1]
        next_key_ord = ord(max_key_char) + 1
        
        added_labels = []
        added_distractors = []

        for distractor in distractors[:self.num_distractors]:
            new_label = chr(next_key_ord)
            modified.options[new_label] = distractor
            added_labels.append(new_label)
            added_distractors.append(distractor)
            next_key_ord += 1
            print(f"[INFO] Added distractor: {new_label}) {distractor}")

        # Update metadata
        update_attack_metadata(modified, self.name, {
            "distractors_added": added_distractors,
            "options_added": added_labels,
            "attacker_model": self.model_id,
            "num_distractors_requested": self.num_distractors
        })

        return modified
    
    def _parse_distractors(self, text: str) -> List[str]:
        """
        Parse numbered list of distractors from LLM response.
        
        Supports formats:
        - "1) Distractor text"
        - "1. Distractor text"
        
        Args:
            text: LLM response text
            
        Returns:
            List of parsed distractor strings
        """
        lines = text.split("\n")
        distractors = []
        
        for line in lines:
            line = line.strip()
            # Match lines starting with digit followed by ) or .
            if line and len(line) > 0:
                # Check if starts with digit and has ) or . within first 4 chars
                if line[0].isdigit() and (")" in line[:4] or "." in line[:4]):
                    # Extract text after the first ) or .
                    if ")" in line:
                        distractor_text = line.split(")", 1)[1].strip()
                    elif "." in line:
                        distractor_text = line.split(".", 1)[1].strip()
                    else:
                        continue
                    
                    if distractor_text:  # Only add non-empty distractors
                        distractors.append(distractor_text)
        
        return distractors



class BiasManipulationStrategy(AttackStrategy):
    """
    Uses an LLM to introduce cognitive biases into the question.

    This strategy injects bias-inducing sentences that reference incorrect
    options, making the question more deceptive by exploiting common cognitive
    biases in medical reasoning.

    Supported bias types:
    - Self-diagnosis: Patient convinced of wrong answer
    - Recency: Recent case suggesting wrong answer
    - Confirmation: Prior experience supporting wrong answer
    - Frequency: Statistical bias toward wrong answer
    - Cultural: Cultural beliefs favoring wrong answer
    - Status quo: Institutional practice of wrong answer
    - False-consensus: Colleagues agreeing on wrong answer
    - Defensive-medicine: Caution leading to wrong answer

    Key features:
    - Selects specified number of bias styles (1-8)
    - Generates contextually appropriate bias sentences
    - Integrates bias naturally into question text
    - References a wrong answer to mislead testee

    Attributes:
        name: Always "bias_manipulation"
        model_id: Model ID for generating biased questions
        model: Model instance from pool
        system_prompt: System instructions (built from BIAS_STYLES)
        config: Generation configuration (default: temperature=0.2)
        n_bias_styles: Number of bias styles to inject (1-8)

    Example:
        Original: "A patient presents with symptoms. What is the diagnosis?"
        Modified: "A patient presents with symptoms. Recently, there was a patient
                   with similar symptoms diagnosed with [wrong answer]. What is the diagnosis?"
    """

    def __init__(
        self,
        model_id: str,
        model_pool: ModelPool,
        n_bias_styles: int = 1,
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize the bias manipulation strategy.

        Args:
            model_id: Model to use for generating biased questions (required)
            model_pool: Shared model pool (required)
            n_bias_styles: Number of bias styles to inject (1-8)
            config: Optional generation config (default: temperature=0.2)

        Raises:
            ValueError: If n_bias_styles is not between 1 and 8
        """
        # Validate n_bias_styles before calling super
        if not (1 <= n_bias_styles <= 8):
            raise ValueError("n_bias_styles must be between 1 and 8")

        super().__init__(
            name="bias_manipulation",
            model_id=model_id,
            model_pool=model_pool,
            system_prompt=build_bias_manipulation_system_prompt(n_bias_styles),
            config=config or GenerationConfig(temperature=0.2, max_tokens=2048)
        )
        self.n_bias_styles = n_bias_styles

    def to_dict(self) -> Dict[str, Any]:
        """Export strategy configuration for logging."""
        strategy_dict = super().to_dict()
        strategy_dict["n_bias_styles"] = self.n_bias_styles
        return strategy_dict

    def apply(self, test_case: TestCase, context: Dict[str, Any]) -> TestCase:
        """
        Introduce cognitive bias into the question.
        
        Args:
            test_case: Original test case
            context: Additional context (can specify 'incorrect_option_label')
            
        Returns:
            Modified test case with bias-inducing sentences inserted
        """
        modified = copy.deepcopy(test_case)
        
        # Validate we have options
        if not modified.options or not modified.correct_answer:
            print(f"[WARNING] Cannot {self.name}: missing options or correct_answer")
            return modified
        
        correct_labels = parse_answer_label_set(
            modified.correct_answer,
            valid_labels=modified.options.keys()
        )
        wrong_choices = {
            label: text for label, text in modified.options.items()
            if label not in correct_labels
        }
        
        if not wrong_choices:
            print(f"[WARNING] Cannot {self.name}: no wrong choices available")
            return modified
        
        # Get incorrect option (from context or random)
        incorrect_option_label = context.get('incorrect_option_label')
        if incorrect_option_label and incorrect_option_label in wrong_choices:
            incorrect_option_text = wrong_choices[incorrect_option_label]
        else:
            incorrect_option_label = random.choice(list(wrong_choices.keys()))
            incorrect_option_text = wrong_choices[incorrect_option_label]
        
        print(f"[INFO] Using incorrect option for bias: {incorrect_option_label}) {incorrect_option_text}")

        # Build user prompt
        user_prompt = build_bias_manipulation_user_prompt(
            test_case=modified,
            incorrect_option_text=incorrect_option_text
        )

        # Generate biased version
        response = self.model.generate(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
            config=self.config
        )
        
        
        # Parse JSON response
        try:
            parsed_response = self._parse_json_response(response.final_answer)

            print(f"[INFO] LLM parsed response for {self.name}:\n{parsed_response}")

            bias_styles_used = parsed_response["bias_styles"]
            modified_question = parsed_response["modified_question"]
            
            # Validate number of styles
            if len(bias_styles_used) != self.n_bias_styles:
                print(f"[WARNING] Expected {self.n_bias_styles} bias styles, got {len(bias_styles_used)}")
            
            # Update test case
            modified.question = modified_question

            # Update metadata
            update_attack_metadata(modified, self.name, {
                "bias_styles_used": bias_styles_used,
                "n_bias_styles": len(bias_styles_used),
                "incorrect_option_label": incorrect_option_label,
                "incorrect_option_text": incorrect_option_text,
                "original_question": test_case.question,
                "attacker_model": self.model_id
            })

            print(f"[INFO] Successfully applied {len(bias_styles_used)} bias style(s): {bias_styles_used}")
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[ERROR] Failed to parse bias manipulation response: {e}")
            print(f"[ERROR] Raw response: {response.final_answer}")
            update_attack_metadata(modified, self.name, {
                "manipulation_failed": True,
                "failure_category": "parser_error",
                "reason": f"parse error: {str(e)}",
            })
            return modified
        
        return modified
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM using shared utility.
        
        Raises json.JSONDecodeError if response parsing fails.
        """
        return parse_json_response(response_text)


class InvertQuestionAnswerStrategy(AttackStrategy):
    """
    Uses an LLM to invert the final question sentence and complement the answer.

    This strategy identifies the final question sentence (e.g., "Which is most likely?")
    and inverts its meaning (e.g., "Which is not most likely?"). It then complements
    the answer set - if the original correct answer was "A", the new correct answer
    becomes all options except "A" (i.e., "B,C,D").

    Key features:
    - Inverts common patterns: "most likely" → "not most likely"
    - Handles singular vs. plural adjustments
    - Complements the answer set automatically
    - Detects manipulation failures (unchanged questions)
    - Preserves all background context

    Examples of inversions:
    - "most likely" → "not most likely"
    - "most appropriate" → "not most appropriate"
    - "true" → "false"
    - "correct" → "incorrect"
    - "best" → "not best"

    Attributes:
        name: Always "invert_question_answer"
        model_id: Model ID for generating inverted questions
        model: Model instance from pool
        system_prompt: System instructions (set to INVERT_QUESTION_SYSTEM_PROMPT)
        config: Generation configuration (default: temperature=0.2)

    Example:
        Original: "Which treatment is most appropriate?" Answer: A
        Inverted: "Which treatment is not most appropriate?" Answer: B,C,D

    This is particularly effective for questions where the inverted version
    creates a valid multi-answer scenario.
    """

    def __init__(
        self,
        model_id: str,
        model_pool: ModelPool,
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize the invert question answer strategy.

        Args:
            model_id: Model to use for generating inverted questions (required)
            model_pool: Shared model pool (required)
            config: Optional generation config (default: temperature=0.2)
        """
        super().__init__(
            name="invert_question_answer",
            model_id=model_id,
            model_pool=model_pool,
            system_prompt=INVERT_QUESTION_SYSTEM_PROMPT,
            config=config or GenerationConfig(temperature=0.2, max_tokens=2048)
        )
    
    def apply(self, test_case: TestCase, context: Dict[str, Any]) -> TestCase:
        """
        Invert the final question sentence and complement the answer.
        
        Args:
            test_case: Original test case
            context: Additional context (unused)
            
        Returns:
            Modified test case with inverted question and complemented answer,
            or original test case if manipulation failed
        """
        modified = copy.deepcopy(test_case)
        
        # Validate we have options and correct answer
        if not modified.options or not modified.correct_answer:
            print(f"[WARNING] Cannot {self.name}: missing options or correct_answer")
            return modified
        
        original_question = modified.question
        original_correct_answer = modified.correct_answer
        
        # Build prompts using TestCase directly
        user_prompt = build_invert_question_user_prompt(test_case=modified)
        
        # Generate inverted version
        response = self.model.generate(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
            config=self.config
        )
        
        # Parse JSON response
        try:
            parsed_response = self._parse_json_response(response.final_answer)

            
            print(f"[INFO] LLM parsed response for {self.name}:\n{parsed_response}")

            modified_sentence = parsed_response.get("modified_sentence", "")
            entire_question = parsed_response.get("entire_question", original_question)
            
            # Check if manipulation succeeded
            if entire_question.strip() == original_question.strip():
                print("[INFO] Invert manipulation failed: entire question unchanged.")
                update_attack_metadata(modified, self.name, {
                    "manipulation_failed": True,
                    "reason": "entire question unchanged"
                })
                return modified

            # Update question
            modified.question = entire_question

            print(f"[INFO] Modified final sentence: {modified_sentence}")

            # Complement the full answer set, including multi-answer inputs.
            original_labels = parse_answer_label_set(
                original_correct_answer,
                valid_labels=modified.options.keys()
            )
            complement_labels = complement_answer_label_set(
                modified.options,
                original_correct_answer
            )
            if not complement_labels:
                update_attack_metadata(modified, self.name, {
                    "manipulation_failed": True,
                    "failure_category": "precondition_failed",
                    "reason": "inversion produced an empty answer complement",
                })
                return modified

            new_correct_answer = format_answer_label_set(
                complement_labels,
                option_order=modified.options.keys()
            )
            new_answer_text = (
                "All choices except "
                + format_answer_label_set(original_labels, modified.options.keys())
            )

            # Update answer
            #  original_correct_answer = modified.correct_answer
            modified.correct_answer = new_correct_answer

            # Update metadata
            update_attack_metadata(modified, self.name, {
                "original_question": original_question,
                "modified_sentence": modified_sentence,
                "original_correct_answer": original_correct_answer,
                "new_correct_answer": new_correct_answer,
                "new_answer_text": new_answer_text,
                "complement_labels": [
                    label for label in modified.options if label in complement_labels
                ],
                "attacker_model": self.model_id,
                "manipulation_failed": False
            })

            print(f"[INFO] Answer complemented: {original_correct_answer} → {new_correct_answer}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[ERROR] Failed to parse inversion response: {e}")
            print(f"[ERROR] Raw response: {response.final_answer}")
            update_attack_metadata(modified, self.name, {
                "manipulation_failed": True,
                "failure_category": "parser_error",
                "reason": f"parse error: {str(e)}"
            })
            return modified
        
        return modified
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM using shared utility.
        
        Raises json.JSONDecodeError if response parsing fails.
        """
        return parse_json_response(response_text)

class AdjustImpossibleMeasurementStrategy(AttackStrategy):
    """
    Uses an LLM to detect and modify numerical measurements to impossible values.

    This strategy identifies numeric measurements in medical questions (vitals, labs,
    doses, etc.) and changes exactly one to an extreme, impossible value. It then adds
    a new option stating there is false/impossible information, making this the correct
    answer.

    Key features:
    - Detects measurements beyond basic demographics (age, weight, height)
    - Changes exactly one measurement to impossible value
    - Adds new option: "There is false/impossible information in the text"
    - Updates correct answer to the new option
    - Adds note to question text guiding to impossible info option

    Examples of impossible values:
    - Temperature: 38.4°C → 48°C
    - Blood pressure: 125/70 → 490/290 mmHg
    - Heart rate: 105 bpm → 300 bpm
    - Respirations: 18/min → 128/min
    - Platelet count: 200×10^9/L → 3000×10^9/L
    - Medicine dose: 50mg/day → 100g/day

    Attributes:
        name: Always "adjust_impossible_measurement"
        model_id: Model ID for detecting and adjusting measurements
        model: Model instance from pool
        system_prompt: System instructions (set to ADJUST_MEASUREMENT_SYSTEM_PROMPT)
        config: Generation configuration (default: temperature=0.2)

    Example:
        Original: "Patient has BP 120/80, HR 72. What is the diagnosis?"
        Modified: "Patient has BP 490/290, HR 72. What is the diagnosis?
                   (Note: If there is false/impossible information...)"
        New Option F: "There is false/impossible information in the text"
        New Answer: F

    This strategy is particularly effective for testing whether models can detect
    physiologically impossible values and recognize data quality issues.
    """

    def __init__(
        self,
        model_id: str,
        model_pool: ModelPool,
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize the adjust impossible measurement strategy.

        Args:
            model_id: Model to use for detecting and adjusting measurements (required)
            model_pool: Shared model pool (required)
            config: Optional generation config (default: temperature=0.2)
        """
        super().__init__(
            name="adjust_impossible_measurement",
            model_id=model_id,
            model_pool=model_pool,
            system_prompt=ADJUST_MEASUREMENT_SYSTEM_PROMPT,
            config=config or GenerationConfig(temperature=0.2, max_tokens=2048)
        )
    
    def apply(self, test_case: TestCase, context: Dict[str, Any]) -> TestCase:
        """
        Detect and adjust measurements to impossible values.
        
        Args:
            test_case: Original test case
            context: Additional context (unused)
            
        Returns:
            Modified test case with impossible measurement and new option,
            or original test case if no measurements found
        """
        modified = copy.deepcopy(test_case)
        
        # Validate we have options
        if not modified.options:
            print(f"[WARNING] Cannot {self.name}: missing options")
            return modified
        
        original_question = modified.question
        
        # Build prompts using TestCase directly
        user_prompt = build_adjust_measurement_user_prompt(test_case=modified)
        
        # Generate modified version with impossible measurement
        response = self.model.generate(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
            config=self.config
        )
        
        # Parse JSON response
        try:
            parsed_response = self._parse_json_response(response.final_answer)

            print(f"[INFO] LLM parsed response for {self.name}:\n{parsed_response}")

            updated_question = parsed_response.get("updated_question", original_question)
            changed_measurement = parsed_response.get("changed_measurement", "")
            
            # Check if measurement was actually changed
            if updated_question == original_question or not changed_measurement:
                print("[INFO] No measurement changed - no numeric measurements found beyond age")
                update_attack_metadata(modified, self.name, {
                    "measurement_changed": False,
                    "changed_measurement": "",
                    "reason": "no numeric measurements found beyond age",
                    "manipulation_failed": True
                })
                return modified

            # Measurement was changed - update question and add new option
            print(f"[INFO] Impossible measurement changed: {changed_measurement}")

            # Add note to question
            modified.question = (
                updated_question +
                " (Note: If there is false/impossible information in the text, "
                "only choose the option that says so.)"
            )

            # Add new "impossible info" option
            updated_options = modified.options.copy()
            existing_labels = sorted(updated_options.keys())
            last_label = existing_labels[-1]
            next_label = chr(ord(last_label) + 1)

            impossible_info_option = "There is false/impossible information in the text."
            updated_options[next_label] = impossible_info_option

            # Update options and correct answer
            original_correct_answer = modified.correct_answer
            modified.options = updated_options
            modified.correct_answer = next_label

            # Update metadata
            update_attack_metadata(modified, self.name, {
                "measurement_changed": True,
                "changed_measurement": changed_measurement,
                "original_question": original_question,
                "original_correct_answer": original_correct_answer,
                "new_correct_answer": next_label,
                "impossible_info_option": impossible_info_option,
                "attacker_model": self.model_id,
                "manipulation_failed": False
            })

            print(f"[INFO] New option added: {next_label}) {impossible_info_option}")
            print(f"[INFO] Answer changed: {original_correct_answer} → {next_label}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[ERROR] Failed to parse measurement adjustment response: {e}")
            print(f"[ERROR] Raw response: {response.final_answer}")
            update_attack_metadata(modified, self.name, {
                "measurement_changed": False,
                "reason": f"parse error: {str(e)}",
                "manipulation_failed": True,
                "failure_category": "parser_error",
            })
            return modified
        
        return modified
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM using shared utility.
        
        Raises json.JSONDecodeError if response parsing fails.
        """
        return parse_json_response(response_text)
