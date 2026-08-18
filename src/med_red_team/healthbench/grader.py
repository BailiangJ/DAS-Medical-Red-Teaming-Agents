"""
RubricGrader for evaluating completions against rubric items.
"""

from typing import Dict, List, Any, Optional
from med_red_team.models import GenerationConfig
from med_red_team.model_pool import ModelPool
from med_red_team.grader import Grader
from med_red_team.utils import parse_json_response
from med_red_team.healthbench.data import RubricItem, RubricGradeResult
from med_red_team.healthbench.prompts import HEALTHBENCH_GRADER_USER_PROMPT, HEALTHBENCH_GRADER_SYSTEM_PROMPT


class RubricGrader(Grader):
    """
    Grader for evaluating completions against rubric items.
    
    Extends the Grader abstract class to perform rubric-based evaluation.
    Uses an LLM to assess whether responses meet specific rubric criteria.
    """
    
    def __init__(
        self,
        model_id: str,
        model_pool: ModelPool,
        config: Optional[GenerationConfig] = None,
        max_retries: int = 3,
        verbose: bool = False
    ):
        """
        Initialize RubricGrader.
        
        Args:
            model_id: Model identifier (e.g., "gpt-4.1-2025-04-14")
            model_pool: ModelPool instance for efficient model reuse
            config: Optional GenerationConfig (uses default if None)
            max_retries: Maximum number of retries for parsing JSON
            verbose: Whether to print verbose output
        """
        super().__init__(
            name="healthbench_grader",
            model_id=model_id,
            model_pool=model_pool,
            system_prompt=HEALTHBENCH_GRADER_SYSTEM_PROMPT,
            config=config or GenerationConfig(temperature=0.0, max_tokens=4096)
        )
        self.max_retries = max_retries
        self.verbose = verbose
    
    def _format_conversation(self, conversation: List[Dict[str, str]], completion: str) -> str:
        """Format conversation including the completion into a string."""
        full_conversation = conversation + [{"role": "assistant", "content": completion}]
        return "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in full_conversation])
    
    def _grade_single_rubric(
        self,
        conversation: List[Dict[str, str]],
        completion: str,
        rubric_item: RubricItem,
    ) -> RubricGradeResult:
        """
        Grade a completion against a single rubric item.
        
        Args:
            conversation: The conversation history (list of messages)
            completion: The model's completion to evaluate
            rubric_item: The rubric item to evaluate against
            
        Returns:
            RubricGradeResult with criteria_met and explanation
        """
        # Format the conversation with completion
        convo_str = self._format_conversation(conversation, completion)
        
        # Create user prompt by filling in the template
        user_prompt = HEALTHBENCH_GRADER_USER_PROMPT.replace(
            "<<conversation>>", convo_str
        ).replace("<<rubric_item>>", str(rubric_item))
        
        # Query grader model with retry logic
        for attempt in range(self.max_retries):
            # Call grader model using the proper interface
            grading_response = self.model.generate(
                user_prompt=user_prompt,
                system_prompt=self.system_prompt,
                config=self.config
            )
            
            # Extract the final answer (handles CoT if present)
            grading_text = grading_response.final_answer

            # Parse JSON response using common utility
            grading_dict = parse_json_response(
                response_text=grading_text,
                expected_fields={"criteria_met": bool, "explanation": str},
                fallback={},
                verbose=self.verbose
            )

            # Validate response
            if "criteria_met" in grading_dict:
                criteria_met = grading_dict["criteria_met"]
                if criteria_met is True or criteria_met is False:
                    explanation = grading_dict.get("explanation", "No explanation provided")

                    if self.verbose:
                        print(f"✓ Rubric grading successful (attempt {attempt + 1})")

                    return RubricGradeResult(
                        rubric_item=rubric_item,
                        criteria_met=criteria_met,
                        explanation=explanation,
                    )

            if self.verbose:
                print(f"⚠ Grading failed due to bad JSON output (attempt {attempt + 1}/{self.max_retries}), retrying...")
        
        # If all retries failed, return an explicit invalid grading result.
        print(f"❌ ERROR: Failed to parse grading response after {self.max_retries} attempts")
        return RubricGradeResult(
            rubric_item=rubric_item,
            criteria_met=None,
            explanation="ERROR: Failed to parse grading response",
        )
    
    def grade(
        self,
        conversation: List[Dict[str, str]],
        completion: str,
        rubric_items: List[RubricItem],
    ) -> List[RubricGradeResult]:
        """
        Grade a completion against all rubric items.
        
        This is the main grading method that matches the Grader interface pattern.
        
        Args:
            conversation: The conversation history
            completion: The model's completion to evaluate
            rubric_items: List of rubric items to evaluate against
            
        Returns:
            List of RubricGradeResult, one per rubric item
        """
        results = []
        for i, rubric_item in enumerate(rubric_items):
            if self.verbose:
                print(f"Grading rubric {i+1}/{len(rubric_items)}: {rubric_item.criterion[:50]}...")
            
            result = self._grade_single_rubric(conversation, completion, rubric_item)
            results.append(result)
        
        return results
