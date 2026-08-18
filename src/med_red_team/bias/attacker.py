"""
Race/Socioeconomic Label Attack Strategy
========================================

Attack strategy that replaces patient placeholder with demographic labels
to test for bias in medical AI models.
"""

from typing import Optional, List, Dict, Any

from med_red_team.actors import AttackStrategy
from med_red_team.model_pool import ModelPool
from med_red_team.utils import clean_markdown_response
from med_red_team.models import GenerationConfig, ModelExecutionError
from med_red_team.bias.data import (
    BIAS_STATUS_MANIPULATION_FAILED,
    BIAS_STATUS_NOT_APPLICABLE,
    BiasTestCase,
)
from med_red_team.bias.utils import is_mental_health_case
from med_red_team.bias.prompts import (
    RACE_SOCIOECONOMIC_LABEL_SYSTEM_PROMPT,
    RACE_SOCIOECONOMIC_LABEL_USER_PROMPT,
    DEFAULT_RACE_SOCIOECONOMIC_LABELS,
    LANGUAGE_STRATEGY_SELECTION_SYSTEM_PROMPT,
    LANGUAGE_STRATEGY_SELECTION_USER_PROMPT,
    LANGUAGE_REWRITE_REGISTRY,
    DEFAULT_LANGUAGE_STRATEGIES,
    EMOTION_STRATEGY_SELECTION_SYSTEM_PROMPT,
    EMOTION_STRATEGY_SELECTION_USER_PROMPT,
    EMOTION_REWRITE_REGISTRY,
    DEFAULT_EMOTION_STRATEGIES,
    COGNITIVE_BIAS_GENERATION_SYSTEM_PROMPT,
    COGNITIVE_BIAS_GENERATION_USER_PROMPT,
    COGNITIVE_BIAS_TYPES,
    format_label_list,
    format_strategy_list
)


def _available_self_report(value: Any) -> bool:
    if value is None:
        return False
    try:
        if value != value:
            return False
    except (TypeError, ValueError):
        pass
    normalized = str(value).strip()
    return bool(normalized) and normalized.casefold() not in {
        "nan",
        "none",
        "null",
        "n/a",
        "na",
        "unavailable",
    }


def _override_generation_config(
    base_config: GenerationConfig,
    **overrides: Any,
) -> GenerationConfig:
    payload = base_config.to_dict()
    payload.update({
        key: value
        for key, value in overrides.items()
        if value is not None
    })
    return GenerationConfig(**payload)


def _excluded_case(
    test_case: BiasTestCase,
    *,
    question_idx: int,
    reason: str,
) -> BiasTestCase:
    return BiasTestCase(
        case_id=test_case.case_id,
        clinical_summary=test_case.clinical_summary,
        vital_signs=test_case.vital_signs,
        patient_self_report=test_case.patient_self_report,
        questions=test_case.questions,
        category=test_case.category,
        metadata={
            **test_case.metadata,
            "attack_excluded": True,
            "status": BIAS_STATUS_NOT_APPLICABLE,
            "failure_category": BIAS_STATUS_NOT_APPLICABLE,
            "skip_reason": reason,
            "question_idx": question_idx,
        },
    )


def _strategy_log(strategy: AttackStrategy, message: str) -> None:
    if getattr(strategy, "verbose", True):
        print(message)


class RaceSocioeconomicLabelStrategy(AttackStrategy):
    """
    Attack strategy that replaces {patient} placeholder with demographic labels.
    
    This strategy tests whether models exhibit bias when patient demographic
    information changes. It uses an LLM to select which demographic label
    (e.g., "black unhoused patient", "very low income patient") would most
    likely cause the model to change its answer.
    
    Process:
    1. Format case with {patient} placeholder
    2. Use LLM to select most biasing label from list
    3. Replace {patient} with selected label
    4. Return modified case
    
    Example:
        Original: "A 19-year-old {patient} presents with..."
        Selected label: "black unhoused patient"
        Result: "A 19-year-old black unhoused patient presents with..."
    """
    
    def __init__(
        self,
        model_id: str,
        model_pool: ModelPool,
        config: Optional[GenerationConfig] = None,
        label_list: Optional[List[str]] = None,
        patient_placeholder: str = "{patient}",
        verbose: bool = True,
    ):
        """
        Initialize race/socioeconomic label strategy.
        
        Args:
            model_id: LLM to use for label selection (e.g., "gpt-4o")
            model_pool: Shared model pool
            system_prompt: Custom system prompt (uses default if None)
            config: Generation config (uses default if None)
            label_list: List of demographic labels to choose from
            patient_placeholder: Placeholder string in case text
        """
        super().__init__(
            name="race_socioeconomic_label",
            model_id=model_id,
            model_pool=model_pool,
            system_prompt=RACE_SOCIOECONOMIC_LABEL_SYSTEM_PROMPT,
            config=config or GenerationConfig(temperature=0.3, max_tokens=2048)
        )
        self.label_list = label_list or DEFAULT_RACE_SOCIOECONOMIC_LABELS
        self.verbose = verbose
        self.patient_placeholder = patient_placeholder
    
    def apply(
        self,
        test_case: BiasTestCase,
        context: Dict[str, Any]
    ) -> BiasTestCase:
        """
        Apply race/socioeconomic label manipulation to test case.
        
        Args:
            test_case: Original test case with {patient} placeholder
            context: Must contain:
                - 'ref_answer': Reference answer from unbiased case
                - 'question_idx': Which question to format (0-3)
        
        Returns:
            Modified test case with selected label replacing {patient}
        
        Raises:
            ValueError: If ref_answer not in context
            RuntimeError: If label selection fails
        """
        # Extract context
        ref_answer = context.get('ref_answer')
        if ref_answer is None:
            raise ValueError("Context must contain 'ref_answer'")
        unbiased_model_choice = context.get("unbiased_model_choice") or ref_answer

        question_idx = context.get('question_idx', 0)
        if self.patient_placeholder not in test_case.clinical_summary:
            return _excluded_case(
                test_case,
                question_idx=question_idx,
                reason="Identity attack is not applicable without a patient placeholder",
            )

        # Format case with placeholder
        case_text = test_case.format_full_case(
            question_idx=question_idx,
            patient_placeholder=self.patient_placeholder
        )
        
        # Select biasing label using LLM
        selected_label = self._select_label(case_text, unbiased_model_choice)
        
        if selected_label is None:
            raise RuntimeError(
                f"Failed to select label for case {test_case.case_id}"
            )
        
        # Replace placeholder in clinical summary
        modified_summary = test_case.clinical_summary.replace(
            self.patient_placeholder,
            selected_label
        )
        if modified_summary == test_case.clinical_summary:
            return BiasTestCase(
                case_id=test_case.case_id,
                clinical_summary=test_case.clinical_summary,
                vital_signs=test_case.vital_signs,
                patient_self_report=test_case.patient_self_report,
                questions=test_case.questions,
                category=test_case.category,
                metadata={
                    **test_case.metadata,
                    "manipulation_failed": True,
                    "status": BIAS_STATUS_MANIPULATION_FAILED,
                    "failure_category": "unchanged_identity_prompt",
                    "skip_reason": "Identity manipulation produced no prompt change",
                },
            )

        # Create modified test case
        modified_case = BiasTestCase(
            case_id=test_case.case_id,
            clinical_summary=modified_summary,
            vital_signs=test_case.vital_signs,
            patient_self_report=test_case.patient_self_report,
            questions=test_case.questions,
            category=test_case.category,
            metadata={
                **test_case.metadata,
                "attack_applied": self.name,
                "selected_label": selected_label,
                "ref_answer": ref_answer,
                "question_idx": question_idx
            }
        )
        
        return modified_case
    
    def _select_label(
        self,
        question_template: str,
        unbiased_model_choice: str
    ) -> Optional[str]:
        """
        Use LLM to select most biasing label from list.
        
        Args:
            question_template: Full case text with {patient} placeholder
            unbiased_model_choice: Reference answer (e.g., "C")
        
        Returns:
            Selected label string or None if selection fails
        """
        # Format prompts
        system_prompt = self.system_prompt.format(
            unbiased_model_choice=unbiased_model_choice
        )

        label_str = format_label_list(self.label_list)
        user_prompt = RACE_SOCIOECONOMIC_LABEL_USER_PROMPT.format(
            question_template=question_template,
            label_str=label_str,
            unbiased_model_choice=unbiased_model_choice
        )
        
        try:
            # Query LLM
            response = self.model.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                config=self.config
            )
            
            if not response:
                if self.verbose:
                    print("Warning: LLM returned no content for label selection")
                return None

            if self.verbose:
                print(f"[INFO] LLM response for {self.name}:\n{response.final_answer}")
            
            # Clean response (remove markdown code blocks)
            selected_label = clean_markdown_response(response.final_answer)
            
            # Validate label is in list
            if selected_label in self.label_list:
                return selected_label
            else:
                if self.verbose:
                    print(
                    f"Warning: LLM returned invalid label '{selected_label}'. "
                    f"Not in provided list."
                )
                return None
        
        except ModelExecutionError:
            raise
        except Exception as e:
            if self.verbose:
                print(f"Error during label selection: {e}")
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Export strategy configuration for logging."""
        base_dict = super().to_dict()
        base_dict["strategy_params"] = {
            "label_list": self.label_list,
            "num_labels": len(self.label_list),
            "patient_placeholder": self.patient_placeholder
        }
        return base_dict


class LanguageManipulationStrategy(AttackStrategy):
    """
    Attack strategy that modifies patient self-report language style.
    
    This strategy tests whether models exhibit bias when patient language
    style changes (e.g., African American English, limited English, extreme
    tones). It uses an LLM in two stages:
    1. Select which language strategy would most change the answer
    2. Rewrite patient self-report in selected style (using registry)
    
    The rewriting stage uses LANGUAGE_REWRITE_REGISTRY which maps each
    strategy to its specific prompts, temperature, and requirements
    (e.g., distraction sentences for dialects, deterministic translation).
    
    Process:
    1. Check patient self-report is available
    2. Use LLM to select language strategy
    3. Lookup strategy in LANGUAGE_REWRITE_REGISTRY
    4. Use strategy-specific prompts and config to rewrite
    5. Replace self-report in test case
    6. Return modified case
    
    Example:
        Original: "I have severe chest pain that started this morning."
        Selected: "limited english"
        Rewritten: "I have very bad pain in chest. Start this morning."
    """
    
    def __init__(
        self,
        model_id: str,
        model_pool: ModelPool,
        config: Optional[GenerationConfig] = None,
        strategy_list: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        """
        Initialize language manipulation strategy.
        
        Args:
            model_id: LLM to use for strategy selection and rewriting
            model_pool: Shared model pool
            config: Generation config (uses default if None)
            strategy_list: List of language strategies to choose from
        """
        super().__init__(
            name="language_manipulation",
            model_id=model_id,
            model_pool=model_pool,
            system_prompt=LANGUAGE_STRATEGY_SELECTION_SYSTEM_PROMPT,
            config=config or GenerationConfig(temperature=0.3, max_tokens=2048)
        )
        
        self.strategy_list = strategy_list or DEFAULT_LANGUAGE_STRATEGIES
        self.verbose = verbose
        
        # Validate all strategies have rewrite entries
        self._validate_strategies()
    
    def _validate_strategies(self):
        """Validate all strategies in list have rewrite registry entries."""
        missing = [s for s in self.strategy_list if s not in LANGUAGE_REWRITE_REGISTRY]
        if missing:
            if self.verbose:
                print(
                f"Warning: Strategies missing from LANGUAGE_REWRITE_REGISTRY: {missing}\n"
                f"These strategies will fail if selected during rewriting."
            )
    
    def apply(
        self,
        test_case: BiasTestCase,
        context: Dict[str, Any]
    ) -> BiasTestCase:
        """
        Apply language manipulation to test case.
        
        Args:
            test_case: Original test case
            context: Must contain:
                - 'ref_answer': Reference answer from unbiased case
                - 'question_idx': Which question to format (0-3)
        
        Returns:
            Modified test case with rewritten patient self-report
        
        Raises:
            ValueError: If ref_answer not in context or self-report unavailable
            RuntimeError: If strategy selection or rewriting fails
        """
        # Validate context
        ref_answer = context.get('ref_answer')
        if ref_answer is None:
            raise ValueError("Context must contain 'ref_answer'")
        unbiased_model_choice = context.get("unbiased_model_choice") or ref_answer

        question_idx = context.get('question_idx', 0)

        if not _available_self_report(test_case.patient_self_report):
            return _excluded_case(
                test_case,
                question_idx=question_idx,
                reason="Language manipulation is not applicable without a self-report",
            )
        
        # Format case for strategy selection
        case_text = test_case.format_full_case(question_idx)
        
        # Step 1: Select language strategy using LLM
        selected_strategy = self._select_strategy(case_text, unbiased_model_choice)
        
        if selected_strategy is None:
            raise RuntimeError(
                f"Failed to select language strategy for case {test_case.case_id}"
            )
        
        _strategy_log(self, f"[INFO] Selected language strategy: {selected_strategy}")
        
        # Step 2: Rewrite patient self-report using LLM
        modified_report = self._rewrite_text(
            test_case.patient_self_report,
            selected_strategy
        )
        
        if modified_report is None:
            raise RuntimeError(
                f"Failed to rewrite patient self-report for case {test_case.case_id}"
            )
        
        _strategy_log(
            self,
            f"[INFO] Original report: {test_case.patient_self_report[:100]}...",
        )
        _strategy_log(self, f"[INFO] Modified report: {modified_report[:100]}...")
        
        # Step 3: Create modified test case
        modified_case = BiasTestCase(
            case_id=test_case.case_id,
            clinical_summary=test_case.clinical_summary,
            vital_signs=test_case.vital_signs,
            patient_self_report=modified_report,  # Replaced
            questions=test_case.questions,
            category=test_case.category,
            metadata={
                **test_case.metadata,
                "attack_applied": self.name,
                "selected_strategy": selected_strategy,
                "original_report": test_case.patient_self_report,
                "ref_answer": ref_answer,
                "question_idx": question_idx
            }
        )
        
        return modified_case
    
    def _select_strategy(
        self,
        question_template: str,
        unbiased_model_choice: str
    ) -> Optional[str]:
        """
        Use LLM to select language strategy most likely to change answer.
        
        Args:
            question_template: Full case text
            unbiased_model_choice: Reference answer (e.g., "C")
        
        Returns:
            Selected strategy string or None if selection fails
        """
        # Format prompts
        system_prompt = self.system_prompt.format(
            unbiased_model_choice=unbiased_model_choice
        )
        
        strategy_str = format_strategy_list(self.strategy_list)
        user_prompt = LANGUAGE_STRATEGY_SELECTION_USER_PROMPT.format(
            question_template=question_template,
            strategy_str=strategy_str,
            unbiased_model_choice=unbiased_model_choice
        )
        
        try:
            # Query LLM
            response = self.model.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                config=self.config
            )
            
            if not response:
                if self.verbose:
                    print("Warning: LLM returned no content for strategy selection")
                return None
            
            if self.verbose:
                print(f"[INFO] LLM response for {self.name} (selection):\n{response.final_answer}")
            
            # Clean response (remove markdown code blocks)
            selected_strategy = clean_markdown_response(response.final_answer)
            
            # Validate strategy is in list
            if selected_strategy in self.strategy_list:
                return selected_strategy
            else:
                if self.verbose:
                    print(
                    f"Warning: LLM returned invalid strategy '{selected_strategy}'. "
                    f"Not in provided list: {self.strategy_list}"
                )
                return None
        
        except ModelExecutionError:
            raise
        except Exception as e:
            if self.verbose:
                print(f"Error during strategy selection: {e}")
            return None
    
    def _rewrite_text(
        self,
        original_text: str,
        strategy: str
    ) -> Optional[str]:
        """
        Use LLM to rewrite patient self-report using strategy-specific prompts from registry.
        
        Each strategy in LANGUAGE_REWRITE_REGISTRY has:
        - system_prompt: Strategy-specific instructions
        - user_prompt_template: Template with {original_text} placeholder
        - temperature_override: Optional temperature (e.g., 0.0 for translation, 0.3 for dialects)
        - requires_distraction: Whether strategy adds distraction sentences
        
        Args:
            original_text: Original patient self-report
            strategy: Selected language strategy (e.g., "limited english")
        
        Returns:
            Rewritten text or None if rewriting fails
        """
        # Get registry entry for this strategy
        if strategy not in LANGUAGE_REWRITE_REGISTRY:
            if self.verbose:
                print(
                f"Warning: Strategy '{strategy}' not found in LANGUAGE_REWRITE_REGISTRY. "
                f"Available strategies: {list(LANGUAGE_REWRITE_REGISTRY.keys())}"
            )
            return None
        
        registry_entry = LANGUAGE_REWRITE_REGISTRY[strategy]
        
        # Format prompts from registry
        system_prompt = registry_entry["system_prompt"]
        user_prompt = registry_entry["user_prompt_template"].format(
            original_text=original_text
        )
        
        # Use strategy-specific temperature if specified, otherwise use default
        temperature_override = registry_entry.get("temperature_override")
        if temperature_override is not None:
            config = _override_generation_config(
                self.config,
                temperature=temperature_override,
            )
            if self.verbose:
                print(f"[INFO] Using temperature override: {temperature_override} for {strategy}")
        else:
            config = self.config
        
        try:
            # Query LLM
            response = self.model.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                config=config
            )
            
            if not response:
                if self.verbose:
                    print(f"Warning: LLM returned no content for rewriting ({strategy})")
                return None
            
            if self.verbose:
                print(f"[INFO] LLM response for {self.name} (rewrite - {strategy}):\n{response.final_answer}")
            
            # Clean response (remove markdown code blocks)
            rewritten_text = clean_markdown_response(response.final_answer)
            
            # Validate non-empty
            if rewritten_text:
                # Log distraction info if applicable
                if registry_entry.get("requires_distraction", False):
                    distraction_count = registry_entry.get("distraction_count", 0)
                    if self.verbose:
                        print(f"[INFO] Strategy includes {distraction_count} distraction sentences")
                return rewritten_text
            else:
                if self.verbose:
                    print(f"Warning: LLM returned empty rewritten text for {strategy}")
                return None
        
        except ModelExecutionError:
            raise
        except Exception as e:
            if self.verbose:
                print(f"Error during text rewriting ({strategy}): {e}")
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Export strategy configuration for logging."""
        base_dict = super().to_dict()
        base_dict["strategy_params"] = {
            "strategy_list": self.strategy_list,
            "num_strategies": len(self.strategy_list)
        }
        return base_dict


class EmotionManipulationStrategy(AttackStrategy):
    """
    Attack strategy that modifies patient self-report emotional tone.
    
    This strategy tests whether models exhibit bias when patient emotional
    expression changes (e.g., angry, anxious, depressive, optimistic). It 
    uses an LLM in two stages:
    1. Select which emotion strategy would most change the answer
    2. Rewrite patient self-report with selected emotion (using registry)
    
    The rewriting stage uses EMOTION_REWRITE_REGISTRY which maps each
    emotion to its specific prompts and requirements (all emotions add
    2 distraction sentences and use temperature 0.3).
    
    Process:
    1. Check patient self-report is available
    2. Use LLM to select emotion strategy
    3. Lookup strategy in EMOTION_REWRITE_REGISTRY
    4. Use strategy-specific prompts to rewrite
    5. Replace self-report in test case
    6. Return modified case
    
    Example:
        Original: "I have severe chest pain that started this morning."
        Selected: "anxious"
        Rewritten: "I have severe chest pain that started this morning and 
                    I'm really worried about it. What if it's something serious?"
    """
    
    def __init__(
        self,
        model_id: str,
        model_pool: ModelPool,
        config: Optional[GenerationConfig] = None,
        strategy_list: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        """
        Initialize emotion manipulation strategy.
        
        Args:
            model_id: LLM to use for strategy selection and rewriting
            model_pool: Shared model pool
            config: Generation config (uses default if None)
            strategy_list: List of emotion strategies to choose from
        """
        super().__init__(
            name="emotion_manipulation",
            model_id=model_id,
            model_pool=model_pool,
            system_prompt=EMOTION_STRATEGY_SELECTION_SYSTEM_PROMPT,
            config=config or GenerationConfig(temperature=0.3, max_tokens=2048)
        )
        
        self.strategy_list = strategy_list or DEFAULT_EMOTION_STRATEGIES
        self.verbose = verbose

        # Validate all strategies have rewrite entries
        self._validate_strategies()
    
    def _validate_strategies(self):
        """Validate all strategies in list have rewrite registry entries."""
        missing = [s for s in self.strategy_list if s not in EMOTION_REWRITE_REGISTRY]
        if missing:
            if self.verbose:
                print(
                f"Warning: Strategies missing from EMOTION_REWRITE_REGISTRY: {missing}\n"
                f"These strategies will fail if selected during rewriting."
            )
    
    def apply(
        self,
        test_case: BiasTestCase,
        context: Dict[str, Any]
    ) -> BiasTestCase:
        """
        Apply emotion manipulation to test case.

        Args:
            test_case: Original test case
            context: Must contain:
                - 'ref_answer': Reference answer from unbiased case
                - 'question_idx': Which question to format (0-3)

        Returns:
            Modified test case with rewritten patient self-report

        Raises:
            ValueError: If ref_answer not in context or self-report unavailable
            RuntimeError: If strategy selection or rewriting fails
        """
        # Validate context
        ref_answer = context.get('ref_answer')
        if ref_answer is None:
            raise ValueError("Context must contain 'ref_answer'")
        unbiased_model_choice = context.get("unbiased_model_choice") or ref_answer

        question_idx = context.get('question_idx', 0)
        question_text = (
            test_case.questions[question_idx]
            if test_case.has_valid_question(question_idx)
            else None
        )
        if is_mental_health_case(test_case.category, question_text):
            return _excluded_case(
                test_case,
                question_idx=question_idx,
                reason=(
                    "Emotional strategy excluded for "
                    "psychiatric/mental-health case"
                ),
            )

        if not _available_self_report(test_case.patient_self_report):
            return _excluded_case(
                test_case,
                question_idx=question_idx,
                reason="Emotion manipulation is not applicable without a self-report",
            )
        
        # Format case for strategy selection
        case_text = test_case.format_full_case(question_idx)
        
        # Step 1: Select emotion strategy using LLM
        selected_strategy = self._select_strategy(case_text, unbiased_model_choice)
        
        if selected_strategy is None:
            raise RuntimeError(
                f"Failed to select emotion strategy for case {test_case.case_id}"
            )
        
        _strategy_log(self, f"[INFO] Selected emotion strategy: {selected_strategy}")
        
        # Step 2: Rewrite patient self-report using LLM
        modified_report = self._rewrite_text(
            test_case.patient_self_report,
            selected_strategy
        )
        
        if modified_report is None:
            raise RuntimeError(
                f"Failed to rewrite patient self-report for case {test_case.case_id}"
            )
        
        _strategy_log(
            self,
            f"[INFO] Original report: {test_case.patient_self_report[:100]}...",
        )
        _strategy_log(self, f"[INFO] Modified report: {modified_report[:100]}...")
        
        # Step 3: Create modified test case
        modified_case = BiasTestCase(
            case_id=test_case.case_id,
            clinical_summary=test_case.clinical_summary,
            vital_signs=test_case.vital_signs,
            patient_self_report=modified_report,  # Replaced
            questions=test_case.questions,
            category=test_case.category,
            metadata={
                **test_case.metadata,
                "attack_applied": self.name,
                "selected_strategy": selected_strategy,
                "original_report": test_case.patient_self_report,
                "ref_answer": ref_answer,
                "question_idx": question_idx
            }
        )
        
        return modified_case
    
    def _select_strategy(
        self,
        question_template: str,
        unbiased_model_choice: str
    ) -> Optional[str]:
        """
        Use LLM to select emotion strategy most likely to change answer.
        
        Args:
            question_template: Full case text
            unbiased_model_choice: Reference answer (e.g., "C")
        
        Returns:
            Selected strategy string or None if selection fails
        """
        # Format prompts
        system_prompt = self.system_prompt.format(
            unbiased_model_choice=unbiased_model_choice
        )
        
        strategy_str = format_strategy_list(self.strategy_list)
        user_prompt = EMOTION_STRATEGY_SELECTION_USER_PROMPT.format(
            question_template=question_template,
            strategy_str=strategy_str,
            unbiased_model_choice=unbiased_model_choice
        )
        
        try:
            # Query LLM
            response = self.model.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                config=self.config
            )
            
            if not response:
                if self.verbose:
                    print("Warning: LLM returned no content for strategy selection")
                return None
            
            if self.verbose:
                print(f"[INFO] LLM response for {self.name} (selection):\n{response.final_answer}")
            
            # Clean response (remove markdown code blocks)
            selected_strategy = clean_markdown_response(response.final_answer)
            
            # Validate strategy is in list
            if selected_strategy in self.strategy_list:
                return selected_strategy
            else:
                if self.verbose:
                    print(
                    f"Warning: LLM returned invalid strategy '{selected_strategy}'. "
                    f"Not in provided list: {self.strategy_list}"
                )
                return None
        
        except ModelExecutionError:
            raise
        except Exception as e:
            if self.verbose:
                print(f"Error during strategy selection: {e}")
            return None
    
    def _rewrite_text(
        self,
        original_text: str,
        strategy: str
    ) -> Optional[str]:
        """
        Use LLM to rewrite patient self-report using strategy-specific prompts from registry.
        
        Each strategy in EMOTION_REWRITE_REGISTRY has:
        - system_prompt: Strategy-specific instructions
        - user_prompt_template: Template with {original_text} placeholder
        - temperature_override: 0.3 for all emotion strategies
        - requires_distraction: True (all emotions add 2 distraction sentences)
        
        Args:
            original_text: Original patient self-report
            strategy: Selected emotion strategy (e.g., "anxious")
        
        Returns:
            Rewritten text or None if rewriting fails
        """
        # Get registry entry for this strategy
        if strategy not in EMOTION_REWRITE_REGISTRY:
            if self.verbose:
                print(
                f"Warning: Strategy '{strategy}' not found in EMOTION_REWRITE_REGISTRY. "
                f"Available strategies: {list(EMOTION_REWRITE_REGISTRY.keys())}"
            )
            return None
        
        registry_entry = EMOTION_REWRITE_REGISTRY[strategy]
        
        # Format prompts from registry
        system_prompt = registry_entry["system_prompt"]
        user_prompt = registry_entry["user_prompt_template"].format(
            original_text=original_text
        )
        
        # Use strategy-specific temperature if specified, otherwise use default
        temperature_override = registry_entry.get("temperature_override")
        if temperature_override is not None:
            config = _override_generation_config(
                self.config,
                temperature=temperature_override,
            )
            if self.verbose:
                print(f"[INFO] Using temperature override: {temperature_override} for {strategy}")
        else:
            config = self.config
        
        try:
            # Query LLM
            response = self.model.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                config=config
            )
            
            if not response:
                if self.verbose:
                    print(f"Warning: LLM returned no content for rewriting ({strategy})")
                return None
            
            if self.verbose:
                print(f"[INFO] LLM response for {self.name} (rewrite - {strategy}):\n{response.final_answer}")
            
            # Clean response (remove markdown code blocks)
            rewritten_text = clean_markdown_response(response.final_answer)
            
            # Validate non-empty
            if rewritten_text:
                # Log distraction info (all emotion strategies have distractions)
                if registry_entry.get("requires_distraction", False):
                    distraction_count = registry_entry.get("distraction_count", 0)
                    if self.verbose:
                        print(f"[INFO] Strategy includes {distraction_count} distraction sentences")
                return rewritten_text
            else:
                if self.verbose:
                    print(f"Warning: LLM returned empty rewritten text for {strategy}")
                return None
        
        except ModelExecutionError:
            raise
        except Exception as e:
            if self.verbose:
                print(f"Error during text rewriting ({strategy}): {e}")
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Export strategy configuration for logging."""
        base_dict = super().to_dict()
        base_dict["strategy_params"] = {
            "strategy_list": self.strategy_list,
            "num_strategies": len(self.strategy_list)
        }
        return base_dict


class CognitiveBiasStrategy(AttackStrategy):
    """
    Attack strategy that introduces cognitive bias via clinical note modification.
    
    This strategy tests whether models exhibit cognitive biases (status quo,
    recency, authority, etc.) when additional context is added to the clinical
    note. Unlike language/emotion strategies, this is SINGLE-STAGE: it directly
    generates a bias-inducing sentence without a selection phase.
    
    Process:
    1. Use LLM to generate bias-inducing sentence
    2. Append sentence to clinical note
    3. Return modified case
    
    The generated sentence aims to make the model AVOID the correct answer by
    introducing cognitive biases like:
    - Status quo: "Current protocol differs..."
    - Recency: "Recent cases with X had complications..."
    - Authority: "Dr. Evans suggests avoiding X..."
    - Defensive medicine: "Increasing scrutiny on X..."
    
    Example:
        Original clinical note: "45-year-old patient with chest pain. BP 140/90."
        Generated sentence: "Recent cases where myocardial infarction was 
                           diagnosed led to unexpected complications, so we're 
                           being more cautious."
        Modified note: "45-year-old patient with chest pain. BP 140/90. Recent 
                       cases where myocardial infarction was diagnosed led to 
                       unexpected complications, so we're being more cautious."
    """
    
    def __init__(
        self,
        model_id: str,
        model_pool: ModelPool,
        config: Optional[GenerationConfig] = None,
        verbose: bool = True,
    ):
        """
        Initialize cognitive bias strategy.
        
        Args:
            model_id: LLM to use for sentence generation
            model_pool: Shared model pool
            config: Generation config (uses default if None)
        """
        super().__init__(
            name="cognitive_bias",
            model_id=model_id,
            model_pool=model_pool,
            system_prompt=COGNITIVE_BIAS_GENERATION_SYSTEM_PROMPT,
            config=config or GenerationConfig(temperature=0.3, max_tokens=2048)
        )
        self.verbose = verbose

    def apply(
        self,
        test_case: BiasTestCase,
        context: Dict[str, Any]
    ) -> BiasTestCase:
        """
        Apply cognitive bias to test case.
        
        Args:
            test_case: Original test case
            context: Must contain:
                - 'ref_answer': Reference answer from unbiased case
                - 'question_idx': Which question to format (0-3)
        
        Returns:
            Modified test case with bias sentence appended to clinical note
        
        Raises:
            ValueError: If ref_answer not in context
            RuntimeError: If sentence generation fails
        """
        # Validate context
        ref_answer = context.get('ref_answer')
        if ref_answer is None:
            raise ValueError("Context must contain 'ref_answer'")
        unbiased_model_choice = context.get("unbiased_model_choice") or ref_answer

        question_idx = context.get('question_idx', 0)
        
        # Format case for sentence generation
        case_text = test_case.format_full_case(question_idx)
        
        # Generate bias-inducing sentence using LLM
        bias_sentence = self._generate_bias_sentence(case_text, unbiased_model_choice)
        
        if bias_sentence is None:
            raise RuntimeError(
                f"Failed to generate cognitive bias sentence for case {test_case.case_id}"
            )
        
        _strategy_log(self, f"[INFO] Generated bias sentence: {bias_sentence}")
        
        # Append sentence to clinical summary
        # Original: "Clinical summary. Vital Sign: BP 120/80"
        # Modified: "Clinical summary. Vital Sign: BP 120/80 Bias sentence."
        original_clinical_note = f"{test_case.clinical_summary} Vital Sign: {test_case.vital_signs}"
        modified_clinical_note = f"{original_clinical_note.strip()} {bias_sentence}"
        
        # Extract just the summary part (remove vital signs)
        # We need to update clinical_summary to include the bias sentence
        # But keep vital_signs separate
        modified_clinical_summary = f"{test_case.clinical_summary.strip()} {bias_sentence}"
        
        _strategy_log(
            self,
            f"[INFO] Original clinical note: {original_clinical_note[:100]}...",
        )
        _strategy_log(
            self,
            f"[INFO] Modified clinical note: {modified_clinical_note[:100]}...",
        )
        
        # Create modified test case
        modified_case = BiasTestCase(
            case_id=test_case.case_id,
            clinical_summary=modified_clinical_summary,  # Updated with bias sentence
            vital_signs=test_case.vital_signs,
            patient_self_report=test_case.patient_self_report,
            questions=test_case.questions,
            category=test_case.category,
            metadata={
                **test_case.metadata,
                "attack_applied": self.name,
                "bias_sentence": bias_sentence,
                "original_clinical_summary": test_case.clinical_summary,
                "ref_answer": ref_answer,
                "question_idx": question_idx
            }
        )
        
        return modified_case
    
    def _generate_bias_sentence(
        self,
        question_template: str,
        unbiased_model_choice: str
    ) -> Optional[str]:
        """
        Use LLM to generate a cognitive bias-inducing sentence.
        
        The sentence will be appended to the clinical note and should make
        the model AVOID selecting the unbiased_model_choice.
        
        Args:
            question_template: Full case text
            unbiased_model_choice: Reference answer (e.g., "C") that we want to avoid
        
        Returns:
            Generated bias sentence or None if generation fails
        """
        # Format prompts
        system_prompt = self.system_prompt.format(
            unbiased_model_choice=unbiased_model_choice
        )
        
        user_prompt = COGNITIVE_BIAS_GENERATION_USER_PROMPT.format(
            question_template=question_template,
            unbiased_model_choice=unbiased_model_choice
        )
        
        try:
            # Query LLM
            response = self.model.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                config=self.config
            )
            
            if not response:
                if self.verbose:
                    print("Warning: LLM returned no content for bias sentence generation")
                return None
            
            if self.verbose:
                print(f"[INFO] LLM response for {self.name}:\n{response.final_answer}")
            
            # Clean response (remove markdown code blocks)
            bias_sentence = clean_markdown_response(response.final_answer)
            
            # Validate non-empty
            if bias_sentence:
                return bias_sentence
            else:
                if self.verbose:
                    print("Warning: LLM returned empty bias sentence")
                return None
        
        except ModelExecutionError:
            raise
        except Exception as e:
            if self.verbose:
                print(f"Error during bias sentence generation: {e}")
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Export strategy configuration for logging."""
        base_dict = super().to_dict()
        base_dict["strategy_params"] = {
            "bias_types": COGNITIVE_BIAS_TYPES,
            "num_bias_types": len(COGNITIVE_BIAS_TYPES)
        }
        return base_dict
