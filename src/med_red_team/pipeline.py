from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generic, TypeVar, Callable, Tuple
from pathlib import Path
from datetime import datetime

from med_red_team.data import TestCase, EvaluationResult, EvaluationSummary
from med_red_team.testee import Testee
from med_red_team.grader import Grader
from med_red_team.actors import AttackStrategy


# Type aliases for clarity
EvalResult = TypeVar('EvalResult')  # Axis-specific result type
EvalSummary = TypeVar('EvalSummary')  # Axis-specific summary type


class EvaluationPipeline(ABC):
    """
    Abstract base class for two-phase evaluation pipelines.

    All red-teaming axes follow a standardized two-phase pattern:
    1. Baseline phase: Test original cases, save self-contained results
    2. Attack phase: Load baseline results, apply attacks, test again

    This base class enforces the common interface across all axes
    (Privacy, Bias, HealthBench, Robustness).

    Subclasses must implement:
    - run_baseline(): Execute baseline evaluation (phase 1)
    - run_attack(): Execute attack evaluation (phase 2)
    - save_results(): Save results with standardized format

    Example:
        >>> pipeline = MyPipeline(testee, grader, attack_strategies)
        >>>
        >>> # Phase 1: Baseline
        >>> baseline_results, summary = pipeline.run_baseline(test_cases)
        >>> pipeline.save_results(baseline_results, summary, "baseline.json")
        >>>
        >>> # Phase 2: Attack
        >>> attack_results, summary = pipeline.run_attack(baseline_results)
        >>> pipeline.save_results(attack_results, summary, "attack.json")
    """

    def __init__(
        self,
        testee: Testee,
        grader: Grader,
        verbose: bool = True
    ):
        """
        Initialize base evaluation pipeline.

        Args:
            testee: Model to evaluate
            grader: Grader to assess responses
            verbose: Whether to print progress
        """
        self.testee = testee
        self.grader = grader
        self.verbose = verbose
    
    def evaluate(
        self,
        test_cases: List[TestCase],
        progress_callback = None,
        save_every: int = 10,
        **kwargs
    ) -> tuple[List[EvaluationResult], EvaluationSummary]:
        """
        Run complete evaluation pipeline with crash recovery support.

        This is the main entry point that orchestrates the evaluation.
        It calls task-specific methods that subclasses implement.

        Supports periodic checkpointing via progress_callback for crash recovery.

        Args:
            test_cases: List of test cases to evaluate
            progress_callback: Optional callback(results) for periodic saving
            save_every: Save checkpoint every N cases (default: 10)
            **kwargs: Task-specific parameters

        Returns:
            Tuple of (results_list, summary)

        Example:
            >>> results, summary = pipeline.evaluate(
            ...     test_cases,
            ...     progress_callback=callback,
            ...     save_every=10
            ... )
        """
        from med_red_team.utils import evaluate_items_fail_fast

        # Task-specific setup
        self._setup_evaluation(**kwargs)

        # Create evaluation wrapper with stopping criteria
        def _evaluate_with_stopping(test_case, idx):
            """Wrapper that checks stopping criteria before evaluating."""
            # Note: stopping check moved to before evaluation
            # This is slightly different from original but more efficient
            return self.evaluate_single(test_case, idx, **kwargs)

        # Use safe_evaluate_loop for consistent error handling and progress tracking
        results = evaluate_items_fail_fast(
            items=test_cases,
            evaluate_fn=_evaluate_with_stopping,
            description=f"{self.__class__.__name__.replace('Pipeline', '')} evaluation",
            progress_callback=progress_callback,
            save_every=save_every,
            verbose=self.verbose
        )

        # Note: Stopping criteria now handled via max_samples in kwargs
        # Subclasses can filter test_cases before calling evaluate()

        # Create summary
        summary = self.create_summary(results, **kwargs)

        # Final report
        if self.verbose:
            self._print_summary(summary)

        return results, summary
    
    # =========================================================================
    # New Abstract Methods (Required for all pipelines)
    # =========================================================================

    @abstractmethod
    def run_baseline(
        self,
        test_cases: List[TestCase],
        max_samples: Optional[int] = None,
        progress_callback: Optional[Any] = None,
        save_every: int = 10
    ) -> tuple[List[EvaluationResult], EvaluationSummary]:
        """
        Run baseline evaluation (phase 1).

        Test original cases without attacks and save self-contained results.

        Args:
            test_cases: List of test cases to evaluate
            max_samples: Maximum number of samples to process (None = all)
            progress_callback: Optional callback for periodic checkpointing
            save_every: Save checkpoint every N cases

        Returns:
            Tuple of (results_list, summary)
        """
        pass

    @abstractmethod
    def run_attack(
        self,
        baseline_results: List[EvaluationResult],
        progress_callback: Optional[Any] = None,
        save_every: int = 10
    ) -> tuple[List[EvaluationResult], EvaluationSummary]:
        """
        Run attack evaluation (phase 2).

        Load baseline results, apply attacks, and test manipulated cases.

        Args:
            baseline_results: Self-contained baseline results from phase 1
            progress_callback: Optional callback for periodic checkpointing
            save_every: Save checkpoint every N cases

        Returns:
            Tuple of (results_list, summary)
        """
        pass

    @abstractmethod
    def save_results(
        self,
        results: List[EvaluationResult],
        summary: EvaluationSummary,
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save evaluation results with standardized format.

        Args:
            results: List of evaluation results
            summary: Evaluation summary
            output_path: Path to output JSON file
            metadata: Optional metadata dict (can include config, dataset_info, etc.)

        Format:
            {
                "metadata": {...},
                "summary": {...},
                "results": [...]
            }
        """
        pass

    # =========================================================================
    # Common Evaluation Methods (Shared Across All Axes)
    # =========================================================================

    def _run_baseline_common(
        self,
        items: List[Any],
        evaluate_fn: Callable[[Any, int], EvalResult],
        compute_summary_fn: Callable[[List[EvalResult]], EvalSummary],
        print_summary_fn: Callable[[EvalSummary], None],
        max_samples: Optional[int] = None,
        preprocess_fn: Optional[Callable[[List[Any]], List[Any]]] = None,
        progress_callback: Optional[Callable[[List[EvalResult]], None]] = None,
        save_every: int = 10,
        header_title: str = "BASELINE EVALUATION",
        description: str = "Baseline"
    ) -> Tuple[List[EvalResult], EvalSummary]:
        """
        Common baseline evaluation logic shared across all axes.

        This method implements the standard baseline evaluation workflow:
        1. Optional preprocessing (axis-specific filters)
        2. Sample limiting
        3. Header printing
        4. Safe evaluation loop with error handling
        5. Summary computation
        6. Summary printing

        Args:
            items: Items to evaluate (test cases or other evaluation units)
            evaluate_fn: Axis-specific evaluation function (item, idx) -> Result
            compute_summary_fn: Function to compute summary from results
            print_summary_fn: Function to print summary
            max_samples: Optional limit on number of items to process
            preprocess_fn: Optional preprocessing (e.g., filter_single_turn for HealthBench)
            progress_callback: Optional callback for periodic checkpointing
            save_every: Save checkpoint every N items
            header_title: Title for the header (e.g., "BASELINE EVALUATION")
            description: Description for progress bar

        Returns:
            Tuple of (results, summary)

        Example:
            >>> def _evaluate_case(test_case, idx):
            ...     response = self.testee.answer(test_case.question)
            ...     return RobustnessResult(...)
            >>>
            >>> results, summary = self._run_baseline_common(
            ...     items=test_cases,
            ...     evaluate_fn=_evaluate_case,
            ...     compute_summary_fn=lambda r: self._compute_summary(r, baseline_only=True),
            ...     print_summary_fn=self._print_baseline_summary,
            ...     max_samples=100
            ... )
        """
        from med_red_team.utils import evaluate_items_fail_fast

        # Step 1: Axis-specific preprocessing (optional)
        if preprocess_fn:
            items = preprocess_fn(items)

        # Step 2: Limit samples
        if max_samples is not None and len(items) > max_samples:
            items = items[:max_samples]

        # Step 3: Print header
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"{header_title}")
            print(f"{'='*80}")
            print(f"Processing {len(items)} items")

        # Step 4: Safe evaluation loop
        results = evaluate_items_fail_fast(
            items=items,
            evaluate_fn=evaluate_fn,
            description=description,
            progress_callback=progress_callback,
            save_every=save_every,
            verbose=self.verbose
        )

        # Step 5: Compute summary
        summary = compute_summary_fn(results)

        # Step 6: Print summary
        if self.verbose:
            print_summary_fn(summary)

        return results, summary

    def _run_attack_common(
        self,
        baseline_results: List[Any],
        filter_fn: Callable[[Any], bool],
        create_eval_items_fn: Callable[[List[Any]], List[Any]],
        evaluate_fn: Callable[[Any, int], EvalResult],
        compute_summary_fn: Callable[[List[EvalResult]], EvalSummary],
        print_summary_fn: Callable[[EvalSummary], None],
        progress_callback: Optional[Callable[[List[EvalResult]], None]] = None,
        save_every: int = 10,
        header_title: str = "ATTACK EVALUATION",
        description: str = "Attack",
        filter_description: str = "candidates"
    ) -> Tuple[List[EvalResult], EvalSummary]:
        """
        Common attack evaluation logic shared across all axes.

        This method implements the standard attack evaluation workflow:
        1. Filter baseline results for attackable cases
        2. Header printing
        3. Check for empty filtered list
        4. Create evaluation items (cross-product with strategies if needed)
        5. Safe evaluation loop with error handling
        6. Summary computation
        7. Summary printing

        Args:
            baseline_results: Results from baseline evaluation
            filter_fn: Function to filter attackable results (e.g., correct answers only)
            create_eval_items_fn: Function to create evaluation items from filtered results
            evaluate_fn: Axis-specific evaluation function (item, idx) -> Result
            compute_summary_fn: Function to compute summary from results
            print_summary_fn: Function to print summary
            progress_callback: Optional callback for periodic checkpointing
            save_every: Save checkpoint every N items
            header_title: Title for the header
            description: Description for progress bar
            filter_description: Description of filtered items (e.g., "correct answers")

        Returns:
            Tuple of (results, summary)

        Example:
            >>> # Filter to correctly answered cases
            >>> def filter_correct(r):
            ...     return r.original_correct and not r.skipped
            >>>
            >>> # Create evaluation items (cross-product with strategies)
            >>> def create_items(filtered):
            ...     return [(r, s) for r in filtered for s in self.strategies]
            >>>
            >>> results, summary = self._run_attack_common(
            ...     baseline_results=baseline_results,
            ...     filter_fn=filter_correct,
            ...     create_eval_items_fn=create_items,
            ...     evaluate_fn=self._evaluate_attack_case,
            ...     compute_summary_fn=lambda r: self._compute_summary(r, baseline_only=False),
            ...     print_summary_fn=self._print_attack_summary
            ... )
        """
        from med_red_team.utils import evaluate_items_fail_fast

        # Step 1: Filter attackable results
        filtered_results = [r for r in baseline_results if filter_fn(r)]

        # Step 2: Print header
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"{header_title}")
            print(f"{'='*80}")
            print(f"Baseline results: {len(baseline_results)} total")
            print(f"Filtered {filter_description}: {len(filtered_results)}")

        # Step 3: Check for empty filtered list
        if len(filtered_results) == 0:
            if self.verbose:
                print(f"\n[WARNING] No {filter_description} found. Nothing to attack.")
            empty_summary = compute_summary_fn([])
            return [], empty_summary

        # Step 4: Create evaluation items
        eval_items = create_eval_items_fn(filtered_results)

        if self.verbose:
            print(f"Evaluation items: {len(eval_items)}")

        # Step 5: Safe evaluation loop
        results = evaluate_items_fail_fast(
            items=eval_items,
            evaluate_fn=evaluate_fn,
            description=description,
            progress_callback=progress_callback,
            save_every=save_every,
            verbose=self.verbose
        )

        # Step 6: Compute summary
        summary = compute_summary_fn(results)

        # Step 7: Print summary
        if self.verbose:
            print_summary_fn(summary)

        return results, summary

    # =========================================================================
    # Old Abstract Methods (Deprecated - kept for backward compatibility)
    # =========================================================================

    def evaluate_single(
        self,
        test_case: TestCase,
        sample_idx: int,
        **kwargs
    ) -> EvaluationResult:
        """
        DEPRECATED: Use run_baseline() and run_attack() instead.

        This method is kept for backward compatibility with orchestrator pipeline.
        New code should use the separated baseline/attack pattern.
        """
        raise NotImplementedError(
            "evaluate_single() is deprecated. "
            "Use run_baseline() and run_attack() methods instead."
        )

    def create_summary(
        self,
        results: List[EvaluationResult],
        **kwargs
    ) -> EvaluationSummary:
        """
        DEPRECATED: Summaries are now returned directly from run_baseline()/run_attack().

        This method is kept for backward compatibility.
        """
        raise NotImplementedError(
            "create_summary() is deprecated. "
            "Summaries are returned from run_baseline()/run_attack()."
        )
    
    def _setup_evaluation(self, **kwargs):
        """
        Setup before evaluation starts.
        
        Subclasses can override to perform task-specific setup.
        """
        pass
    
    def _should_stop_evaluation(
        self,
        current_idx: int,
        results: List[EvaluationResult],
        **kwargs
    ) -> bool:
        """
        Determine if evaluation should stop early.
        
        Subclasses can override to implement stopping criteria.
        
        Args:
            current_idx: Current sample index
            results: Results so far
            **kwargs: Task-specific parameters
        
        Returns:
            True if should stop, False otherwise
        """
        return False
    
    def _create_iterator(self, test_cases: List[TestCase]):
        """
        Create iterator for test cases.
        
        Subclasses can override to add progress bars, etc.
        
        Args:
            test_cases: List of test cases
        
        Returns:
            Iterator over test cases
        """
        if self.verbose:
            from tqdm import tqdm
            return tqdm(test_cases, desc="Evaluating")
        return test_cases
    
    def _print_progress(self, idx: int, result: EvaluationResult):
        """
        Print progress update.
        
        Subclasses can override for custom progress reporting.
        """
        if self.verbose:
            status = "SKIPPED" if result.skipped else "COMPLETED"
            print(f"[Sample {idx + 1}] {status}")
    
    def _print_summary(self, summary: EvaluationSummary):
        """
        Print final summary.
        
        Subclasses can override for custom summary reporting.
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"{summary.evaluation_type.upper()} EVALUATION SUMMARY")
            print("=" * 70)
            for key, value in summary.to_dict().items():
                print(f"{key}: {value}")
            print("=" * 70)


class ResultsLogger(ABC):
    """
    DEPRECATED: Use pipeline.save_results() instead.

    This class is kept for backward compatibility but is no longer
    the recommended way to save results. New code should use the
    standardized pipeline.save_results() method.

    Migration path:
        OLD:
            logger = RobustnessLogger(output_dir, model_id)
            output_path = logger.save(
                results, summary,
                config=config, testee=testee, ...
            )

        NEW:
            from scripts.utils import generate_output_path

            output_path = generate_output_path(
                log_dir=output_dir,
                testee_model=model_id,
                mode="robustness_baseline",
                ...
            )

            pipeline.save_results(
                results, summary, str(output_path),
                metadata={
                    "config": config.to_dict(),
                    "testee_config": testee.config.to_dict() if testee.config else {},
                    ...
                }
            )

    Benefits of new approach:
        - Single way to save results across all axes
        - Consistent with Privacy/HealthBench pipelines
        - Simpler architecture with less code to maintain
        - Metadata is explicit and extensible via dict

    Each evaluation type may have different logging requirements,
    but all follow this basic structure.
    """
    
    def __init__(
        self,
        output_dir: str,
        evaluation_type: str,
        create_dir: bool = True
    ):
        """
        Initialize logger.
        
        Args:
            output_dir: Directory to save results
            evaluation_type: Type of evaluation (bias/robustness/privacy/hallucination)
            create_dir: Whether to create directory if it doesn't exist
        """
        self.output_dir = Path(output_dir)
        self.evaluation_type = evaluation_type
        
        if create_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def save(
        self,
        results: List[EvaluationResult],
        summary: EvaluationSummary,
        **kwargs
    ) -> Path:
        """
        Save evaluation results.
        
        Args:
            results: List of evaluation results
            summary: Evaluation summary
            **kwargs: Additional data to save
        
        Returns:
            Path to saved file
        """
        pass
    
    def _generate_filename(
        self,
        prefix: str,
        suffix: str = "json"
    ) -> str:
        """
        Generate filename with timestamp.
        
        Args:
            prefix: Filename prefix
            suffix: File extension
        
        Returns:
            Filename string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.evaluation_type}_{prefix}_{timestamp}.{suffix}"