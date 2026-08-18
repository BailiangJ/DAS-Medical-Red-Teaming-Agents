from typing import List, Any, Optional, TypeVar, Callable, Tuple

from med_red_team.data import EvaluationResult, EvaluationSummary
from med_red_team.testee import Testee
from med_red_team.grader import Grader


# Type aliases for clarity
EvalResult = TypeVar('EvalResult')  # Axis-specific result type
EvalSummary = TypeVar('EvalSummary')  # Axis-specific summary type


class EvaluationPipeline:
    """
    Shared helper base for axis-specific evaluation pipelines.

    Concrete axes keep their own public entrypoints and result types. This base
    class exists only to hold the common loop, progress, and summary-printing
    helpers reused by Privacy, Bias, HealthBench, and Robustness. Callers should
    use the concrete pipeline methods exposed by each axis rather than relying on
    a universal ``evaluate()`` contract.

    Concrete pipelines expose their own public run/save APIs. This class only
    provides protected evaluation-loop and reporting helpers.
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
    # Misc shared hooks
    # =========================================================================

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
