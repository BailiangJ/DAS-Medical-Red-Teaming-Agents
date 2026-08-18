"""
Privacy Dataset Loader
======================

Utilities for loading privacy test cases from Excel files.

This module provides functions to load privacy-sensitive test cases
for red-teaming evaluation.

Example:
    >>> from med_red_team.privacy.privacy_dataset_loader import load_privacy_test_cases
    >>>
    >>> test_cases = load_privacy_test_cases(
    ...     file_path="data/RT_Privacy.xlsx",
    ...     sheet_name="Privacy",
    ...     max_samples=100
    ... )
    >>>
    >>> print(f"Loaded {len(test_cases)} test cases")
"""

from typing import List, Optional
from pathlib import Path
import pandas as pd

from med_red_team.privacy.data import PrivacyTestCase


def load_privacy_test_cases(
    file_path: str,
    sheet_name: str = "Privacy",
    prompt_column: str = "Case Plain",
    max_samples: Optional[int] = None,
) -> List[PrivacyTestCase]:
    """
   
    Args:
        file_path: Path to the Excel file containing test cases.
        sheet_name: Sheet to read (default: "Privacy").
        prompt_column: Column containing the prompt text (default: "Case Plain").
        max_samples: Maximum number of valid non-empty cases to load (None = no limit).

    Returns:
        List[PrivacyTestCase]: Sequence of test cases with sequential IDs,
        preserved category/diagnosis (if present), and cleaned prompts.

    Raises:
        FileNotFoundError: If the Excel file doesn't exist.
        KeyError: If the sheet or prompt column doesn't exist.
        ValueError: If no valid (non-empty) prompts are found after filtering.
    """
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be >= 0 or None")
    if max_samples == 0:
        return []

    path = Path(file_path)

    # Validate file exists
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    print(f"\n[Loading] Reading Excel: {path}")
    print(f"  Sheet: {sheet_name}")
    print(f"  Prompt column: {prompt_column}")
    print(f"  Max samples: {max_samples}")

    # Load Excel sheet (give a clear error if the sheet is missing)
    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
    except ValueError as e:
        raise KeyError(f"Sheet '{sheet_name}' not found in {path}") from e

    # Validate prompt column exists
    if prompt_column not in df.columns:
        available_columns = ", ".join(map(str, df.columns))
        raise KeyError(
            f"Column '{prompt_column}' not found in sheet '{sheet_name}'. "
            f"Available columns: {available_columns}"
        )

    # Build PrivacyTestCase list, filtering blanks before applying the valid-case limit.
    test_cases: List[PrivacyTestCase] = []
    count = 0
    for _, row in df.iterrows():
        prompt = row[prompt_column]
        if pd.isna(prompt) or not str(prompt).strip():
            continue  # skip empty/whitespace-only prompts

        count += 1
        test_cases.append(
            PrivacyTestCase(
                case_id=f"case_{count:03d}",              # mirrors class-based ID format
                sample_number=count,
                original_prompt=str(prompt).strip(),
                category=row.get("Category", "Unknown"),  # preserve domain labels if present
                diagnosis=row.get("Diagnosis", "Unknown")
            )
        )
        if max_samples is not None and count >= max_samples:
            break

    if not test_cases:
        raise ValueError(
            f"No valid test cases loaded from {path}. "
            f"Check that '{prompt_column}' contains non-empty values."
        )

    print(f"  Loaded {len(test_cases)} valid test cases")

    return test_cases
