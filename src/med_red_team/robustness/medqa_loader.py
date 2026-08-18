"""
MedQA Dataset Loader with TestCase Conversion
==============================================

This module provides functions to load MedQA dataset and convert samples
to TestCase objects for use with the red-teaming framework.
"""

import hashlib
import json
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

# Assuming these imports from your framework
from med_red_team.data import TestCase


def _stable_sample_id(sample: Dict[str, Any]) -> str:
    """Return a process-stable fallback ID derived from MedQA row content."""
    payload = {
        "question": sample.get("question", ""),
        "options": sample.get("options", {}),
        "answer_idx": sample.get("answer_idx"),
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    return f"medqa_{digest}"


def sample_to_test_case(sample: Dict[str, Any], sample_id: Optional[str] = None) -> TestCase:
    """
    Convert a MedQA sample to a TestCase object.
    
    Args:
        sample: MedQA sample dictionary with keys:
            - question (str): Question text
            - options (dict): Options dictionary {label: text}
            - answer_idx (str): Correct answer label
            - answer (str, optional): Correct answer text
            - meta_info (str, optional): Dataset metadata
        sample_id: Optional ID for the test case. If not provided,
                   uses a stable content hash of the sample
    
    Returns:
        TestCase object
        
    Example:
        >>> sample = {
        ...     'question': 'A 67-year-old man with transitional cell carcinoma...',
        ...     'answer': 'Cross-linking of DNA',
        ...     'options': {
        ...         'A': 'Inhibition of thymidine synthesis',
        ...         'B': 'Inhibition of proteasome',
        ...         'C': 'Hyperstabilization of microtubules',
        ...         'D': 'Generation of free radicals',
        ...         'E': 'Cross-linking of DNA'
        ...     },
        ...     'meta_info': 'step1',
        ...     'answer_idx': 'E'
        ... }
        >>> test_case = sample_to_test_case(sample, sample_id="medqa_001")
        >>> print(test_case.question[:50])
        A 67-year-old man with transitional cell carcinoma
    """
    # Generate a process-stable ID if not provided.
    if sample_id is None:
        sample_id = _stable_sample_id(sample)
    
    # Extract fields
    question = sample['question']
    options = sample['options']
    correct_answer = sample['answer_idx']
    
    # Optional metadata
    metadata = {}
    if 'meta_info' in sample:
        metadata['dataset'] = sample['meta_info']
    if 'answer' in sample:
        metadata['answer_text'] = sample['answer']
    
    # Create TestCase
    test_case = TestCase(
        id=sample_id,
        question=question,
        options=options,
        correct_answer=correct_answer,
        task_type="multiple_choice",
        metadata=metadata
    )
    
    return test_case


def load_medqa(
    dataset_path: str,
    convert_to_test_cases: bool = True,
    limit: Optional[int] = None
) -> Tuple[List, Dict[str, Any]]:
    """
    Load MedQA dataset from a JSONL file.

    Args:
        dataset_path: Path to the JSONL file containing MedQA questions
        convert_to_test_cases: If True, convert samples to TestCase objects.
                               If False, return raw dictionaries.
        limit: Optional limit on number of samples to load

    Returns:
        Tuple of (test_data, dataset_info)
        - test_data: List[TestCase] or List[Dict]
        - dataset_info: Dict with metadata about the dataset

    Example:
        >>> # Load as TestCase objects
        >>> test_cases, info = load_medqa('data/medqa_test.jsonl', convert_to_test_cases=True)
        >>> print(f"Loaded {len(test_cases)} test cases")
        >>> print(f"First test case ID: {test_cases[0].id}")

        >>> # Load as raw dictionaries
        >>> test_samples, info = load_medqa('data/medqa_test.jsonl', convert_to_test_cases=False)
        >>> print(test_samples[0]['question'][:50])
    """
    test_data = []

    # Load data from JSONL file
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            if limit is not None and idx >= limit:
                break

            sample = json.loads(line)

            if convert_to_test_cases:
                # Convert to TestCase with generated ID
                # Use the file stem (filename without extension) in the ID for traceability
                file_stem = dataset_path.stem
                test_case = sample_to_test_case(sample, sample_id=f"medqa_{file_stem}_{idx}")
                test_data.append(test_case)
            else:
                # Keep as dictionary
                test_data.append(sample)

    # Create dataset info
    dataset_info = {
        "source": str(dataset_path),
        "filename": dataset_path.name,
        "total_loaded": len(test_data),
        "limit_applied": limit
    }

    return test_data, dataset_info
