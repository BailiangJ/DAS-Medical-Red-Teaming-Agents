"""Offline metrics for hallucination detector artifacts."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from statistics import NormalDist
from typing import Any, Iterable, Literal, Sequence

from .schemas import normalize_merged_codes


UncertainPolicy = Literal["negative", "positive", "drop"]


def _get_value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def label_to_binary(value: Any) -> int | None:
    """Convert dataset labels to 0/1; positive numeric labels become 1."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return 1 if value > 0 else 0
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return 1 if float(stripped) > 0 else 0
        except ValueError as exc:
            raise ValueError(f"Cannot convert label {value!r} to binary") from exc
    raise ValueError(f"Cannot convert {type(value).__name__} label to binary")


def merged_code_to_binary(
    merged_codes: Any,
    *,
    uncertain_policy: UncertainPolicy = "positive",
) -> int | None:
    """Convert merged hallucination codes to a binary detection."""
    normalized = normalize_merged_codes(merged_codes)
    if isinstance(normalized, list):
        return 1
    if normalized == "0":
        return 0
    if normalized == "0.5":
        if uncertain_policy == "negative":
            return 0
        if uncertain_policy == "positive":
            return 1
        if uncertain_policy == "drop":
            return None
    raise ValueError(f"Unknown uncertain_policy: {uncertain_policy}")


def prediction_counts(
    values: Iterable[Any],
    *,
    uncertain_policy: UncertainPolicy = "positive",
) -> dict[str, int]:
    """Count raw, evaluated, positive, negative, and policy-dropped predictions."""
    raw = 0
    positive = 0
    negative = 0
    dropped = 0
    for value in values:
        raw += 1
        prediction = merged_code_to_binary(
            value,
            uncertain_policy=uncertain_policy,
        )
        if prediction is None:
            dropped += 1
        elif prediction == 1:
            positive += 1
        else:
            negative += 1
    return {
        "raw": raw,
        "evaluated": positive + negative,
        "positive": positive,
        "negative": negative,
        "dropped": dropped,
    }


def confusion_counts(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    uncertain_policy: UncertainPolicy = "positive",
) -> dict[str, int]:
    """Return tp/fp/tn/fn after dropping rows with unavailable labels/predictions."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "dropped": 0, "total": 0}
    for truth, pred in zip(y_true, y_pred):
        truth_binary = label_to_binary(truth)
        pred_binary = merged_code_to_binary(pred, uncertain_policy=uncertain_policy)
        if truth_binary is None or pred_binary is None:
            counts["dropped"] += 1
            continue
        counts["total"] += 1
        if truth_binary == 1 and pred_binary == 1:
            counts["tp"] += 1
        elif truth_binary == 0 and pred_binary == 1:
            counts["fp"] += 1
        elif truth_binary == 0 and pred_binary == 0:
            counts["tn"] += 1
        elif truth_binary == 1 and pred_binary == 0:
            counts["fn"] += 1
    return counts


def accuracy_precision_recall_f1(counts: dict[str, int]) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1 from confusion counts."""
    tp = counts.get("tp", 0)
    fp = counts.get("fp", 0)
    tn = counts.get("tn", 0)
    fn = counts.get("fn", 0)
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def binary_classification_report(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    uncertain_policy: UncertainPolicy = "positive",
) -> dict[str, Any]:
    """Return confusion counts plus accuracy/precision/recall/F1."""
    counts = confusion_counts(y_true, y_pred, uncertain_policy=uncertain_policy)
    return {"counts": counts, "metrics": accuracy_precision_recall_f1(counts)}


def cohen_kappa(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    uncertain_policy: UncertainPolicy = "positive",
) -> float:
    """Compute Cohen's kappa for binary labels/predictions."""
    pairs: list[tuple[int, int]] = []
    for truth, pred in zip(y_true, y_pred):
        truth_binary = label_to_binary(truth)
        pred_binary = merged_code_to_binary(pred, uncertain_policy=uncertain_policy)
        if truth_binary is not None and pred_binary is not None:
            pairs.append((truth_binary, pred_binary))
    total = len(pairs)
    if total == 0:
        return 0.0

    agree = sum(1 for truth, pred in pairs if truth == pred) / total
    true_pos_rate = sum(1 for truth, _ in pairs if truth == 1) / total
    true_neg_rate = 1 - true_pos_rate
    pred_pos_rate = sum(1 for _, pred in pairs if pred == 1) / total
    pred_neg_rate = 1 - pred_pos_rate
    expected = true_pos_rate * pred_pos_rate + true_neg_rate * pred_neg_rate
    if expected == 1:
        return 1.0 if agree == 1 else 0.0
    return (agree - expected) / (1 - expected)


def positive_rates_by(
    records: Iterable[Any],
    group_key: str,
    *,
    codes_key: str = "merged_codes",
    uncertain_policy: UncertainPolicy = "positive",
) -> dict[str, dict[str, float | int]]:
    """Compute positive detection rates grouped by a record field."""
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "positive": 0, "dropped": 0})
    for record in records:
        group = _get_value(record, group_key, "unknown")
        key = "unknown" if group is None else str(group)
        pred = merged_code_to_binary(_get_value(record, codes_key), uncertain_policy=uncertain_policy)
        if pred is None:
            grouped[key]["dropped"] += 1
            continue
        grouped[key]["total"] += 1
        grouped[key]["positive"] += pred

    return {
        key: {
            "total": value["total"],
            "positive": value["positive"],
            "dropped": value["dropped"],
            "positive_rate": value["positive"] / value["total"] if value["total"] else 0.0,
        }
        for key, value in sorted(grouped.items())
    }


def per_model_rates(
    records: Iterable[Any],
    *,
    model_key: str = "model",
    codes_key: str = "merged_codes",
    uncertain_policy: UncertainPolicy = "positive",
) -> dict[str, dict[str, float | int]]:
    """Compute positive detection rates by model."""
    return positive_rates_by(records, model_key, codes_key=codes_key, uncertain_policy=uncertain_policy)


def per_category_rates(
    records: Iterable[Any],
    *,
    category_key: str = "category",
    codes_key: str = "merged_codes",
    uncertain_policy: UncertainPolicy = "positive",
) -> dict[str, dict[str, float | int]]:
    """Compute positive detection rates by category."""
    return positive_rates_by(records, category_key, codes_key=codes_key, uncertain_policy=uncertain_policy)


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must satisfy 0 <= successes <= total")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if total == 0:
        return (0.0, 0.0)

    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def chi_square_test(table: Sequence[Sequence[int | float]], *, correction: bool = True) -> dict[str, Any]:
    """Run scipy.stats.chi2_contingency if scipy is installed."""
    try:
        from scipy.stats import chi2_contingency  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "scipy is required for chi_square_test; install the optional analysis dependency "
            "or call this function only in environments with scipy available"
        ) from exc

    chi2, p_value, dof, expected = chi2_contingency(table, correction=correction)
    expected_list = expected.tolist() if hasattr(expected, "tolist") else expected
    return {"chi2": float(chi2), "p_value": float(p_value), "dof": int(dof), "expected": expected_list}


def rogan_gladen_correction(
    apparent_prevalence: float,
    sensitivity: float,
    specificity: float,
    *,
    clip: bool = True,
) -> float:
    """Correct apparent prevalence for imperfect sensitivity/specificity."""
    for name, value in {
        "apparent_prevalence": apparent_prevalence,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    denominator = sensitivity + specificity - 1
    if denominator <= 0:
        raise ValueError("sensitivity + specificity must be greater than 1")
    corrected = (apparent_prevalence + specificity - 1) / denominator
    if clip:
        return min(1.0, max(0.0, corrected))
    return corrected


def rogan_gladen_from_counts(
    observed_positive: int,
    total: int,
    sensitivity: float,
    specificity: float,
    *,
    clip: bool = True,
) -> dict[str, float | int]:
    """Return apparent and Rogan-Gladen corrected prevalence from counts."""
    if total < 0 or observed_positive < 0 or observed_positive > total:
        raise ValueError("observed_positive and total must satisfy 0 <= observed_positive <= total")
    apparent = observed_positive / total if total else 0.0
    corrected = rogan_gladen_correction(apparent, sensitivity, specificity, clip=clip)
    return {
        "observed_positive": observed_positive,
        "total": total,
        "apparent_prevalence": apparent,
        "corrected_prevalence": corrected,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    if probability <= 0:
        return sorted_values[0]
    if probability >= 1:
        return sorted_values[-1]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def rogan_gladen_monte_carlo_interval(
    observed_positive: int,
    total: int,
    sensitivity: float,
    specificity: float,
    *,
    sensitivity_counts: tuple[int, int] | None = None,
    specificity_counts: tuple[int, int] | None = None,
    draws: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
    clip: bool = True,
) -> dict[str, Any]:
    """Deterministic Monte Carlo interval for Rogan-Gladen corrected prevalence."""
    if total < 0 or observed_positive < 0 or observed_positive > total:
        raise ValueError("observed_positive and total must satisfy 0 <= observed_positive <= total")
    if draws <= 0:
        raise ValueError("draws must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(draws):
        apparent = rng.betavariate(observed_positive + 1, total - observed_positive + 1) if total else 0.0
        sens = sensitivity
        spec = specificity
        if sensitivity_counts is not None:
            sens_success, sens_total = sensitivity_counts
            sens = rng.betavariate(sens_success + 1, sens_total - sens_success + 1)
        if specificity_counts is not None:
            spec_success, spec_total = specificity_counts
            spec = rng.betavariate(spec_success + 1, spec_total - spec_success + 1)
        if sens + spec <= 1:
            continue
        values.append(rogan_gladen_correction(apparent, sens, spec, clip=clip))

    values.sort()
    alpha = 1 - confidence
    point = rogan_gladen_from_counts(
        observed_positive,
        total,
        sensitivity,
        specificity,
        clip=clip,
    )
    return {
        **point,
        "confidence": confidence,
        "seed": seed,
        "draws_requested": draws,
        "draws_used": len(values),
        "interval": (
            _quantile(values, alpha / 2),
            _quantile(values, 1 - alpha / 2),
        ),
    }


__all__ = [
    "UncertainPolicy",
    "accuracy_precision_recall_f1",
    "binary_classification_report",
    "chi_square_test",
    "cohen_kappa",
    "confusion_counts",
    "label_to_binary",
    "merged_code_to_binary",
    "per_category_rates",
    "per_model_rates",
    "positive_rates_by",
    "prediction_counts",
    "rogan_gladen_correction",
    "rogan_gladen_from_counts",
    "rogan_gladen_monte_carlo_interval",
    "wilson_interval",
]
