from __future__ import annotations

from pathlib import Path

import pytest

from med_red_team.hallucination.config import ArtifactSummaryConfig
from med_red_team.hallucination.metrics import (
    accuracy_precision_recall_f1,
    binary_classification_report,
    chi_square_test,
    cohen_kappa,
    confusion_counts,
    merged_code_to_binary,
    rogan_gladen_correction,
    rogan_gladen_from_counts,
    rogan_gladen_monte_carlo_interval,
    wilson_interval,
)
from med_red_team.hallucination.schemas import normalize_merged_codes
from scripts.hallucination.summarize_artifacts import summarize


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "hallucination"


@pytest.mark.parametrize(
    ("merged_codes", "policy", "expected"),
    [
        ("0", "positive", 0),
        ("0.5", "positive", 1),
        ("0.5", "negative", 0),
        ("0.5", "drop", None),
        (["1"], "positive", 1),
        (["1A", "5E"], "negative", 1),
        ("1A, 5E", "positive", 1),
    ],
)
def test_merged_code_binary_policies(merged_codes, policy, expected):
    assert merged_code_to_binary(merged_codes, uncertain_policy=policy) == expected


def test_merged_code_normalization_accepts_subcodes_and_sorts_roots():
    assert normalize_merged_codes("5E,1A,1B,0.5") == ["1", "5"]
    assert normalize_merged_codes([]) == "0"
    assert normalize_merged_codes("0.5") == "0.5"


def test_confusion_metrics_and_cohen_kappa_with_uncertain_drop_policy():
    y_true = [1, 0, 0, 1, 1, 0]
    y_pred = [["1"], ["2"], "0", "0", "0.5", "0.5"]

    counts = confusion_counts(y_true, y_pred, uncertain_policy="drop")
    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1, "dropped": 2, "total": 4}

    metrics = accuracy_precision_recall_f1(counts)
    assert metrics == {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert binary_classification_report(y_true, y_pred, uncertain_policy="drop") == {
        "counts": counts,
        "metrics": metrics,
    }
    assert cohen_kappa(y_true, y_pred, uncertain_policy="drop") == pytest.approx(0.0)


def test_wilson_interval_and_rogan_gladen_helpers_are_deterministic():
    low, high = wilson_interval(40, 100)
    assert low == pytest.approx(0.3094, abs=5e-4)
    assert high == pytest.approx(0.4979, abs=5e-4)
    assert wilson_interval(0, 0) == (0.0, 0.0)

    corrected = rogan_gladen_correction(0.4, sensitivity=0.9, specificity=0.8)
    assert corrected == pytest.approx(2 / 7)
    assert rogan_gladen_from_counts(40, 100, 0.9, 0.8) == {
        "observed_positive": 40,
        "total": 100,
        "apparent_prevalence": 0.4,
        "corrected_prevalence": pytest.approx(2 / 7),
        "sensitivity": 0.9,
        "specificity": 0.8,
    }

    first = rogan_gladen_monte_carlo_interval(
        40,
        100,
        0.9,
        0.8,
        sensitivity_counts=(90, 100),
        specificity_counts=(80, 100),
        draws=500,
        seed=123,
    )
    second = rogan_gladen_monte_carlo_interval(
        40,
        100,
        0.9,
        0.8,
        sensitivity_counts=(90, 100),
        specificity_counts=(80, 100),
        draws=500,
        seed=123,
    )
    assert first == second
    assert first["draws_requested"] == 500
    assert first["draws_used"] > 0
    assert first["interval"][0] <= first["corrected_prevalence"] <= first["interval"][1]


def test_optional_scipy_chi_square_behavior():
    table = [[10, 5], [3, 12]]
    try:
        result = chi_square_test(table)
    except ImportError as exc:
        assert "scipy is required for chi_square_test" in str(exc)
    else:
        assert set(result) == {"chi2", "p_value", "dof", "expected"}
        assert result["dof"] == 1
        assert 0.0 <= result["p_value"] <= 1.0


def test_current_primary_direct_metrics_and_provenance_reporting():
    summary = summarize(
        ArtifactSummaryConfig(artifact_root=str(ARTIFACT_ROOT)),
        uncertain_policy="positive",
        include_chi_square=False,
        include_rogan_gladen=False,
    )

    primary = summary["native_detector_metrics"]["primary_openai_gpt4o_o3"]
    assert primary["stanford_positive_131"]["counts"] == {
        "tp": 109,
        "fp": 0,
        "tn": 0,
        "fn": 22,
        "dropped": 0,
        "total": 131,
    }
    assert primary["healthbench_negative_manuscript_129"]["counts"] == {
        "tp": 0,
        "fp": 25,
        "tn": 104,
        "fn": 0,
        "dropped": 0,
        "total": 129,
    }
    assert primary["combined_validation"]["counts"] == {
        "tp": 109,
        "fp": 25,
        "tn": 104,
        "fn": 22,
        "dropped": 0,
        "total": 260,
    }
    assert primary["combined_validation"]["metrics"] == {
        "accuracy": pytest.approx(213 / 260),
        "precision": pytest.approx(109 / 134),
        "recall": pytest.approx(109 / 131),
        "f1": pytest.approx(0.8226415094339622),
    }

    assert summary["known_caveats"] == [
        "Generated-response artifacts use Stanford prompt rows.",
        "Per-model detection coverage is recorded by the denominator fields.",
        "Bundle membership is defined by manifest.files.",
        "manifest.json is not self-hashed.",
    ]
