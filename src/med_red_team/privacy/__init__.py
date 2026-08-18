"""
Privacy Red-Teaming Module
===========================

This module provides the public Privacy-axis API for evaluating medical AI
privacy safeguards. Lower-level metadata, resume, scoring, and stage-handoff
helpers remain available from their concrete submodules.
"""

from med_red_team.privacy.attacker import (
    CombinedDisguiseStrategy,
    FocusDistractionStrategy,
    ImplicitDisguiseStrategy,
    PrivacyWarningStrategy,
    WellIntentionStrategy,
)
from med_red_team.privacy.config import PrivacyConfig
from med_red_team.privacy.data import (
    EXPECTED_PHI_IDENTIFIER_KEYS,
    PrivacyGradingResult,
    PrivacyResult,
    PrivacySummary,
    PrivacyTestCase,
)
from med_red_team.privacy.generator_PHI import (
    DEFAULT_BASIC_IDENTIFIER_KEYS,
    DEFAULT_EXTRA_IDENTIFIER_KEYS,
    IDENTIFIERS_TEMPLATE,
    PHIGenerator,
)
from med_red_team.privacy.grader import PrivacyGrader, RuleBasedPrivacyGrader
from med_red_team.privacy.pipeline import PrivacyPipeline
from med_red_team.privacy.privacy_dataset_loader import load_privacy_test_cases

__all__ = [
    "CombinedDisguiseStrategy",
    "DEFAULT_BASIC_IDENTIFIER_KEYS",
    "DEFAULT_EXTRA_IDENTIFIER_KEYS",
    "EXPECTED_PHI_IDENTIFIER_KEYS",
    "FocusDistractionStrategy",
    "IDENTIFIERS_TEMPLATE",
    "ImplicitDisguiseStrategy",
    "PHIGenerator",
    "PrivacyConfig",
    "PrivacyGrader",
    "PrivacyGradingResult",
    "PrivacyPipeline",
    "PrivacyResult",
    "PrivacySummary",
    "PrivacyTestCase",
    "PrivacyWarningStrategy",
    "RuleBasedPrivacyGrader",
    "WellIntentionStrategy",
    "load_privacy_test_cases",
]
