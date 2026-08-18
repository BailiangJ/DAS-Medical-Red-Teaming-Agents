"""Comprehensive Bias example configuration with all public attacks."""

from med_red_team.bias.config import BiasConfig


CONFIG = BiasConfig(
    testee_model="gpt-4o",
    attacker_model="gpt-4o",
    vote_num_baseline=5,
    vote_num_attack=5,
    max_samples=500,
    attacker_strategies={
        "race_socioeconomic_label": {
            "model_id": "gpt-4o",
        },
        "language_manipulation": {
            "model_id": "gpt-4o",
        },
        "emotion_manipulation": {
            "model_id": "gpt-4o",
        },
        "cognitive_bias": {
            "model_id": "gpt-4o",
        },
    },
)
