"""Fast Bias example configuration for quick smoke runs."""

from med_red_team.bias.config import BiasConfig


CONFIG = BiasConfig(
    testee_model="gpt-4.1-mini",
    attacker_model="gpt-4.1-mini",
    vote_num_baseline=5,
    vote_num_attack=5,
    max_samples=10,
    attacker_strategies={
        "race_socioeconomic_label": {
            "model_id": "gpt-4.1-mini",
        },
    },
)
