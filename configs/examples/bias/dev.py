"""Small Bias example configuration for local development."""

from med_red_team.bias.config import BiasConfig


CONFIG = BiasConfig(
    testee_model="gemini-2.5-flash",
    attacker_model="gemini-2.5-flash",
    vote_num_baseline=5,
    vote_num_attack=5,
    max_samples=2,
    attacker_strategies={
        "race_socioeconomic_label": {
            "model_id": "gemini-2.5-flash",
        },
    },
)
