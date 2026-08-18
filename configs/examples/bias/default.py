"""Default Bias example configuration."""

from med_red_team.bias.config import BiasConfig


CONFIG = BiasConfig(
    testee_model="gpt-4o",
    attacker_model="gpt-4o",
    vote_num_baseline=5,
    vote_num_attack=5,
    attacker_strategies={
        "race_socioeconomic_label": {
            "model_id": "gpt-4o",
        },
    },
)
