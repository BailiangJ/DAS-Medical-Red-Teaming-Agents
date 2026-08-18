from med_red_team.models import GenerationConfig
from med_red_team.privacy.config import PrivacyConfig


CONFIG = PrivacyConfig(
    testee_model="gpt-4o",
    testee_config=GenerationConfig(temperature=0.0, max_tokens=4096),
    grader_model="gpt-4o",
    grader_config=GenerationConfig(temperature=0.0, max_tokens=4096),
    generator_model="gpt-4o",
    generator_config=GenerationConfig(temperature=1.0, max_tokens=4096),
    max_samples=81,
    num_attempts=3,
    attacker_strategies={
        "combined_disguise": {
            "model_id": "gpt-4o",
            "config": GenerationConfig(temperature=0.0, max_tokens=4096),
        }
    },
)
