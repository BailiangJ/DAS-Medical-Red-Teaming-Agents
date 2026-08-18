from med_red_team.healthbench.config import HealthBenchConfig
from med_red_team.models import GenerationConfig


CONFIG = HealthBenchConfig(
    testee_model="qwen3",
    testee_config=GenerationConfig(temperature=0.0, max_tokens=4096),
    grader_model="gpt-4.1",
    grader_config=GenerationConfig(temperature=0.0, max_tokens=2048),
    filter_single_turn=True,
    attacker_strategies={
        "impossible_measurement": {
            "model_id": "o3",
            "config": GenerationConfig(temperature=0.7, max_tokens=4096),
        }
    },
    max_samples=192,
)
