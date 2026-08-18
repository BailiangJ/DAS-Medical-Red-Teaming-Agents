from med_red_team.healthbench.config import HealthBenchConfig
from med_red_team.models import GenerationConfig


CONFIG = HealthBenchConfig(
    testee_model="medgemma",
    testee_config=GenerationConfig(temperature=0.0, max_tokens=4096),
    grader_model="gpt-4.1",
    grader_config=GenerationConfig(temperature=0.0, max_tokens=2048),
    filter_single_turn=True,
    attacker_strategies={},
    max_samples=192,
)
