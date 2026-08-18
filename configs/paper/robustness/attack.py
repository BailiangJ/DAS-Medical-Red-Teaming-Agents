from med_red_team.models import GenerationConfig
from med_red_team.robustness.config import RobustnessConfig

from configs.paper.robustness.baseline import SYSTEM_PROMPT


STRATEGY_CATALOG = {
    "add_none_of_the_above": {},
    "replace_correct_answer_with_none": {},
    "add_distraction_sentence": {
        "model_id": "gpt-4o",
        "config": GenerationConfig(temperature=0.2, max_tokens=2048),
    },
    "generate_distractor_options": {
        "model_id": "gpt-4o",
        "num_distractors": 2,
        "config": GenerationConfig(temperature=0.3, max_tokens=2048),
    },
    "bias_manipulation": {
        "model_id": "gpt-4o",
        "n_bias_styles": 3,
        "config": GenerationConfig(temperature=0.3, max_tokens=2048),
    },
    "invert_question_answer": {
        "model_id": "gpt-4o",
        "config": GenerationConfig(temperature=0.2, max_tokens=2048),
    },
    "adjust_impossible_measurement": {
        "model_id": "gpt-4o",
        "config": GenerationConfig(temperature=0.2, max_tokens=2048),
    },
}

DEFAULT_STRATEGIES = (
    "generate_distractor_options",
    "invert_question_answer",
    "bias_manipulation",
    "add_distraction_sentence",
)

CONFIG = RobustnessConfig(
    testee_model="gpt-4o",
    testee_config=GenerationConfig(temperature=0.0, max_tokens=256),
    testee_system_prompt=SYSTEM_PROMPT,
    max_samples=100,
    dataset_path="data/medqa_test.jsonl",
    attacker_strategies={name: STRATEGY_CATALOG[name] for name in DEFAULT_STRATEGIES},
)
CONFIG.strategy_catalog = STRATEGY_CATALOG
