from med_red_team.bias.config import BiasConfig
from med_red_team.models import GenerationConfig

from configs.paper.bias.baseline import CONFIG as BASELINE, SYSTEM_PROMPT


CONFIG = BiasConfig(
    testee_model=BASELINE.testee_model,
    testee_config=GenerationConfig(temperature=0.0, max_tokens=256),
    testee_system_prompt=SYSTEM_PROMPT,
    data_file=BASELINE.data_file,
    sheet_name="Bias",
    max_samples=None,
    vote_num_baseline=5,
    vote_num_attack=5,
    use_baseline_cache=True,
    baseline_cache_file=None,
    log_dir="logs",
    cache_dir="cache",
    attacker_strategies={
        name: {
            "model_id": "gpt-4o",
            "config": GenerationConfig(temperature=0.3, max_tokens=2048),
        }
        for name in (
            "race_socioeconomic_label",
            "language_manipulation",
            "emotion_manipulation",
            "cognitive_bias",
        )
    },
)
