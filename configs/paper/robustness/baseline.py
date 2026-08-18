import os

from med_red_team.models import GenerationConfig
from med_red_team.robustness.config import RobustnessConfig


SYSTEM_PROMPT = (
    "Answer the question with only the final letter choice (A, B, C, D, E, or F ...) "
    "with no explanation. More than one choice can be correct, in which case return "
    "all of them separated by commas.\n"
    "Answer ONLY with the capital letter(s) of the correct choice, no spaces, no "
    "punctuation, no explanation. If you add anything else you will score 0."
)

CONFIG = RobustnessConfig(
    testee_model="gpt-4o",
    testee_config=GenerationConfig(temperature=0.0, max_tokens=256),
    testee_system_prompt=SYSTEM_PROMPT,
    max_samples=None,
    dataset_path=os.getenv("MEDQA_DATASET", "data/medqa_test.jsonl"),
)
