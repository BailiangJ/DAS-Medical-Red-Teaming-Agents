import os

from med_red_team.models import GenerationConfig
from med_red_team.privacy.config import PrivacyConfig


CONFIG = PrivacyConfig(
    testee_model="gpt-4o",
    testee_system_prompt_mode="hard",
    testee_config=GenerationConfig(temperature=0.0, max_tokens=4096),
    grader_model="gpt-4o",
    grader_config=GenerationConfig(temperature=0.0, max_tokens=4096),
    generator_model="gpt-4o",
    generator_config=GenerationConfig(temperature=1.0, max_tokens=4096),
    max_samples=81,
    num_attempts=3,
    data_file=os.getenv("PRIVACY_DATASET", "data/RT_Privacy.xlsx"),
    sheet_name="Privacy",
    prompt_column="Case Plain",
)
