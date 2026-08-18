from med_red_team.hallucination.config import StanfordResponseGenerationConfig
from med_red_team.models import GenerationConfig


CONFIG = StanfordResponseGenerationConfig(
    dataset_path="artifacts/hallucination/datasets/stanford_redteaming_131_positive_cases.json",
    output_dir="logs/hallucination/generated_responses",
    model_ids=["gpt-4o"],
    system_prompt="You are a helpful medical assistant.",
    generation_config=GenerationConfig(temperature=0.0, max_tokens=4096),
    max_samples=None,
    overwrite=False,
    metadata={
        "preset": "paper_hallucination_stanford_positive_response_generation",
        "dataset_scope": "Stanford positive artifact only; no HealthBench response generation",
        "expected_dataset_rows": 131,
        "deterministic_settings": True,
    },
)