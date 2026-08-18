from configs.paper.hallucination.response_generation import CONFIG as PAPER_CONFIG
from med_red_team.hallucination.config import StanfordResponseGenerationConfig


data = PAPER_CONFIG.to_dict()
data["max_samples"] = 2
data["output_dir"] = "logs/hallucination/generated_responses"
data["metadata"] = {
    **data.get("metadata", {}),
    "preset": "example_hallucination_stanford_positive_response_generation",
    "example_samples": 2,
}
CONFIG = StanfordResponseGenerationConfig.from_dict(data)
