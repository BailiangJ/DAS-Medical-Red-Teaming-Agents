from configs.paper.hallucination.detector_validation_healthbench_negative_131 import (
    CONFIG as PAPER_CONFIG,
)
from med_red_team.hallucination.config import DetectorExecutionConfig


data = PAPER_CONFIG.to_dict()
data["max_samples"] = 2
data["output_path"] = "logs/hallucination/detector/example_openai_gpt4o_o3_healthbench_negative_2_v2.json"
data["metadata"] = {
    **data.get("metadata", {}),
    "preset": "example_hallucination_detector_validation_healthbench_negative_131",
    "example_samples": 2,
}
CONFIG = DetectorExecutionConfig.from_dict(data)
