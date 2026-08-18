from configs.paper.healthbench.baseline import CONFIG as PAPER_CONFIG
from med_red_team.healthbench.config import HealthBenchConfig


data = PAPER_CONFIG.to_dict()
data["testee_model"] = "qwen3"
data["max_samples"] = 2
CONFIG = HealthBenchConfig.from_dict(data)
