from configs.paper.robustness.baseline import CONFIG as PAPER_CONFIG
from med_red_team.robustness.config import RobustnessConfig


data = PAPER_CONFIG.to_dict()
data["max_samples"] = 2
CONFIG = RobustnessConfig.from_dict(data)
