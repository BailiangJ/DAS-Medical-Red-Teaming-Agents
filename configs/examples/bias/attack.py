from configs.paper.bias.attack import CONFIG as PAPER_CONFIG
from med_red_team.bias.config import BiasConfig


data = PAPER_CONFIG.to_dict()
data["max_samples"] = 2
data["use_baseline_cache"] = False
CONFIG = BiasConfig.from_dict(data)
