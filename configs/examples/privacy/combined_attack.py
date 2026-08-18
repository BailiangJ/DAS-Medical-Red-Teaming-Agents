from configs.paper.privacy.combined_attack import CONFIG as PAPER_CONFIG
from med_red_team.privacy.config import PrivacyConfig


data = PAPER_CONFIG.to_dict()
data["max_samples"] = 2
CONFIG = PrivacyConfig.from_dict(data)
