from configs.paper.hallucination.artifact_summary import CONFIG as PAPER_CONFIG
from med_red_team.hallucination.config import ArtifactSummaryConfig


data = PAPER_CONFIG.to_dict()
data["max_samples"] = 2
data["raise_on_error"] = False
CONFIG = ArtifactSummaryConfig.from_dict(data)
