from med_red_team.hallucination.config import ArtifactSummaryConfig


CONFIG = ArtifactSummaryConfig(
    artifact_root="artifacts/hallucination",
    manifest_name="manifest.json",
    validate_hashes=True,
    validate_json=True,
    validate_counts=True,
    scan_public_safety=True,
    raise_on_error=False,
    max_samples=None,
)
