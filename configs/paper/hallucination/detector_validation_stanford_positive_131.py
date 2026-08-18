from med_red_team.hallucination.config import DetectorExecutionConfig


CONFIG = DetectorExecutionConfig(
    input_path="artifacts/hallucination/datasets/stanford_redteaming_131_positive_cases.json",
    output_path="logs/hallucination/detector/openai_gpt4o_o3_stanford_positive_131_v2.json",
    backend="openai",
    input_format="native",
    orchestrator_model="gpt-4o",
    sub_agent_model="o3",
    search_agent_model="o3",
    max_samples=None,
    start_idx=0,
    end_idx=None,
    ignore_existing=False,
    overwrite=False,
    metadata={
        "preset": "paper_hallucination_detector_validation_stanford_positive_131",
        "validation_input": "artifacts/hallucination/datasets/stanford_redteaming_131_positive_cases.json",
        "expected_dataset_rows": 131,
    },
)
