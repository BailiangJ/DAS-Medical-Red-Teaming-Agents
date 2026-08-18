# Manuscript-oriented presets

These presets provide runnable configurations for the published evaluation settings. Datasets, filtered case lists, provider credentials, and generated outputs are supplied separately.

Always record the exact dataset source, local hash, filtering procedure, model revision, and CLI overrides used for a run. Unknown values below must not be inferred from a preset filename.

## Robustness

- Dataset: [MedQA](https://github.com/jind11/MedQA), cited as Jin et al. (2021).
- Paper adversarial scale: 100 first-round-correct MedQA items per evaluated model.
- Baseline: run the selected MedQA test split, then retain baseline-correct questions for the attack stage.
- Orchestrator: up to five attack rounds over the six-tool catalog; invalid or failed manipulations remain separate from the target-tested denominator.
- Dataset identity: the published method does not specify a total baseline row count or a single canonical reformatted JSONL. Record the exact official split or re-upload identifier and its hash.
- Terms: obtain the dataset from an authorized upstream source and verify the current terms for any reformatted or hosted copy you use.

`configs/paper/robustness/attack.py` limits the fixed attack run to 100 eligible items. The baseline preset leaves the source row count uncapped; attack eligibility depends on baseline correctness for the selected model.

## Privacy

- Dataset: author-curated `RT_Privacy.xlsx`; it is not redistributed here.
- Paper scale: 81 privacy-trap scenarios.
- Categories: unauthorized disclosure (16), minimum necessary/over-sharing (13), overheard/overseen (10), misdirected email (8), personal devices without safeguards (9), no valid healthcare reason (7), public PHI disclosure (10), and accidental social-media PHI release (8).
- Stages: baseline and hard-warning evaluation start from all 81; individual attacks use hard-baseline-safe cases; the combined stage uses the intersection safe under all four individual disguises.
- Manuscript-level inclusion gate: run the privacy attacks only for models whose hard-warning jailbreak ratio is below 80%; record this selection outside the per-run preset.
- Survivor counts are model-dependent and are not fixed in a preset.
- Caveat: the workbook is not redistributed here, and scenario-inspiration web articles do not establish redistribution rights for derived prompts. Obtain explicit permission before distributing the workbook or derived case text.

## Bias

- Dataset: mixed-source `RT_Bias.xlsx`; it is not redistributed here.
- Paper scale: 415 multiple-choice question instances.
- Composition: 304 physician-screened/adapted items from [`mamuto11/LLMs_Bias_Bench`](https://huggingface.co/datasets/mamuto11/LLMs_Bias_Bench), plus 111 clinician-authored scenarios.
- Protocol: five baseline votes and five attack votes; all four attack families are enabled.
- Emotional manipulation does not apply to psychiatric or mental-health cases. The excluded count depends on the input dataset and category normalization.
- Terms: verify upstream terms and attribution requirements for the adapted subset, and obtain a separate authorship/release statement for clinician-authored cases.
- The figure 415 counts question instances rather than workbook rows. Because a workbook row may contain multiple questions, the preset leaves `max_samples` unset.

## HealthBench

- Dataset: OpenAI [HealthBench](https://huggingface.co/datasets/openai/healthbench), Consensus subset.
- Public source file referenced by the manuscript workflow: `consensus_2025-05-09-20-00-46.jsonl`.
- Paper scale: sample 200 Consensus cases, discard 8 after physician rubric filtering, and evaluate a final seed set of 192.
- Filtering: `filter_single_turn=True`; retain clinically meaningful rubrics with sufficiently clear binary pass/fail boundaries and discard subjective style criteria.
- Attacks: narrative distraction, cognitive bias, and impossible measurement, one strategy per run. MCQ-specific tools are not used.
- Terms: verify current upstream terms at download time and record them with the authorized source copy.
- Contamination caveat: do not expose benchmark examples in public documentation, screenshots, logs, generated HTML, or committed artifacts. Record the exact source file, filtered-file hash, filtering procedure, and selected 192 identifiers.
- Exact reproduction requires the sampling seed, the selected 192 case identifiers, and the filtered JSONL SHA-256. Record these values with the experiment artifacts.

Counts used for judge-fidelity or human-validity subsamples are validation counts, not the main evaluation seed counts.

## Hallucination

- Artifact root: `artifacts/hallucination`. The manifest defines file membership, hashes, sizes, counts, and provenance metadata.
- Stanford positive source: `artifacts/hallucination/datasets/stanford_redteaming_131_positive_cases.json`, 131 rows.
- HealthBench detector datasets: the current source file has 131 rows; together with 131 Stanford-positive rows, the OpenAI native validation artifacts contain 260 rows (131 Stanford + 129 of 131 HealthBench).
- Generated responses: 15 files contain 1,965 responses over the 131 Stanford-positive prompts.
- Generated-response detections: existing OpenAI detector outputs total 1,918 rows, with per-model denominators ranging from 123 to 131. Use `manifest.json` denominator metadata instead of assuming every model has 131 detections.
- Detector architecture: `HallucinationDetectionPipeline` uses `HallucinationGrader` over the `HallucinationDetector` Protocol. OpenAI Agents and Claude Agent SDK detector implementations are available.
- Presets: `artifact_summary.py` is offline; `response_generation.py`, `detector_validation_stanford_positive_131.py`, and `detector_validation_healthbench_negative_131.py` support live planning and execution. Live outputs use v2 envelopes stored outside `artifacts/hallucination`.
- Redistribution: review Stanford, HealthBench, and provider-output provenance and applicable terms before sharing raw rows or derived outputs. This README does not assert redistribution rights.
