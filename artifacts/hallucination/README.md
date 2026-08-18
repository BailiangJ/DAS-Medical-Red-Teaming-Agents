# Hallucination artifact bundle

This directory contains the manifest-declared datasets, detector outputs, generated responses, and provenance metadata used by the Hallucination axis.

## Contents

- `datasets/stanford_redteaming_131_positive_cases.json`: Stanford-positive source dataset, 131 rows.
- `datasets/healthbench_131_negative_cases.json`: HealthBench-negative source dataset, 131 rows.
- `native_detector/openai_gpt4o_o3/`: OpenAI GPT-4o/O3 native detector outputs.
- `native_detector/cross_vendor_claude/`: Claude detector outputs and run metadata.
- `generated_responses/stanford_positive_131/`: 15 generated-response files containing 1,965 rows.
- `generated_response_detection/openai_gpt4o_o3/`: 15 generated-response detector files containing 1,918 rows. Per-file coverage is recorded in `manifest.json`.

## Scope

Bundle membership is defined by `manifest.json`. Source paths are logical repository-relative labels.

## Validation

The manifest records file hashes, sizes, row counts, source labels, and copy or derivation modes. Provider-free validation checks these values before summarization.
