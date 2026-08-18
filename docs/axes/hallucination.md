# Hallucination

The hallucination axis is the fifth public v2 axis. The manifest-declared bundle under `artifacts/hallucination` provides datasets, detector outputs, generated responses, and provenance metadata for provider-free validation and summarization.

Live outputs use flexible v2 envelopes over compatible configured datasets or selected subsets and must be stored outside `artifacts/hallucination`. The supplied OpenAI and Claude configurations are runnable examples; live users may select compatible models and inputs.

## Architecture

The public axis has three layers:

1. **Immutable artifact bundle.** `artifacts/hallucination/manifest.json` records file hashes, row counts, legacy schema roles, logical source labels, known caveats, and per-model denominators. The bundled JSON files preserve their source schemas for provenance rather than being rewritten into the shared v2 result envelope.
2. **Offline validation and summarization.** `scripts.hallucination.validate_artifacts` validates manifest hashes, JSON parsing, manifest-declared file/count/denominator arithmetic, the physical-file allowlist, and public-safety scans. `scripts.hallucination.summarize_artifacts` requires that integrity validation first, computes each metric on the available identity-aligned population, and reports missing, overlap, policy-dropped, null/blank/error, and evaluated denominators. These commands never call providers, models, or detectors.
3. **Framework pipelines and thin runners.** `HallucinationResponseGenerationPipeline` uses `Testee` plus `ModelPool` for configurable prompt datasets and selected subsets. `HallucinationDetectionPipeline` grades valid available native or generated responses and exposes source, eligible, success, and failure coverage. It may consume partial generation checkpoints without treating missing rows as negatives. CLI modules load configs, plan runs, and dispatch these package APIs; detector execution requires `--execute-live`.

Provider orchestration is exposed through the `HallucinationDetector` Protocol. When available, specialist execution metadata is recorded in the `execution_audit` field.

## Datasets and counts

| Component | Public path | Count | Role |
| --- | --- | ---: | --- |
| Stanford positives | `artifacts/hallucination/datasets/stanford_redteaming_131_positive_cases.json` | 131 rows | Positive native detector validation and source prompts for generated responses. |
| HealthBench current negatives | `artifacts/hallucination/datasets/healthbench_131_negative_cases.json` | 131 rows | Native near-ideal negatives for detector validation only. |
| Manuscript-primary native validation | Stanford 131 + HealthBench 129 | 260 rows | Validation total for the OpenAI GPT-4o/o3 primary native detector artifacts; the HealthBench 129-row coverage is read directly from `native_detector/openai_gpt4o_o3/healthbench_negative_manuscript_129/detailed_outputs.json`. |
| Generated responses | `artifacts/hallucination/generated_responses/stanford_positive_131/` | 15 files, 1,965 rows | Existing responses from 15 models prompted only with the 131 Stanford-positive prompts. |
| Generated-response detections | `artifacts/hallucination/generated_response_detection/openai_gpt4o_o3/` | 15 files, 1,918 rows | Existing detector outputs over generated Stanford responses. Per-model denominators range from 123 to 131. |

HealthBench contributes native near-ideal negatives only for detector validation. There are no HealthBench generated-response artifacts in the public hallucination bundle and none should be introduced when reproducing the release.

## Artifact layout

Key bundle paths:

- `artifacts/hallucination/manifest.json`: authoritative hashes, counts, denominator metadata, source labels, and caveats.
- `artifacts/hallucination/README.md`: package-level artifact notes.
- `artifacts/hallucination/native_detector/openai_gpt4o_o3/`: manuscript-primary native detector outputs.
- `artifacts/hallucination/native_detector/cross_vendor_claude/`: Claude detector outputs and run metadata.

Files under `artifacts/hallucination` are release artifacts. Live runs write v2 outputs elsewhere and record their configuration, input provenance, and provider metadata in the result envelope.

## Detector artifact groups

- **OpenAI:** GPT-4o orchestrator with o3 specialist models.
- **Claude:** Claude Agent SDK detector outputs for the same validation datasets.
- Bundle membership is defined by `manifest.json`.

## Data and provenance caveats

- HealthBench current negatives have 131 rows, while the manuscript-primary OpenAI HealthBench detector output covers 129 of those `row_idx` values (excluded: 8 and 129).
- Generated-response detector populations are file-specific. Summaries report each artifact's available/missing coverage, and `uncertain_policy="drop"` uses a separate evaluated denominator that excludes uncertain rows from both positive and negative counts.

## Commands

Run these commands from the repository root.

Offline artifact validation:

```bash
python -m scripts.hallucination.validate_artifacts \
  --manifest artifacts/hallucination/manifest.json
```

Offline artifact summarization:

```bash
python -m scripts.hallucination.summarize_artifacts \
  --config configs/paper/hallucination/artifact_summary.py
```

Optional offline JSON summary:

```bash
python -m scripts.hallucination.summarize_artifacts \
  --config configs/paper/hallucination/artifact_summary.py \
  --json
```

Validate a response-generation plan without constructing a model or writing output:

```bash
python -m scripts.hallucination.run_baseline \
  --config configs/paper/hallucination/response_generation.py \
  --validate-only
```

Dry-run detector validation without calling a backend:

```bash
python -m scripts.hallucination.run_detector \
  --config configs/paper/hallucination/detector_validation_stanford_positive_131.py
```

To execute live generation, omit `--validate-only`; to execute live detection, add `--execute-live`. Generation subsets, interrupted checkpoints, and unequal model populations are valid: reusable rows bind by stable identity plus exact prompt, failed rows are retryable, and outputs are serialized in current-plan order. Detector metrics use successful rows only and report source, eligible, excluded, successful, failed, pending, and policy-dropped populations separately. Detector runs write v2 envelopes and can resume compatible state with `--resume` or automatic output/checkpoint discovery. Live runs require credentials and optional dependencies, can incur provider cost, and must write outside the artifact bundle.

## Optional dependencies

- Offline validation and summarization use the base/dev install and are provider-free.
- Live hallucination generation or default OpenAI detector execution requires `python -m pip install -e ".[hallucination]"` plus the relevant provider credentials. Claude detector execution is an explicit opt-in via `python -m pip install -e ".[hallucination-claude]"`.
- Optional chi-square or manuscript-analysis checks use `python -m pip install -e ".[analysis]"`.

## Provenance and redistribution

The artifact manifest uses logical source labels rather than private absolute paths. It records hashes and copy/derivation modes, but it does not settle redistribution rights. Review Stanford, HealthBench, and provider-output provenance, current terms of use, and model-output handling policies before sharing raw rows or derived outputs.
