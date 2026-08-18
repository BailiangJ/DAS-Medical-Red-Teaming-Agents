# Data and provenance

Most evaluation datasets are not redistributed in this repository. The manifest-declared Hallucination bundle under `artifacts/hallucination` supports provider-free integrity validation and summarization.

Before running an axis or redistributing an input or output artifact, obtain the data from an authorized source and verify current terms of use, redistribution restrictions, and permitted handling of model outputs. The presence of provenance metadata does not establish redistribution rights. Keep sensitive data and generated artifacts outside Git unless they are listed in a release manifest.

## Robustness / MedQA

The robustness loader expects JSONL multiple-choice cases with a stable case identifier, question text, options, and correct answer label(s). The loader validates labels against the actual option set.

The paper preset uses `MEDQA_DATASET` when set and otherwise points to `data/medqa_test.jsonl` as a local placeholder.

## Privacy

The privacy loader expects an Excel workbook and configurable sheet/prompt-column names. The framework generates synthetic identifiers for evaluation; do not replace these with real patient identifiers.

The paper preset uses `PRIVACY_DATASET` when set and otherwise points to `data/RT_Privacy.xlsx` as a local placeholder. The repository does not include that workbook.

## Bias

The bias loader expects an Excel workbook containing the axis fields used to build clinical cases, questions, category labels, and reference information. Preserve the `{patient}` placeholder where required by identity attacks.

The paper preset uses `BIAS_DATASET` when set and otherwise points to `data/RT_Bias.xlsx` as a local placeholder. The repository does not include that workbook.

## HealthBench

HealthBench uses JSONL records containing a prompt identifier, conversation messages, rubric items with signed point values, and optional ideal/reference completion data. Dataset filtering, including single-turn selection, is recorded in the effective config.

Obtain HealthBench from its official source and follow its terms. Do not commit downloaded source files, filtered copies, generated attack JSONL, grading exports, or HTML review artifacts.

## Hallucination

Hallucination uses the manifest-declared artifact bundle at `artifacts/hallucination`.

Included source and validation data:

- Stanford positive source: `artifacts/hallucination/datasets/stanford_redteaming_131_positive_cases.json`, 131 rows.
- HealthBench near-ideal negatives: `artifacts/hallucination/datasets/healthbench_131_negative_cases.json` contains the current 131-row source subset. HealthBench is used only as native near-ideal negatives for detector validation; there is no HealthBench generated-response set.
- The bundled OpenAI native detector artifacts contain 260 validation rows: 131 Stanford-positive and 129 HealthBench-negative rows.

Generated-response artifacts are Stanford-only: 15 models were prompted with the same 131 Stanford-positive prompts, yielding 1,965 existing responses. Existing generated-response detections total 1,918 rows, and per-model denominators range from 123 to 131. Use `artifacts/hallucination/manifest.json` for file hashes, row counts, and source labels.

The bundle retains its source schemas. Live Hallucination runs write v2 envelopes outside `artifacts/hallucination`. Review Stanford and HealthBench provenance, upstream terms, model-output handling permissions, and redistribution constraints before sharing raw rows or derived outputs.

## Recommended provenance record

For each run, record at least:

- source URL or dataset registry identifier;
- upstream version, release date, or commit;
- local cryptographic hash;
- terms reference;
- selected split and any filtering;
- number of source, eligible, and evaluated cases;
- transformations performed before evaluation.

Never place protected health information, provider request logs, or private prompts in a public result file.
