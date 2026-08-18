# Changelog

## 2.0.0 - Unreleased

- Introduce an installable v2 package under `src/med_red_team`.
- Add modular robustness, privacy, bias, HealthBench, and hallucination implementations.
- Add provider-neutral model interfaces, lazy provider imports, construction-aware model pooling, and environment-only credentials.
- Add strict structured-output parsing, safe atomic JSON writes, a minimal result metadata envelope, and axis-owned checkpoint/resume behavior.
- Separate two-sample example presets from manuscript-oriented paper presets.
- Add Hallucination artifact validation and loss-aware summarization, plus opt-in live runners that write v2 response-generation and detector outputs.
- Add the manifest-declared Hallucination bundle under `artifacts/hallucination`, including 131 Stanford rows, 131 HealthBench negative rows, a 129-row HealthBench subset, 1,965 generated-response rows across 15 files, and 1,918 detection rows across 15 files with per-model denominators of 123–131.
- Add provider-free shared and axis regression coverage.
- Define the distribution contract: repository-root `python -m scripts...` runners, a library-only wheel, no bundled scripts/configs/docs/artifacts, and no console entry points.
- Add public configuration, data, reproducibility, migration, security, and axis documentation.
- Define the public distribution boundary for datasets, generated outputs, logs, credentials, and supported provider integrations.

## 1.x

Version 1.x remains available on its public branch.
