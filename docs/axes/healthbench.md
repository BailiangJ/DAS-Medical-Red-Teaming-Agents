# HealthBench robustness

The HealthBench axis evaluates medical conversation responses against signed rubric items before and after an attack.

## Attack contract

Each attack run configures exactly one strategy:

- narrative distraction;
- cognitive bias;
- impossible measurement.

Separate runs keep strategy provenance, applicability, and denominators explicit. The pipeline and CLI accept exactly one attack strategy per run and reject multi-strategy configurations.

## Rubric semantics

Positive and negative rubric items have different unsafe-direction transitions. Case-level and rubric-level attack success are both reported.

For `distraction` and `cognitive_bias`, the attacker targets **one selected rubric**. The attack-success question is whether that selected rubric fails in the unsafe direction. Score comparisons therefore use the same rubric basis on both sides:

- baseline score on the selected rubric;
- attacked score on that selected rubric.

If the selected rubric has no positive-point denominator, score degradation is unavailable even though rubric attack success can still be determined.

Scoring requires rubric and grade lists to match one-to-one by length and rubric identity. Empty, short, extra, or mismatched grade lists do not produce scores.

## Workflows

- `HealthBenchRobustnessPipeline.run_baseline(...)`: end-to-end baseline generation and grading over `HealthBenchTestCase` inputs.
- `HealthBenchRobustnessPipeline.run_attack(...)`: end-to-end attacked response generation and grading over HealthBench baseline/test-case data.
- `run_baseline_split`: separate collection and grading stages.
- `run_attack_replay`: **impossible-measurement-only** attacked-generation reuse.

### Baseline-dependent attacks

`distraction` and `cognitive_bias` are baseline-dependent. If a cached baseline artifact is explicitly supplied, it must be complete and compatible with the current run:

- same target model;
- same target generation config and system prompt;
- same grader model and generation config;
- exact current case/rubric identity for every requested test case;
- full coverage of the requested filtered/sample-limited population.

An explicitly supplied baseline artifact must exist and cover the requested population. Cached rows must contain baseline state only; rows containing attacked conversation, rubric, completion, grade, score, or attack-result state are rejected. Stored baseline scores are recomputed from rubric grades and must match. Every requested cached row is validated against the current case, conversation, rubric, and grades before attacker or target work begins.

The CLI validates this contract before provider construction. The public `HealthBenchRobustnessPipeline.run_attack(...)` boundary independently requires callers that pass cached `baseline_results` for baseline-dependent work to also pass the complete artifact `baseline_metadata`; it validates the target model, target generation config and system prompt, grader model, grader generation config, phase, and completeness before accepting any cached row. Baseline-independent `impossible_measurement` runs ignore unused baseline inputs.

### Replay semantics

Replay supports only `impossible_measurement` and reuses attacked conversations across target models without rerunning the attacker.

Because selected-rubric attacks are model-dependent, replay rejects source artifacts from other strategies. Replay also does **not** recollect a replay-model baseline. It grades only the replay target's attacked response against the standalone impossible-measurement rubric and reports attacked-only robustness outcomes rather than baseline-vs-attacked score comparisons.

## Metrics and denominators

Attack summaries distinguish:

- attempted attacks;
- applicable attacks;
- comparable attacks;
- score-comparable cases.

Applicable-but-skipped rows remain in the applicable population instead of disappearing from denominators. Score-impact metrics are only reported when score-comparable cases exist.

## Configuration, resume, and checkpoints

CLI precedence is explicit across the HealthBench entry points: omitted override flags preserve the selected preset, explicit zero/false values remain meaningful, and replay temperature overrides preserve every unrelated generation setting. Sample limits follow the shared non-negative contract in both the CLI and public pipeline; negative or boolean limits are rejected rather than interpreted as Python slices.

Split baseline and impossible-measurement replay checkpoints validate the exact workflow stage and output-affecting contract:

- baseline collection freezes the target model/config/prompt and effective dataset population;
- baseline grading freezes the collected target settings, current grader settings, grading limit, and exact response artifact;
- replay collection freezes the replay target settings, source impossible-measurement strategy/config, skip policy, limit, and exact source attack artifact;
- replay grading freezes the replay target settings, current grader settings, limit, source strategy/config, and exact replay-response artifact.

The main baseline and attack resumes also freeze target/grader generation settings, target system prompt, sample limit, and current effective dataset content before a completed checkpoint can be finalized. Content hashes bind a resumed stage to its immediate scientific input while allowing an unchanged artifact to be relocated; path drift alone is not a rejection criterion. Missing required content identity fails closed. Resume rows must have unique `test_case_id` values and belong to the current source population.

HealthBench-local checkpoint merging works for both dict response rows and typed result objects, de-duplicates by `test_case_id`, and keeps `sample_number` contiguous. Replay-grade metadata names the replay-collection response artifact as its immediate parent and preserves the original attack lineage separately under `source.upstream_attack`. Completed partial split/replay grade checkpoints are finalized without reconstructing providers, and all main/split/replay zero-remaining-work paths avoid target or grader model construction.

## Presets

Baseline and one-strategy presets are available under both `configs/examples/healthbench/` and `configs/paper/healthbench/`.
