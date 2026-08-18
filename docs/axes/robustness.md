# Robustness

The Robustness axis evaluates whether a model that answers an original medical multiple-choice question correctly remains correct after an adversarial transformation. Results are paired: each attacked outcome retains the target's original response and correctness together with the complete manipulated question, ordered options, correct answer, and target response.

## Workflows

- `run_baseline`: evaluate original questions for one target model. This runner is original-question-only.
- `run_attack`: generate a validated fixed attack chain and evaluate it on that target's baseline-correct cases in one run.
- `run_generate_attacks`: generate a target-independent attacked dataset without evaluating a testee.
- `run_attack_replay`: combine a completed original baseline with a completed generated-attack artifact, then evaluate the reusable attacks on that target's baseline-correct cases.
- `run_orchestrator`: adaptively plan attacks with bounded retries and tool-policy validation.

## Reusing generated attacks across targets

Generate a reusable attack artifact once:

```bash
python -m scripts.robustness.run_generate_attacks \
  --config configs/examples/robustness/attack.py \
  --dataset /path/to/medqa_test.jsonl
```

For each target model, first produce its original-question baseline:

```bash
python -m scripts.robustness.run_baseline \
  --config configs/examples/robustness/baseline.py \
  --dataset /path/to/medqa_test.jsonl \
  --testee-model model-B
```

Then pair that baseline with the generated attacks:

```bash
python -m scripts.robustness.run_attack_replay \
  --baseline-results /path/to/model-B-robustness-baseline.json \
  --attacked-dataset /path/to/generated-attacks.json
```

The replay runner accepts no configuration or model/strategy overrides. Target generation settings, system prompt, and grader are inherited from the baseline artifact. Strategy order, strategy configuration, and row-level attack metadata are inherited from the generated artifact.

`run_baseline` evaluates original questions. A paired Robustness evaluation uses a baseline that establishes pre-attack correctness for the selected target.

## Replay population semantics

Replay requires a complete strict-v2 `phase="baseline"`, `evaluation_mode="original"` artifact and a complete strict-v2 `phase="attack_generation"` artifact derived from the same source dataset content.

The runner:

1. traverses generated attack rows in artifact order;
2. joins them to baseline rows by exact `test_case_id`;
3. verifies exact equality of the original question, ordered options, and correct answer;
4. keeps only rows satisfying the canonical baseline-correct eligibility rule;
5. applies `--max-samples` after that eligible join; and
6. freezes the resulting ordered population before failure, no-op, and resume filtering.

The population is never backfilled. If one selected transformation failed generation, made no material change, was already completed in a resume artifact, or produces an invalid target response, a later attack row does not replace it.

A generated row with `manipulation_failed=true` is always a source failure, even if it contains a partial transformation. A row declared successful but with unchanged question, options, and correct answer is recorded as `invalid_noop`. Both remain paired to the target's correct baseline row, remain in the fixed eligible denominator, and skip the target call.

Replay resume compatibility is bound to both input artifact content hashes, the target and grader settings, generated strategy provenance, source dataset hash, replay sample limit, selected population size, and ordered population fingerprint. Partial upstream artifacts, incompatible resumes, duplicate IDs, and content mismatches are rejected before provider construction.

## Metrics

Answer comparison is set-aware and validates labels against the attacked case's actual options. Option-changing and correct-answer-changing transformations are therefore graded against the complete attacked multiple-choice payload rather than the original answer key.

Robustness summaries distinguish:

- **attack coverage** = valid target-tested attacks / eligible baseline-correct population;
- **conditional attack success** = fooled / valid target-tested attacks;
- **conditional robustness** = target-safe / valid target-tested attacks; and
- **end-to-end attack success** = fooled / eligible baseline-correct population.

Failed/no-op manipulations and invalid target outputs remain visible in coverage and end-to-end denominators rather than disappearing. Replay checkpoints use the same frozen eligible denominator as the final artifact, so partial progress does not inflate rates.

Tool-policy checks for live fixed and orchestrated attacks reject unknown, duplicate, incompatible, or invalid transformations. Malformed target output is recorded as a parser error and is never counted as a successful attack. Provider execution failures remain fail-fast.

## Presets

- `configs/examples/robustness/baseline.py`
- `configs/examples/robustness/attack.py`
- manuscript-oriented equivalents under `configs/paper/robustness/`

The adaptive orchestrator requires the optional `openai-agents` dependency included in the API extra.
