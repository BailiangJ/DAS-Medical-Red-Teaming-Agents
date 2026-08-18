# Configuration

Configuration stays readable and axis-local. There is no universal experiment config class.

## Preset contract

A Python preset must expose exactly one public axis configuration object named `CONFIG`:

```python
from med_red_team.models import GenerationConfig
from med_red_team.robustness.config import RobustnessConfig

CONFIG = RobustnessConfig(
    testee_model="gpt-4o",
    testee_config=GenerationConfig(temperature=0.0, max_tokens=256),
    dataset_path="data/medqa_test.jsonl",
    max_samples=2,
)
```

The loader rejects a missing `CONFIG` and rejects an object of the wrong axis type. It does not scan a module for the first compatible object.

## Preset groups

- `configs/examples/` contains two-sample wiring checks. These still call the configured models unless tests inject fakes; hallucination artifact-summary presets are offline.
- `configs/paper/` contains manuscript-oriented models, generation settings, strategy sets, filters, voting rules, artifact manifests, and evaluation limits.

Most dataset files are not included. Hallucination presets point to the public immutable artifact bundle under `artifacts/hallucination`; other paths in presets are placeholders or can be supplied through documented environment variables and command-line overrides.

## Precedence

Runners apply settings in this order:

1. defaults from the axis dataclass;
2. the selected preset's `CONFIG`;
3. explicit command-line arguments.

The final effective config is stored under `metadata.config` in result files. Public runner options use hyphenated CLI spelling (`--baseline-results`, `--data-file`, `--max-samples`, `--individual-attack-results`, and similar); result metadata keys such as `baseline_results` remain axis-specific JSON names and are not CLI aliases.

## Sample limits

The public convention is consistent across runners:

- `None`: process all eligible items;
- `0`: process no items and avoid constructing models where the runner can determine this early;
- positive integer: process at most that many eligible items.

For hallucination, `None` on the paper response-generation preset means all 131 Stanford-positive rows. The artifact-summary preset also accepts `max_samples` for quick offline smoke checks, but manuscript-oriented summaries should use all rows.

## Model construction versus generation

`InfrastructureConfig` controls construction-time settings such as model path, tensor parallelism, memory limits, dtype, quantization, and seed. These values participate in `ModelPool` cache identity.

`GenerationConfig` is call-time state. Temperature, token limits, stop sequences, and reasoning settings are passed to `generate()` and must not be supplied to `ModelPool.get_model()`.

## Attack presets

Attack runners must receive a non-empty strategy configuration, either through an attack preset or explicit CLI strategy selection. A baseline preset with no strategies must fail clearly when used as an attack config.

HealthBench attack presets configure exactly one strategy per run. Privacy separates individual and combined stages. Robustness validates fixed-chain tool compatibility. Bias retains its own strategy catalog and voting configuration. Bias example presets such as `default.py`, `fast.py`, `dev.py`, and `comprehensive.py` live under `configs/examples/bias/`; the source package defines `BiasConfig` but does not carry a preset catalog.

## Hallucination presets

Hallucination uses three preset roles:

- `configs/paper/hallucination/artifact_summary.py` validates and summarizes `artifacts/hallucination/manifest.json` offline. It never calls providers.
- `configs/paper/hallucination/response_generation.py` provides a runnable response-generation configuration. The live framework accepts compatible configured prompt/response datasets or subsets; `--validate-only` plans the selected rows without constructing a model.
- `configs/paper/hallucination/detector_validation_stanford_positive_131.py` and `configs/paper/hallucination/detector_validation_healthbench_negative_131.py` provide OpenAI GPT-4o orchestrator and o3 specialist settings for the two native validation inputs. The detector command is a dry run unless `--execute-live` is supplied. Claude execution requires the `hallucination-claude` extra plus `--backend claude`. Detector runs use `HallucinationDetectionPipeline`, write shared result envelopes, and accept `--resume` and `--save-every` controls.

The package API separates `HallucinationResponseGenerationPipeline` from `HallucinationDetectionPipeline`; the latter uses `HallucinationGrader` over a `HallucinationDetector`. OpenAI Agents and Claude Agent SDK detector implementations are available.

## Credentials

Presets contain model identifiers, not credentials. Provider keys are read only from environment variables listed in `.env.example`. Hallucination validation/summarization and detector dry runs require no credentials; live response generation or live detector execution requires the relevant optional extras and environment credentials.
