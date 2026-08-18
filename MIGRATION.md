# Migrating from Earlier Public Releases

This guide is for users moving from the original script-oriented public code, local copies of the manuscript experiment scripts, or other earlier checkouts into the current public paper repository.

The current repository uses an installable `med_red_team` package, module-based runners, environment credentials, structured result artifacts, and axis-specific configuration presets. It does not require local credential modules or repository-path edits in user code.

## Main changes

- Install the package from the `src/` layout with `python -m pip install -e ".[dev]"` when using the runners and presets.
- Run axis entry points with `python -m scripts.<axis>.<runner>` from the repository root.
- Use `med_red_team.*` imports for library code.
- Place reusable Python presets under `configs/examples/` or `configs/paper/` and export one object named `CONFIG`.
- Supply credentials only through environment variables.
- Read effective runtime configuration from `metadata.config` in result files.
- Treat summary rates as numeric fractions, not formatted percentage strings.
- Use sibling `.inprogress` checkpoints where a runner supports resume.
- Keep Hallucination release artifacts under `artifacts/hallucination/` immutable; write new live outputs to an untracked location.

## Imports

Use the installable package:

```python
from med_red_team import ModelPool, Testee
from med_red_team.robustness import RobustnessConfig, RobustnessPipeline
```

The root `med_red_team` API contains model factory/config entry points, `ModelPool`, and `Testee`. Axis-specific APIs live under `med_red_team.robustness`, `med_red_team.privacy`, `med_red_team.bias`, `med_red_team.healthbench`, and `med_red_team.hallucination`.

Lower-level validators, prompt constants, metadata helpers, and summary helpers remain importable from concrete submodules when needed, but they are not root/package-level API exports.

Earlier imports such as `from red_team...`, `from models...`, or direct imports from old root-level axis scripts should be replaced with package or concrete-submodule imports.

## Running evaluations

Run modules from the repository root:

```bash
python -m scripts.robustness.run_baseline --help
python -m scripts.privacy.run_baseline --help
python -m scripts.bias.run_baseline --help
python -m scripts.healthbench.run_baseline --help
python -m scripts.hallucination.validate_artifacts --help
```

Do not add repository paths to `sys.path` in user code. The runners expect to be invoked as modules from the repository checkout.

## Configuration

A preset must export `CONFIG` of the expected axis type. The loader rejects a missing `CONFIG` and rejects an object of the wrong type; it does not scan a module for the first compatible object.

Public presets are split into:

- `configs/examples/` for small wiring checks and usage examples;
- `configs/paper/` for manuscript-oriented models, generation settings, strategy sets, filters, voting rules, artifact manifests, and evaluation limits.

Runners apply settings in this order:

1. defaults from the axis dataclass;
2. the selected preset's `CONFIG`;
3. explicit command-line arguments.

## Credentials

Remove imports of local credential modules. Set provider credentials through environment variables such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, and `HF_TOKEN`.

For one shell session, use `export NAME="..."`. For persistent Bash sessions, add the relevant exports to `~/.bash_profile` or `~/.bashrc` and reload with `source ~/.bash_profile` or `source ~/.bashrc`. For a local `.env` workflow, copy `.env.example` to an ignored `.env`, edit it locally, and load it with `set -a; source .env; set +a`.

Use `python -m scripts.check_provider_env` to check which supported credential variables are available without printing secrets.

Never copy a populated credential file into the repository.

## Results

Result files use a small shared metadata convention plus axis-specific payloads. The common metadata records the schema identifier, axis, phase, effective configuration, model information, dataset/source provenance, and partial-run status.

Axis data remains axis-specific:

- standard baseline outputs generally use `results`;
- Privacy attack outputs may contain `baseline_results` and `attack_results`;
- HealthBench split/replay outputs use workflow-specific keys;
- Hallucination live response-generation outputs use the shared envelope, while bundled detector outputs, generated responses, generated-response detections, and provenance manifests retain their source schemas under `artifacts/hallucination`.

Bias vote-distribution uncertainty is named vote entropy (`ref_vote_entropy`, `manipulated_vote_entropy`) because the value is Shannon entropy over valid canonical votes, not model perplexity. Hallucination artifact counts and subsets are defined by `artifacts/hallucination/manifest.json`.

## Resume

Runners that support resume write sibling `.inprogress` checkpoints with atomic writes. Resume compatibility is validated before work begins, and duplicate handling uses an axis-owned identity key. Do not reuse a partial file unless it has been explicitly checked against the current run configuration and source data.
