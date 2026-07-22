# DAS Medical Red-Teaming Agents v2

This branch hosts the modular v2 implementation of Dynamic, Automatic, and Systematic red-teaming for medical language models.

The public v2 release covers:

- MedQA robustness
- Privacy
- Bias and fairness
- HealthBench robustness

Hallucination evaluation is not included in this branch.

> This branch is under release preparation. See `docs/` for configuration, data, and reproducibility guidance as it is finalized.

## Development install

```bash
python -m pip install -e ".[dev]"
```

Provider integrations are optional:

```bash
python -m pip install -e ".[api]"
```

API credentials are read from environment variables. Copy `.env.example` for the supported variable names; never commit populated credentials.

## Configuration sets

- `configs/examples/`: small, low-cost smoke configurations
- `configs/paper/`: manuscript-oriented reproduction configurations

Each axis retains its own readable configuration dataclass and runtime behavior.

## Tests

```bash
python -m pytest -m "not live"
```

Live provider smoke tests are opt-in and are never part of the default test run.
