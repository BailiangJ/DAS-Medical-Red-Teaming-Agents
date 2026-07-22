# Migrating from v1 to v2

The v2 branch is a clean modular implementation and does not preserve v1 script-path compatibility.

Key changes:

- Installable `med_red_team` package under `src/`.
- Axis-specific runners under `scripts/`.
- Example and paper configurations are separated.
- Baseline and attack outputs use a versioned metadata envelope.
- API keys are accepted only through environment variables.

Detailed command and result-schema migration notes will be completed before the v2 release.
