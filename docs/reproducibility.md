# Reproducibility

A reproducible run requires more than a model name. Archive the full result metadata, exact config preset, dataset identity, package versions, and provider/model revision where available.

## Minimum record

Record:

- repository commit and release tag;
- Python version and installed package versions;
- exact preset path plus all CLI overrides;
- serialized `metadata.config`;
- model IDs and provider-reported model versions;
- construction settings, including seed and local model path/revision;
- dataset source, split, hash, filters, and eligible count;
- for hallucination artifact analyses, `artifacts/hallucination/manifest.json`, the artifact-root hash/check status, legacy schema role, and the exact Stanford/HealthBench subset counts used;
- result schema version and axis phase;
- interruption/resume history;
- date and provider region if relevant.

## Presets

Use `configs/paper/` for manuscript-oriented runs. Example presets process at most two samples where live execution is supported and are not manuscript-equivalent.

Hallucination presets include an offline artifact-summary preset plus live-capable response-generation and detector-validation presets. Offline validation and summarization do not call providers. The detector runner is a dry run unless `--execute-live` is passed, and the generation runner supports `--validate-only`. Claude execution requires the Claude extra and `--backend claude`.

Copy a preset when changing scientific settings and keep the resulting file with the experiment record. Every custom preset must still export `CONFIG`.

## Determinism

A fixed infrastructure seed improves repeatability for supported local runtimes but cannot make all hosted APIs deterministic. Record provider behavior, generation parameters, and repeated-attempt counts. Do not assume temperature zero guarantees identical outputs across provider revisions.

Hallucination artifact analyses use the files and denominators declared in `artifacts/hallucination/manifest.json`. The manifest records dataset counts, subset membership, Prompt/Response comparisons, and per-model coverage.

## Checkpoints

Interrupted runs write a sibling `.inprogress` file. Resume only when common metadata and axis-local identity checks pass. The axis controls its deduplication key:

- robustness: case or case/strategy identity depending on the workflow;
- privacy: case and staged strategy/attempt semantics;
- bias: case, question, and strategy;
- HealthBench: test case plus collect/grade/replay stage semantics.
- hallucination: response-generation identity uses the configured stable row identifier; native detector identity uses that identifier; generated-response detector identity uses explicit source identity when available. Resume validates identity-to-input binding and model settings. Bundled artifacts are analysis inputs, not checkpoints.

Do not edit a checkpoint manually unless the modification is documented and validated.

## Verification before publication

Run the following verification commands from a clean repository checkout. Each expected check should complete successfully.

### Compile and provider-free tests

```bash
python -m compileall src scripts configs tests
python -m pytest -m "not live"
```

The default pytest invocation is provider-free. Live checks are opt-in only, after installing the relevant extras and configuring credentials:

```bash
python -m pytest -m live
```

Do not make live tests part of the default command or CI gate.

### Source-tree runner smoke checks

`python -m scripts...` requires the source checkout and repository root. Exercise every Python script module's help path without constructing a model or starting a run:

```bash
while IFS= read -r file; do
  module=${file%.py}
  module=${module//\//.}
  python -m "$module" --help >/dev/null
done < <(find scripts -type f -name '*.py' ! -name '__init__.py' | sort)
```

This includes the axis runners, HealthBench split/replay tools, hallucination artifact tools, and auxiliary source-tree utilities. Keep documented CLI spelling hyphenated (`--baseline-results`, `--data-file`, `--max-samples`, and similar); do not add underscore aliases to examples or release notes.

### Zero-work and validation-only checks

Where a runner accepts a sample limit, exercise `--max-samples 0` with a safe example or paper config and confirm it exits without constructing a provider model or writing an unintended runtime result. Use the non-generating checks that are specific to each workflow:

```bash
python -m scripts.hallucination.run_baseline \
  --config configs/paper/hallucination/response_generation.py \
  --validate-only

python -m scripts.hallucination.run_detector \
  --config configs/paper/hallucination/detector_validation_stanford_positive_131.py
```

The detector command is a provider-free dry run unless `--execute-live` is supplied. The generation command performs plan validation with `--validate-only`. Live outputs must be written outside `artifacts/hallucination`. For split or staged runners, use the documented zero-work mode where available and verify that resume/checkpoint paths are not created unexpectedly.

### Hallucination bundle validation

The artifact manifest defines bundle membership and integrity metadata. Validate JSON parsing, every declared SHA-256, manifest totals and row counts, dataset subset metadata, public-safety scans, and the physical-file allowlist:

```bash
python -m scripts.hallucination.validate_artifacts \
  --manifest artifacts/hallucination/manifest.json \
  --json
```

The manifest records 131 Stanford source rows, 131 HealthBench negative rows, 15 generated-response files with 1,965 rows, and 15 generated-response detector files with 1,918 rows. Detection denominators are file-specific and range from 123 to 131. Validation also rejects undeclared files, local absolute paths, credential-like tokens, and references excluded by the artifact policy. Offline summarization remains provider-free:

```bash
python -m scripts.hallucination.summarize_artifacts \
  --config configs/paper/hallucination/artifact_summary.py
```

### Clean wheel contract

Build and install a wheel outside the checkout in a clean virtual environment. Confirm that the wheel imports `med_red_team` but does not provide `scripts`, `configs`, `docs`, `artifacts`, or console entry points:

```bash
tmpdir=$(mktemp -d)
python -m build --wheel --outdir "$tmpdir/dist"
python -m venv "$tmpdir/venv"
"$tmpdir/venv/bin/python" -m pip install --no-deps "$tmpdir/dist/"*.whl
(
  cd "$tmpdir"
  "$tmpdir/venv/bin/python" - <<'PY'
import importlib.util
import med_red_team

assert importlib.util.find_spec("med_red_team") is not None
assert importlib.util.find_spec("scripts") is None
assert importlib.util.find_spec("configs") is None
PY
)
if unzip -l "$tmpdir/dist/"*.whl | grep -E '(^|/)(scripts|configs|docs|artifacts)/|/(bin|entry_points)'; then
  echo "Unexpected non-library wheel contents or entry-point files" >&2
  exit 1
fi
```

The wheel-content check should produce no output. Inspect wheel `METADATA` and `RECORD` as needed; this release has no console entry points.

### Release-tree and runtime-output scans

Use the provider-free release-tree scanner to identify source-tree runtime output, sensitive material, credentials, model weights, and files outside the Hallucination artifact manifest. The scanner checks both the physical tree and git-tracked paths:

```bash
python -m scripts.validate_release_tree \
  --manifest artifacts/hallucination/manifest.json \
  --json
python -m scripts.validate_release_tree \
  --manifest artifacts/hallucination/manifest.json \
  --tracked-only --json
```

The scanner checks runtime-output paths, sensitive-data suffixes, local absolute paths, credential-like values, and model-weight suffixes. The following shell commands perform targeted checks:

```bash
# Release-tree directories and runtime outputs outside the distribution contract.
if git ls-files | grep -E '(^|/)(private|internal|tmp|logs|outputs|runs|checkpoints)/|\.inprogress$'; then
  echo "Release-tree or runtime-output path outside the distribution contract" >&2
  exit 1
fi

# Private absolute paths and credential-like tokens.
git grep -nE '(/home/[^/]+/|/Users/[^/]+/|[A-Za-z]:\\Users\\|sk-[A-Za-z0-9_-]{12,}|AIza[0-9A-Za-z_-]{20,}|hf_[A-Za-z0-9]{20,})' \
  -- ':!.env.example' ':!.github/workflows/test.yml' && exit 1 || true

# Model weights and large binary checkpoints must not be tracked.
git ls-files | grep -E '\.(safetensors|ckpt|pt|pth|bin|onnx|gguf)$' && exit 1 || true
```

Treat a match as a validation failure unless the file is declared by an artifact manifest. Store runtime outputs outside the release tree.

### Build metadata and whitespace

Build distributions from a source tree without existing `build/`, `dist/`, or `*.egg-info/` directories, then verify generated metadata and entry points:

```bash
find . -maxdepth 3 \( -name build -o -name dist -o -name '*.egg-info' \) -print
python -m build --sdist --wheel
python -m pip check

git diff --check
```

The first command is a review point: a release checkout should not carry stale `build/`, `dist/`, or `*.egg-info/` directories. Re-run the wheel import check after the final rebuild and use `git diff --check` as the last textual-integrity gate. Also confirm that baseline outputs are accepted by their corresponding attack runners without committing the resulting runtime files.
