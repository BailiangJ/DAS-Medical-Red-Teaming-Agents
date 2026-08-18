#!/usr/bin/env python3
"""Generate model responses from configurable prompt/response datasets.

The runner writes live outputs in the v2 result envelope and does not modify
bundled Hallucination artifacts.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

from med_red_team.hallucination.config import (
    StanfordResponseGenerationConfig,
    load_config,
)
from med_red_team.hallucination.pipeline import HallucinationResponseGenerationPipeline
from med_red_team.shared.io import to_jsonable


DEFAULT_CONFIG_PATH = "configs/examples/hallucination/response_generation.py"
_UNSET = object()


def _parse_max_samples(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"none", "all", "null"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max samples must be an integer, 'all', or 'none'") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("max samples must be None/all, 0, or a positive integer")
    return parsed


def _json_dumps(value: Any) -> str:
    return json.dumps(to_jsonable(value), indent=2, ensure_ascii=False, sort_keys=True)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("._") or "model"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate responses for a configurable hallucination prompt dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.hallucination.run_baseline --validate-only
  python -m scripts.hallucination.run_baseline \
      --config configs/examples/hallucination/response_generation.py \
      --validate-only --json
  python -m scripts.hallucination.run_baseline \
      --config configs/paper/hallucination/response_generation.py \
      --model-id gpt-4o
        """,
    )
    parser.add_argument("--config", help=f"Config .py/.json path (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--dataset-path", help="Override config.dataset_path")
    parser.add_argument("--output-dir", help="Override config.output_dir")
    parser.add_argument("--output", help="Write to this exact output JSON path instead of auto-naming")
    parser.add_argument("--model-id", help="Override config.model_ids with a single model id")
    parser.add_argument("--system-prompt", help="Override config.system_prompt")
    parser.add_argument(
        "--max-samples",
        type=_parse_max_samples,
        default=_UNSET,
        help="Selected row cap: all/none for every row, 0 for no work, or a positive integer",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing --output if it already exists")
    parser.add_argument("--resume", help="Resume from a v2 result/checkpoint JSON file")
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint every N processed rows (default: 10)")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate dataset/config and print planned counts/hashes without constructing a model or writing results",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON for validation/summary output")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return parser.parse_args(argv)


def load_generation_config(args: argparse.Namespace) -> StanfordResponseGenerationConfig:
    if args.config:
        config = load_config(args.config, StanfordResponseGenerationConfig)
    elif args.dataset_path:
        config = StanfordResponseGenerationConfig(dataset_path=args.dataset_path)
    else:
        config = load_config(DEFAULT_CONFIG_PATH, StanfordResponseGenerationConfig)
    data = config.to_dict()

    if args.dataset_path:
        data["dataset_path"] = args.dataset_path
    if args.output_dir:
        data["output_dir"] = args.output_dir
    if args.model_id:
        data["model_ids"] = [args.model_id]
    if args.system_prompt is not None:
        data["system_prompt"] = args.system_prompt
    if args.max_samples is not _UNSET:
        data["max_samples"] = args.max_samples
    if args.overwrite:
        data["overwrite"] = True

    return StanfordResponseGenerationConfig.from_dict(data)


def _auto_output_path(config: StanfordResponseGenerationConfig, model_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(config.output_dir) / f"results_{_safe_slug(model_id)}_{timestamp}_v2.json"


def print_plan(plan: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(_json_dumps({"validation": plan}))
        return
    print("Hallucination response-generation plan")
    print(f"  Model: {plan['model_id']}")
    print(f"  Dataset: {plan['dataset_path']}")
    print(f"  Dataset SHA256: {plan['dataset_sha256']}")
    print(f"  Total rows: {plan['total_rows']}")
    print(f"  Rows with positive labels: {plan['positive_rows']}")
    print(f"  Max samples: {'all' if plan['max_samples'] is None else plan['max_samples']}")
    print(f"  Planned rows: {plan['planned_count']}")
    print(f"  Planned prompts SHA256: {plan['planned_prompts_sha256']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_generation_config(args)
        pipeline = HallucinationResponseGenerationPipeline()
        plan = pipeline.build_generation_plan(config)
        if args.validate_only:
            print_plan(plan, as_json=args.json)
            return 0

        if not args.quiet and not args.json:
            print_plan(plan, as_json=False)
        output_path = Path(args.output) if args.output else _auto_output_path(config, str(plan["model_id"]))
        run_kwargs = {
            "output_path": output_path,
            "plan": plan,
            "resume_path": Path(args.resume) if args.resume else None,
            "save_every": args.save_every,
            "quiet": args.quiet or args.json,
        }
        if args.json:
            # Keep provider/model progress on stderr so stdout remains valid JSON.
            with redirect_stdout(sys.stderr):
                result = pipeline.run(config, **run_kwargs)
        else:
            result = pipeline.run(config, **run_kwargs)
        if args.json:
            print(_json_dumps({"result": result}))
        elif not args.quiet:
            print("[Done]")
            for key, value in result.items():
                print(f"  {key}: {value}")
        return 0
    except Exception as exc:
        if args.json:
            print(
                _json_dumps(
                    {"error": str(exc), "error_type": type(exc).__name__}
                )
            )
        else:
            print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
