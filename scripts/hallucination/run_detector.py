#!/usr/bin/env python3
"""Explicit live hallucination detector runner.

The optional provider SDKs are imported only after --execute-live is supplied.
Without that flag this command validates the input/output plan and exits without
calling detectors.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from typing import Any

from med_red_team.hallucination.config import DetectorExecutionConfig, load_config
from med_red_team.hallucination.pipeline import HallucinationDetectionPipeline
from med_red_team.shared.io import to_jsonable


DEFAULT_CONFIG_PATH = "configs/examples/hallucination/detector_validation_stanford_positive_131.py"
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or dry-run the live hallucination detector over explicit JSON inputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.hallucination.run_detector \
      --config configs/examples/hallucination/detector_validation_stanford_positive_131.py
  python -m scripts.hallucination.run_detector \
      --config configs/paper/hallucination/detector_validation_healthbench_negative_131.py \
      --backend openai --execute-live
        """,
    )
    parser.add_argument("--config", help=f"DetectorExecutionConfig .py/.json path (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--input-path", help="Override config.input_path")
    parser.add_argument("--output-path", help="Override config.output_path")
    parser.add_argument("--backend", choices=["openai", "claude"], help="Detector backend override")
    parser.add_argument("--input-format", choices=["auto", "native", "generated"], help="Input format override")
    parser.add_argument("--orchestrator-model", help="Override orchestrator model")
    parser.add_argument("--sub-agent-model", help="Override specialist/sub-agent model")
    parser.add_argument("--search-agent-model", help="Override OpenAI search-capable specialist model")
    parser.add_argument("--max-samples", type=_parse_max_samples, default=_UNSET, help="all/none, 0, or positive row cap")
    parser.add_argument("--start-idx", type=int, help="Inclusive zero-based start index")
    parser.add_argument("--end-idx", type=int, help="Inclusive zero-based end index")
    parser.add_argument("--ignore-existing", action="store_true", help="Regenerate selected keys even if already in output")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing the output JSON file")
    parser.add_argument("--resume", help="Resume from a compatible v2 detector envelope")
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Checkpoint every N processed rows (default: 10; 0 disables periodic checkpoints)",
    )
    parser.add_argument(
        "--execute-live",
        action="store_true",
        help="Actually call the selected detector backend. Omit for a no-SDK dry run.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args(argv)


def _load_config_from_args(args: argparse.Namespace) -> DetectorExecutionConfig:
    if args.config:
        config = load_config(args.config, DetectorExecutionConfig)
        data = config.to_dict()
    elif args.input_path and args.output_path:
        data = {"input_path": args.input_path, "output_path": args.output_path}
    else:
        config_path = Path(DEFAULT_CONFIG_PATH)
        if not config_path.is_file():
            raise ValueError("Either --config or both --input-path and --output-path are required")
        config = load_config(config_path, DetectorExecutionConfig)
        data = config.to_dict()

    if args.input_path:
        data["input_path"] = args.input_path
    if args.output_path:
        data["output_path"] = args.output_path
    if args.backend:
        backend_changed = args.backend != data.get("backend", "openai")
        data["backend"] = args.backend
        if backend_changed and not args.orchestrator_model:
            data["orchestrator_model"] = ""
        if backend_changed and not args.sub_agent_model:
            data["sub_agent_model"] = ""
    if args.input_format:
        data["input_format"] = args.input_format
    if args.orchestrator_model:
        data["orchestrator_model"] = args.orchestrator_model
    if args.sub_agent_model:
        data["sub_agent_model"] = args.sub_agent_model
    if args.search_agent_model:
        data["search_agent_model"] = args.search_agent_model
    if args.max_samples is not _UNSET:
        data["max_samples"] = args.max_samples
    if args.start_idx is not None:
        data["start_idx"] = args.start_idx
    if args.end_idx is not None:
        data["end_idx"] = args.end_idx
    if args.ignore_existing:
        data["ignore_existing"] = True
    if args.overwrite:
        data["overwrite"] = True

    return DetectorExecutionConfig.from_dict(data)


def _print_plan(plan: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(_json_dumps({"plan": plan}))
        return
    print("Hallucination detector plan")
    print(f"  Backend: {plan['backend']}")
    print(f"  Input: {plan['input_path']}")
    print(f"  Output: {plan['output_path']}")
    print(f"  Input format: {plan['input_format']}")
    print(f"  Total input rows: {plan['total_input_rows']}")
    print(f"  Planned rows: {plan['planned_count']}")
    print(f"  Orchestrator model: {plan['orchestrator_model']}")
    print(f"  Sub-agent model: {plan['sub_agent_model']}")
    print(f"  Search-agent model: {plan['search_agent_model']}")
    print("  Live execution: no (pass --execute-live to call the backend)")


def _execute_live(
    config: DetectorExecutionConfig,
    *,
    plan: dict[str, Any] | None = None,
    resume_path: Path | None = None,
    save_every: int = 10,
    quiet: bool = False,
) -> dict[str, Any]:
    pipeline = HallucinationDetectionPipeline()
    plan = plan or pipeline.build_detection_plan(config)
    kwargs = {
        "plan": plan,
        "resume_path": resume_path,
        "save_every": save_every,
        "quiet": quiet,
    }
    if plan["input_format"] == "native":
        return pipeline.run_native_validation(config, **kwargs)
    return pipeline.run_generated_response_detection(config, **kwargs)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = _load_config_from_args(args)
        plan = HallucinationDetectionPipeline().build_detection_plan(config)
        if not args.execute_live:
            _print_plan(plan, as_json=args.json)
            return 0

        plan["will_execute_live"] = True
        if not args.json:
            print("[Detector] Executing live backend. Provider credentials and SDKs are required.")
        live_kwargs = {
            "plan": plan,
            "resume_path": Path(args.resume) if args.resume else None,
            "save_every": args.save_every,
            "quiet": args.json,
        }
        if args.json:
            with redirect_stdout(sys.stderr):
                result = _execute_live(config, **live_kwargs)
        else:
            result = _execute_live(config, **live_kwargs)
        if args.json:
            print(_json_dumps({"plan": plan, "result": result}))
        else:
            print("[Done]")
            print(_json_dumps(result))
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
