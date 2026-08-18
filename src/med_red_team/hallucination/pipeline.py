"""Package-owned hallucination response-generation and detection pipelines."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from med_red_team.model_pool import ModelPool
from med_red_team.models import GenerationConfig, ModelExecutionError, ModelResponse
from med_red_team.models.utils.response_utils import get_parse_error_info, has_parse_error
from med_red_team.shared.checkpoints import checkpoint_path, remove_checkpoint, validate_resume_metadata
from med_red_team.shared.io import (
    atomic_write_json,
    build_result_envelope,
    ensure_result_metadata,
    paths_alias,
    read_json,
    require_distinct_paths,
    validate_output_path,
    validate_result_metadata,
)
from med_red_team.testee import Testee

from .config import (
    DetectorExecutionConfig,
    StanfordResponseGenerationConfig,
    resolve_max_samples,
)
from .data import (
    GeneratedResponseEnvelope,
    GeneratedResponseRecord,
    HallucinationDataError,
    PromptResponseRecord,
    dataset_sha256,
    generated_response_exclusion_reason,
    load_native_prompt_response_json,
    load_prompt_responses,
    sha256_json_value,
)
from .detectors.base import (
    DetectorOutputError,
    HallucinationDetector,
    SourceFormat,
    extract_prompt_response,
    summarize_detector_rows,
)
from .grader import HallucinationGrader
from .schemas import OrchestratorOutput


class BaselineValidationError(ValueError):
    """Raised when the input dataset/config is not safe for generation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_one_model(config: StanfordResponseGenerationConfig) -> str:
    if len(config.model_ids) != 1:
        raise BaselineValidationError(
            "response generation constructs exactly one model per run; "
            "use a config with one model_id"
        )
    model_id = config.model_ids[0]
    if not isinstance(model_id, str) or not model_id.strip():
        raise BaselineValidationError("model_id must be a non-empty string")
    return model_id


def _identity_from_generation_result(row: dict[str, Any]) -> str:
    if "row_idx" in row:
        return str(row["row_idx"])
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if "row_idx" in metadata:
        return str(metadata["row_idx"])
    if "sample_id" in row:
        return str(row["sample_id"])
    raise ValueError("Existing result row lacks row_idx/sample_id identity")


def _length_stats(values: Iterable[int | float]) -> dict[str, float | int]:
    items = list(values)
    if not items:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    items.sort()
    midpoint = len(items) // 2
    median = (
        items[midpoint]
        if len(items) % 2
        else (items[midpoint - 1] + items[midpoint]) / 2
    )
    return {
        "min": items[0],
        "max": items[-1],
        "mean": sum(items) / len(items),
        "median": median,
    }


def _generation_row_is_reusable(row: dict[str, Any]) -> bool:
    if row.get("error"):
        return False
    response = row.get("response")
    if not isinstance(response, str) or not response.strip():
        return False
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    response_metadata = (
        metadata.get("model_response_metadata")
        if isinstance(metadata.get("model_response_metadata"), dict)
        else {}
    )
    return not bool(metadata.get("parse_error") or response_metadata.get("parse_error"))


def _generation_result_summary(
    results: list[dict[str, Any]],
    *,
    planned_count: int,
    source_count: int | None = None,
    ignored_resume_count: int = 0,
) -> dict[str, Any]:
    successes = [row for row in results if _generation_row_is_reusable(row)]
    failures = [row for row in results if not _generation_row_is_reusable(row)]
    response_times: list[float] = []
    for row in results:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        response_time = metadata.get("response_time")
        if isinstance(response_time, (int, float)):
            response_times.append(float(response_time))

    summary: dict[str, Any] = {
        "source_count": source_count if source_count is not None else planned_count,
        "planned_count": planned_count,
        "available_count": len(results),
        "result_count": len(results),
        "successful_count": len(successes),
        "completed_count": len(successes),
        "failed_count": len(failures),
        "pending_count": max(0, planned_count - len(results)),
        "detector_eligible_count": len(successes),
        "ignored_resume_count": ignored_resume_count,
        "prompt_length_stats": _length_stats(
            len(str(row.get("prompt", ""))) for row in results
        ),
        "response_length_stats": _length_stats(
            len(str(row.get("response", ""))) for row in successes
        ),
        "total_prompts": planned_count,
        "total_responses": len(successes),
    }
    if response_times:
        summary["response_time_stats"] = _length_stats(response_times)
    return summary


def _generation_row(
    record: PromptResponseRecord,
    *,
    model_id: str,
    generation_config: GenerationConfig,
    response: ModelResponse | None,
    elapsed: float | None,
    error: Exception | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp": _utc_now(),
        "model": model_id,
        "row_idx": record.row_idx,
        "sample_id": record.identity,
        "source_dataset": record.source_path,
        "source_model": record.model,
        "source_label": record.label,
        "source_additional_comments": record.additional_comments,
        "generation_config": asdict(generation_config),
    }
    if elapsed is not None:
        metadata["response_time"] = elapsed
    if response is not None:
        metadata["model_response_metadata"] = dict(response.metadata or {})
        if response.reasoning:
            metadata["reasoning"] = response.reasoning

    parse_failed = response is not None and has_parse_error(response)
    if error is None and response is not None and not parse_failed:
        generated_text = response.final_answer
        if isinstance(generated_text, str) and generated_text.strip():
            error_payload = None
        else:
            generated_text = ""
            error_payload = {
                "type": "EmptyResponseError",
                "message": "Generation returned no nonblank final answer",
            }
    elif parse_failed and response is not None:
        parse_info = get_parse_error_info(response)
        generated_text = ""
        error_payload = {
            "type": "ResponseParseError",
            "message": parse_info.get(
                "error_message",
                "Generation response could not be parsed",
            ),
        }
    else:
        generated_text = ""
        error_payload = {
            "type": type(error).__name__ if error is not None else "UnknownError",
            "message": (
                str(error)
                if error is not None
                else "Generation did not return a response"
            ),
        }

    return {
        "prompt": record.prompt,
        "response": generated_text,
        "raw_response": response.raw_text if response is not None else "",
        "reasoning": response.reasoning if response is not None else None,
        "row_idx": record.row_idx,
        "sample_id": record.identity,
        "metadata": metadata,
        "error": error_payload,
    }


def _is_systemic_generation_error(exc: BaseException) -> bool:
    return isinstance(exc, (ModelExecutionError, ImportError))


class HallucinationResponseGenerationPipeline:
    """Generate responses for configured prompt records."""

    def __init__(
        self,
        *,
        model_pool_factory: Callable[[], Any] = ModelPool,
        testee_type: type[Any] = Testee,
    ) -> None:
        self.model_pool_factory = model_pool_factory
        self.testee_type = testee_type

    def build_generation_plan(
        self,
        config: StanfordResponseGenerationConfig,
    ) -> dict[str, Any]:
        model_id = _require_one_model(config)
        dataset_path = Path(config.dataset_path)
        try:
            records = load_native_prompt_response_json(
                dataset_path,
                prompt_key=config.prompt_key,
                response_key=config.response_key,
                row_id_key=config.row_id_key,
                parse_label=False,
            )
        except HallucinationDataError as exc:
            raise BaselineValidationError(str(exc)) from exc

        planned_count = resolve_max_samples(config.max_samples, len(records))
        planned_records = records[:planned_count]
        positive_rows = sum(
            1 for record in records if record.label is not None and record.label > 0
        )
        return {
            "model_id": model_id,
            "dataset_path": dataset_path.as_posix(),
            "dataset_sha256": dataset_sha256(dataset_path),
            "total_rows": len(records),
            "positive_rows": positive_rows,
            "max_samples": config.max_samples,
            "source_keys": {
                "prompt": config.prompt_key,
                "response": config.response_key,
                "identity": config.row_id_key,
            },
            "planned_count": planned_count,
            "planned_identities": [record.identity for record in planned_records],
            "planned_prompts_sha256": sha256_json_value(
                [
                    {"identity": record.identity, "prompt": record.prompt}
                    for record in planned_records
                ]
            ),
        }

    @staticmethod
    def _load_existing_envelope(
        path: str | Path | None,
        *,
        config: StanfordResponseGenerationConfig | None = None,
        plan: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        if path is None:
            return None, []
        payload = read_json(path)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError(
                f"Resume/checkpoint file must be a v2 envelope with a results list: {path}"
            )

        if config is not None and plan is not None:
            metadata = payload.get("metadata")
            summary = payload.get("summary")
            if not isinstance(metadata, dict) or not isinstance(summary, dict):
                raise ValueError(
                    f"Resume/checkpoint file requires v2 metadata and summary objects: {path}"
                )
            validate_result_metadata(metadata)
            if metadata.get("schema_version") != "2.0":
                raise ValueError(
                    f"Unsupported Hallucination resume schema_version: "
                    f"{metadata.get('schema_version')!r}"
                )
            validate_resume_metadata(
                {
                    "schema_version": "2.0",
                    "axis": "hallucination",
                    "phase": "response_generation",
                },
                metadata,
            )
            models = metadata.get("models")
            resume_config = metadata.get("config")
            if not isinstance(models, dict) or models.get("target_model") != plan["model_id"]:
                raise ValueError("Checkpoint is incompatible with this run (target model differs)")
            expected_config = config.to_dict()
            if not isinstance(resume_config, dict):
                raise ValueError("Checkpoint is incompatible with this run (config metadata is missing)")
            for key in ("system_prompt", "generation_config"):
                if resume_config.get(key) != expected_config.get(key):
                    raise ValueError(f"Checkpoint is incompatible with this run ({key} differs)")

        return payload, [dict(row) for row in payload["results"]]

    @staticmethod
    def _build_envelope(
        *,
        config: StanfordResponseGenerationConfig,
        model_id: str,
        plan: dict[str, Any],
        results: list[dict[str, Any]],
        output_path: Path,
        is_partial: bool,
        ignored_resume_count: int = 0,
    ) -> dict[str, Any]:
        metadata = ensure_result_metadata(
            {
                "schema_version": "2.0",
                "axis": "hallucination",
                "phase": "response_generation",
                "mode": "baseline",
                "target_model": model_id,
                "models": {"target_model": model_id},
                "dataset": {
                    "path": plan["dataset_path"],
                    "sha256": plan["dataset_sha256"],
                    "total_rows": plan["total_rows"],
                    "positive_rows": plan["positive_rows"],
                    "planned_count": plan["planned_count"],
                    "planned_prompts_sha256": plan["planned_prompts_sha256"],
                    "source_keys": plan["source_keys"],
                },
                "source": {
                    "pipeline": "HallucinationResponseGenerationPipeline",
                    "script": "scripts.hallucination.run_baseline",
                },
                "config": config.to_dict(),
                "output_path": output_path.as_posix(),
                "is_partial": is_partial,
                "generated_at": _utc_now(),
            },
            axis="hallucination",
        )
        return build_result_envelope(
            metadata=metadata,
            summary=_generation_result_summary(
                results,
                planned_count=int(plan["planned_count"]),
                source_count=int(plan["total_rows"]),
                ignored_resume_count=ignored_resume_count,
            ),
            data=results,
            data_key="results",
        )

    def run(
        self,
        config: StanfordResponseGenerationConfig,
        *,
        output_path: Path,
        plan: dict[str, Any] | None = None,
        resume_path: Path | None = None,
        save_every: int = 10,
        quiet: bool = False,
    ) -> dict[str, Any]:
        if save_every < 0:
            raise ValueError("save_every must be non-negative")
        require_distinct_paths(
            config.dataset_path,
            output_path,
            input_label="Generation dataset",
            output_label="Generation output",
        )
        if plan is None:
            plan = self.build_generation_plan(config)
        else:
            current_plan = self.build_generation_plan(config)
            plan_keys = (
                "model_id",
                "planned_count",
                "planned_identities",
                "planned_prompts_sha256",
                "max_samples",
                "source_keys",
            )
            if any(plan.get(key) != current_plan.get(key) for key in plan_keys):
                raise ValueError("Generation plan is stale or incompatible with the current config/input")
            plan = current_plan
        model_id = str(plan["model_id"])
        planned_count = int(plan["planned_count"])
        if planned_count == 0:
            return {
                "output_path": None,
                "planned_count": 0,
                "result_count": 0,
                "completed_count": 0,
                "failed_count": 0,
                "message": (
                    "max_samples=0 selected no work; no model was constructed "
                    "and no result was written"
                ),
            }

        planned_records = load_native_prompt_response_json(
            config.dataset_path,
            max_samples=planned_count,
            prompt_key=config.prompt_key,
            response_key=config.response_key,
            row_id_key=config.row_id_key,
            parse_label=False,
        )
        planned_by_identity = {record.identity: record for record in planned_records}

        def generation_candidate_successes(path: Path) -> int | None:
            try:
                _, rows = self._load_existing_envelope(
                    path,
                    config=config,
                    plan=plan,
                )
            except (OSError, TypeError, ValueError):
                return None
            seen: set[str] = set()
            successes = 0
            for row in rows:
                try:
                    identity = _identity_from_generation_result(row)
                except ValueError:
                    return None
                if identity in seen:
                    return None
                seen.add(identity)
                current = planned_by_identity.get(identity)
                if current is None:
                    continue
                if row.get("prompt") != current.prompt:
                    return None
                if _generation_row_is_reusable(row):
                    successes += 1
            return successes

        checkpoint = checkpoint_path(output_path)
        selected_resume_path: Path | None = None
        if resume_path is not None:
            selected_resume_path = resume_path
        elif not config.overwrite:
            if output_path.exists():
                if output_path.is_dir():
                    raise IsADirectoryError(f"Output path is a directory: {output_path}")
                if output_path.is_symlink():
                    raise ValueError(f"Refusing to use symlink output path: {output_path}")
                selected_resume_path = output_path
                if checkpoint.exists():
                    if checkpoint.is_symlink():
                        raise ValueError(
                            f"Refusing to use symlink checkpoint path: {checkpoint}"
                        )
                    final_successes = generation_candidate_successes(output_path)
                    checkpoint_successes = generation_candidate_successes(checkpoint)
                    if checkpoint_successes is not None and (
                        final_successes is None
                        or checkpoint_successes > final_successes
                    ):
                        selected_resume_path = checkpoint
            elif checkpoint.exists():
                selected_resume_path = checkpoint

        if selected_resume_path is not None:
            if selected_resume_path.is_dir():
                raise IsADirectoryError(f"Resume path is a directory: {selected_resume_path}")
            if selected_resume_path.is_symlink():
                raise ValueError(f"Refusing to use symlink resume path: {selected_resume_path}")

        existing_payload, existing_results = self._load_existing_envelope(
            selected_resume_path,
            config=config,
            plan=plan,
        )
        existing_by_identity: dict[str, dict[str, Any]] = {}
        ignored_resume_count = 0
        for row in existing_results:
            identity = _identity_from_generation_result(row)
            if identity in existing_by_identity:
                raise ValueError(
                    f"Resume/checkpoint contains duplicate generation identity {identity!r}"
                )
            current_record = planned_by_identity.get(identity)
            if current_record is None:
                ignored_resume_count += 1
                continue
            if row.get("prompt") != current_record.prompt:
                raise ValueError(
                    "Resume/checkpoint prompt mismatch for generation identity "
                    f"{identity!r}"
                )
            existing_by_identity[identity] = row

        if ignored_resume_count:
            import warnings

            warnings.warn(
                f"Ignoring {ignored_resume_count} resume rows outside the current generation plan",
                RuntimeWarning,
                stacklevel=2,
            )
        reusable_by_identity = {
            identity: row
            for identity, row in existing_by_identity.items()
            if _generation_row_is_reusable(row)
        }
        generated_by_identity = dict(existing_by_identity)

        def ordered_results() -> list[dict[str, Any]]:
            return [
                generated_by_identity[record.identity]
                for record in planned_records
                if record.identity in generated_by_identity
            ]

        def write_partial_checkpoint() -> None:
            checkpoint_results = ordered_results()
            atomic_write_json(
                checkpoint,
                self._build_envelope(
                    config=config,
                    model_id=model_id,
                    plan=plan,
                    results=checkpoint_results,
                    output_path=output_path,
                    is_partial=True,
                    ignored_resume_count=ignored_resume_count,
                ),
            )

        remaining_records = [
            record
            for record in planned_records
            if record.identity not in reusable_by_identity
        ]

        if not quiet:
            print(f"[Generation] Planned rows: {planned_count}")
            if existing_results:
                print(f"[Generation] Resuming with {len(existing_results)} existing rows")
            print(f"[Generation] Rows to generate: {len(remaining_records)}")

        if not remaining_records and existing_payload is not None:
            metadata = existing_payload.get("metadata")
            if (
                selected_resume_path is not None and paths_alias(selected_resume_path, output_path)
                and isinstance(metadata, dict)
                and not metadata.get("is_partial")
                and ignored_resume_count == 0
            ):
                summary = existing_payload.get(
                    "summary",
                    _generation_result_summary(
                        ordered_results(),
                        planned_count=planned_count,
                        source_count=int(plan["total_rows"]),
                        ignored_resume_count=ignored_resume_count,
                    ),
                )
                return {
                    "output_path": output_path.as_posix(),
                    **summary,
                    "message": "all planned rows were already complete; no model was constructed",
                }
            if selected_resume_path is None or not paths_alias(selected_resume_path, output_path):
                validate_output_path(
                    output_path,
                    overwrite=(
                        config.overwrite
                        or selected_resume_path is not None
                        and paths_alias(selected_resume_path, checkpoint)
                        and output_path.exists()
                    ),
                )
            payload = self._build_envelope(
                config=config,
                model_id=model_id,
                plan=plan,
                results=ordered_results(),
                output_path=output_path,
                is_partial=False,
                ignored_resume_count=ignored_resume_count,
            )
            atomic_write_json(output_path, payload)
            remove_checkpoint(output_path)
            return {
                "output_path": output_path.as_posix(),
                **payload["summary"],
                "message": "all planned rows were already present; finalized without model construction",
            }

        if selected_resume_path is None or not paths_alias(selected_resume_path, output_path):
            validate_output_path(
                output_path,
                overwrite=(
                    config.overwrite
                    or selected_resume_path is not None and paths_alias(selected_resume_path, checkpoint)
                    and output_path.exists()
                ),
            )
        else:
            validate_output_path(output_path, overwrite=True)

        model_pool: Any | None = None
        try:
            model_pool = self.model_pool_factory()
            testee = self.testee_type(
                model_id=model_id,
                model_pool=model_pool,
                system_prompt=config.system_prompt,
                config=config.generation_config,
            )
            processed_since_checkpoint = 0
            for position, record in enumerate(remaining_records, start=1):
                if not quiet:
                    print(
                        f"[Generation] {position}/{len(remaining_records)} "
                        f"row_idx={record.row_idx}"
                    )
                started = time.monotonic()
                response: ModelResponse | None = None
                error: Exception | None = None
                try:
                    response = testee.answer(
                        record.prompt,
                        config=config.generation_config,
                    )
                except Exception as exc:
                    if _is_systemic_generation_error(exc):
                        if not quiet:
                            print(f"[Generation] row_idx={record.row_idx} aborted: {exc}")
                        raise
                    error = exc
                    if not quiet:
                        print(f"[Generation] row_idx={record.row_idx} failed: {exc}")
                elapsed = time.monotonic() - started

                generated_by_identity[str(record.row_idx)] = _generation_row(
                    record,
                    model_id=model_id,
                    generation_config=config.generation_config,
                    response=response,
                    elapsed=elapsed,
                    error=error,
                )
                processed_since_checkpoint += 1
                if save_every > 0 and processed_since_checkpoint >= save_every:
                    write_partial_checkpoint()
                    processed_since_checkpoint = 0
        except Exception as exc:
            if _is_systemic_generation_error(exc):
                write_partial_checkpoint()
            raise
        finally:
            if model_pool is not None:
                model_pool.clear()

        payload = self._build_envelope(
            config=config,
            model_id=model_id,
            plan=plan,
            results=ordered_results(),
            output_path=output_path,
            is_partial=False,
            ignored_resume_count=ignored_resume_count,
        )
        atomic_write_json(output_path, payload)
        remove_checkpoint(output_path)
        return {"output_path": output_path.as_posix(), **payload["summary"]}


def _load_detection_records(
    config: DetectorExecutionConfig,
) -> tuple[
    list[PromptResponseRecord] | list[GeneratedResponseRecord],
    SourceFormat,
    dict[str, Any],
]:
    try:
        loaded = load_prompt_responses(
            config.input_path,
            source_format=config.input_format,
            max_samples=None,
            prompt_key=config.prompt_key,
            response_key=config.response_key,
            row_id_key=config.row_id_key,
            parse_label=False,
        )
    except HallucinationDataError as exc:
        raise ValueError(str(exc)) from exc
    if isinstance(loaded, GeneratedResponseEnvelope):
        eligible: list[GeneratedResponseRecord] = []
        exclusion_counts: dict[str, int] = {}
        excluded_identities: list[str] = []
        for record in loaded.responses:
            reason = generated_response_exclusion_reason(record)
            if reason is None:
                eligible.append(record)
            else:
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
                excluded_identities.append(record.identity)
        metadata = loaded.metadata if isinstance(loaded.metadata, dict) else {}
        dataset = metadata.get("dataset") if isinstance(metadata.get("dataset"), dict) else {}
        planned = (
            dataset.get("planned_count")
            if isinstance(dataset.get("planned_count"), int)
            else loaded.statistics.get("planned_count")
            if isinstance(loaded.statistics.get("planned_count"), int)
            else loaded.statistics.get("total_prompts")
            if isinstance(loaded.statistics.get("total_prompts"), int)
            else len(loaded.responses)
        )
        available = len(loaded.responses)
        return eligible, "generated", {
            "source_planned_count": planned,
            "source_available_count": available,
            "source_successful_count": len(eligible),
            "source_failed_count": available - len(eligible),
            "source_pending_count": max(0, planned - available),
            "source_is_partial": bool(metadata.get("is_partial")),
            "excluded_count": available - len(eligible),
            "excluded_by_reason": exclusion_counts,
            "excluded_identities": excluded_identities,
        }
    return loaded, "native", {
        "source_planned_count": len(loaded),
        "source_available_count": len(loaded),
        "source_successful_count": len(loaded),
        "source_failed_count": 0,
        "source_pending_count": 0,
        "source_is_partial": False,
        "excluded_count": 0,
        "excluded_by_reason": {},
        "excluded_identities": [],
    }


def _record_plan_value(
    record: PromptResponseRecord | GeneratedResponseRecord,
) -> dict[str, Any]:
    return {
        "identity": record.identity,
        "prompt": record.prompt,
        "response": record.response,
        "source_format": record.source_format,
    }


def _detector_models(config: DetectorExecutionConfig) -> dict[str, Any]:
    models = {
        "orchestrator_model": config.orchestrator_model,
        "sub_agent_model": config.sub_agent_model,
    }
    if config.backend == "openai":
        models["search_agent_model"] = config.search_agent_model
    return models


def _detector_result_identity(row: dict[str, Any], *, source_format: SourceFormat) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    source_identity = metadata.get("source_identity")
    if source_identity is not None:
        return str(source_identity)
    if source_format == "native":
        row_idx = row.get("row_idx", metadata.get("row_idx"))
        if row_idx is None:
            raise ValueError("Existing native detector row lacks row_idx/source_identity")
        return str(row_idx)
    model = metadata.get("model") or row.get("model") or row.get("Model")
    if not model:
        raise ValueError("Existing generated detector row lacks model identity")
    source_ordinal = metadata.get("source_ordinal")
    if source_ordinal is None:
        raise ValueError(
            "Existing generated detector row lacks metadata.source_ordinal/source_identity"
        )
    return f"{model}:{source_ordinal}"


def _detector_row_is_reusable(row: dict[str, Any]) -> bool:
    if row.get("detection_error"):
        return False
    try:
        OrchestratorOutput.model_validate(
            {
                "merged_codes": row["merged_codes"],
                "rationale": row["rationale"],
                "agent_decisions": row["agent_decisions"],
            }
        )
    except (KeyError, TypeError, ValueError):
        return False
    return True


class HallucinationDetectionPipeline:
    """Grade native or generated prompt-response pairs with one detector."""

    def __init__(
        self,
        grader: HallucinationGrader | None = None,
        *,
        grader_factory: Callable[[DetectorExecutionConfig], HallucinationGrader] | None = None,
    ) -> None:
        self.grader = grader
        self.grader_factory = grader_factory or self._create_grader_from_config

    @staticmethod
    def _create_grader_from_config(
        config: DetectorExecutionConfig,
    ) -> HallucinationGrader:
        if config.backend == "openai":
            from .detectors.openai_agents import OpenAIAgentsHallucinationDetector

            detector: HallucinationDetector = OpenAIAgentsHallucinationDetector(
                orchestrator_model=config.orchestrator_model,
                sub_agent_model=config.sub_agent_model,
                search_agent_model=config.search_agent_model,
            )
        elif config.backend == "claude":
            from .detectors.claude_agent_sdk import ClaudeAgentSDKHallucinationDetector

            detector = ClaudeAgentSDKHallucinationDetector(
                orchestrator_model=config.orchestrator_model,
                sub_agent_model=config.sub_agent_model,
            )
        else:
            raise ValueError(f"Unsupported backend: {config.backend}")
        return HallucinationGrader(detector)

    def build_detection_plan(
        self,
        config: DetectorExecutionConfig,
    ) -> dict[str, Any]:
        require_distinct_paths(
            config.input_path,
            config.output_path,
            input_label="Detection input",
            output_label="Detection output",
        )
        records, source_format, source_info = _load_detection_records(config)
        row_slice = config.resolve_row_slice(len(records))
        selected_records = records[row_slice]
        phase = (
            "native_validation"
            if source_format == "native"
            else "generated_response_detection"
        )
        existing_output_rows: int | None = None
        output_path = Path(config.output_path)
        if output_path.is_file() and not output_path.is_symlink():
            payload = read_json(output_path)
            if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                existing_output_rows = len(payload["results"])
            elif isinstance(payload, list):
                existing_output_rows = len(payload)

        return {
            "backend": config.backend,
            "input_path": config.input_path,
            "input_sha256": dataset_sha256(config.input_path),
            "output_path": config.output_path,
            "input_format": source_format,
            "phase": phase,
            "total_input_rows": source_info["source_available_count"],
            "eligible_input_rows": len(records),
            **source_info,
            "selected_start": row_slice.start or 0,
            "selected_stop_exclusive": row_slice.stop or 0,
            "planned_count": len(selected_records),
            "planned_identities": [record.identity for record in selected_records],
            "planned_rows_sha256": sha256_json_value(
                [_record_plan_value(record) for record in selected_records]
            ),
            "existing_output_rows": existing_output_rows,
            "will_execute_live": False,
            "orchestrator_model": config.orchestrator_model,
            "sub_agent_model": config.sub_agent_model,
            "search_agent_model": (
                config.search_agent_model if config.backend == "openai" else None
            ),
        }

    def run_native_validation(
        self,
        config: DetectorExecutionConfig,
        *,
        plan: dict[str, Any] | None = None,
        grader: HallucinationGrader | None = None,
        resume_path: Path | None = None,
        save_every: int = 10,
        quiet: bool = False,
    ) -> dict[str, Any]:
        return self._run_detection(
            config,
            expected_format="native",
            plan=plan,
            grader=grader,
            resume_path=resume_path,
            save_every=save_every,
            quiet=quiet,
        )

    def run_generated_response_detection(
        self,
        config: DetectorExecutionConfig,
        *,
        plan: dict[str, Any] | None = None,
        grader: HallucinationGrader | None = None,
        resume_path: Path | None = None,
        save_every: int = 10,
        quiet: bool = False,
    ) -> dict[str, Any]:
        return self._run_detection(
            config,
            expected_format="generated",
            plan=plan,
            grader=grader,
            resume_path=resume_path,
            save_every=save_every,
            quiet=quiet,
        )

    @staticmethod
    def _load_resume(
        path: Path,
        *,
        config: DetectorExecutionConfig,
        plan: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        payload = read_json(path)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError(
                "Detector resume/checkpoint must be a v2 envelope; legacy bare-list "
                f"outputs are read-only artifacts: {path}"
            )
        metadata = payload.get("metadata")
        summary = payload.get("summary")
        if not isinstance(metadata, dict) or not isinstance(summary, dict):
            raise ValueError(
                f"Detector resume/checkpoint requires v2 metadata and summary objects: {path}"
            )
        validate_result_metadata(metadata)
        if metadata.get("schema_version") != "2.0":
            raise ValueError(
                f"Unsupported Hallucination detector resume schema_version: "
                f"{metadata.get('schema_version')!r}"
            )
        validate_resume_metadata(
            {
                "schema_version": "2.0",
                "axis": "hallucination",
                "phase": plan["phase"],
            },
            metadata,
        )
        if metadata.get("backend") != config.backend:
            raise ValueError("Checkpoint is incompatible with this run (backend differs)")
        if metadata.get("models") != _detector_models(config):
            raise ValueError("Checkpoint is incompatible with this run (detector models differ)")
        dataset = metadata.get("dataset")
        if not isinstance(dataset, dict) or dataset.get("source_format") != plan["input_format"]:
            raise ValueError("Checkpoint is incompatible with this run (source format differs)")
        resume_config = metadata.get("config")
        if not isinstance(resume_config, dict):
            raise ValueError("Checkpoint is incompatible with this run (config metadata is missing)")
        return payload, [dict(row) for row in payload["results"]]

    @staticmethod
    def _build_detection_envelope(
        *,
        config: DetectorExecutionConfig,
        plan: dict[str, Any],
        results: list[dict[str, Any]],
        output_path: Path,
        grader: HallucinationGrader | None,
        is_partial: bool,
        detector_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        detector = grader.detector if grader is not None else None
        detector_to_dict = getattr(detector, "to_dict", None)
        resolved_detector_metadata = (
            dict(detector_metadata)
            if isinstance(detector_metadata, dict)
            else detector_to_dict()
            if callable(detector_to_dict)
            else {
                "class": detector.__class__.__name__ if detector is not None else None,
                "backend": config.backend,
                "role": "live_detector",
            }
        )
        metadata = ensure_result_metadata(
            {
                "schema_version": "2.0",
                "axis": "hallucination",
                "phase": plan["phase"],
                "mode": "detector",
                "backend": config.backend,
                "detector": resolved_detector_metadata,
                "models": _detector_models(config),
                "dataset": {
                    "path": plan["input_path"],
                    "sha256": plan["input_sha256"],
                    "source_format": plan["input_format"],
                    "total_rows": plan["total_input_rows"],
                    "source_planned_count": plan["source_planned_count"],
                    "source_available_count": plan["source_available_count"],
                    "source_successful_count": plan["source_successful_count"],
                    "source_failed_count": plan["source_failed_count"],
                    "source_pending_count": plan["source_pending_count"],
                    "source_is_partial": plan["source_is_partial"],
                    "eligible_input_rows": plan["eligible_input_rows"],
                    "excluded_count": plan["excluded_count"],
                    "excluded_by_reason": plan["excluded_by_reason"],
                    "excluded_identities": plan["excluded_identities"],
                    "selected_start": plan["selected_start"],
                    "selected_stop_exclusive": plan["selected_stop_exclusive"],
                    "planned_count": plan["planned_count"],
                    "planned_rows_sha256": plan["planned_rows_sha256"],
                },
                "source": {
                    "pipeline": "HallucinationDetectionPipeline",
                    "script": "scripts.hallucination.run_detector",
                },
                "config": config.to_dict(),
                "output_path": output_path.as_posix(),
                "is_partial": is_partial,
                "generated_at": _utc_now(),
            },
            axis="hallucination",
        )
        summary = summarize_detector_rows(results)
        summary.update(
            {
                "source_planned_count": plan["source_planned_count"],
                "source_available_count": plan["source_available_count"],
                "source_successful_count": plan["source_successful_count"],
                "source_failed_count": plan["source_failed_count"],
                "source_pending_count": plan["source_pending_count"],
                "source_is_partial": plan["source_is_partial"],
                "eligible_input_rows": plan["eligible_input_rows"],
                "excluded_count": plan["excluded_count"],
                "excluded_by_reason": plan["excluded_by_reason"],
                "planned_count": plan["planned_count"],
            }
        )
        return build_result_envelope(
            metadata=metadata,
            summary=summary,
            data=results,
            data_key="results",
        )

    @staticmethod
    def _result_row(
        record: PromptResponseRecord | GeneratedResponseRecord,
        grading: Any,
        *,
        source_format: SourceFormat,
    ) -> dict[str, Any]:
        row = dict(record.source_row)
        metadata_value = row.get("metadata")
        metadata = dict(metadata_value) if isinstance(metadata_value, dict) else {}
        metadata["source_identity"] = record.identity
        row["prompt"] = record.prompt
        row["response"] = record.response
        if source_format == "native":
            metadata.setdefault("row_idx", record.row_idx)
        else:
            metadata.setdefault("model", record.model)
            metadata["source_ordinal"] = record.ordinal
        row["metadata"] = metadata
        row.update(
            {
                "merged_codes": grading.merged_codes,
                "rationale": grading.rationale,
                "agent_decisions": [dict(value) for value in grading.agent_decisions],
                "execution_audit": (
                    dict(grading.execution_audit)
                    if isinstance(grading.execution_audit, dict)
                    else None
                ),
            }
        )
        return row

    @staticmethod
    def _error_result_row(
        record: PromptResponseRecord | GeneratedResponseRecord,
        error: DetectorOutputError,
        *,
        source_format: SourceFormat,
    ) -> dict[str, Any]:
        row = dict(record.source_row)
        for key in ("merged_codes", "rationale", "agent_decisions"):
            row.pop(key, None)
        metadata_value = row.get("metadata")
        metadata = dict(metadata_value) if isinstance(metadata_value, dict) else {}
        metadata["source_identity"] = record.identity
        row["prompt"] = record.prompt
        row["response"] = record.response
        if source_format == "native":
            metadata.setdefault("row_idx", record.row_idx)
        else:
            metadata.setdefault("model", record.model)
            if record.ordinal is not None:
                metadata["source_ordinal"] = record.ordinal
        row["metadata"] = metadata
        row["detection_error"] = {
            "type": error.error_type,
            "message": str(error),
            "retryable": True,
        }
        return row

    def _run_detection(
        self,
        config: DetectorExecutionConfig,
        *,
        expected_format: SourceFormat,
        plan: dict[str, Any] | None,
        grader: HallucinationGrader | None,
        resume_path: Path | None,
        save_every: int,
        quiet: bool,
    ) -> dict[str, Any]:
        if save_every < 0:
            raise ValueError("save_every must be non-negative")
        if config.overwrite and resume_path is not None:
            raise ValueError("overwrite=True cannot be combined with an explicit resume path")

        if plan is None:
            plan = self.build_detection_plan(config)
        else:
            current_plan = self.build_detection_plan(config)
            plan_keys = (
                "backend",
                "output_path",
                "input_format",
                "phase",
                "selected_start",
                "selected_stop_exclusive",
                "planned_count",
                "planned_identities",
                "planned_rows_sha256",
                "orchestrator_model",
                "sub_agent_model",
                "search_agent_model",
            )
            if any(plan.get(key) != current_plan.get(key) for key in plan_keys):
                raise ValueError("Detection plan is stale or incompatible with the current config/input")
            plan = current_plan
        if plan["input_format"] != expected_format:
            raise ValueError(
                f"{plan['phase']} input is {plan['input_format']!r}; "
                f"this method requires {expected_format!r} input"
            )
        records, source_format, source_info = _load_detection_records(config)
        row_slice = config.resolve_row_slice(len(records))
        selected_records = records[row_slice]
        if len(selected_records) != int(plan["planned_count"]):
            raise ValueError("Detection plan no longer matches the selected input rows")

        if not selected_records:
            return {
                "output_path": None,
                "input_format": source_format,
                "processed": 0,
                "total_outputs": 0,
                "summary": summarize_detector_rows([]),
                "message": (
                    "max_samples=0 or the selected slice contains no rows; no SDK "
                    "was imported and no output was written"
                ),
            }

        selected_by_identity = {record.identity: record for record in selected_records}

        def detection_candidate_state(
            path: Path,
        ) -> tuple[int, bool] | None:
            try:
                payload, rows = self._load_resume(
                    path,
                    config=config,
                    plan=plan,
                )
            except (OSError, TypeError, ValueError):
                return None
            seen: set[str] = set()
            successes = 0
            for row in rows:
                try:
                    identity = _detector_result_identity(
                        row,
                        source_format=source_format,
                    )
                except ValueError:
                    return None
                if identity in seen:
                    return None
                seen.add(identity)
                current = selected_by_identity.get(identity)
                if current is None:
                    continue
                try:
                    stored_prompt, stored_response = extract_prompt_response(row)
                except ValueError:
                    return None
                if (
                    stored_prompt != current.prompt
                    or stored_response != current.response
                ):
                    return None
                if _detector_row_is_reusable(row):
                    successes += 1
            metadata = payload.get("metadata")
            is_partial = (
                isinstance(metadata, dict)
                and bool(metadata.get("is_partial"))
            )
            return successes, is_partial

        output_path = Path(config.output_path)
        checkpoint = checkpoint_path(output_path)
        selected_resume_path: Path | None = None
        if not config.overwrite:
            if resume_path is not None:
                selected_resume_path = resume_path
            elif output_path.exists():
                if output_path.is_dir():
                    raise IsADirectoryError(f"Output path is a directory: {output_path}")
                if output_path.is_symlink():
                    raise ValueError(f"Refusing to use symlink output path: {output_path}")
                selected_resume_path = output_path
                if checkpoint.exists():
                    if checkpoint.is_symlink():
                        raise ValueError(
                            f"Refusing to use symlink checkpoint path: {checkpoint}"
                        )
                    final_state = detection_candidate_state(output_path)
                    checkpoint_state = detection_candidate_state(checkpoint)
                    if checkpoint_state is not None and (
                        final_state is None
                        or config.ignore_existing and checkpoint_state[1]
                        or checkpoint_state[0] > final_state[0]
                    ):
                        selected_resume_path = checkpoint
            elif checkpoint.exists():
                selected_resume_path = checkpoint

        existing_payload: dict[str, Any] | None = None
        existing_results: list[dict[str, Any]] = []
        if selected_resume_path is not None:
            if selected_resume_path.is_dir():
                raise IsADirectoryError(f"Resume path is a directory: {selected_resume_path}")
            if selected_resume_path.is_symlink():
                raise ValueError(f"Refusing to use symlink resume path: {selected_resume_path}")
            existing_payload, existing_results = self._load_resume(
                selected_resume_path,
                config=config,
                plan=plan,
            )

        existing_by_identity: dict[str, dict[str, Any]] = {}
        ignored_resume_count = 0
        for row in existing_results:
            identity = _detector_result_identity(row, source_format=source_format)
            if identity in existing_by_identity:
                raise ValueError(
                    f"Detector resume contains duplicate source identity {identity!r}"
                )
            current_record = selected_by_identity.get(identity)
            if current_record is None:
                ignored_resume_count += 1
                continue
            try:
                stored_prompt, stored_response = extract_prompt_response(row)
            except ValueError as exc:
                raise ValueError(
                    f"Detector resume source payload is invalid for identity {identity!r}"
                ) from exc
            if (
                stored_prompt != current_record.prompt
                or stored_response != current_record.response
            ):
                raise ValueError(
                    f"Detector resume source mismatch for identity {identity!r}"
                )
            existing_by_identity[identity] = row

        if ignored_resume_count:
            import warnings

            warnings.warn(
                f"Ignoring {ignored_resume_count} detector resume rows outside the current plan",
                RuntimeWarning,
                stacklevel=2,
            )
        resume_metadata = (
            existing_payload.get("metadata")
            if isinstance(existing_payload, dict)
            else None
        )
        resume_is_partial = (
            isinstance(resume_metadata, dict)
            and bool(resume_metadata.get("is_partial"))
        )
        resume_detector_metadata = (
            resume_metadata.get("detector")
            if isinstance(resume_metadata, dict)
            and isinstance(resume_metadata.get("detector"), dict)
            else None
        )
        reusable_by_identity = (
            {}
            if config.ignore_existing and not resume_is_partial
            else {
                identity: row
                for identity, row in existing_by_identity.items()
                if _detector_row_is_reusable(row)
            }
        )
        remaining_records = [
            record
            for record in selected_records
            if record.identity not in reusable_by_identity
        ]

        if not quiet:
            print(f"[Detection] Planned rows: {len(selected_records)}")
            if existing_results:
                print(f"[Detection] Compatible existing rows: {len(existing_results)}")
            print(f"[Detection] Rows to process: {len(remaining_records)}")

        if not remaining_records and existing_payload is not None:
            metadata = existing_payload.get("metadata")
            if (
                selected_resume_path is not None and paths_alias(selected_resume_path, output_path)
                and isinstance(metadata, dict)
                and not metadata.get("is_partial")
                and ignored_resume_count == 0
            ):
                return {
                    "output_path": output_path.as_posix(),
                    "input_format": source_format,
                    "processed": 0,
                    "total_outputs": len(existing_results),
                    "summary": existing_payload.get(
                        "summary",
                        summarize_detector_rows(existing_results),
                    ),
                    "message": "all planned rows were already complete; no SDK was imported",
                }

        if selected_resume_path is None or not paths_alias(selected_resume_path, output_path):
            validate_output_path(
                output_path,
                overwrite=(
                    config.overwrite
                    or selected_resume_path is not None and paths_alias(selected_resume_path, checkpoint)
                    and output_path.exists()
                ),
            )
        else:
            validate_output_path(output_path, overwrite=True)

        active_grader = grader or self.grader
        if remaining_records and active_grader is None:
            active_grader = self.grader_factory(config)

        results_by_identity = (
            {}
            if config.ignore_existing and not resume_is_partial
            else dict(existing_by_identity)
        )
        processed = 0

        def ordered_results() -> list[dict[str, Any]]:
            return [
                results_by_identity[record.identity]
                for record in selected_records
                if record.identity in results_by_identity
            ]

        try:
            for position, record in enumerate(remaining_records, start=1):
                if not quiet:
                    print(
                        f"[Detection] {position}/{len(remaining_records)} "
                        f"identity={record.identity}"
                    )
                if active_grader is None:
                    raise RuntimeError("Detection grader was not constructed")
                try:
                    grading = active_grader.grade_prompt_response(
                        record.prompt,
                        record.response,
                        context={
                            "source_identity": record.identity,
                            "source_format": source_format,
                            "source_path": record.source_path,
                        },
                    )
                    results_by_identity[record.identity] = self._result_row(
                        record,
                        grading,
                        source_format=source_format,
                    )
                except DetectorOutputError as exc:
                    results_by_identity[record.identity] = self._error_result_row(
                        record,
                        exc,
                        source_format=source_format,
                    )
                processed += 1
                if save_every > 0 and processed % save_every == 0:
                    atomic_write_json(
                        checkpoint,
                        self._build_detection_envelope(
                            config=config,
                            plan=plan,
                            results=ordered_results(),
                            output_path=output_path,
                            grader=active_grader,
                            is_partial=True,
                            detector_metadata=(
                                resume_detector_metadata
                                if active_grader is None
                                else None
                            ),
                        ),
                    )
        except Exception:
            atomic_write_json(
                checkpoint,
                self._build_detection_envelope(
                    config=config,
                    plan=plan,
                    results=ordered_results(),
                    output_path=output_path,
                    grader=active_grader,
                    is_partial=True,
                    detector_metadata=(
                        resume_detector_metadata if active_grader is None else None
                    ),
                ),
            )
            raise

        final_results = ordered_results()
        payload = self._build_detection_envelope(
            config=config,
            plan=plan,
            results=final_results,
            output_path=output_path,
            grader=active_grader,
            is_partial=False,
            detector_metadata=(
                resume_detector_metadata if active_grader is None else None
            ),
        )
        atomic_write_json(output_path, payload)
        remove_checkpoint(output_path)
        return {
            "output_path": output_path.as_posix(),
            "input_format": source_format,
            "processed": processed,
            "total_outputs": len(final_results),
            "summary": payload["summary"],
        }


__all__ = [
    "BaselineValidationError",
    "HallucinationDetectionPipeline",
    "HallucinationResponseGenerationPipeline",
]
