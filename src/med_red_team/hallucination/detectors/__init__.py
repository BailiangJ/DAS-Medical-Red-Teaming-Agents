"""Hallucination detector interfaces and provider implementations."""

from .base import (
    HallucinationDetector,
    SourceFormat,
    coerce_orchestrator_output,
    extract_prompt_response,
    format_detector_payload,
    infer_row_source_format,
    merge_orchestrator_output_into_row,
    read_input_rows,
    row_identity,
    summarize_detector_rows,
)
from .claude_agent_sdk import ClaudeAgentSDKHallucinationDetector
from .openai_agents import OpenAIAgentsHallucinationDetector


__all__ = [
    "ClaudeAgentSDKHallucinationDetector",
    "HallucinationDetector",
    "OpenAIAgentsHallucinationDetector",
    "SourceFormat",
    "coerce_orchestrator_output",
    "extract_prompt_response",
    "format_detector_payload",
    "infer_row_source_format",
    "merge_orchestrator_output_into_row",
    "read_input_rows",
    "row_identity",
    "summarize_detector_rows",
]
