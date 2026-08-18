"""Claude Agent SDK implementation of the hallucination detector Protocol."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import json
from typing import Any

from ..prompts import (
    citation_description,
    citation_prompt,
    context_description,
    context_prompt,
    hallucination_description,
    hallucination_prompt,
    instruction_description,
    instruction_prompt,
    medfact_description,
    medfact_prompt,
    orchestrator_prompt,
    reasoning_description,
    reasoning_prompt,
    safety_description,
    safety_prompt,
)
from ..schemas import OrchestratorOutput
from .audit import (
    SPECIALIST_NAME_TO_CODE,
    attach_execution_audit,
    observed_specialist_result,
)
from .base import DetectorOutputError, format_detector_payload


DEFAULT_SUB_AGENT_MODEL = "inherit"
DEFAULT_ORCHESTRATOR_MODEL = "claude-sonnet-4-20250514"


def _import_claude_agent_sdk() -> tuple[Any, Any, Any, Any, Any]:
    """Import claude-agent-sdk only when a live detector function is invoked."""
    try:
        from claude_agent_sdk import (  # type: ignore[import-not-found]
            AgentDefinition,
            AssistantMessage,
            ClaudeAgentOptions,
            TextBlock,
            query,
        )
    except ImportError as exc:
        raise ImportError(
            "claude-agent-sdk is required for live Claude detector runs. "
            "Install the project's hallucination extra and provide credentials via environment variables."
        ) from exc
    return AgentDefinition, AssistantMessage, ClaudeAgentOptions, TextBlock, query


def clean_json_string(text: str) -> str:
    """Extract the first balanced JSON object from common markdown-wrapped output."""
    text = text.strip()
    if "```json" in text:
        start_idx = text.find("```json") + 7
        end_idx = text.find("```", start_idx)
        if end_idx != -1:
            text = text[start_idx:end_idx].strip()
    elif text.startswith("```"):
        text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    start_idx = text.find("{")
    if start_idx == -1:
        return text
    text = text[start_idx:]
    brace_count = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[: index + 1]
    return text


def create_specialist_agents(sub_agent_model: str = DEFAULT_SUB_AGENT_MODEL) -> dict[str, Any]:
    """Create specialist definitions for the Claude Task tool."""
    AgentDefinition, _, _, _, _ = _import_claude_agent_sdk()
    instruction_prompt_fixed = instruction_prompt.replace("{DATE}", "{{DATE}}").replace(
        "{NAME}", "{{NAME}}"
    )
    return {
        "MedFactChecker": AgentDefinition(
            description=medfact_description,
            prompt=medfact_prompt,
            tools=["WebSearch", "WebFetch"],
            model=sub_agent_model,
        ),
        "CitationVerifier": AgentDefinition(
            description=citation_description,
            prompt=citation_prompt,
            tools=["WebSearch", "WebFetch"],
            model=sub_agent_model,
        ),
        "ReasoningAuditor": AgentDefinition(
            description=reasoning_description,
            prompt=reasoning_prompt,
            tools=[],
            model=sub_agent_model,
        ),
        "ContextKeeper": AgentDefinition(
            description=context_description,
            prompt=context_prompt,
            tools=[],
            model=sub_agent_model,
        ),
        "SafetyGuardian": AgentDefinition(
            description=safety_description,
            prompt=safety_prompt,
            tools=["WebSearch", "WebFetch"],
            model=sub_agent_model,
        ),
        "InstructionWatcher": AgentDefinition(
            description=instruction_description,
            prompt=instruction_prompt_fixed,
            tools=[],
            model=sub_agent_model,
        ),
        "HallucinationScout": AgentDefinition(
            description=hallucination_description,
            prompt=hallucination_prompt,
            tools=[],
            model=sub_agent_model,
        ),
    }


def create_orchestrator_options(
    orchestrator_model: str = DEFAULT_ORCHESTRATOR_MODEL,
    sub_agent_model: str = DEFAULT_SUB_AGENT_MODEL,
) -> Any:
    """Create Claude Agent SDK options for the hallucination orchestrator."""
    _, _, ClaudeAgentOptions, _, _ = _import_claude_agent_sdk()
    return ClaudeAgentOptions(
        agents=create_specialist_agents(sub_agent_model=sub_agent_model),
        allowed_tools=["Agent", "Task"],
        disallowed_tools=["Write", "Edit", "Bash"],
        system_prompt=orchestrator_prompt.strip(),
        model=orchestrator_model,
        permission_mode="default",
    )


async def _query_output(prompt: str, response: str, orchestrator_options: Any) -> OrchestratorOutput:
    _, AssistantMessage, _, TextBlock, query = _import_claude_agent_sdk()
    final_assistant_text: str | None = None
    terminal_message: Any | None = None
    calls: dict[str, str] = {}
    tool_results: dict[str, tuple[Any, bool]] = {}

    def inspect_blocks(blocks: Any) -> list[str]:
        texts: list[str] = []
        if not isinstance(blocks, list):
            return texts
        for block in blocks:
            if isinstance(block, TextBlock) and block.text:
                texts.append(block.text)
                continue
            block_name = block.__class__.__name__
            if block_name == "ToolUseBlock" or (
                hasattr(block, "id") and hasattr(block, "name") and hasattr(block, "input")
            ):
                tool_name = getattr(block, "name", None)
                tool_input = getattr(block, "input", None)
                call_id = getattr(block, "id", None)
                if tool_name in {"Agent", "Task"} and isinstance(tool_input, Mapping):
                    specialist = tool_input.get("subagent_type")
                    if (
                        isinstance(call_id, str)
                        and isinstance(specialist, str)
                        and specialist in SPECIALIST_NAME_TO_CODE
                    ):
                        calls[call_id] = specialist
                continue
            if block_name == "ToolResultBlock" or hasattr(block, "tool_use_id"):
                call_id = getattr(block, "tool_use_id", None)
                if isinstance(call_id, str):
                    tool_results[call_id] = (
                        getattr(block, "content", None),
                        bool(getattr(block, "is_error", False)),
                    )
        return texts

    async for message in query(
        prompt=format_detector_payload(prompt, response),
        options=orchestrator_options,
    ):
        message_name = message.__class__.__name__
        content = getattr(message, "content", None)
        texts = inspect_blocks(content)
        if isinstance(message, AssistantMessage) and not getattr(
            message, "parent_tool_use_id", None
        ):
            if texts:
                final_assistant_text = "".join(texts)
        if message_name == "ResultMessage":
            terminal_message = message

    if terminal_message is not None and bool(
        getattr(terminal_message, "is_error", False)
    ):
        raise RuntimeError("Claude detector query ended with an error result")

    terminal_value = None
    if terminal_message is not None:
        terminal_value = getattr(terminal_message, "structured_output", None)
        if terminal_value is None:
            terminal_value = getattr(terminal_message, "result", None)
    chosen = terminal_value if terminal_value is not None else final_assistant_text
    if chosen is None:
        raise DetectorOutputError(
            "missing_orchestrator_output",
            "Claude detector returned no terminal structured output",
        )

    try:
        if isinstance(chosen, Mapping):
            output = OrchestratorOutput.model_validate(dict(chosen))
        else:
            output = OrchestratorOutput.model_validate(
                json.loads(clean_json_string(str(chosen)))
            )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DetectorOutputError(
            "invalid_orchestrator_output",
            "Claude detector returned an invalid structured output",
        ) from exc

    observed = None
    if calls:
        observed = []
        for call_id, specialist in calls.items():
            result_value, failed = tool_results.get(call_id, (None, False))
            entry = observed_specialist_result(
                specialist,
                result_value,
                failed=failed,
            )
            if entry is not None:
                observed.append(entry)
    attach_execution_audit(output, observed)
    return output


class ClaudeAgentSDKHallucinationDetector:
    """Claude Agent SDK hallucination detector."""

    backend = "claude"
    detector_role = "live_detector"

    def __init__(
        self,
        orchestrator_model: str = DEFAULT_ORCHESTRATOR_MODEL,
        sub_agent_model: str = DEFAULT_SUB_AGENT_MODEL,
        *,
        orchestrator_options: Any | None = None,
    ) -> None:
        self.orchestrator_model = orchestrator_model
        self.sub_agent_model = sub_agent_model
        self._orchestrator_options = orchestrator_options

    def _get_orchestrator_options(self) -> Any:
        if self._orchestrator_options is None:
            self._orchestrator_options = create_orchestrator_options(
                orchestrator_model=self.orchestrator_model,
                sub_agent_model=self.sub_agent_model,
            )
        return self._orchestrator_options

    async def adetect(
        self,
        prompt: str,
        response: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> OrchestratorOutput:
        """Asynchronously classify one prompt-response pair."""
        del metadata
        return await _query_output(prompt, response, self._get_orchestrator_options())

    def detect(
        self,
        prompt: str,
        response: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> OrchestratorOutput:
        """Synchronous Protocol adapter that also works inside an active event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.adetect(prompt, response, metadata=metadata))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                lambda: asyncio.run(
                    self.adetect(prompt, response, metadata=metadata)
                )
            )
            return future.result()

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "backend": self.backend,
            "role": self.detector_role,
            "orchestrator_model": self.orchestrator_model,
            "sub_agent_model": self.sub_agent_model,
        }

__all__ = [
    "ClaudeAgentSDKHallucinationDetector",
    "DEFAULT_ORCHESTRATOR_MODEL",
    "DEFAULT_SUB_AGENT_MODEL",
    "_import_claude_agent_sdk",
    "clean_json_string",
    "create_orchestrator_options",
    "create_specialist_agents",
]
