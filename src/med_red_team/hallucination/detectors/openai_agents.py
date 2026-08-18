"""OpenAI Agents implementation of the hallucination detector Protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import warnings

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
from ..schemas import (
    BaseAgentOutput,
    CitationOutput,
    ContextOutput,
    HallucinationOutput,
    InstructionOutput,
    MedFactOutput,
    OrchestratorOutput,
    ReasoningOutput,
    SafetyOutput,
)
from .audit import (
    SPECIALIST_NAME_TO_CODE,
    attach_execution_audit,
    observed_specialist_result,
)
from .base import DetectorOutputError, coerce_orchestrator_output, format_detector_payload


DEFAULT_MODEL = "gpt-4o"
DEFAULT_SEARCH_AGENT_MODEL = "o3"
PAPER_TESTED_MODELS = {"gpt-4o", "o3", "o4-mini"}
SUPPORTED_MODELS = PAPER_TESTED_MODELS


def _import_openai_agents() -> tuple[Any, Any, Any, Any]:
    """Import openai-agents only when a live detector function is invoked."""
    try:
        from agents import Agent, ModelSettings, Runner, WebSearchTool  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "openai-agents is required for live OpenAI detector runs. "
            "Install the project's hallucination extra and provide credentials via environment variables."
        ) from exc
    return Agent, ModelSettings, Runner, WebSearchTool


def parse_response(output: BaseAgentOutput | OrchestratorOutput) -> tuple[str, str]:
    """Extract a string classification and rationale from a structured output."""
    if isinstance(output, OrchestratorOutput):
        return str(output.merged_codes), output.rationale
    if isinstance(output, BaseAgentOutput):
        return str(output.classification), output.reasoning
    raise ValueError(f"Invalid output type: {type(output).__name__}")


def get_model_settings(model: str, *, tool_required: bool = False) -> Any:
    """Return openai-agents ModelSettings for a model id."""
    _, ModelSettings, _, _ = _import_openai_agents()
    tool_choice = "required" if tool_required else "auto"
    if model == "gpt-4o":
        return ModelSettings(tool_choice=tool_choice, temperature=0.0)
    return ModelSettings(tool_choice=tool_choice)


def make_agent(
    name: str,
    prompt: str,
    tool_description: str,
    output_type: type[BaseAgentOutput],
    *,
    uses_search: bool = False,
    model: str = DEFAULT_MODEL,
    search_agent_model: str = DEFAULT_SEARCH_AGENT_MODEL,
) -> Any:
    """Create a specialist tool agent. Requires openai-agents at call time."""
    Agent, _, _, WebSearchTool = _import_openai_agents()
    agent_model = search_agent_model if uses_search else model
    search_tool = None
    if uses_search and agent_model != "o3":
        search_tool = WebSearchTool(search_context_size="medium")

    return Agent(
        name=name,
        instructions=prompt.strip(),
        model=agent_model,
        tools=[search_tool] if search_tool else [],
        model_settings=get_model_settings(agent_model, tool_required=search_tool is not None),
        output_type=output_type,
    ).as_tool(
        tool_name=name,
        tool_description=tool_description.strip(),
    )


def create_orchestrator(
    orchestrator_model: str = DEFAULT_MODEL,
    sub_agent_model: str = DEFAULT_MODEL,
    *,
    search_agent_model: str = DEFAULT_SEARCH_AGENT_MODEL,
) -> Any:
    """Create the seven-specialist hallucination orchestrator."""
    Agent, _, _, _ = _import_openai_agents()
    specialist_tools = [
        make_agent(
            "MedFactChecker",
            medfact_prompt,
            medfact_description,
            MedFactOutput,
            uses_search=True,
            model=sub_agent_model,
            search_agent_model=search_agent_model,
        ),
        make_agent(
            "CitationVerifier",
            citation_prompt,
            citation_description,
            CitationOutput,
            uses_search=True,
            model=sub_agent_model,
            search_agent_model=search_agent_model,
        ),
        make_agent(
            "ReasoningAuditor",
            reasoning_prompt,
            reasoning_description,
            ReasoningOutput,
            model=sub_agent_model,
        ),
        make_agent(
            "ContextKeeper",
            context_prompt,
            context_description,
            ContextOutput,
            model=sub_agent_model,
        ),
        make_agent(
            "SafetyGuardian",
            safety_prompt,
            safety_description,
            SafetyOutput,
            uses_search=True,
            model=sub_agent_model,
            search_agent_model=search_agent_model,
        ),
        make_agent(
            "InstructionWatcher",
            instruction_prompt,
            instruction_description,
            InstructionOutput,
            model=sub_agent_model,
        ),
        make_agent(
            "HallucinationScout",
            hallucination_prompt,
            hallucination_description,
            HallucinationOutput,
            model=sub_agent_model,
        ),
    ]

    return Agent(
        name="MedFact Orchestrator",
        instructions=orchestrator_prompt.strip(),
        tools=specialist_tools,
        model=orchestrator_model,
        model_settings=get_model_settings(orchestrator_model),
        output_type=OrchestratorOutput,
    )


def _item_value(item: Any, name: str) -> Any:
    value = getattr(item, name, None)
    if value is not None:
        return value
    raw = getattr(item, "raw_item", None)
    if isinstance(raw, Mapping):
        return raw.get(name)
    return getattr(raw, name, None)


def _observed_openai_specialists(result: Any) -> list[Any] | None:
    items = getattr(result, "new_items", None)
    if not isinstance(items, list):
        return None
    calls: dict[str, str] = {}
    outputs: dict[str, Any] = {}
    for index, item in enumerate(items):
        item_type = str(getattr(item, "type", "")).casefold()
        class_name = item.__class__.__name__.casefold()
        call_id = _item_value(item, "call_id")
        key = str(call_id) if call_id is not None else f"item-{index}"
        if "toolcalloutput" in class_name or "tool_call_output" in item_type:
            outputs[key] = getattr(item, "output", None)
            continue
        if "toolcall" in class_name or "tool_call" in item_type:
            tool_name = _item_value(item, "tool_name") or _item_value(item, "name")
            if isinstance(tool_name, str) and tool_name in SPECIALIST_NAME_TO_CODE:
                calls[key] = tool_name

    if not calls:
        return None
    observed = []
    for key, name in calls.items():
        entry = observed_specialist_result(name, outputs.get(key))
        if entry is not None:
            observed.append(entry)
    return observed


class OpenAIAgentsHallucinationDetector:
    """OpenAI Agents hallucination detector."""

    backend = "openai"
    detector_role = "live_detector"

    def __init__(
        self,
        orchestrator_model: str = DEFAULT_MODEL,
        sub_agent_model: str = DEFAULT_MODEL,
        *,
        search_agent_model: str = DEFAULT_SEARCH_AGENT_MODEL,
        orchestrator: Any | None = None,
    ) -> None:
        self.orchestrator_model = orchestrator_model
        self.sub_agent_model = sub_agent_model
        self.search_agent_model = search_agent_model
        self._orchestrator = orchestrator
        if self.search_agent_model == "o3":
            warnings.warn(
                "External retrieval is unavailable for the configured verification specialist.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _get_orchestrator(self) -> Any:
        if self._orchestrator is None:
            self._orchestrator = create_orchestrator(
                orchestrator_model=self.orchestrator_model,
                sub_agent_model=self.sub_agent_model,
                search_agent_model=self.search_agent_model,
            )
        return self._orchestrator

    def detect(
        self,
        prompt: str,
        response: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> OrchestratorOutput:
        """Run the seven-specialist orchestrator over one prompt-response pair."""
        del metadata
        _, _, Runner, _ = _import_openai_agents()
        result = Runner.run_sync(
            self._get_orchestrator(),
            format_detector_payload(prompt, response),
        )
        try:
            output = coerce_orchestrator_output(result.final_output)
        except (TypeError, ValueError) as exc:
            raise DetectorOutputError(
                "invalid_orchestrator_output",
                "OpenAI detector returned an invalid structured output",
            ) from exc
        attach_execution_audit(
            output,
            _observed_openai_specialists(result),
        )
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "backend": self.backend,
            "role": self.detector_role,
            "orchestrator_model": self.orchestrator_model,
            "sub_agent_model": self.sub_agent_model,
            "search_agent_model": self.search_agent_model,
        }

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SEARCH_AGENT_MODEL",
    "OpenAIAgentsHallucinationDetector",
    "SUPPORTED_MODELS",
    "_import_openai_agents",
    "create_orchestrator",
    "get_model_settings",
    "make_agent",
    "parse_response",
]
