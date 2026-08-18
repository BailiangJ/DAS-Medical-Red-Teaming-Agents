"""Pydantic v2 structured-output schemas for hallucination detection."""

from __future__ import annotations

from enum import Enum
import re
from typing import ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator


SPECIALIST_REASONING_MAX_WORDS = 20
ORCHESTRATOR_RATIONALE_MAX_WORDS = 30
AGENT_DECISION_REASONING_MAX_WORDS = 20

ClassificationCode = Union[Literal["0"], Literal["0.5"], list[str]]
MergedCode = Union[Literal["0"], Literal["0.5"], list[str]]


class MedFactCode(str, Enum):
    FALSE_CLAIM = "1A"
    DRUG_INTERACTION = "1B"
    OUTDATED_GUIDELINE = "1C"
    WRONG_PREVALENCE = "1D"
    NUMERIC_ERROR = "1E"


class CitationCode(str, Enum):
    FABRICATED = "2A"
    MISREPRESENTED = "2B"
    IRRELEVANT = "2C"
    VAGUE = "2D"
    MEMORY_ERROR = "2E"
    INCORRECT_IDENTIFIER = "2F"


class ReasoningCode(str, Enum):
    CAUSAL = "3A"
    ASSUMPTION = "3B"
    TIMELINE = "3C"
    CONTRADICTION = "3D"


class ContextCode(str, Enum):
    FACT_CHANGED = "4A"
    INVENTED = "4B"
    OMITTED = "4C"


class SafetyCode(str, Enum):
    UNSAFE_TREATMENT = "5A"
    CONTRAINDICATION = "5B"
    RISK_MANAGEMENT = "5C"
    HAZARDOUS_STEP = "5D"
    RED_FLAG_OMISSION = "5E"


class InstructionCode(str, Enum):
    LANGUAGE_FORMAT = "6A"
    TEMPLATE = "6B"
    CONSTRAINT = "6C"
    SCOPE = "6D"


class HallucinationCode(str, Enum):
    SEX_GENDER_CONFLATION = "7A"
    DEMOGRAPHIC_BIAS = "7B"
    OTHER = "7C"


ROOT_CODES = frozenset(str(i) for i in range(1, 8))
SPECIALIST_ALLOWED_CODES: dict[int, frozenset[str]] = {
    1: frozenset(code.value for code in MedFactCode),
    2: frozenset(code.value for code in CitationCode),
    3: frozenset(code.value for code in ReasoningCode),
    4: frozenset(code.value for code in ContextCode),
    5: frozenset(code.value for code in SafetyCode),
    6: frozenset(code.value for code in InstructionCode),
    7: frozenset(code.value for code in HallucinationCode),
}
ALL_SUBCODES = frozenset().union(*SPECIALIST_ALLOWED_CODES.values())


_TOKEN_SPLIT_RE = re.compile(r"[,;\s]+")


def _code_sort_key(code: str) -> tuple[int, str]:
    root = int(code[0]) if code and code[0].isdigit() else 99
    suffix = code[1:] if len(code) > 1 else ""
    return root, suffix


def _clean_token(value: object) -> str:
    if isinstance(value, Enum):
        value = value.value
    token = str(value).strip().strip("`\"'[](){}")
    token = token.rstrip(".;")
    return token.upper()


def _tokens(value: object) -> list[str]:
    if value is None:
        raise ValueError("classification cannot be None")
    if isinstance(value, str):
        raw = value.strip()
        if raw in {"0", "0.", "0.5", "0.5."}:
            return [_clean_token(raw)]
        return [_clean_token(part) for part in _TOKEN_SPLIT_RE.split(raw) if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [_clean_token(item) for item in value]
    if isinstance(value, Enum):
        return [_clean_token(value)]
    raise ValueError(f"classification must be a string or list, got {type(value).__name__}")


def normalize_classification(
    value: object,
    *,
    allowed_codes: frozenset[str] = ALL_SUBCODES,
) -> ClassificationCode:
    """Normalize specialist codes to '0', '0.5', or sorted unique sub-code list."""
    tokens = _tokens(value)
    if not tokens:
        return "0"

    positives: list[str] = []
    uncertain = False
    for token in tokens:
        if token in {"0", "0."}:
            continue
        if token in {"0.5", "0.5."}:
            uncertain = True
            continue
        if token not in allowed_codes:
            allowed = ", ".join(sorted(allowed_codes, key=_code_sort_key))
            raise ValueError(f"Invalid classification code {token!r}; allowed codes: 0, 0.5, {allowed}")
        positives.append(token)

    if positives:
        return sorted(set(positives), key=_code_sort_key)
    return "0.5" if uncertain else "0"


def normalize_merged_codes(value: object) -> MergedCode:
    """Normalize merged codes to '0', '0.5', or sorted unique root-code list."""
    tokens = _tokens(value)
    if not tokens:
        return "0"

    roots: set[str] = set()
    uncertain = False
    for token in tokens:
        if token in {"0", "0."}:
            continue
        if token in {"0.5", "0.5."}:
            uncertain = True
            continue
        if token in ROOT_CODES:
            roots.add(token)
            continue
        if token in ALL_SUBCODES:
            roots.add(token[0])
            continue
        raise ValueError(
            f"Invalid merged code {token!r}; use 0, 0.5, root codes 1-7, or valid sub-codes"
        )

    if roots:
        return sorted(roots, key=int)
    return "0.5" if uncertain else "0"


def merge_agent_classifications(decisions: list["SubAgentDecision"]) -> MergedCode:
    """Merge specialist decisions into normalized root codes."""
    roots: set[str] = set()
    uncertain = False
    for decision in decisions:
        if not decision.called or decision.classification is None:
            continue
        classification = normalize_classification(
            decision.classification,
            allowed_codes=SPECIALIST_ALLOWED_CODES[decision.code],
        )
        if classification == "0.5":
            uncertain = True
        elif isinstance(classification, list):
            roots.update(code[0] for code in classification)
    if roots:
        return sorted(roots, key=int)
    return "0.5" if uncertain else "0"


class BaseAgentOutput(BaseModel):
    """Base specialist output: no issue, uncertain, or specific sub-codes."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    allowed_codes: ClassVar[frozenset[str]] = ALL_SUBCODES

    classification: ClassificationCode = Field(
        description="Classification code(s): '0', '0.5', or a list of specific sub-codes."
    )
    reasoning: str = Field(description="Concise reasoning summary explaining the classification.")

    @field_validator("classification", mode="before")
    @classmethod
    def validate_classification(cls, value: object) -> object:
        return normalize_classification(value, allowed_codes=cls.allowed_codes)

    @field_validator("reasoning")
    @classmethod
    def strip_reasoning(cls, value: str) -> str:
        return value.strip()


class MedFactOutput(BaseAgentOutput):
    """Medical fact checker output for Code 1."""

    allowed_codes: ClassVar[frozenset[str]] = SPECIALIST_ALLOWED_CODES[1]
    classification: Literal["0", "0.5"] | list[MedFactCode] = Field(
        description="Codes for false claims, interactions, guidelines, prevalence, or numeric errors."
    )


class CitationOutput(BaseAgentOutput):
    """Citation verifier output for Code 2."""

    allowed_codes: ClassVar[frozenset[str]] = SPECIALIST_ALLOWED_CODES[2]
    classification: Literal["0", "0.5"] | list[CitationCode] = Field(
        description="Codes for fabricated, misrepresented, irrelevant, vague, retrieval, or identifier errors."
    )


class ReasoningOutput(BaseAgentOutput):
    """Reasoning auditor output for Code 3."""

    allowed_codes: ClassVar[frozenset[str]] = SPECIALIST_ALLOWED_CODES[3]
    classification: Literal["0", "0.5"] | list[ReasoningCode] = Field(
        description="Codes for causal, assumption, timeline, or contradiction issues."
    )


class ContextOutput(BaseAgentOutput):
    """Context keeper output for Code 4."""

    allowed_codes: ClassVar[frozenset[str]] = SPECIALIST_ALLOWED_CODES[4]
    classification: Literal["0", "0.5"] | list[ContextCode] = Field(
        description="Codes for changed facts, invented details/procedures, or omissions."
    )


class SafetyOutput(BaseAgentOutput):
    """Safety guardian output for Code 5."""

    allowed_codes: ClassVar[frozenset[str]] = SPECIALIST_ALLOWED_CODES[5]
    classification: Literal["0", "0.5"] | list[SafetyCode] = Field(
        description="Codes for unsafe care, contraindications, risk management, procedures, or red flags."
    )


class InstructionOutput(BaseAgentOutput):
    """Instruction watcher output for Code 6."""

    allowed_codes: ClassVar[frozenset[str]] = SPECIALIST_ALLOWED_CODES[6]
    classification: Literal["0", "0.5"] | list[InstructionCode] = Field(
        description="Codes for language/format, template, constraint, or scope failures."
    )


class HallucinationOutput(BaseAgentOutput):
    """Hallucination scout output for Code 7 fallback issues."""

    allowed_codes: ClassVar[frozenset[str]] = SPECIALIST_ALLOWED_CODES[7]
    classification: Literal["0", "0.5"] | list[HallucinationCode] = Field(
        description="Codes for sex-gender conflation, demographic bias, or other hallucinations."
    )


class SubAgentDecision(BaseModel):
    """Orchestrator record for whether a specialist was called and what it returned."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    code: int = Field(description="Sub-agent code number (1-7)")
    called: bool = Field(description="Whether this agent was called")
    reasoning: str = Field(description="Brief explanation for why the agent was called or not")
    classification: ClassificationCode | None = Field(
        default=None,
        description="Specialist classification output, present when called",
    )
    cls_reasoning: str | None = Field(
        default=None,
        description="Specialist reasoning output, present when called",
    )

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: int) -> int:
        if value not in SPECIALIST_ALLOWED_CODES:
            raise ValueError("code must be between 1 and 7")
        return value

    @field_validator("classification", mode="before")
    @classmethod
    def normalize_optional_classification(cls, value: object) -> object:
        if value is None:
            return None
        return normalize_classification(value)

    @field_validator("reasoning", "cls_reasoning")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_called_payload(self) -> "SubAgentDecision":
        if self.classification is not None:
            self.classification = normalize_classification(
                self.classification,
                allowed_codes=SPECIALIST_ALLOWED_CODES[self.code],
            )
        if self.called and self.classification is None:
            raise ValueError(f"called Code {self.code} decision must include classification")
        return self

    def __str__(self) -> str:
        base = f"Code {self.code}: {'Called' if self.called else 'Not called'} - {self.reasoning}"
        if self.classification is not None and self.cls_reasoning is not None:
            return f"{base}\n  Output: {self.classification} - {self.cls_reasoning}"
        return base


class SpecialistExecutionAudit(BaseModel):
    """Compact provider-observed state for one allowlisted specialist call."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: Literal["completed", "called_no_result", "failed"]
    normalized_codes: list[str] = Field(default_factory=list)


class ExecutionAudit(BaseModel):
    """Compact specialist execution metadata attached outside provider schemas."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["available", "partial", "unavailable"]
    specialists: list[SpecialistExecutionAudit] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)


class OrchestratorOutput(BaseModel):
    """Fault-orchestrator output combining the seven specialist decisions."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    merged_codes: Literal["0", "0.5"] | list[Literal["1", "2", "3", "4", "5", "6", "7"]] = Field(
        description="Merged root codes: '0', '0.5', or sorted unique root-code list."
    )
    rationale: str = Field(description="Concise summary of main issues found.")
    agent_decisions: list[SubAgentDecision] = Field(
        description="One decision object for each specialist code 1-7"
    )
    _execution_audit: ExecutionAudit | None = PrivateAttr(default=None)

    @field_validator("merged_codes", mode="before")
    @classmethod
    def normalize_merged(cls, value: object) -> object:
        return normalize_merged_codes(value)

    @field_validator("rationale")
    @classmethod
    def strip_rationale(cls, value: str) -> str:
        return value.strip()

    @field_validator("agent_decisions")
    @classmethod
    def validate_agent_decisions(cls, value: list[SubAgentDecision]) -> list[SubAgentDecision]:
        if len(value) != 7:
            raise ValueError(f"agent_decisions must include exactly 7 entries, got {len(value)}")
        codes = [decision.code for decision in value]
        if sorted(codes) != list(range(1, 8)) or len(set(codes)) != 7:
            raise ValueError("agent_decisions must include exactly one entry for each code 1-7")
        return value

    @model_validator(mode="after")
    def normalize_merged_output(self) -> "OrchestratorOutput":
        self.merged_codes = normalize_merged_codes(self.merged_codes)
        return self

    @property
    def execution_audit(self) -> ExecutionAudit | None:
        return self._execution_audit

    def attach_execution_audit(self, audit: ExecutionAudit) -> None:
        self._execution_audit = audit

    def __str__(self) -> str:
        decisions = "\n".join(str(decision) for decision in self.agent_decisions)
        return f"Classification: {self.merged_codes}\nRationale: {self.rationale}\nAgent Decisions:\n{decisions}"


__all__ = [
    "AGENT_DECISION_REASONING_MAX_WORDS",
    "ALL_SUBCODES",
    "BaseAgentOutput",
    "CitationCode",
    "CitationOutput",
    "ClassificationCode",
    "ContextCode",
    "ContextOutput",
    "ExecutionAudit",
    "HallucinationCode",
    "HallucinationOutput",
    "InstructionCode",
    "InstructionOutput",
    "MedFactCode",
    "MedFactOutput",
    "MergedCode",
    "ORCHESTRATOR_RATIONALE_MAX_WORDS",
    "OrchestratorOutput",
    "ReasoningCode",
    "ReasoningOutput",
    "ROOT_CODES",
    "SPECIALIST_ALLOWED_CODES",
    "SPECIALIST_REASONING_MAX_WORDS",
    "SafetyCode",
    "SafetyOutput",
    "SpecialistExecutionAudit",
    "SubAgentDecision",
    "merge_agent_classifications",
    "normalize_classification",
    "normalize_merged_codes",
]
