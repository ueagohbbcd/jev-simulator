"""Public Jev request shapes; deployment limits are checked before inference."""
from typing import Annotated, Literal

from pydantic import Field, JsonValue

from .config import InferenceConfig, Strict
from .images import MAX_IMAGES

Content = str | dict[str, JsonValue] | list[JsonValue]


class NoulCriteria(Strict):
    true: Content = "Yes"
    false: Content = "No"


class NoulQuestion(Strict):
    type: Literal["noul"]
    instructions: Content
    criteria: NoulCriteria = Field(default_factory=NoulCriteria)


class ChoiceQuestion(Strict):
    type: Literal["choice"]
    instructions: Content
    criteria: dict[str, Content | None] = Field(min_length=2, max_length=255)


class ScoreQuestion(Strict):
    type: Literal["score"]
    instructions: Content
    criteria: list[Content] = Field(min_length=2, max_length=10)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class ImageInput(Strict):
    data: str = Field(min_length=1)
    type: Literal["image/jpeg", "image/png", "image/webp", "image/gif"] | None = None


class SystemOneRequest(Strict):
    model: str = Field(min_length=1)
    state: Content
    questions: dict[str, Question] = Field(min_length=1)
    images: list[ImageInput] = Field(default_factory=list, max_length=MAX_IMAGES)


class Usage(Strict):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class NoulAnswer(Strict):
    type: Literal["noul"]
    noul: float = Field(ge=0, le=1, allow_inf_nan=False)


class ChoiceAnswer(Strict):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class ScoreAnswer(Strict):
    type: Literal["score"]
    score: float = Field(ge=0, allow_inf_nan=False)
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class SystemOneResponse(Strict):
    model: str
    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]
    usage: Usage


class EvaluationRequest(Strict):
    request: SystemOneRequest
    execution: InferenceConfig
    dry_run: bool = False


class ApplicabilityWarning(Strict):
    code: str
    field: str
    message: str


class PlannedBranch(Strict):
    question_id: str
    index: int
    mapping: dict[str, str]
    messages: list[dict[str, JsonValue]]
    pair: list[str] | None


class EvaluationPlan(Strict):
    request_count: int
    requests: list[PlannedBranch]


class EvaluationDryRunResponse(Strict):
    dry_run: Literal[True]
    config_id: str
    execution: InferenceConfig
    warnings: list[ApplicabilityWarning]
    plan: EvaluationPlan


class EvaluationResultResponse(Strict):
    dry_run: Literal[False]
    config_id: str
    execution: InferenceConfig
    warnings: list[ApplicabilityWarning]
    result: SystemOneResponse


EvaluationResponse = Annotated[
    EvaluationDryRunResponse | EvaluationResultResponse, Field(discriminator="dry_run")
]
