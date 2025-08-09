from typing import Annotated

from annotated_types import Len
from pydantic import BaseModel
from pydantic.types import StringConstraints

from ayapi.models import EmbeddingModel

# To avoid overloading the server, we impose some length limits on input data
type EmbeddingSentence = Annotated[str, StringConstraints(max_length=200)]
type EmbeddingSentences = Annotated[
    list[EmbeddingSentence], Len(max_length=50)
]


class EmbeddingRequest(BaseModel):
    """Text strings to embed and hyper parameters."""

    model: EmbeddingModel = EmbeddingModel.QWEN_06B
    sentences: EmbeddingSentences


class EmbeddingResponse(BaseModel):
    """One vector per input text in the same order."""

    embeddings: list[list[float]]


class RateLimitExceededError(BaseModel):
    """Error message returned if the API is called too frequently."""

    error: str
