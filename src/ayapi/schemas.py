from typing import Annotated

from pydantic import BaseModel
from pydantic.types import StringConstraints

from ayapi.models import EmbeddingModel


class EmbeddingRequest(BaseModel):
    """Text strings to embed and hyper parameters."""

    model: EmbeddingModel = EmbeddingModel.NOMIC
    sentences: list[Annotated[str, StringConstraints(max_length=200)]]


class EmbeddingResponse(BaseModel):
    """One vector per input text in the same order."""

    embeddings: list[list[float]]
