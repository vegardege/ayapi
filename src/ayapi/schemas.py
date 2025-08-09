from typing import Annotated, Literal

from annotated_types import Len
from pydantic import BaseModel
from pydantic.types import StringConstraints

from ayapi.models import EmbeddingModel

# To avoid overloading the server, we impose some length limits on input data
type EmbeddingSentence = Annotated[str, StringConstraints(max_length=200)]
type EmbeddingSentences = Annotated[
    list[EmbeddingSentence], Len(max_length=10)
]

type Precision = Literal["float32", "int8", "uint8", "binary", "ubinary"]


class EmbeddingRequest(BaseModel):
    """Text strings to embed and hyper parameters."""

    # The full name of a supported embedding model.
    model: EmbeddingModel = EmbeddingModel.GTE_MULTILINGUAL_BASE

    # List of sentences we want to create embeddings for.
    sentences: EmbeddingSentences

    # The numeric format for the output embeddings.
    # float32: Full precision, highest accuracy, highest memory use.
    # int8 / uint8: 8-bit quantized ints, smaller size, slight accuracy loss.
    # binary / ubinary: 1-bit per dimension, huge compression, bigger loss in
    #   precision. Often for approximate nearest neighbor search.
    precision: Precision = "float32"

    # Normalized vectors make cosine similarity equivalent to dot product.
    normalize_embeddings: bool = False

    # Cuts off the vector to the first N dimensions. Can reduce storage and
    # speed up search, but you lose information.
    truncate_dim: int | None = None

    # Splits long inputs into smaller chunks before encoding.
    chunk_size: int | None = None


class EmbeddingResponse(BaseModel):
    """Vector embeddings corresponding to input sentences."""

    # One embedding per input sentence in the same order.
    embeddings: list[list[float]]


class RateLimitExceededError(BaseModel):
    """Error message returned if the API is called too frequently."""

    error: str
