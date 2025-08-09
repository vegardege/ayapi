from fastapi import FastAPI, HTTPException, Request, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ayapi.models import EMBEDDING_MODELS
from ayapi.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    RateLimitExceededError,
)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="AyAPI",
    description="FastAPI wrapper for Python-only AI functionality",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post(
    "/embeddings",
    responses={
        status.HTTP_429_TOO_MANY_REQUESTS: {"model": RateLimitExceededError},
    },
    tags=["Embeddings"],
)
@limiter.limit("5/second")
async def embeddings(
    request: Request,
    input: EmbeddingRequest,
) -> EmbeddingResponse:
    """Create embeddings from text strings using the specified model and
    hyper parameters."""
    if model := EMBEDDING_MODELS.get(input.model):
        embeddings = model.encode(input.sentences, convert_to_numpy=True)
        return EmbeddingResponse(embeddings=embeddings.tolist())
    else:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Embedding model not found",
        )
