from fastapi import FastAPI, HTTPException, status

from ayapi.models import EMBEDDING_MODELS
from ayapi.schemas import EmbeddingRequest, EmbeddingResponse

app = FastAPI(
    title="AyAPI",
    description="FastAPI wrapper for Python-only AI functionality",
)


@app.post("/embedding")
def embedding(input: EmbeddingRequest) -> EmbeddingResponse:
    """Convert a single input string to an embedding with the chosen model.

    Currently only supports nomic-embed-text-v1.5.
    """
    if model := EMBEDDING_MODELS.get(input.model):
        embeddings = model.encode(input.sentences, convert_to_numpy=True)
        return EmbeddingResponse(embeddings=embeddings.tolist())
    else:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Embedding model not found",
        )
