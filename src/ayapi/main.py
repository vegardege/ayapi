from fastapi import FastAPI, HTTPException, status

from ayapi.models import MODELS

app = FastAPI(
    title="AyAPI",
    description="FastAPI wrapper for Python-only AI functionality",
)


@app.post("/embedding")
def embedding(input: str) -> list[float]:
    """Convert a single input string to an embedding with the chosen model.

    Currently only supports nomic-embed-text-v1.5.
    """
    if model := MODELS.get("nomic"):
        return model.encode(input, convert_to_numpy=True).tolist()
    else:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Model not found",
        )
