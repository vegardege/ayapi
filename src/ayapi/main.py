from fastapi import FastAPI

app = FastAPI(
    title="AyAPI",
    description="FastAPI wrapper for Python-only AI functionality",
)


@app.get("/")
def ayapi() -> dict[str, str]:
    """Test function"""
    return {"message": "Hello from AyAPI"}
