from sentence_transformers import SentenceTransformer

MODELS: dict[str, SentenceTransformer] = {
    "nomic": SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
    )
}
