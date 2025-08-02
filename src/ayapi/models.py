from enum import StrEnum

from sentence_transformers import SentenceTransformer


class EmbeddingModel(StrEnum):
    """The embedding models supported by `ayapi`"""

    NOMIC = "nomic"


EMBEDDING_MODELS: dict[EmbeddingModel, SentenceTransformer] = {
    EmbeddingModel.NOMIC: SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
    )
}
