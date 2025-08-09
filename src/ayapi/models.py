from enum import StrEnum

from sentence_transformers import SentenceTransformer


class EmbeddingModel(StrEnum):
    """The embedding models supported by `ayapi`"""

    QWEN_06B = "Qwen/Qwen3-Embedding-0.6B"
    GTE_MULTILINGUAL_BASE = "Alibaba-NLP/gte-multilingual-base"


EMBEDDING_MODELS: dict[EmbeddingModel, SentenceTransformer] = {
    EmbeddingModel.QWEN_06B: SentenceTransformer("Qwen/Qwen3-Embedding-0.6B"),
    EmbeddingModel.GTE_MULTILINGUAL_BASE: SentenceTransformer(
        "Alibaba-NLP/gte-multilingual-base",
        trust_remote_code=True,
    ),
}
