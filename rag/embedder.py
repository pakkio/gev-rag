"""STEP 3 (ingest) / STEP 1 (query): EMBED — text -> vector of numbers.

Texts with similar meaning land close together in vector space, so
"How do I get a refund?" ends up near "Our return policy allows...".
Runs locally on CPU (ONNX, no PyTorch). First use downloads ~70 MB.
"""
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384 dimensions
_CACHE = Path(__file__).resolve().parent.parent / "models"
_model = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(MODEL_NAME, cache_dir=str(_CACHE))
    return _model


def _normalize(vectors: np.ndarray) -> np.ndarray:
    # Unit-length vectors make cosine similarity a plain dot product.
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embed_passages(texts: list) -> np.ndarray:
    return _normalize(np.array(list(_get_model().passage_embed(texts)), dtype=np.float32))


def embed_query(text: str) -> np.ndarray:
    # bge models use a slightly different encoding for queries vs passages.
    return _normalize(np.array(list(_get_model().query_embed([text])), dtype=np.float32))[0]
