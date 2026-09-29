"""STEP 3 (ingest) / STEP 1 (query): EMBED — text -> vector of numbers.

Texts with similar meaning land close together in vector space, so
"How do I get a refund?" ends up near "Our return policy allows...".
Runs locally via ONNX (no PyTorch): on the GPU when onnxruntime-gpu can load CUDA,
otherwise on CPU. First use downloads ~70 MB.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384 dimensions
_CACHE = Path(__file__).resolve().parent.parent / "models"
_model = None
DEVICE = "cpu"


def _get_model() -> TextEmbedding:
    global _model, DEVICE
    if _model is None:
        if "CUDAExecutionProvider" in ort.get_available_providers():
            ort.preload_dlls()  # CUDA/cuDNN come from the nvidia-* pip wheels, not a system install
            _model = TextEmbedding(MODEL_NAME, cache_dir=str(_CACHE),
                                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            DEVICE = "cuda"
        else:
            _model = TextEmbedding(MODEL_NAME, cache_dir=str(_CACHE))
    return _model


def _normalize(vectors: np.ndarray) -> np.ndarray:
    # Unit-length vectors make cosine similarity a plain dot product.
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embed_passages(texts: list) -> np.ndarray:
    model = _get_model()
    # On CPU, fastembed's default batch_size=256 makes onnxruntime hold ~2.4 GB and never
    # release it; 16 gives the same throughput at ~500 MB. The GPU benefits from bigger batches.
    batch_size = 64 if DEVICE == "cuda" else 16
    return _normalize(np.array(list(model.passage_embed(texts, batch_size=batch_size)), dtype=np.float32))


def embed_query(text: str) -> np.ndarray:
    # bge models use a slightly different encoding for queries vs passages.
    return _normalize(np.array(list(_get_model().query_embed([text])), dtype=np.float32))[0]
