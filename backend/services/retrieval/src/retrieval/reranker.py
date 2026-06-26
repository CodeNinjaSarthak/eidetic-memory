"""Local cross-encoder reranker for demo query endpoint.

Loads cross-encoder/ms-marco-MiniLM-L-6-v2 at startup. Prefers ONNX INT8 at
ONNX_MODEL_PATH if available; falls back to full CrossEncoder from HuggingFace cache.
If sentence-transformers is not installed, rerank() returns the input unchanged.
"""

import asyncio
import logging
import os
from typing import Any

from storage.models import MemoryFact

logger = logging.getLogger(__name__)

ONNX_MODEL_PATH = "/tmp/reranker-onnx-int8/model.onnx"
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_local_cross_encoder: Any = None
_ort_session: Any = None
_ort_tokenizer: Any = None
_reranker_loaded = False


def load_reranker() -> None:
    """Load the reranker at app startup.

    Tries ONNX INT8 first (faster), then full CrossEncoder.
    Logs a warning and sets _reranker_loaded=False if unavailable.
    """
    global _local_cross_encoder, _ort_session, _ort_tokenizer, _reranker_loaded

    if os.path.exists(ONNX_MODEL_PATH):
        try:
            import onnxruntime as ort  # type: ignore[import]
            from transformers import AutoTokenizer  # type: ignore[import]

            logger.info("Loading ONNX INT8 reranker from %s", ONNX_MODEL_PATH)
            _ort_tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(ONNX_MODEL_PATH))
            _ort_session = ort.InferenceSession(
                ONNX_MODEL_PATH, providers=["CPUExecutionProvider"]
            )
            _reranker_loaded = True
            logger.info("ONNX INT8 reranker loaded.")
            return
        except Exception as exc:
            logger.warning("ONNX reranker load failed (%s), falling back to CrossEncoder", exc)

    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import]
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — reranking disabled. "
            "Install with: uv add --package retrieval sentence-transformers"
        )
        return

    logger.info("Loading local cross-encoder (%s)...", _CROSS_ENCODER_MODEL)
    _local_cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)
    _reranker_loaded = True
    logger.info("Local cross-encoder loaded.")


async def rerank(query: str, facts: list[MemoryFact], top_k: int) -> list[MemoryFact]:
    """Rerank facts with the local cross-encoder, returning top_k by score descending.

    Runs CPU-bound inference in a thread pool executor so it doesn't block the event loop.
    Returns input[:top_k] unchanged if the reranker was not loaded.
    """
    if not _reranker_loaded or not facts:
        return facts[:top_k]

    documents = [f.content for f in facts]

    def _predict() -> list[float]:
        if _ort_session is not None:
            enc = _ort_tokenizer(
                [query] * len(documents),
                documents,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512,
            )
            logits = _ort_session.run(None, dict(enc))[0]
            return logits[:, 0].tolist()
        pairs = [[query, doc] for doc in documents]
        return _local_cross_encoder.predict(pairs, batch_size=32).tolist()

    loop = asyncio.get_running_loop()
    scores: list[float] = await loop.run_in_executor(None, _predict)

    sorted_pairs = sorted(zip(facts, scores, strict=False), key=lambda x: x[1], reverse=True)
    return [fact for fact, _ in sorted_pairs[:top_k]]
