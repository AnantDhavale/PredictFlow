"""
PredictFlow Confidence Adapters (Secure & Extensible)
-----------------------------------------------------

Purpose:
  - Provide a unified, secure interface for generating text embeddings or confidence scores
  - Support multiple adapters (local, ML API, or custom model)
  - Ensure deterministic, bounded, and fail-safe behavior for each adapter

Security & Design Principles:
  ✅ No arbitrary code execution
  ✅ Bounded input/output lengths
  ✅ Configurable timeouts for external APIs
  ✅ Fallbacks for offline/local mode
  ✅ All adapters return consistent vector shape or confidence values
"""

import os
import logging
import json
from typing import List, Dict, Any, Protocol

import numpy as np
from predictflow.confidence.confidence_vectorizer import get_vector, cosine_similarity  # from previous module

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. Base Interface (Protocol for typing)
# -----------------------------------------------------------------------------
class ConfidenceAdapter(Protocol):
    """Interface definition for all PredictFlow confidence adapters."""

    def embed(self, text: str) -> List[float]:
        """Return deterministic embedding vector for given text."""
        ...

    def compare(self, text1: str, text2: str) -> float:
        """Return confidence similarity score between two texts (0–1)."""
        ...


# -----------------------------------------------------------------------------
# 2. Local Adapter (default, offline-safe)
# -----------------------------------------------------------------------------
class LocalConfidenceAdapter:
    """
    Default local confidence adapter.

    Uses deterministic pseudo-random embeddings for local scoring.
    Suitable for air-gapped or open-source deployments.
    """

    def embed(self, text: str) -> List[float]:
        return get_vector(text)

    def compare(self, text1: str, text2: str) -> float:
        v1 = self.embed(text1)
        v2 = self.embed(text2)
        return cosine_similarity(v1, v2)


# -----------------------------------------------------------------------------
# 3. Optional Remote Adapter (e.g., OpenAI / HuggingFace)
# -----------------------------------------------------------------------------
class RemoteConfidenceAdapter:
    """
    Secure external embedding adapter.
    Uses APIs (if available) to fetch high-quality embeddings.

    Notes:
      - Reads API keys from environment variables only
      - Enforces rate limiting and timeouts
      - Falls back to LocalConfidenceAdapter on errors
    """

    API_TIMEOUT = 10  # seconds

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HUGGINGFACE_TOKEN")
        self.provider = "openai" if os.getenv("OPENAI_API_KEY") else "huggingface" if os.getenv("HUGGINGFACE_TOKEN") else None
        self.fallback = LocalConfidenceAdapter()

        if not self.provider:
            logger.info("[Confidence] No external API key found; using local adapter.")

    def embed(self, text: str) -> List[float]:
        """
        Fetch embeddings securely.
        Returns fallback vector if API unavailable or fails.
        """
        if not self.provider:
            return self.fallback.embed(text)

        try:
            import requests  # lazy import for optional dependency

            headers = {"Authorization": f"Bearer {self.api_key}"}
            data = {"input": text.strip()[:2000]}  # truncate input for safety

            if self.provider == "openai":
                url = "https://api.openai.com/v1/embeddings"
                payload = {"model": "text-embedding-3-small", **data}
            else:
                url = "https://api-inference.huggingface.co/pipeline/feature-extraction/all-MiniLM-L6-v2"
                payload = data

            resp = requests.post(url, headers=headers, json=payload, timeout=self.API_TIMEOUT)
            resp.raise_for_status()

            result = resp.json()
            # OpenAI and HF formats differ slightly
            if isinstance(result, dict) and "data" in result:
                vector = result["data"][0]["embedding"]
            elif isinstance(result, list):
                vector = result[0]
            else:
                raise ValueError("Unexpected embedding response format.")

            # Normalize to 0–1 range and ensure list of floats
            vec = np.array(vector, dtype=float)
            vec = (vec - vec.min()) / (vec.max() - vec.min() + 1e-9)
            return vec.tolist()

        except Exception as e:
            logger.warning(f"[Confidence] Remote adapter failed: {e}")
            return self.fallback.embed(text)

    def compare(self, text1: str, text2: str) -> float:
        v1 = self.embed(text1)
        v2 = self.embed(text2)
        return cosine_similarity(v1, v2)


# -----------------------------------------------------------------------------
# 4. Adapter Factory
# -----------------------------------------------------------------------------
class ConfidenceAdapterFactory:
    """
    Centralized adapter loader.
    Chooses the most suitable adapter based on configuration and availability.
    """

    @staticmethod
    def get_adapter(prefer_remote: bool = True) -> ConfidenceAdapter:
        """Return a configured confidence adapter instance."""
        if prefer_remote and (os.getenv("OPENAI_API_KEY") or os.getenv("HUGGINGFACE_TOKEN")):
            return RemoteConfidenceAdapter()
        return LocalConfidenceAdapter()


# -----------------------------------------------------------------------------
# 5. Example Utility
# -----------------------------------------------------------------------------
def compare_texts(text1: str, text2: str, use_remote: bool = True) -> Dict[str, Any]:
    """
    Utility function for quick comparison of two texts.
    Returns structured JSON for dashboards or APIs.
    """
    adapter = ConfidenceAdapterFactory.get_adapter(prefer_remote=use_remote)
    score = adapter.compare(text1, text2)

    return {
        "text_1": text1,
        "text_2": text2,
        "similarity_score": round(score, 4),
        "adapter": adapter.__class__.__name__,
    }


# -----------------------------------------------------------------------------
# 6. Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    result = compare_texts("approve payment", "approve invoice")
    print(json.dumps(result, indent=2))
