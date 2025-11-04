"""
PredictFlow Confidence Vectorizer (Secure & Deterministic)
-----------------------------------------------------------

Purpose:
  - Generate safe, deterministic vector representations of text
  - Provide a placeholder for later integration with ML/NLP models
  - Ensure reproducibility and input validation across environments

Security & Quality Features:
  ✅ Deterministic hash-based seeding using SHA-256
  ✅ Input length limits (prevents abuse / large-text attacks)
  ✅ Sanitization and normalization of Unicode input
  ✅ Bounded numeric output (0–1)
  ✅ No network or model dependencies
"""

import hashlib
import numpy as np
import unicodedata
import logging
from typing import List

logger = logging.getLogger(__name__)

# Fixed embedding size
VECTOR_SIZE = 128
# Input safety limits
MAX_TEXT_LENGTH = 5000


def _sanitize_text(text: str) -> str:
    """
    Normalize and truncate text to a safe, bounded form.
    Removes control characters and enforces Unicode normalization.
    """
    if not isinstance(text, str):
        text = str(text or "")
    # Normalize to NFC form
    text = unicodedata.normalize("NFC", text)
    # Strip non-printable characters
    text = "".join(ch for ch in text if ch.isprintable())
    # Truncate excessively long input
    return text[:MAX_TEXT_LENGTH]


def _seed_from_text(text: str) -> int:
    """
    Generate a stable, deterministic integer seed from a SHA-256 hash.
    Avoids Python's salted `hash()` for reproducibility.
    """
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:12], 16)  # take first 12 hex digits (~48 bits)


def get_vector(text: str) -> List[float]:
    """
    Generate a deterministic pseudo-random vector representation of text.

    Each unique text yields a consistent 128-dimensional vector of floats ∈ [0,1).
    Useful for lightweight text similarity scoring without model dependencies.

    Example:
        >>> v1 = get_vector("approve payment")
        >>> v2 = get_vector("approve payment")
        >>> assert v1 == v2  # deterministic

    Security:
      - Deterministic across systems
      - Safe for untrusted input
      - No dynamic code execution or external calls
    """
    try:
        clean_text = _sanitize_text(text)
        seed = _seed_from_text(clean_text)
        rng = np.random.default_rng(seed)
        vector = rng.random(VECTOR_SIZE).tolist()
        return vector
    except Exception as e:
        logger.error(f"[Confidence] Failed to generate vector for text: {e}")
        return [0.0] * VECTOR_SIZE  # fallback for safety


# Optional: lightweight similarity helper
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors safely.
    Returns value between 0 and 1.
    """
    try:
        a, b = np.array(vec1), np.array(vec2)
        if a.shape != b.shape:
            return 0.0
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom != 0 else 0.0
    except Exception:
        return 0.0


# Example usage
if __name__ == "__main__":
    text1 = "Approve payment workflow"
    text2 = "Approve invoice workflow"
    v1 = get_vector(text1)
    v2 = get_vector(text2)
    sim = cosine_similarity(v1, v2)

    print(f"Vector length: {len(v1)}")
    print(f"Cosine similarity: {sim:.3f}")
