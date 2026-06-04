"""Reference 4-mer count embeddings (CPU baseline for regression checks)."""
from __future__ import annotations

import numpy as np

from .base import BaseEmbeddingExtractor

_ALPHABET = "ACGT"
_K = 4
_DIM = len(_ALPHABET) ** _K  # 256


def _kmer_index(kmer: str) -> int | None:
    idx = 0
    for ch in kmer.upper():
        if ch not in "ACGT":
            return None
        idx = idx * 4 + "ACGT".index(ch)
    return idx


class ExampleKmerExtractor(BaseEmbeddingExtractor):
    """L1-normalized 4-mer frequency vectors (CPU-only, deterministic)."""

    def __init__(self, name_model: str = "kmer-4", device: str = "cpu"):
        self.name_model = name_model
        self.device = device

    def extract_embeddings(self, sequences: list[str], batch_size: int = 8) -> np.ndarray:
        out = np.zeros((len(sequences), _DIM), dtype=np.float32)
        for i, seq in enumerate(sequences):
            vec = out[i]
            s = (seq or "").upper()
            for j in range(len(s) - _K + 1):
                ix = _kmer_index(s[j : j + _K])
                if ix is not None:
                    vec[ix] += 1.0
            n = vec.sum()
            if n > 0:
                vec /= n
        return out
