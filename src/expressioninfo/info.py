"""
Lower bounds on the mutual information I(G; X) between a gene's identity G and its measured expression pattern X,
computed from a blinded decoder (retrieval) evaluated on held-out genes. Both bounds assume G is uniform over the
N candidate genes, so H(G) = log2(N).

- Fano: from top-1 accuracy alone. Valid for any decoder, but loose: it ignores how close the wrong guesses were.
- Cross-entropy (Barber-Agakov / variational): I(G; X) >= H(G) - E[-log2 q(g | x)] for ANY distribution q that was
  not fitted on the evaluated genes. Uses the full predicted probabilities, so it is usually much tighter.
"""
from __future__ import annotations

import numpy as np


def binary_entropy(p: float | np.ndarray) -> float | np.ndarray:
    """h(p) in bits; h(0) = h(1) = 0."""
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1 - 1e-16)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.where((p <= 1e-300) | (p >= 1 - 1e-16), 0.0, h)


def fano_lower_bound(top1_accuracy: float, n: int) -> float:
    """I(G; X) >= log2 N - h(Pe) - Pe * log2(N - 1), Pe = 1 - top-1 accuracy. Clipped at 0 (the bound can go negative)."""
    if n < 2:
        return 0.0
    pe = 1.0 - float(top1_accuracy)
    bound = np.log2(n) - float(binary_entropy(pe)) - pe * np.log2(n - 1)
    return max(0.0, float(bound))


def ce_lower_bound(log2_q_true: np.ndarray, n: int | np.ndarray) -> float:
    """I(G; X) >= mean(log2 N_i + log2 q(true gene | x_i)). N may differ per gene (e.g. a restricted candidate set);
    then this bounds the information given the restriction. NOT clipped: a negative value means the decoder is worse
    calibrated than guessing uniformly, which is itself informative."""
    log2_q_true = np.asarray(log2_q_true, dtype=float)
    return float(np.mean(np.log2(np.asarray(n, dtype=float)) + log2_q_true))
