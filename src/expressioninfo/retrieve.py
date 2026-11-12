"""
Blinded identity retrieval: for each gene's pattern in measurement A, score every gene's pattern in measurement B and
ask where the true gene lands. Scores become probabilities with a softmax whose temperature is fitted on a separate
set of calibration genes, so the information bounds in info.py are evaluated on genes the decoder never saw.

Everything works on "codes": genes x d matrices whose row dot products are the similarity. transform() builds the
codes for Spearman / Pearson / cosine; technical_codes() builds codes whose dot product is a (shifted) negative
squared distance on purely technical features. The N x N similarity matrix is never held in memory at once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import rankdata

LN2 = np.log(2.0)
METRICS = ("spearman", "pearson", "cosine")


def transform(x: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Rows = genes. Returns (codes, ok) with unit-norm rows so that codes_a @ codes_b.T is the chosen similarity:
    spearman = Pearson correlation of within-gene ranks, pearson = correlation of the values, cosine = uncentred.
    `ok` is False for rows with zero variance (no pattern at all); their codes are 0."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    z = rankdata(x, axis=1).astype(np.float64) if metric == "spearman" else x.astype(np.float64)
    if metric != "cosine":
        z = z - z.mean(1, keepdims=True)
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    ok = norm[:, 0] > 1e-12
    return (z / np.where(norm > 1e-12, norm, 1.0)).astype(np.float32), ok


def technical_codes(feat_a: np.ndarray, feat_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Technical-only baseline. feat_* = genes x k measured technical features (e.g. overall mean expression and
    detection rate in that measurement), standardised with B's statistics. Codes are built so that
    a_code . b_code = 2 a.b - |b|^2 = -|a - b|^2 + |a|^2; the |a|^2 term is constant per query, so ranks and softmax
    are those of the negative squared Euclidean distance."""
    mu, sd = feat_b.mean(0), feat_b.std(0) + 1e-12
    a, b = (feat_a - mu) / sd, (feat_b - mu) / sd
    ca = np.hstack([2 * a, np.ones((len(a), 1))])
    cb = np.hstack([b, -(b ** 2).sum(1, keepdims=True)])
    return ca.astype(np.float32), cb.astype(np.float32)


def _block_scores(ca, cb, rows, true_idx, tau, bins):
    """Retrieval statistics for one block of query genes."""
    s = (ca[rows] @ cb.T).astype(np.float64)                      # block x N similarities
    r = np.arange(len(rows))
    t = s[r, true_idx[rows]]
    if bins is not None:                                          # restrict candidates to the true gene's bin
        s = np.where(bins[None, :] == bins[true_idx[rows]][:, None], s, -np.inf)
    n = np.isfinite(s).sum(1)
    greater, equal = (s > t[:, None]).sum(1), (s == t[:, None]).sum(1) - 1
    rank = greater + 0.5 * equal                                  # 0 = best; ties shared, never optimistic
    log2q = (t / tau - logsumexp(s / tau, axis=1)) / LN2
    return rank, log2q, n, s.argmax(1), t


def fit_temperature(ca, cb, true_idx, cal_rows, bins=None, grid=None) -> float:
    """Temperature minimising the mean cross-entropy of the TRUE gene on the calibration genes only."""
    grid = np.logspace(-3, 2, 61) if grid is None else grid
    s = (ca[cal_rows] @ cb.T).astype(np.float64)
    r = np.arange(len(cal_rows))
    t = s[r, true_idx[cal_rows]]
    if bins is not None:
        s = np.where(bins[None, :] == bins[true_idx[cal_rows]][:, None], s, -np.inf)
    ce = [np.mean(logsumexp(s / tau, axis=1) - t / tau) for tau in grid]
    i = int(np.argmin(ce))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]   # refine around the grid minimum
    fine = np.geomspace(lo, hi, 41)
    ce = [np.mean(logsumexp(s / tau, axis=1) - t / tau) for tau in fine]
    return float(fine[int(np.argmin(ce))])


@dataclass
class RetrievalResult:
    per_gene: pd.DataFrame      # test genes only
    tau: float
    n_candidates: int


def retrieve(ca: np.ndarray, cb: np.ndarray, genes: pd.Index, cal_mask: np.ndarray, true_idx: np.ndarray | None = None,
             bins: np.ndarray | None = None, block: int = 1024) -> RetrievalResult:
    """Query = row i of ca, candidates = all rows of cb, truth = row true_idx[i] (identity unless permuted for the
    null). The temperature is fitted on cal_mask genes; statistics are reported for the other (test) genes.
    `bins` (one label per gene) restricts each query's candidates to genes in the same bin as its true gene."""
    n = len(genes)
    true_idx = np.arange(n) if true_idx is None else np.asarray(true_idx)
    cal_rows, test_rows = np.flatnonzero(cal_mask), np.flatnonzero(~cal_mask)
    tau = fit_temperature(ca, cb, true_idx, cal_rows, bins)
    out = []
    for start in range(0, len(test_rows), block):
        rows = test_rows[start:start + block]
        rank, log2q, ncand, top, t = _block_scores(ca, cb, rows, true_idx, tau, bins)
        out.append(pd.DataFrame({"gene": genes[rows], "true_match": genes[true_idx[rows]], "rank": rank,
                                 "log2_q_true": log2q, "n_candidates": ncand, "top_match": genes[top], "sim_true": t}))
    per_gene = pd.concat(out, ignore_index=True)
    return RetrievalResult(per_gene=per_gene, tau=tau, n_candidates=n)


def summarize(res: RetrievalResult) -> dict:
    """Accuracy and both information bounds for a retrieval result (test genes)."""
    from .info import ce_lower_bound, fano_lower_bound
    g = res.per_gene
    top1 = float((g["rank"] == 0).mean())
    n_med = int(np.median(g["n_candidates"]))
    return {"n_test_genes": len(g), "n_candidates_median": n_med, "max_bits": float(np.log2(n_med)), "tau": res.tau,
            "top1": top1, "top10": float((g["rank"] < 10).mean()), "median_rank": float(g["rank"].median()),
            "mrr": float((1 / (g["rank"] + 1)).mean()),
            "bits_fano": fano_lower_bound(top1, n_med),
            "bits_ce": ce_lower_bound(g["log2_q_true"], g["n_candidates"])}
