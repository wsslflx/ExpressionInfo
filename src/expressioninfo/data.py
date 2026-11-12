"""
Reading the two measurements of an expression atlas and putting them on a common footing.

A "measurement" is an AnnData with
  obs  = samples (e.g. "CL:0000236|UBERON:0002048" = cell type | tissue), with n_cells / n_donors columns
  var  = genes, indexed by Ensembl ID without version suffix
  X    = mean log1p(CP10k) expression per sample and gene
  layers["frac_expr"] = fraction of the sample's cells in which the gene was detected (count > 0)
as written by scripts/build_pseudobulk.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def strip_version(ids) -> pd.Index:
    """ENSG00000141510.17 -> ENSG00000141510"""
    return pd.Index([str(i).split(".")[0] for i in ids])


def read_measurement(path: str | Path) -> ad.AnnData:
    a = ad.read_h5ad(path)
    a.var_names = strip_version(a.var_names)
    if not a.var_names.is_unique:
        a = a[:, ~a.var_names.duplicated()].copy()
    return a


@dataclass
class AlignedPair:
    """Two measurements of the same genes over the same samples. Rows = genes, columns = samples."""
    genes: pd.Index
    samples: pd.Index
    a: np.ndarray            # genes x samples, mean log1p CP10k in measurement A
    b: np.ndarray            # same for B
    frac_a: np.ndarray       # genes x samples, detection rate in A
    frac_b: np.ndarray
    report: dict = field(default_factory=dict)


def align(a: ad.AnnData, b: ad.AnnData, min_frac: float = 0.05, min_samples: int = 1,
          genes: pd.Index | None = None) -> AlignedPair:
    """Keep samples and genes present in both measurements, and genes detected (frac_expr >= min_frac) in at least
    `min_samples` samples in BOTH measurements. A gene that is silent everywhere has a flat pattern that every other
    silent gene matches equally well, so it is excluded and counted in the report instead of silently dropped.
    `genes` optionally restricts the universe (e.g. protein-coding genes)."""
    samples = a.obs_names.intersection(b.obs_names)
    shared = a.var_names.intersection(b.var_names)
    rep = {"samples_a": a.n_obs, "samples_b": b.n_obs, "samples_shared": len(samples),
           "genes_a": a.n_vars, "genes_b": b.n_vars, "genes_shared": len(shared)}
    if genes is not None:
        shared = shared.intersection(genes)
        rep["genes_after_universe_filter"] = len(shared)
    a, b = a[samples, shared], b[samples, shared]
    xa, xb = np.asarray(a.X, dtype=np.float32).T, np.asarray(b.X, dtype=np.float32).T
    fa = np.asarray(a.layers["frac_expr"], dtype=np.float32).T
    fb = np.asarray(b.layers["frac_expr"], dtype=np.float32).T
    detected = ((fa >= min_frac).sum(1) >= min_samples) & ((fb >= min_frac).sum(1) >= min_samples)
    rep.update({"min_frac": min_frac, "min_samples": min_samples, "genes_not_detected_excluded": int((~detected).sum()),
                "genes_kept": int(detected.sum())})
    return AlignedPair(genes=pd.Index(shared)[detected], samples=pd.Index(samples), a=xa[detected], b=xb[detected],
                       frac_a=fa[detected], frac_b=fb[detected], report=rep)


def gene_properties(p: AlignedPair, min_frac: float = 0.05) -> pd.DataFrame:
    """Per-gene descriptors used to break results down (averaged over the two measurements, i.e. gene properties,
    not properties of one noisy measurement): mean expression, mean detection rate, breadth (share of samples where
    the gene is detected) and tau tissue-specificity (0 = uniform, 1 = one sample only)."""
    mean_expr = (p.a.mean(1) + p.b.mean(1)) / 2
    det = (p.frac_a.mean(1) + p.frac_b.mean(1)) / 2
    breadth = (((p.frac_a >= min_frac).mean(1)) + ((p.frac_b >= min_frac).mean(1))) / 2
    x = (p.a + p.b) / 2
    mx = x.max(1, keepdims=True)
    tau = np.where(mx[:, 0] > 0, (1 - x / np.where(mx > 0, mx, 1)).sum(1) / max(x.shape[1] - 1, 1), np.nan)
    return pd.DataFrame({"mean_expr": mean_expr, "detection": det, "breadth": breadth, "tau": tau}, index=p.genes)
