"""Sanity checks on synthetic data where the true amount of identity information is known."""
import numpy as np
import pandas as pd
import pytest

from expressioninfo.info import binary_entropy, ce_lower_bound, fano_lower_bound
from expressioninfo.retrieve import retrieve, summarize, technical_codes, transform


def test_binary_entropy():
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.0) == 0.0 and binary_entropy(1.0) == 0.0


def test_fano_extremes():
    n = 1000
    assert fano_lower_bound(1.0, n) == pytest.approx(np.log2(n))
    assert fano_lower_bound(1 / n, n) == pytest.approx(0.0, abs=1e-9)   # chance accuracy proves nothing


def test_ce_bound_uniform_q_is_zero():
    n = 512
    assert ce_lower_bound(np.full(10, -np.log2(n)), n) == pytest.approx(0.0)


def _patterns(n_genes=600, n_samples=40, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.gamma(1.0, 1.0, size=(n_genes, n_samples))
    a = base + noise * rng.standard_normal(base.shape)
    b = base + noise * rng.standard_normal(base.shape)
    return a, b, pd.Index([f"g{i}" for i in range(n_genes)])


@pytest.mark.parametrize("metric", ["spearman", "pearson", "cosine"])
def test_identical_measurements_give_near_max_bits(metric):
    a, b, genes = _patterns(noise=0.0)
    cal = np.random.default_rng(1).random(len(genes)) < 0.2
    s = summarize(retrieve(transform(a, metric)[0], transform(b, metric)[0], genes, cal))
    assert s["top1"] == 1.0
    assert s["bits_fano"] == pytest.approx(np.log2(len(genes)))
    assert s["bits_ce"] > 0.9 * np.log2(len(genes))


def test_independent_measurements_give_no_bits():
    rng = np.random.default_rng(3)
    a, b = rng.gamma(1.0, 1.0, (600, 40)), rng.gamma(1.0, 1.0, (600, 40))   # no shared signal at all
    genes = pd.Index([f"g{i}" for i in range(600)])
    cal = rng.random(600) < 0.2
    s = summarize(retrieve(transform(a, "spearman")[0], transform(b, "spearman")[0], genes, cal))
    assert s["bits_ce"] < 0.05 and s["top1"] < 0.02


def test_more_noise_fewer_bits_and_permutation_kills_signal():
    cal = np.random.default_rng(1).random(600) < 0.2
    bits = []
    for noise in (0.3, 1.0, 3.0):
        a, b, genes = _patterns(noise=noise)
        bits.append(summarize(retrieve(transform(a, "pearson")[0], transform(b, "pearson")[0], genes, cal))["bits_ce"])
    assert bits[0] > bits[1] > bits[2]
    a, b, genes = _patterns(noise=0.3)
    perm = np.random.default_rng(2).permutation(len(genes))
    s = summarize(retrieve(transform(a, "pearson")[0], transform(b, "pearson")[0], genes, cal, true_idx=perm))
    assert s["bits_ce"] < 0.05


def test_ties_are_not_counted_as_hits():
    a = np.ones((50, 10)); b = np.ones((50, 10))                       # every gene identical: no information
    ca, cb = np.ones((50, 1), np.float32), np.ones((50, 1), np.float32)
    genes = pd.Index([f"g{i}" for i in range(50)])
    cal = np.zeros(50, bool); cal[:10] = True
    s = summarize(retrieve(ca, cb, genes, cal))
    assert s["top1"] == 0.0 and s["median_rank"] == pytest.approx(24.5)
    assert s["bits_ce"] == pytest.approx(0.0, abs=1e-6)


def test_technical_codes_rank_like_distance():
    rng = np.random.default_rng(4)
    fa, fb = rng.normal(size=(30, 2)), rng.normal(size=(30, 2))
    ca, cb = technical_codes(fa, fb)
    mu, sd = fb.mean(0), fb.std(0) + 1e-12
    za, zb = (fa - mu) / sd, (fb - mu) / sd
    d = ((za[:, None, :] - zb[None, :, :]) ** 2).sum(-1)
    assert (np.argsort(-(ca @ cb.T), axis=1)[:, 0] == np.argsort(d, axis=1)[:, 0]).all()


def test_expression_bins_restrict_candidates():
    a, b, genes = _patterns(noise=1.0)
    bins = np.arange(len(genes)) % 10
    cal = np.random.default_rng(1).random(len(genes)) < 0.2
    r = retrieve(transform(a, "pearson")[0], transform(b, "pearson")[0], genes, cal, bins=bins)
    assert (r.per_gene["n_candidates"] == 60).all()
