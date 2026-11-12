#!/usr/bin/env python3
"""
RQ1: how many bits does a gene's expression pattern carry about which gene it is?

Takes two measurements of the same samples (A and B, e.g. from scripts/build_pseudobulk.py), and for each gene asks
whether its pattern in A finds the same gene among all genes' patterns in B. Reports accuracy and two lower bounds on
I(gene; pattern) (see src/expressioninfo/info.py), evaluated on test genes; the softmax temperature is fitted on a
separate 20% of calibration genes.

Runs, for each similarity metric:
  observed            the real retrieval
  permutation null    the truth relabelled at random (--n-perm times): what the bounds give with no information
  expression-matched  each gene competes only against genes of the same mean-expression decile: what is left
                      once overall expression level cannot help (a technical fingerprint check)
  technical only      retrieval from overall mean expression and detection rate alone (no pattern)
and breaks the observed result down by expression breadth, tissue specificity (tau) and expression level.

Writes results/rq1__<name>__<metric>.json and results/rq1__<name>__<metric>__genes.parquet.

    python scripts/run_retrieval.py --a data/processed/pseudobulk__census_ct_tissue__A.h5ad \
                                    --b data/processed/pseudobulk__census_ct_tissue__B.h5ad --name census_ct_tissue
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from expressioninfo.data import RESULTS, align, gene_properties, read_measurement  # noqa: E402
from expressioninfo.info import ce_lower_bound  # noqa: E402
from expressioninfo.retrieve import METRICS, retrieve, summarize, technical_codes, transform  # noqa: E402


def bootstrap(per_gene: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict:
    """95% intervals over test genes for top-1 accuracy and the cross-entropy bound."""
    top1, ce = [], []
    r, q, n = per_gene["rank"].to_numpy(), per_gene["log2_q_true"].to_numpy(), per_gene["n_candidates"].to_numpy()
    for _ in range(n_boot):
        i = rng.integers(0, len(r), len(r))
        top1.append((r[i] == 0).mean()); ce.append(ce_lower_bound(q[i], n[i]))
    return {"top1_ci95": [float(np.quantile(top1, 0.025)), float(np.quantile(top1, 0.975))],
            "bits_ce_ci95": [float(np.quantile(ce, 0.025)), float(np.quantile(ce, 0.975))]}


def breakdown(g: pd.DataFrame, col: str, n_bins: int = 5) -> list[dict]:
    """Observed result per quantile bin of one gene property (pointwise bits = log2 N + log2 q(true))."""
    b = pd.qcut(g[col], n_bins, duplicates="drop")
    rows = []
    for k, s in g.groupby(b, observed=True):
        rows.append({"bin": str(k), "n_genes": len(s), "top1": float((s["rank"] == 0).mean()),
                     "top10": float((s["rank"] < 10).mean()), "median_rank": float(s["rank"].median()),
                     "bits_ce": ce_lower_bound(s["log2_q_true"], s["n_candidates"])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--metric", action="append", choices=METRICS, help="repeatable; default: all three")
    ap.add_argument("--genes", default=None, help="optional file with one Ensembl ID per line: restrict the gene universe")
    ap.add_argument("--min-frac", type=float, default=0.05, help="a gene counts as detected in a sample at this detection rate")
    ap.add_argument("--cal-frac", type=float, default=0.2, help="share of genes used only to fit the temperature")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args()
    metrics = args.metric or list(METRICS)
    rng = np.random.default_rng(args.seed)

    universe = pd.Index([l.strip().split(".")[0] for l in open(args.genes) if l.strip()]) if args.genes else None
    pair = align(read_measurement(args.a), read_measurement(args.b), min_frac=args.min_frac, genes=universe)
    print(json.dumps(pair.report, indent=2))
    props = gene_properties(pair, args.min_frac)
    n = len(pair.genes)
    cal_mask = rng.random(n) < args.cal_frac
    expr_bins = pd.qcut(props["mean_expr"].rank(method="first"), 10, labels=False).to_numpy()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # technical-only baseline: does not depend on the metric
    tech_a = np.column_stack([pair.a.mean(1), pair.frac_a.mean(1)])
    tech_b = np.column_stack([pair.b.mean(1), pair.frac_b.mean(1)])
    tech = summarize(retrieve(*technical_codes(tech_a, tech_b), pair.genes, cal_mask))
    print(f"technical only: top-1 {tech['top1']:.4f}, bits (CE) {tech['bits_ce']:.3f}")

    for metric in metrics:
        ca, ok_a = transform(pair.a, metric)
        cb, ok_b = transform(pair.b, metric)
        if not (ok_a & ok_b).all():
            raise SystemExit(f"{int((~(ok_a & ok_b)).sum())} genes have a flat pattern after the detection filter")
        obs = retrieve(ca, cb, pair.genes, cal_mask)
        res = {"observed": summarize(obs)}
        res["observed"].update(bootstrap(obs.per_gene, args.n_boot, rng))
        print(f"\n[{metric}] observed: top-1 {res['observed']['top1']:.4f}, top-10 {res['observed']['top10']:.4f}, "
              f"bits CE {res['observed']['bits_ce']:.3f}, Fano {res['observed']['bits_fano']:.3f} (max {res['observed']['max_bits']:.2f})")

        perm = [summarize(retrieve(ca, cb, pair.genes, cal_mask, true_idx=rng.permutation(n))) for _ in range(args.n_perm)]
        res["permutation_null"] = {k: {"mean": float(np.mean([p[k] for p in perm])), "max": float(np.max([p[k] for p in perm]))}
                                   for k in ("top1", "top10", "bits_ce", "bits_fano")}
        print(f"[{metric}] permutation null: bits CE mean {res['permutation_null']['bits_ce']['mean']:.3f}, "
              f"max {res['permutation_null']['bits_ce']['max']:.3f}")

        matched = retrieve(ca, cb, pair.genes, cal_mask, bins=expr_bins)
        res["expression_matched"] = summarize(matched)
        tech_m = summarize(retrieve(*technical_codes(tech_a, tech_b), pair.genes, cal_mask, bins=expr_bins))
        res["expression_matched"]["technical_only_within_bins"] = tech_m
        print(f"[{metric}] expression-matched: top-1 {res['expression_matched']['top1']:.4f}, bits CE "
              f"{res['expression_matched']['bits_ce']:.3f} of max {res['expression_matched']['max_bits']:.2f} "
              f"(technical only within bins: {tech_m['bits_ce']:.3f})")
        res["technical_only"] = tech

        g = obs.per_gene.join(props, on="gene")
        g["bits"] = np.log2(g["n_candidates"]) + g["log2_q_true"]
        g = g.merge(matched.per_gene[["gene", "rank", "log2_q_true", "n_candidates"]].rename(
            columns={"rank": "rank_matched", "log2_q_true": "log2_q_true_matched", "n_candidates": "n_candidates_matched"}), on="gene")
        res["by_breadth"] = breakdown(g, "breadth"); res["by_tau"] = breakdown(g, "tau"); res["by_mean_expr"] = breakdown(g, "mean_expr")

        meta = {"name": args.name, "metric": metric, "a": args.a, "b": args.b, "align": pair.report, "n_genes": n,
                "n_calibration_genes": int(cal_mask.sum()), "args": vars(args),
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (out / f"rq1__{args.name}__{metric}.json").write_text(json.dumps({**meta, **res}, indent=2, default=float))
        g.to_parquet(out / f"rq1__{args.name}__{metric}__genes.parquet", index=False)
    print(f"\nwrote results to {out}")


if __name__ == "__main__":
    main()
