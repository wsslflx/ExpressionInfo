#!/usr/bin/env python3
"""
RQ1 step 1: two independent measurements of the same atlas, from CELLxGENE Census.

The Census collections (a collection = one publication, usually one set of donors) are split at random into two
halves, A and B. In each half, cells are pooled into samples = cell type x tissue, and every sample is summarised as
  X                    mean log1p(CP10k) per gene   (CP10k from the cell's raw counts and its raw_sum)
  layers["frac_expr"]  fraction of the sample's cells in which the gene was detected (raw count > 0)
Only samples that pass the minimum cell and donor counts in BOTH halves are written, so A and B cover the same
samples and a gene's pattern in A and in B come from disjoint collections.

Cells: primary data only (no duplicates across datasets), healthy tissue (disease = normal), 10x assays by default
(other chemistries change the expression profile), cell type and tissue must be ontology terms. To keep the run
tractable, at most --max-cells-per-dataset cells are taken per sample and dataset, and at most
--max-cells-per-sample per sample and half; a mean over a few thousand cells is already stable.

Writes to data/processed/:
  pseudobulk__<name>__A.h5ad, pseudobulk__<name>__B.h5ad, pseudobulk__<name>__meta.json

Run it where the Census can be streamed quickly (the server); a full human run reads millions of cells.
Smoke test (minutes):
    python scripts/build_pseudobulk.py --name smoke --tissue lung --max-cells-per-sample 200
Full run:
    python scripts/build_pseudobulk.py --name census_ct_tissue
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from expressioninfo.data import DATA_PROCESSED  # noqa: E402

NORMAL = "PATO:0000461"
OBS_COLS = ["soma_joinid", "dataset_id", "donor_id", "assay", "cell_type_ontology_term_id", "cell_type",
            "tissue_general_ontology_term_id", "tissue_general", "tissue_ontology_term_id", "tissue", "raw_sum"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def select_cells(obs: pd.DataFrame, args, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Filter cells, assign halves by collection, define samples, subsample. Returns (cells, samples, split info)."""
    tissue_col = "tissue_general_ontology_term_id" if args.tissue_level == "general" else "tissue_ontology_term_id"
    tissue_lab = "tissue_general" if args.tissue_level == "general" else "tissue"
    n0 = len(obs)
    keep = obs["cell_type_ontology_term_id"].str.startswith("CL:") & obs[tissue_col].str.startswith("UBERON:")
    if args.assay_contains:
        keep &= obs["assay"].str.contains(args.assay_contains, case=False, regex=False)
    keep &= obs["raw_sum"] > 0
    obs = obs[keep].copy()
    log(f"cells after filters: {len(obs):,} of {n0:,}")

    # halves by collection: all datasets of one publication go to the same half
    colls = np.array(sorted(obs["collection_id"].unique()))
    rng.shuffle(colls)
    half_of = {c: ("A" if i < len(colls) // 2 else "B") for i, c in enumerate(colls)}
    obs["half"] = obs["collection_id"].map(half_of)
    obs["sample"] = obs["cell_type_ontology_term_id"] + "|" + obs[tissue_col]
    obs["donor"] = obs["dataset_id"].astype(str) + "/" + obs["donor_id"].astype(str)

    stats = (obs.groupby(["sample", "half"], observed=True)
             .agg(n_cells_total=("soma_joinid", "size"), n_donors=("donor", "nunique"), n_datasets=("dataset_id", "nunique"),
                  n_collections=("collection_id", "nunique"))
             .reset_index())
    ok = stats[(stats["n_cells_total"] >= args.min_cells) & (stats["n_donors"] >= args.min_donors)]
    both = ok.groupby("sample")["half"].nunique()
    samples_ok = both[both == 2].index
    log(f"samples (cell type x tissue): {stats['sample'].nunique():,} in total, {len(samples_ok):,} pass "
        f">= {args.min_cells} cells and >= {args.min_donors} donors in BOTH halves")
    obs = obs[obs["sample"].isin(samples_ok)]

    # subsample: per sample x half x dataset, then per sample x half
    obs = (obs.sample(frac=1.0, random_state=int(rng.integers(2**31)))
           .groupby(["sample", "half", "dataset_id"], observed=True).head(args.max_cells_per_dataset)
           .groupby(["sample", "half"], observed=True).head(args.max_cells_per_sample))
    used = obs.groupby(["sample", "half"], observed=True).size().rename("n_cells_used").reset_index()
    samples = stats[stats["sample"].isin(samples_ok)].merge(used, on=["sample", "half"])
    labels = obs.drop_duplicates("sample").set_index("sample")[["cell_type", tissue_lab]].rename(columns={tissue_lab: "tissue"})
    samples = samples.join(labels, on="sample")
    split = {"n_collections": len(colls), "collections_A": sorted(c for c, h in half_of.items() if h == "A"),
             "collections_B": sorted(c for c, h in half_of.items() if h == "B")}
    log(f"cells to read: {len(obs):,}")
    return obs, samples, split


def accumulate(exp, cells: pd.DataFrame, var: pd.DataFrame, group_keys: list[tuple[str, str]]):
    """Stream the raw counts of the selected cells and sum log1p(CP10k) and detections per (sample, half) x gene."""
    import tiledbsoma as soma

    gidx = {k: i for i, k in enumerate(group_keys)}
    cells = cells.sort_values("soma_joinid")
    coords = cells["soma_joinid"].to_numpy()
    cell_group = np.array([gidx[(s, h)] for s, h in zip(cells["sample"], cells["half"])])
    libsize = cells["raw_sum"].to_numpy(dtype=np.float64)
    var_pos = np.full(int(var["soma_joinid"].max()) + 1, -1, dtype=np.int64)
    var_pos[var["soma_joinid"].to_numpy()] = np.arange(len(var))
    G, V = len(group_keys), len(var)
    sums = np.zeros((G, V), dtype=np.float64)
    hits = np.zeros((G, V), dtype=np.float32)
    n_entries, t0 = 0, time.time()
    with exp.axis_query(measurement_name="RNA", obs_query=soma.AxisQuery(coords=(coords,))) as q:
        for tbl in q.X("raw").tables():
            o = tbl["soma_dim_0"].to_numpy()
            v = var_pos[tbl["soma_dim_1"].to_numpy()]
            x = tbl["soma_data"].to_numpy().astype(np.float64)
            pos = np.searchsorted(coords, o)
            g = cell_group[pos]
            val = np.log1p(x / libsize[pos] * 1e4)
            for arr, data in ((sums, val), (hits, (x > 0).astype(np.float32))):
                m = sp.coo_matrix((data, (g, v)), shape=(G, V)).tocsr()
                m.sum_duplicates()
                rows = np.repeat(np.arange(G), np.diff(m.indptr))
                arr[rows, m.indices] += m.data          # no duplicate (row, col) pairs left, so += is exact
            n_entries += len(x)
            log(f"  {n_entries:,} non-zero counts processed, up to cell {pos.max() + 1:,} of {len(coords):,} ({time.time() - t0:,.0f} s)")
    return sums, hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="output name, e.g. census_ct_tissue")
    ap.add_argument("--organism", default="homo_sapiens", choices=["homo_sapiens", "mus_musculus"])
    ap.add_argument("--census-version", default="2025-11-08", help="pinned for reproducibility; 'stable' = latest")
    ap.add_argument("--tissue-level", default="general", choices=["general", "fine"],
                    help="tissue_general (coarse organ, more overlap between halves) or the fine tissue term")
    ap.add_argument("--tissue", default=None, help="restrict to one tissue_general label (smoke tests), e.g. lung")
    ap.add_argument("--assay-contains", default="10x", help="keep assays whose name contains this ('' = all)")
    ap.add_argument("--min-cells", type=int, default=50, help="per sample and half")
    ap.add_argument("--min-donors", type=int, default=2, help="per sample and half")
    ap.add_argument("--max-cells-per-dataset", type=int, default=500)
    ap.add_argument("--max-cells-per-sample", type=int, default=2000, help="per sample and half")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(DATA_PROCESSED))
    args = ap.parse_args()

    import cellxgene_census

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        summary = census["census_info"]["summary"].read().concat().to_pandas()
        build = dict(zip(summary["label"], summary["value"].astype(str)))
        datasets = census["census_info"]["datasets"].read(column_names=["dataset_id", "collection_id"]).concat().to_pandas()
        exp = census["census_data"][args.organism]
        flt = f"is_primary_data == True and disease_ontology_term_id == '{NORMAL}'"
        if args.tissue:
            flt += f" and tissue_general == '{args.tissue}'"
        log(f"reading cell metadata ({args.organism}, Census {build.get('census_build_date', args.census_version)})")
        obs = exp.obs.read(value_filter=flt, column_names=OBS_COLS).concat().to_pandas()
        for c in obs.columns:                       # Census returns categoricals; plain strings keep joins simple
            if isinstance(obs[c].dtype, pd.CategoricalDtype):
                obs[c] = obs[c].astype(str)
        obs = obs.merge(datasets, on="dataset_id", how="left")
        if obs["collection_id"].isna().any():
            raise SystemExit("some datasets have no collection_id in census_info/datasets")
        cells, samples, split = select_cells(obs, args, rng)
        if cells.empty:
            raise SystemExit("no sample passes the filters in both halves; relax --min-cells / --min-donors")
        var = exp.ms["RNA"].var.read(column_names=["soma_joinid", "feature_id", "feature_name"]).concat().to_pandas()
        keys = [(s, h) for h in ("A", "B") for s in sorted(samples["sample"].unique())]
        sums, hits = accumulate(exp, cells, var, keys)

    n_used = samples.set_index(["sample", "half"])["n_cells_used"]
    var_df = pd.DataFrame({"feature_name": var["feature_name"].to_numpy()}, index=pd.Index(var["feature_id"].to_numpy(), name="feature_id"))
    for half in ("A", "B"):
        rows = [i for i, (s, h) in enumerate(keys) if h == half]
        names = [keys[i][0] for i in rows]
        n = np.array([n_used[(s, half)] for s in names], dtype=np.float64)[:, None]
        obs_df = samples[samples["half"] == half].set_index("sample").loc[names].drop(columns="half")
        a = ad.AnnData(X=(sums[rows] / n).astype(np.float32), obs=obs_df, var=var_df.copy())
        a.layers["frac_expr"] = (hits[rows] / n).astype(np.float32)
        a.write_h5ad(out / f"pseudobulk__{args.name}__{half}.h5ad", compression="gzip")
    meta = {"name": args.name, "args": vars(args), "census_build": build, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_samples": len(keys) // 2, "n_genes": len(var), "n_cells_read": len(cells),
            "cells_per_half": cells["half"].value_counts().to_dict(), **split}
    (out / f"pseudobulk__{args.name}__meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"wrote {meta['n_samples']} samples x {len(var):,} genes per half to {out}")


if __name__ == "__main__":
    main()
