# ExpressionInfo

How many bits does a gene's expression pattern across anatomical sites carry about **which gene it is** (identity,
RQ1) and about **what it does** (GO function, RQ2)?

Information is measured from blinded prediction on held-out genes, against a permutation null, as two lower bounds:

- **Fano**: from top-1 accuracy alone; simple but loose.
- **Cross-entropy**: I(G; X) >= log2 N - E[-log2 q(true gene | x)], with q a softmax over similarities whose
  temperature is fitted on separate calibration genes. Uses the full ranking, usually much tighter.

## Setup

```bash
conda env create -f environment.yml
conda activate ExpressionInfo
pytest            # synthetic checks: identical measurements give ~log2 N bits, independent ones ~0
```

## RQ1, step 1: two halves of CELLxGENE Census

```bash
# smoke test (one tissue, a few minutes)
python scripts/build_pseudobulk.py --name smoke_lung --tissue lung --max-cells-per-sample 200 --max-cells-per-dataset 100
python scripts/run_retrieval.py --a data/processed/pseudobulk__smoke_lung__A.h5ad \
                                --b data/processed/pseudobulk__smoke_lung__B.h5ad --name smoke_lung --n-perm 5

# full run (server: streams millions of cells)
python scripts/build_pseudobulk.py --name census_ct_tissue
python scripts/run_retrieval.py --a data/processed/pseudobulk__census_ct_tissue__A.h5ad \
                                --b data/processed/pseudobulk__census_ct_tissue__B.h5ad --name census_ct_tissue
```

`build_pseudobulk.py` splits Census **collections** (publications) at random into halves A and B, pools healthy,
primary, 10x cells into samples = cell type x tissue, and keeps samples with enough cells and donors in both halves.

`run_retrieval.py` reports, per similarity metric (Spearman, Pearson, cosine):

| result | question |
|---|---|
| observed | how many bits identify the gene across the two halves |
| permutation null | what the bounds give with no information (should be ~0) |
| expression-matched | bits left when a gene only competes against genes of the same expression decile |
| technical only | bits from overall mean expression and detection rate alone (the technical floor) |
| by breadth / tau / expression | which genes carry their identity |

## Layout

```
src/expressioninfo/   data.py (I/O, alignment, gene properties)  retrieve.py (codes, retrieval)  info.py (bounds)
scripts/              build_pseudobulk.py  run_retrieval.py
tests/                synthetic checks
data/raw, data/processed, results/   (gitignored)
```

## Next steps (RQ1)

1. HPA single-cell types vs CELLxGENE (cross-atlas, cell types mapped to CL).
2. Organ level: Bgee vs CELLxGENE aggregated to UBERON organs.
3. Human vs mouse, 1:1 orthologs only.
4. Paralog analysis of the confused gene pairs (feeds RQ3).
