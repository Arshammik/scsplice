"""Multi-modal multigedipy run: gene expression + splicing on samples A01 + B01.

Demonstrates the splikit-py -> multigedipy bind for a real dual-modality
analysis. Splicing data comes from splikit-py (M1/M2 from Solo.out/SJ/);
gene expression comes from scanpy reading Solo.out/Gene/filtered/.

The two modalities share the SAME cells (same Gene/filtered whitelist), so
the two AnnDatas are aligned cell-for-cell before being handed to
``multigedipy.MultiGEDIModel``.

Run on c170:
    cd /home/arsham79/projects/rrg-hsn/arsham79/splikitpy
    module load eigen/3.4.0
    /home/arsham79/projects/rrg-hsn/arsham79/multigedipy_pkg/.venv/bin/python \\
        validation/e2e/run_multimodal_pipeline.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp


SAMPLES = [
    ("A01", Path("/home/arsham79/projects/rrg-hsn/arsham79/alt_splicing/results/star_solo_out/L8TX_181211_01_A01_S01_L003")),
    ("B01", Path("/home/arsham79/projects/rrg-hsn/arsham79/alt_splicing/results/star_solo_out/L8TX_181211_01_B01_S01_L003")),
]
N_LATENT = 20
N_TOP_HVE = 10_000          # splicing events kept for the multigedipy run
N_TOP_HVG = 5_000           # gene-expression genes kept (Seurat-style HVG)
MAX_ITER = 30
OUT = Path(__file__).resolve().parent / "data" / "multimodal_run.h5ad"


def _step(msg: str) -> None:
    print(f"[multimodal] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1.  Load splicing via splikit-py.  Returns AnnData with layers["M1"],
#          layers["M2"], var carrying chr/start/end/strand/group_id/group_kind/
#          group_count/row_names_mtx, obs carrying barcode + sample_id.
# ──────────────────────────────────────────────────────────────────────────────
def load_splicing() -> ad.AnnData:
    import splikit as splk

    sj_dirs = [base / "Solo.out" / "SJ" for _sid, base in SAMPLES]
    sample_ids = [sid for sid, _base in SAMPLES]

    _step(f"splk.io.read_starsolo({len(sj_dirs)} samples, internal whitelist)")
    a = splk.io.read_starsolo(
        sj_dirs=sj_dirs,
        sample_ids=sample_ids,
        use_internal_whitelist=True,    # uses Solo.out/Gene/filtered/barcodes.tsv per sample
        keep_multi_mapped=False,
        min_counts=1,
        verbose=True,
    )
    _step(f"  splicing adata: {a.n_obs} cells × {a.n_vars} events "
          f"(M1 nnz={a.layers['M1'].nnz:,})")

    _step("splk.tl.make_m2(n_threads=8)")
    splk.tl.make_m2(a, n_threads=8)
    _step(f"  M2 nnz={a.layers['M2'].nnz:,}")

    _step(f"splk.pp.highly_variable_events(min_row_sum=10, n_top={N_TOP_HVE})")
    splk.pp.highly_variable_events(
        a, min_row_sum=10, n_top=N_TOP_HVE,
        sample_key="sample_id", n_threads=8, inplace=True,
    )
    a = a[:, a.var["highly_variable"]].copy()
    _step(f"  splicing post-HVE: {a.n_obs} × {a.n_vars}")

    # Subsetting var invalidates M2; recompute on the HVE-filtered subset.
    splk.tl.make_m2(a, n_threads=8)
    return a


# ──────────────────────────────────────────────────────────────────────────────
# Step 2.  Load gene expression for the same two samples from
#          Solo.out/Gene/filtered/.  scanpy.read_10x_mtx gives a per-sample
#          AnnData; concatenate across samples and apply HVG.
# ──────────────────────────────────────────────────────────────────────────────
def load_gene_expression() -> ad.AnnData:
    import scanpy as sc

    per_sample = []
    for sid, base in SAMPLES:
        gene_dir = base / "Solo.out" / "Gene" / "filtered"
        _step(f"sc.read_10x_mtx({gene_dir})")
        a = sc.read_10x_mtx(gene_dir, var_names="gene_ids", make_unique=True)
        # Append the sample suffix on obs_names so the full anndata.concat keeps
        # cells distinguishable (matches splikit-py's `<barcode>-<sample_id>`).
        a.obs_names = a.obs_names + f"-{sid}"
        a.obs["sample_id"] = sid
        per_sample.append(a)
        _step(f"  {sid}: {a.n_obs} × {a.n_vars}")

    _step("anndata.concat(axis=0)")
    a = ad.concat(per_sample, axis=0, join="outer", merge="same",
                  uns_merge="same", index_unique=None)
    a.obs["sample_id"] = a.obs["sample_id"].astype("category")
    _step(f"  pooled gene adata: {a.n_obs} × {a.n_vars}")

    # Seurat-style HVG via scanpy. We need raw counts later for multigedipy
    # (it applies log1p internally), so cache raw counts in layers['counts'].
    a.layers["counts"] = a.X.copy()
    _step(f"sc.pp.highly_variable_genes(flavor='seurat_v3', n_top_genes={N_TOP_HVG})")
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=N_TOP_HVG,
                                batch_key="sample_id", layer="counts")
    a = a[:, a.var["highly_variable"]].copy()
    _step(f"  gene post-HVG: {a.n_obs} × {a.n_vars}")
    return a


# ──────────────────────────────────────────────────────────────────────────────
# Step 3.  Align cells: keep only barcodes shared between the splicing adata
#          and the gene-expression adata, in identical order. multigedipy can
#          handle partial-overlap (per-modality sample_vec), but the
#          cell-paired multi-modal interpretation requires identical cells.
# ──────────────────────────────────────────────────────────────────────────────
def align_cells(spl: ad.AnnData, gex: ad.AnnData) -> tuple[ad.AnnData, ad.AnnData]:
    common = sorted(set(spl.obs_names) & set(gex.obs_names))
    _step(f"cell intersection: {len(common)} (splicing: {spl.n_obs}, "
          f"gene: {gex.n_obs})")
    spl = spl[common, :].copy()
    gex = gex[common, :].copy()
    # Sanity: both AnnDatas now have identical obs_names + sample_id.
    assert (spl.obs_names == gex.obs_names).all()
    assert (np.asarray(spl.obs["sample_id"]) == np.asarray(gex.obs["sample_id"])).all()
    return spl, gex


# ──────────────────────────────────────────────────────────────────────────────
# Step 4.  Hand to multigedipy.  multigedipy's per-modality matrices are
#          (J × N) = events × cells; AnnData is (N × J), so we transpose at
#          the boundary. sample_vec is the per-cell sample label, length N.
# ──────────────────────────────────────────────────────────────────────────────
def run_multigedipy(spl: ad.AnnData, gex: ad.AnnData):
    import multigedipy as mg
    from multigedipy import MultiGEDIModel

    sample_vec = np.asarray(spl.obs["sample_id"].astype(str))

    # Splicing inputs as events × cells, sparse CSC float64.
    M1 = sp.csc_matrix(spl.layers["M1"].T, dtype=np.float64)
    M2 = sp.csc_matrix(spl.layers["M2"].T, dtype=np.float64)

    # Gene-expression input as genes × cells, sparse CSC float64. multigedipy
    # applies log1p internally (obs_type='M'), so we pass raw counts.
    gex_X = gex.layers["counts"] if "counts" in gex.layers else gex.X
    G = sp.csc_matrix(gex_X.T, dtype=np.float64)

    _step(f"MultiGEDIModel(K={N_LATENT}, mode='Bsphere', seed=42)")
    model = MultiGEDIModel(K=N_LATENT, mode="Bsphere", seed=42,
                           num_threads=8, verbose=1)

    _step(f"add_modality(name='gene_expression', obs_type='M', shape={G.shape})")
    model.add_modality(
        name="gene_expression",
        data=G,
        sample_vec=sample_vec,
        obs_type="M",
        orthoZ=True,
    )

    _step(f"add_modality(name='splicing', obs_type='M_list', M1={M1.shape} M2={M2.shape})")
    model.add_modality(
        name="splicing",
        data=(M1, M2),
        sample_vec=sample_vec,
        obs_type="M_list",
        orthoZ=False,
        is_si_fixed=True,
    )

    _step(f"model.train(iterations={MAX_ITER})")
    t0 = time.monotonic()
    model.train(iterations=MAX_ITER, track_interval=1)
    _step(f"  train elapsed: {time.monotonic() - t0:.1f}s")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Step 5.  Extract outputs.  Per-modality Z (gene loadings), the shared cell
#          embedding (Bi concatenated across samples), and per-modality
#          sigma² for diagnostics.  Save back into a unified MuData (or just
#          stash on the splicing adata) so downstream scanpy / scvi-tools
#          can consume it.
# ──────────────────────────────────────────────────────────────────────────────
def collect_outputs(model, spl: ad.AnnData, gex: ad.AnnData) -> ad.AnnData:
    Z_gex = model.get_Z("gene_expression")
    Z_spl = model.get_Z("splicing")
    Bi_gex = model.get_Bi("gene_expression")  # list of (K, Ni) per sample
    Bi_spl = model.get_Bi("splicing")
    shared = np.concatenate([b for b in model.get_shared_Bi()], axis=1).T  # (n_obs, K)

    _step(f"  Z(gene_expression) {Z_gex.shape}; Z(splicing) {Z_spl.shape}")
    _step(f"  shared cell embedding {shared.shape}")

    # Stash everything on the splicing adata (it's the more bespoke object).
    spl.obsm["X_multigedi"] = shared
    spl.varm["multigedi_Z"] = Z_spl
    gex.varm["multigedi_Z"] = Z_gex
    spl.uns["multigedi"] = {
        "params": {"K": N_LATENT, "mode": "Bsphere", "max_iter": MAX_ITER},
        "sigma2_gene_expression": float(model.get_sigma2("gene_expression")),
        "sigma2_splicing": float(model.get_sigma2("splicing")),
    }

    # Optional: persist alongside the splicing adata.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    spl.write_h5ad(OUT)
    _step(f"  wrote {OUT}")
    return spl


def main() -> int:
    spl = load_splicing()
    gex = load_gene_expression()
    spl, gex = align_cells(spl, gex)
    model = run_multigedipy(spl, gex)
    collect_outputs(model, spl, gex)
    _step("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
