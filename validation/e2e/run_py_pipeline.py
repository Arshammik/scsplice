"""End-to-end Python pipeline on samples A01 + B01.

Builds the splicing AnnData from the two STARsolo dirs, recomputes M2, and
saves to ``validation/e2e/data/py_pipeline.h5ad`` for comparison against the
R reference (``run_r_pipeline.R``).

Run on c170 (or any node with the splikit-py editable install):

    cd /home/arsham79/projects/rrg-hsn/arsham79/splikitpy
    module load eigen/3.4.0
    python validation/e2e/run_py_pipeline.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import splikit as splk


SAMPLE_DIRS = [
    Path("/home/arsham79/projects/rrg-hsn/arsham79/alt_splicing/results/star_solo_out/L8TX_181211_01_A01_S01_L003/Solo.out/SJ"),
    Path("/home/arsham79/projects/rrg-hsn/arsham79/alt_splicing/results/star_solo_out/L8TX_181211_01_B01_S01_L003/Solo.out/SJ"),
]
SAMPLE_IDS = ["A01", "B01"]
OUT = Path(__file__).resolve().parent / "data" / "py_pipeline.h5ad"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for d in SAMPLE_DIRS:
        if not d.exists():
            print(f"[ERROR] STARsolo dir not found: {d}", file=sys.stderr)
            return 1

    t0 = time.monotonic()
    print(f"[py] read_starsolo on {len(SAMPLE_DIRS)} samples (cells: 6.8M raw -> "
          "Gene/filtered whitelist auto-applied per sample)...", flush=True)
    adata = splk.io.read_starsolo(
        sj_dirs=SAMPLE_DIRS,
        sample_ids=SAMPLE_IDS,
        use_internal_whitelist=True,
        keep_multi_mapped=False,
        min_counts=1,
        verbose=True,
    )
    print(f"[py]   adata: {adata.n_obs} cells x {adata.n_vars} events "
          f"(M1 nnz={adata.layers['M1'].nnz:,}); "
          f"elapsed {time.monotonic() - t0:.1f}s", flush=True)

    t1 = time.monotonic()
    print("[py] make_m2(n_threads=8)...", flush=True)
    splk.tl.make_m2(adata, n_threads=8)
    print(f"[py]   M2 nnz={adata.layers['M2'].nnz:,}; "
          f"elapsed {time.monotonic() - t1:.1f}s", flush=True)

    print(f"[py] writing {OUT}...", flush=True)
    adata.write_h5ad(OUT)
    print(f"[py] DONE. Total elapsed: {time.monotonic() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
