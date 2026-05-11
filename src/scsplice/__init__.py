"""scsplice: single-cell alternative-splicing analysis on AnnData.

Public API surface (v1.0):

    scsplice.io.read_starsolo               # splicing (M1 / LJV-grouped)
    scsplice.io.read_starsolo_gene          # gene expression (X)
    scsplice.io.read_starsolo_velocyto      # spliced / unspliced / ambiguous
    scsplice.tl.make_m2
    scsplice.tl.pseudo_correlation
    scsplice.pp.highly_variable_events
"""

from importlib.metadata import PackageNotFoundError, version

from . import io, pl, pp, tl
from ._settings import settings

try:
    __version__ = version("scsplice")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__", "io", "pl", "pp", "tl", "settings"]
