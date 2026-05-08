"""splikit-py: single-cell alternative-splicing analysis on AnnData.

Public API surface (v1.0):

    splikit.io.read_starsolo               # splicing (M1 / LJV-grouped)
    splikit.io.read_starsolo_gene          # gene expression (X)
    splikit.io.read_starsolo_velocyto      # spliced / unspliced / ambiguous
    splikit.tl.make_m2
    splikit.tl.pseudo_correlation
    splikit.pp.highly_variable_events
"""

from importlib.metadata import PackageNotFoundError, version

from . import io, pl, pp, tl
from ._settings import settings

try:
    __version__ = version("splikit-py")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__", "io", "pl", "pp", "tl", "settings"]
