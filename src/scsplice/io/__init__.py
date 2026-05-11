"""scsplice.io — readers for STARsolo / 10x output."""

from scsplice.io._starsolo import read_starsolo
from scsplice.io._starsolo_gene import read_starsolo_gene
from scsplice.io._starsolo_velocyto import read_starsolo_velocyto

__all__ = ["read_starsolo", "read_starsolo_gene", "read_starsolo_velocyto"]
