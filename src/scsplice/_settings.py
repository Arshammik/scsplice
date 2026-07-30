"""Global scsplice settings (scanpy-style).

A single :class:`_Settings` instance is exposed as :data:`scsplice.settings`.
Attributes can be set directly to control package behaviour without passing
per-call keyword arguments.

The settings are intentionally minimal; per-call ``verbose`` and
``n_threads`` arguments still work and override the defaults below.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Settings:
    """Mutable container for global scsplice knobs.

    Attributes
    ----------
    verbosity
        Logging verbosity level (0 = errors only, 1 = warnings, 2 = info,
        3 = debug). Default: 1. Mirrors :class:`scanpy._settings.Verbosity`.
    n_jobs
        Default OpenMP thread count for kernels that accept ``n_threads``.
        Per-call ``n_threads=`` overrides this. ``-1`` defers to
        ``OMP_NUM_THREADS`` / ``omp_get_max_threads()``. Default: 1.
    """

    verbosity: int = 1
    n_jobs: int = 1

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        return f"scsplice.settings(verbosity={self.verbosity}, n_jobs={self.n_jobs})"


settings = _Settings()


__all__ = ["settings"]
