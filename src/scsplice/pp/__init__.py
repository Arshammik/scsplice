"""scsplice.pp — preprocessing (per-event filters, HVE selection)."""

from scsplice.pp._hve import highly_variable_events

__all__ = ["highly_variable_events"]
