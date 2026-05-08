# Release notes

## v0.1.0.dev0 (in development)

Initial pre-alpha development release. Not yet published to PyPI.

**Scope:** four public functions — `read_starsolo`, `make_m2`, `highly_variable_events`, `pseudo_correlation` — backed by pybind11-wrapped Eigen3 C++ kernels.

**Cross-language parity:** M2 bit-exact vs. R splikit; HVE deviance `rtol=1e-10`; pseudo-correlation `rtol=1e-7`. The full regression suite lives on the [`validation` branch](https://github.com/Arshammik/splikitpy/tree/validation).

For the full commit history, see the [GitHub repository](https://github.com/Arshammik/splikitpy/commits/main).

---

Formal versioned release notes will be published here starting with v1.0.
