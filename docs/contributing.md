# Contributing

## Development setup

```bash
# Clone
git clone https://github.com/Arshammik/splikitpy
cd splikitpy

# Install Eigen3 (required for the C++ extension)
sudo apt-get install libeigen3-dev     # Debian / Ubuntu
# brew install eigen                   # macOS

# Editable install with all development dependencies
pip install -e ".[test,docs]"
```

The `scikit-build-core` backend compiles the C++ extension during `pip install`. Subsequent edits to Python files take effect immediately (no rebuild needed); edits to `.cpp` files require re-running `pip install -e .`.

## Running the tests

```bash
# All tests (skips r_required and openmp by default if markers not satisfied)
pytest

# With coverage
pytest --cov=splikit --cov-report=term-missing

# Cross-language equivalence tests (require R + rpy2 to regenerate fixtures)
pytest -m r_required -v

# Fast subset only
pytest tests/ -k "not slow" -v
```

## Building the docs locally

```bash
# Serve with live reload
mkdocs serve --dev-addr 0.0.0.0:8000

# Build (strict: warns become errors for broken links)
mkdocs build --strict

# Clean build artifacts
rm -rf site/
```

The docs site is at `http://localhost:8000` after `mkdocs serve`.

## Code style

- Ruff for linting and import sorting: `ruff check . && ruff format .`
- Type hints in all public function signatures.
- NumPy-style docstrings for all public functions. Examples must be runnable.

## Project structure

```
src/splikit/
  io/          read_starsolo and STARsolo parsing internals
  tl/          make_m2, pseudo_correlation
  pp/          highly_variable_events
  pl/          (reserved for v2.0 plotting)
  _core/       validators, shared utilities
  _splikit_cpp (pybind11 extension, built from src/cpp/)
tests/
  test_make_m2.py
  test_hve.py
  test_pseudo_correlation.py
  test_r_equivalence_real.py   (requires r_reference_merfish.h5)
  r_export/                    R scripts that generate reference fixtures
docs/
  api/         auto-generated from docstrings via mkdocstrings
  tutorials/   hand-written notebooks
  how-to/      recipe pages
  explanation/ conceptual background
```

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Make your changes, add tests, and run `pytest` and `mkdocs build --strict`.
3. Open a pull request. The CI workflow runs tests and `mkdocs build --strict` on each PR.

Please do not open PRs that add new public functions without NumPy-style docstrings and at least one test.
