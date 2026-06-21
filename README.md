# Pattern Finding Playground

Python foundation to write and test pattern finding algorithms.

👉 See [notebooks/examples.ipynb](notebooks/examples.ipynb) for examples of how to use the framework.

## Layout

- `src/` contains importable Python code used by the notebooks.
- `src/data_io/` contains ROOT dictionary loading, file metadata helpers, geometry header access, and RNTuple dataset loading.
- `src/figures/` contains reusable figure classes and figure-specific utilities.
- `src/pipeline/stages/` contains both the base `Stage` class and concrete stage implementations.
- `src/algorithms/vertex/type_scoring/` contains vertex type hypotheses and their scorers.
- `notebooks/` contains the user-facing analysis notebooks.
- `.artifacts/` contains tracked inspection helpers and notes.
- `.data/` contains local generated ROOT files and is intentionally ignored.
- `.deprecated/` contains old notebooks/scripts and is intentionally ignored.
