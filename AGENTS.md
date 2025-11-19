# Repository Guidelines

## Project Structure & Organization
- `Code/`: Python (`new.py`, `sbm_super_efficiency.py`) and R (`sbm_super_efficiency.R`) implementations of the SBM super-efficiency models.
- `Data/`: Input Excel/CSV files and example datasets. Treat large or sensitive source data as external and avoid committing new confidential files.
- `Result/`: Generated model outputs (CSV/XLSX). These are derived artifacts and can be regenerated from code and data.
- `Model/` and `Ref/`: Academic papers and methodological references.
- `SPSSres/`: Intermediate SPSS analysis outputs.

## Development, Build & Run
- Use Python 3.10+ and keep scripts compatible with the standard library where possible.
- Typical commands:
  - `python Code/new.py` – prepare data and run the main analysis workflow.
  - `python Code/sbm_super_efficiency.py` – run the SBM super-efficiency model.
  - `Rscript Code/sbm_super_efficiency.R` – execute the R reference implementation.
- When adding dependencies, document them in `requirements.txt` or at the top of the script.

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP 8 style. Use `snake_case` for functions/variables, `CapWords` for classes, and lowercase `snake_case` filenames.
- R: Prefer 2–4 spaces, `snake_case` for functions, and informative variable names.
- Keep functions small and focused; avoid hard-coded absolute paths—use relative paths based on the project root.

## Testing Guidelines
- There is no formal test suite yet; when adding tests, prefer `pytest`.
- Place tests under `Code/tests/` (e.g., `test_sbm_super_efficiency.py`) and name tests `test_*`.
- Run tests with `pytest Code` and aim to cover core model logic and data-processing helpers.

## Commit & Pull Request Guidelines
- Write clear, imperative commit messages (e.g., `Add SBM efficiency metrics export`).
- Group related changes in a single commit; separate unrelated refactors or data updates.
- For pull requests, describe the goal, note impacted scripts (`Code/*.py`, `Code/*.R`) and data/result files, and attach key before/after metrics or sample output when relevant.

## Agent & Automation Notes
- Do not overwrite source data in `Data/`; write new outputs to `Result/` or a new file.
- Prefer adding new scripts over heavily modifying reference implementations without explanation.

