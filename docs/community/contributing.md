# Contributing

Contributions are welcome! Please open an issue to discuss major changes before submitting a PR.

## Development setup

1. Create the Conda environment and install the package in editable mode:
   - `conda env create -f environment.yml`
   - `conda activate jlc`
   - `python -m pip install -e .`

2. Run and iterate locally. Add or update tests where appropriate.

3. Commit messages should be descriptive. Prefer small, focused PRs.

## Code style and linting

Follow standard Python practices. If linters/formatters are added later (e.g., black, ruff), CI will enforce them.

## Reporting bugs and requesting features

Open an issue with a clear description, steps to reproduce (if applicable), and expected vs. actual behavior. Include environment details (`python --version`, OS, etc.).

## Pull request checklist

- [ ] Code compiles and runs locally
- [ ] New functionality is documented in the docs (User Guide/Algorithms/README as appropriate)
- [ ] Add tests or examples when reasonable
- [ ] Update CHANGELOG if user-facing behavior or CLI changes
