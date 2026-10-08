# Contributing

Use uv 0.9.17 or newer on Linux x86_64. Python 3.9–3.12 is supported; `.python-version` selects 3.11 by default. The committed `uv.lock` is the dependency source of truth.

```bash
uv sync --locked --extra cpu --extra export
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts
uv run --no-sync ruff format --check src tests scripts
uv build --no-sources
```

For A40 development, replace `--extra cpu` with `--extra cuda` and add `--extra kinematics`. Never enable both accelerators. Add runtime dependencies to `pyproject.toml`, development tools to its `dev` group, and regenerate `uv.lock` with `uv lock`. Commit both files and verify a clean checkout. Tests must create their own synthetic inputs and must not require local data, plate-model archives, `processed/` or `outputs/`.

Add numerical regression tests when changing observation operators, coordinate conventions, gradients, source terms or conservation signs. A synthetic manufactured solution must exercise new physical constraints before claims are made about real data. Do not use independent test scores to select hyperparameters.

Keep package code in `src/norn_earth`; scripts should call public APIs. Configuration paths must be portable and documented. Checkpoints must remain standalone or change their schema version with an explicit migration policy. Never modify original data as part of training or preparation.

Do not commit raw observations, external plate-model archives, personal paths, virtual environments, model checkpoints or run outputs. Put actual validation evidence in `docs/validation.md` with its scope and data assumptions. Clearly distinguish mathematical correctness, successful execution and geological validation.
