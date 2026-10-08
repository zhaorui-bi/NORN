# Contributing

Use Python 3.9–3.12 and install `.[sfno,kinematics,export,dev]` in an isolated environment. The tested A40 stack is recorded in `requirements/a40-cu121.txt`; CPU CI installs PyTorch from the official CPU index.

Before proposing a change, run:

```bash
python -m pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
python -m build
```

Add numerical regression tests when changing observation operators, coordinate conventions, gradients, source terms or conservation signs. A synthetic manufactured solution must exercise new physical constraints before claims are made about real data. Do not use independent test scores to select hyperparameters.

Keep package code in `src/norn_earth`; scripts should call public APIs. Configuration paths must be portable and documented. Checkpoints must remain standalone or change their schema version with an explicit migration policy. Never modify original data as part of training or preparation.

Do not commit raw observations, external plate-model archives, personal paths, virtual environments, model checkpoints or run outputs. Put actual validation evidence in `docs/validation.md` with its scope and data assumptions. Clearly distinguish mathematical correctness, successful execution and geological validation.
