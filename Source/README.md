# Source — DBSCAN Reproduction Experiments

From-scratch Python/NumPy implementation of **DBSCAN** (Ester et al., KDD-96) and **CLARANS** (Ng & Han, 1994), plus reproduction and limitation experiments.

## Setup

### Using `uv` + `poe` (Recommended)

From the repository root:

```bash
uv sync --package dbscan-repro --group dev
```

### Compatibility Option: Plain Python + `pip`

If `uv` is unavailable (or when running `Source/` standalone):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Source/requirements.txt
pip install -e Source --no-deps
```

Or from the repository root using the compatibility runner:

```bash
python scripts/compat.py install-source --dev
```

## Running Tests and Linting

```bash
# Preferred (uv + poe)
uv run --no-sync poe test
uv run --no-sync poe lint

# Plain-Python fallback
python scripts/compat.py test
python scripts/compat.py lint
```

## Running Experiments

All experiments read `Source/experiments/configs/default.yaml` by default (seed `42`) and write CSV/PNG outputs to `Source/results/`.

```bash
# Preferred (uv + poe)
uv run --no-sync poe exp-01
uv run --no-sync poe exp-02
uv run --no-sync poe exp-03
uv run --no-sync poe exp-04

# Plain-Python fallback
python scripts/compat.py exp all
```
