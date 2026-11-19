# AGENTS.md

Shared project contract for coding agents assisting the 5-member **IntroML-DBSCAN** team (CSC14005, FIT HCMUS; deadline **2026-12-19/20**, freeze **2026-12-18**).

## Locked Decisions & Scope

- **Paper**: Ester et al. (KDD-96), *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise* (`Docs/dbscan.pdf`).
- **Core implementations**: From-scratch NumPy `DBSCANScratch` (`brute` and `kdtree`) validated against `sklearn.cluster.DBSCAN`, plus a from-scratch `CLARANS` baseline.
- **Reproduction scope (Full)**:
  1. DBSCAN vs. CLARANS clustering quality on recreated synthetic DB1–3.
  2. Sorted $k$-dist graph for `Eps` selection.
  3. Runtime scaling at paper sample sizes (`[1252, 2503, 3910, 5213, 6256]`) using a 2D spatial substitute for SEQUOIA 2000; document the substitution and compare scaling trends/ratios rather than 1996 wall-clock seconds.
- **Bonus scope**: Limitation experiments (varying density, high-dimensional distance concentration, `Eps`/`MinPts` sensitivity, optional OPTICS/HDBSCAN comparison), with a modern 2D/spatial dataset as backup.
- **Languages**: English Typst report (`Report/`) and English Manim video with AI voiceover and subtitles (`Manim/`).

## Environment & Gotchas

- **Toolchain**: Python `3.12` managed via `uv` workspace (`Source` + `Manim` members), `poethepoet` (`poe` tasks in `pyproject.toml`), `ruff`, `pytest`, and `typst` CLI. Use `uv` + `poe` for all development; `scripts/compat.py` is strictly a plain-Python fallback for environments without `uv`/`poe`.
- **Source-only sync**: Run `uv sync --package dbscan-repro --group dev` (or `uv run --no-sync poe sync-source` once synced) when working on experiments/tests so `uv` skips `Manim` system dependencies (Cairo/Pango/LaTeX). Run `uv sync --all-packages --group dev` (`poe sync`) only when working on `Manim`.
- **Use `--no-sync` on checks**: Run `uv run --no-sync poe test` / `uv run --no-sync poe lint` (or activate `.venv` and run `poe test` / `poe lint`) so `uv run` does not attempt to build `dbscan-manim` in a `Source`-only environment.
- **Keep `requirements.txt` synced**: Whenever `Source/pyproject.toml` dependencies change, run `uv run --no-sync poe export-req` to regenerate `Source/requirements.txt`.
- **Typst**: `typst` compiles `Report/main.typ` to `Report/Report.pdf` (`uv run --no-sync poe report`). CI compiles it on every push/PR; locally install `typst-cli` if missing.

## Code & Experiment Conventions

- **Scripts over notebooks**: Keep experiments as CLI scripts in `Source/experiments/` driven by YAML configs (`Source/experiments/configs/default.yaml`).
- **Reproducibility**: Initialize RNG via `set_seed(cfg.seed)` (`DEFAULT_SEED = 42`), and record seeds, configs, and library versions alongside outputs in `Source/results/` (CSV/PNG).
- **DBSCAN paper semantics**: `N_Eps(p)` includes `p` itself (consistent with `MinPts = 4` in KDD-96). Border-point cluster assignment depends on point traversal order—note any order-dependent differences when testing against `sklearn`.
- **Video assets**: Generate all diagrams/plots in Manim (no raw screenshots from `Docs/dbscan.pdf`), include on-screen citations, and include subtitles throughout.
- **AI disclosure**: Log every AI session (including TTS and coding assistants) in your member file `Docs/ai-usage/M<1-5>.md`; each entry must contain 3 sections: `Date`, `Tool`, and `Prompt`. Summarize all tools and purposes in `Report/sections/05_ai_use.typ`.

## Team Ownership & Workflow

Work on feature branches (`feat/<topic>`), open PRs into `main` using `.github/pull_request_template.md`, and obtain at least 1 peer review.

| Member | Code / Asset Ownership | Report Sections | Reviews |
|---|---|---|---|
| **M1 (Lead)** | Workspace/CI, Typst template, integration, submission zip | §1.1.1, §1.1.3, `04_team.typ`, `05_ai_use.typ` | M2 |
| **M2** | `dbscan.py` (brute + KD-tree), `kdist.py`, `sklearn` equivalence tests | §1.2.2 code analysis | M3 |
| **M3** | `clarans.py`, `datasets.py` (DB1–3 + SEQUOIA sub), `metrics.py`, `plotting.py`, experiments `01`–`03` | §1.2.1 setup, §1.2.3 comparison | M4 |
| **M4** | Limitation/sensitivity/backup-dataset experiment `04` | §1.1.2 course links, `03_bonus.typ` | M5 |
| **M5** | `Manim/scenes/`, `script.md`, AI voice, subtitles, upload (`url.txt` + `#fithcmus #machinelearning #ai #ml`) | §1.1.2 core method/lemmas/complexity | M1 |

Modify only files within your assigned scope unless coordinating a shared interface change via PR.

## Verification Gate (Before Opening or Merging a PR)

1. `uv run --no-sync poe lint` passes with zero errors.
2. `uv run --no-sync poe test` passes, with unit/equivalence tests covering any newly implemented behavior.
3. `Source/requirements.txt` is regenerated via `uv run --no-sync poe export-req` if `Source/pyproject.toml` changed.
4. `typst compile Report/main.typ Report/Report.pdf` succeeds if `Report/` changed.

## Context Pointers (Read On Demand)

- **Course rubric, video rules, submission zip structure**: `Docs/ML - Course Project.pdf`
- **Algorithm definitions, lemmas, experiment specs**: `Docs/dbscan.pdf`
- **Default hyperparameters, seeds, scaling sizes**: `Source/experiments/configs/default.yaml`
- **Task definitions & CI workflow**: `pyproject.toml` (`[tool.poe.tasks]`), `.github/workflows/ci.yml`
- **Plain-Python compatibility fallback**: `scripts/compat.py`
- **Manim prerequisites & video storyboard**: `Manim/README.md`, `Manim/script.md`
