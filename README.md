# IntroML-DBSCAN

Course project for **CSC14005: Introduction to Machine Learning** (FIT, HCMUS): theoretical analysis, from-scratch reproduction, limitation study, and Manim visualization of **DBSCAN** (*A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise*, Ester et al., KDD-96).

## Repository Structure

```text
IntroML-DBSCAN/
├── Docs/            # Project specification and reference paper
├── Source/          # From-scratch DBSCAN/CLARANS, experiments, tests, requirements.txt
├── Manim/           # Manim visualization scenes, narration script, subtitles/assets
├── Report/          # Typst source for the final report (compiles to Report.pdf)
├── scripts/         # Plain-Python compatibility runner (fallback only)
├── pyproject.toml   # uv workspace root, ruff/pytest config, and poethepoet tasks
└── url.txt          # Public URL to the uploaded 5–10 min visualization video
```

## Quickstart (Recommended: `uv` + `poe`)

Install `uv` (`https://docs.astral.sh/uv/`), then sync the `Source` package and dev tools (`poethepoet`, `pytest`, `ruff`):

```bash
uv sync --package dbscan-repro --group dev
```

Run workspace tasks via `poe` (use `--no-sync` before `poe` or activate `.venv` so `uv` does not force a `Manim` build on `Source`-only setups):

```bash
# Run tests and linter
uv run --no-sync poe test
uv run --no-sync poe lint

# Run experiments (01–04)
uv run --no-sync poe exp-01

# Regenerate Source/requirements.txt after Source dependency changes
uv run --no-sync poe export-req
```

To also install Manim (requires system libraries listed in `Manim/README.md`):

```bash
uv run --no-sync poe sync
uv run --no-sync poe video-preview
```

To compile the Typst report (`Report/main.typ` -> `Report/Report.pdf`), first install the `typst` CLI:

- **Cargo (cross-platform)**: `cargo install --locked typst-cli`
- **macOS / Linux (Homebrew)**: `brew install typst`
- **Windows (Winget)**: `winget install --id Typst.Typst`
- **Linux packages / prebuilt binaries**: install `typst` via your distro package manager (e.g., `sudo pacman -S typst`, `sudo apt install typst`) or download a release binary from `https://github.com/typst/typst/releases`.

Then compile with:

```bash
uv run --no-sync poe report
```

## Compatibility Fallback (Plain Python, No `uv`/`poe`)

For grading or environments without `uv`/`poethepoet`, use [scripts/compat.py](scripts/compat.py) inside an activated Python 3.12+ virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python scripts/compat.py install-source --dev
python scripts/compat.py test
python scripts/compat.py lint
python scripts/compat.py exp 01
```

## Team Roles & Workflow

- **M1 — Phùng Bảo Khang (24127052) (Lead / Report)**: Scaffold, `uv`/CI, Typst report integration, §1.1.1 & §1.1.3, submission packaging.
- **M2 — Nguyễn Minh Đức (24127345) (Core Algorithms)**: From-scratch DBSCAN (brute-force + KD-tree), `kdist`, `sklearn` equivalence tests, §1.2.2.
- **M3 — Nguyễn Phạm Tuấn Đạt (24127342) (Paper Reproduction)**: CLARANS baseline, synthetic DB1–3, SEQUOIA substitute scaling study, §1.2.1 & §1.2.3.
- **M4 — Nguyễn Hoàng Bảo Long (24127201) (Bonus & Course Links)**: Limitation experiments, sensitivity analysis, backup dataset, course links, §1.2.4.
- **M5 — Văn Phú Đức (24127346) (Manim Video)**: Storyboard, Manim animations, English AI voiceover, subtitles, upload, §1.1.2 core theory.

Branching policy: work on feature branches (`feat/<topic>`), open a PR into `main`, pass CI (`ruff` + `pytest` + Typst build), and get at least 1 peer review before merging.
