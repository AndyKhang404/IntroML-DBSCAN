# Manim — Visualization Video

Manim animations for the 5–10 minute explanatory video on **DBSCAN** (Ester et al., KDD-96).

## System Prerequisites

Install system dependencies before syncing `dbscan-manim`:

- **Fedora / RHEL**: `sudo dnf install cairo-devel pango-devel pkg-config ffmpeg texlive-scheme-medium`
- **Ubuntu / Debian**: `sudo apt install libcairo2-dev libpango1.0-dev pkg-config ffmpeg texlive-latex-extra`
- **macOS**: `brew install cairo pango pkg-config ffmpeg`

## Setup with `uv`

From the repository root:

```bash
uv sync --package dbscan-manim
```

## Rendering

Low-quality preview:

```bash
uv run --package dbscan-manim manim -pql Manim/scenes/intro.py IntroScene
```

Final 1080p60 render:

```bash
uv run --package dbscan-manim manim -pqh Manim/scenes/intro.py IntroScene
```
