# Video Storyboard & Narration Script

- **Target length**: 5–10 minutes (aim ~7–8 minutes)
- **Format**: 1080p, 16:9 (YouTube)
- **Language**: English narration (AI Voice — declare tool in `Report/sections/05_ai_use.typ`) + burned-in/SRT subtitles throughout
- **Required hashtags**: `#fithcmus #machinelearning #ai #ml`
- **Visual rules**: All figures generated via Manim (no raw screenshots from the paper); cite sources on-screen when referenced and list full references at the end.

## Scene Breakdown

| Scene | File | Topic | Target Duration | Status |
|---|---|---|---|---|
| 01 | `scenes/intro.py` | Motivation: why k-means/CLARANS fail on arbitrary shapes & noise | 1:00 | Starter |
| 02 | `scenes/definitions.py` | $\varepsilon$-neighborhood, Core / Border / Noise points | 1:30 | Planned |
| 03 | `scenes/reachability.py` | Direct density-reachability, density-reachability, density-connectivity | 1:30 | Planned |
| 04 | `scenes/algorithm.py` | RegionQuery expansion & cluster growth step-by-step | 1:30 | Planned |
| 05 | `scenes/kdist.py` | Sorted $k$-dist graph heuristic for choosing $\varepsilon$ | 1:00 | Planned |
| 06 | `scenes/limitations.py` | Varying density & high-dimensional limitations + wrap-up & refs | 1:00 | Planned |
