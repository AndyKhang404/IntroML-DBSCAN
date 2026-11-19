## Summary
<!-- What does this PR implement or update? -->

## Role & Related Section
- Owner: (M1 / M2 / M3 / M4 / M5)
- Report section(s) affected:

## Reproducibility Checklist
- [ ] Fixed seed (`DEFAULT_SEED = 42` or documented in YAML config) used for any stochastic code
- [ ] Config and output paths (`Source/results/` or `Report/figures/`) documented
- [ ] `uv run --no-sync poe lint` (`ruff check Source Manim scripts`) passes
- [ ] `uv run --no-sync poe test` (`pytest`) passes
- [ ] `typst compile Report/main.typ Report/Report.pdf` succeeds (if report changed)
