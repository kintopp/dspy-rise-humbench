# Claude Code Project Instructions

## Scoring Comparability

All benchmark scoring implementations must match the upstream `humanities_data_benchmark` repo exactly. Do NOT add normalization, fuzzy matching, or other enhancements that the upstream benchmark doesn't have, even when the upstream approach has known limitations. Document limitations in memory notes but keep scoring identical for leaderboard comparability. When scoring appears broken, investigate whether the issue is in the prompt (fixable) or the metric (document only).

## Benchmarks

Each benchmark is a self-contained package under `benchmarks/` exporting a standard interface. Scripts under `scripts/` work generically via `--benchmark` flag and dynamic imports. See `MEMORY.md` for the key differences table.
