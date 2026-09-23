# Example 1: scalar GLE with multiscale memory

Run from this directory after installing the root requirements:

```bash
python run_all.py all
python run_all.py sweep
```

The first command runs exact-reference validation, data generation, diffusion
model training, LSTM training, one-step evaluation, long rollouts, conditional
forecasts, predictive-rank diagnostics, and figure generation. The second runs
the separate LSTM context-length and training-seed study used in Table 1(b).
Full runs use the settings in `config.py`.

Individual stages are `prevet`, `data`, `train`, `lstm`, `onestep`, `eval`,
`fans`, `stats`, `rank`, and `figs`. Training includes memory-conditioned,
memoryless, and raw-history models. The longer raw-window variant is optional
and is not required for the manuscript pipeline.

Manuscript outputs include:

- `figs/ex4_memory.pdf`: memory representation and predictive coordinates.
- `figs/ex4_compare.pdf`: closed-loop correlations and velocity-change statistics.
- `figs/ex4_ensemble.pdf`: conditional forecast means and standard deviations.
- `out/ensemble_stats.json`: aggregate conditional forecast diagnostics.
- `out/lstm_sweep/sweep.json`: LSTM context-length study.

The `ex4_` filename prefix is historical and refers to manuscript Example 1.
After completing the numerical stages, redraw the main figures with:

```bash
python make_figs.py --plot
python evaluate.py --plot
python ensemble_stats.py plot
```

These commands require existing outputs; no data or checkpoints are distributed.
