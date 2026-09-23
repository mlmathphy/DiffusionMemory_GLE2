# Example 3: intermittent scrape-off-layer fluctuations

This example generates a two-population filtered Poisson process with finite
pulse rise time and observation noise. Only the combined signal is observed.
A particle filter supplies conditional reference ensembles for evaluation.

Run from this directory after installing the root requirements:

```bash
python run_all.py all
python run_all.py physics
python run_all.py figs
```

The default pipeline generates synthetic records, trains the memory-conditioned
diffusion flow map and Gaussian-head LSTM, and evaluates conditional and
closed-loop statistics. `physics` computes burst statistics and phase portraits
from cached trajectories. `figs` generates the conditional forecast figure.
Configuration and seeds are declared in `config.py`.

Manuscript figures are written to `figs/`:

- `ex3_phase.pdf`: signal–increment phase portrait.
- `ex3_cloud.pdf`: conditional forecast ensembles.
- `ex3_bursts.pdf`: averaged burst waveform and peak-amplitude exceedance.
- `ex3_rollout.pdf`: autocorrelation and stationary density.

The separate `prevet.py` script validates the synthetic reference process and
particle-filter diagnostics; it is not a prerequisite of `run_all.py all`.
Use `python run_all.py smoke` for a reduced mechanical pipeline check.

After a completed full run, redraw figures from saved outputs with:

```bash
python evaluate.py --plot
python make_figs.py --plot
python make_physics_figs.py --plot
```

To redraw only the burst figure without writing numerical records:

```bash
python make_physics_figs.py --plot --bursts-only
```

Plot-only modes require generated local outputs. No data, checkpoints, or
rollout caches are included in this distribution.
