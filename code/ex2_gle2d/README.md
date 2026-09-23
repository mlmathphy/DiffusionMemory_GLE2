# Example 2: coupled two-dimensional viscoelastic GLE

Run from this directory after installing the root requirements:

```bash
python run_all.py all
python run_all.py markov
python run_all.py lstm
python run_all.py eval
python run_all.py paperfigs
python run_all.py finalfigs
```

The default `all` sequence validates the exact model, generates trajectories,
checks the estimated predictive coordinates, trains the memory model, and
evaluates it. It does not train the baselines. The subsequent `markov` and
`lstm` stages train the memoryless diffusion and full-covariance Gaussian LSTM
baselines; rerunning `eval` then compares all models.

`paperfigs` generates and caches conditional forecast ensembles. `finalfigs`
uses those ensembles and the evaluation summary to write the four manuscript
figures to `final_figs/`:

- `ex2_vacf.pdf`
- `ex2_evolution2d.pdf`
- `ex2_evolution1d.pdf`
- `ex2_msd.pdf`

The memory bank uses reference-assisted selection as described in the paper;
its two-dimensional increment predictor is fitted from training data.
`scout.py` is required by the final pipeline for exact-reference and statistical
helper functions; running its exploratory search is not required.

For a reduced mechanical check, use `python run_all.py smoke`; it writes to
separate smoke-output directories and does not reproduce full-run results.
After a full run, `python run_all.py finalfigs` redraws the final figures using
saved outputs without retraining. No data, checkpoints, or forecast caches are
included in this distribution.
