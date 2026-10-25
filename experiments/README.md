# Filter experiments

Monte Carlo comparison of every filter in SequentialFilters on the two-body
orbit from the EKF demo. The filters are the EKF, UKF, square-root EKF,
square-root UKF, conjugate unscented filter (CUT4), and square-root CUT4.

Errors are truth minus estimate. Position and velocity RMSE, NEES, NIS,
innovation autocorrelation, and the divergence rate come from the diagnosis
functions in `src/diagnosis.jl`. The posterior Cramér–Rao bound is the
Tichavský, Muravchik, and Nehorai (1998) recursion on the noise-free reference
trajectory, using the scenario Jacobians. NEES and NIS confidence bands are a
Wilson–Hilferty approximation of the 95% interval for the mean of M
chi-squared samples.

## Reproduce

```julia
julia --project=experiments experiments/run_twobody.jl
```

- Seed: `20261024`. The measurement-noise sweep uses `20261025`.
- Monte Carlo runs: `M = 100` for the main comparison.
- Divergence threshold: `50` km on the Euclidean position error. A non-finite
  estimate or covariance also counts as diverged. The rate is reported even
  when it is zero.
- Measurement-noise scales: `0.5`, `1`, `2`, and `5`, with `M = 20`.
- Time averages use the posterior at each measurement instant. RMSE and NEES
  plots also include the initial time.

Optional arguments: `--M`, `--sweep-M`, `--seed`, and `--skip-sweep`.

`initial_error_scale`, `steps_per_orbit`, `n_orbits`, and `model_mismatch` are
arguments of `twobody_scenario`. The executed sweep changes only the
measurement-noise scale. That scale multiplies the measurement-noise standard
deviation and the `R` supplied to each filter.

## Output

`experiments/output/` is not committed. The run writes:

- `twobody_table.csv` and `twobody_table.txt`: time-averaged position RMSE,
  time-averaged velocity RMSE, divergence rate, and mean runtime per filter.
  Runtime is the mean elapsed time of one run after one untimed warmup.
- `sensitivity_measurement_noise.csv`
- `experiments_rmse.png`: position and velocity RMSE of every filter and the
  posterior CRLB
- `experiments_nees.png`: averaged NEES and the chi-squared band for
  `M * n_x` degrees of freedom
- `experiments_nis.png`: averaged NIS and the band for `M * n_y` degrees of
  freedom
- `experiments_autocorrelation.png`: mean innovation autocorrelation and the
  `±1.96/sqrt(N)` bound
