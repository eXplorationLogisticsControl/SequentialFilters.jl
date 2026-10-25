# Filter experiments

## Square-root CUT benchmarks

`experiments/run_srcut_benchmarks.jl` compares full-covariance CUT, square-root CUT, the square-root cubature Kalman filter, the UKF, and the square-root UKF on two scenarios that are not the two-body demo.

```bash
julia --project=experiments experiments/run_srcut_benchmarks.jl
```

Optional arguments are `--M` (default 200), `--seed` (default 20261025), `--illcond-only`, and `--longarc-only`. Tables and working figures are written to `experiments/output/srcut/`. The numeric summary is `experiments/srcut_results.md`.

The ill-conditioned case is a planar trajectory with position in kilometres, velocity in km/s, and a drag coefficient. Two range measurements have a 1 cm standard deviation. The position prior is swept from 1e3 to 1e8 times that standard deviation, in Float64 and Float32. CUT-6 is included because the dynamics are a closed-form step.

The long-arc case is angles-only orbit determination. A 400 km LEO and a 500 by 25000 km HEO both use J2 and drag. Passes are separated by a coast from a few hours to several days, using the next visible pass after the requested gap. The posterior CRLB averages the Tichavský covariance bound over the Monte Carlo truths. The long-arc sweep uses CUT-4. A 6-state CUT-6 set has 137 sigma points; order 6 is in the ill-conditioned sweep and in the linear-Gaussian unit test.

## Two-body Monte Carlo

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
