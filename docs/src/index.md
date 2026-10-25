# SequentialFilters.jl

Sequential filters for nonlinear state estimation.

| Filter | Stored uncertainty | Noise |
| :--- | :--- | :--- |
| [`ExtendedKalmanFilter`](@ref) | covariance `P` | additive, through an ODE propagation |
| [`UnscentedKalmanFilter`](@ref) | covariance `P` | additive |
| [`ConjugateUnscentedKalmanFilter`](@ref) | covariance `P` | additive, CUT weights |
| [`SquareRootExtendedKalmanFilter`](@ref) | lower factor `S` with `P = S*S'` | additive |
| [`SquareRootUnscentedKalmanFilter`](@ref) | lower factor `S` with `P = S*S'` | additive |
| [`SquareRootConjugateUnscentedKalmanFilter`](@ref) | lower factor `S` with `P = S*S'` | additive, CUT weights |
| [`SquareRootCubatureKalmanFilter`](@ref) | lower factor `S` with `P = S*S'` | additive, `2n` cubature points |
| [`ParticleFilter`](@ref) | weighted particles; `x` and `P` are sample moments | additive, bootstrap SIR |

The square-root filters keep a lower-triangular factor with a positive diagonal. Predict and update rebuild that factor with a thin QR, and the square-root unscented filter also uses a Cholesky rank-one downdate when a weight is negative. They do not form `P` and call `cholesky` on every step. `covariance` rebuilds `P = S*S'` when a full covariance is needed, for example to draw the `±3√P[i,i]` bands in the demos.

## Tutorials

The [two-body orbit](@ref twobody-tutorial) page walks through the shared scenario and the EKF demo.

Each tutorial starts with why that filter exists and the papers to read, then the predict and update equations, then the same calls the two-body demos make. A quantity ``X`` after prediction is the prior ``X_{k|k-1}``; after the measurement update it is the posterior ``X_{k|k}``. The posterior carried into the next predict is ``X_{k-1|k-1}``.

- [Extended Kalman filter (EKF)](@ref ekf-tutorial)
- [Unscented Kalman filter (UKF)](@ref ukf-tutorial)
- [Conjugate unscented filter (CUT)](@ref cut-tutorial)
- [Square-root EKF (SREKF)](@ref srekf-tutorial)
- [Square-root UKF (SRUKF)](@ref srukf-tutorial)
- [Square-root CUT (SRCUT)](@ref srcut-tutorial)
- [Square-root CKF (SRCKF)](@ref srckf-tutorial)
- [Bootstrap particle filter (PF)](@ref pf-tutorial)

## Demos and cost

The demos share the two-body problem in `demo/twobody/demo_ekf.jl`: a 7000 km orbit, range and range-rate from a rotating ground station, and the same initial covariance and process-noise scale. Each script saves an error figure (estimate minus truth, with `±3√P[i,i]` bands sampled along the predicted `ODESolution`) and prints elapsed time, process CPU time, and allocations. Multi-point filters propagate with [`EnsemblePropagation`](@ref). The default `ensemblealg` is [`EnsembleSerial`](@ref).

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=demo demo/twobody/demo_ekf.jl
julia --project=demo demo/twobody/demo_ukf.jl
julia --project=demo demo/twobody/demo_cut.jl
julia --project=demo demo/twobody/demo_srekf.jl
julia --project=demo demo/twobody/demo_srukf.jl
julia --project=demo demo/twobody/demo_srcut.jl
julia --project=demo demo/twobody/demo_srckf.jl
julia --project=demo demo/twobody/demo_particle.jl
julia --project=demo demo/lorenz/demo_ekf.jl
julia --project=demo demo/lorenz/demo_ukf.jl
julia --project=demo demo/lorenz/demo_cut.jl
julia --project=demo demo/lorenz/demo_srekf.jl
julia --project=demo demo/lorenz/demo_srukf.jl
julia --project=demo demo/lorenz/demo_srcut.jl
julia --project=demo demo/lorenz/demo_srckf.jl
julia --project=demo demo/lorenz/demo_particle.jl
```

`demo/lorenz/lorenz63.jl` is the Lorenz-63 system (σ = 10, ρ = 28, β = 8/3), not a Lorentz force. Each Lorenz script uses that shared `NonlinearODEModel`, a step of 0.02, 2000 steps, additive process noise `Q = 10⁻³ I`, and measurements of `x` and `z`. The initial state `[1, 1, 1]` is off the attractor.

On the two-body timing, both members of a pair propagate the same dynamics, so that comparison includes the ODE. A second linear kernel (6 states, 2 measurements, 5000 steps, no ODE) isolates the predict and update cost. The printed `square-root / covariance` ratios are that cost. Process CPU time sums every thread, so it can exceed elapsed time when a covariance product uses more than one BLAS thread. The square-root return on these dense, low-dimensional steps is the factor: stored `S` stays positive definite. The ratios show what the QR and downdate cost relative to forming `P` directly.
