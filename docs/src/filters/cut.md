# [Conjugate unscented filter (CUT)](@id cut-tutorial)

`ConjugateUnscentedKalmanFilter` is the conjugate unscented Kalman filter of Adurthi, Singla, and Singh, with additive process and measurement noise. It stores the mean `x` and the covariance `P`. No Jacobians are required.

Sigma points and weights come from [`cut_points`](@ref). The default `order` is `4`. CUT4 is defined for every state dimension `n ≥ 1`. CUT6 is defined for `2 ≤ n ≤ 9`. Order 8 is not implemented. Each point carries one weight, and that same weight is used for the mean and the covariance. On the supported dimensions the weights are nonnegative. The two-body demo is `demo/demo_cut.jl` and uses order 4.

## Motivation

The scaled [unscented transform](@ref ukf-tutorial) uses ``2n+1`` points and matches a Gaussian through degree 3. When ``f`` or ``h`` has significant higher-order content, the leftover moments bias the predicted mean and covariance. A tensor-product Gauss–Hermite grid would fix that, but the node count is exponential in ``n``. The conjugate unscented transform adds points on conjugate axes—every coordinate ``\pm r``, or every pair, or every triple—so the cubature is exact for standard-normal moments through total degree 5 (CUT4) or 7 (CUT6), with far fewer nodes than that grid.

Use CUT when the UKF's ``2n+1`` points are not enough and a particle filter is more than the problem needs. The cost is still polynomial in ``n`` (``2n + 2^n`` points for CUT4 when ``n \ge 3``) but grows faster than the UKF. CUT6 is tabulated only for ``2 \le n \le 9``. This package stores ``P`` itself and uses one cubature weight per point for both the mean and the covariance.

**Further reading.**

- Adurthi, N., Singla, P., and Singh, T. (2013). [The Conjugate Unscented Transform — An Approach to Evaluate Finite-Dimensional Integrals](https://ieeexplore.ieee.org/document/6314970). *Proc. American Control Conference*, 5584–5589.
- Adurthi, N., Singla, P., and Singh, T. (2018). [Conjugate Unscented Transformation: Applications to Estimation and Control](https://doi.org/10.1115/1.4037783). *ASME Journal of Dynamic Systems, Measurement, and Control*, 140(3), 030907. The radii and weights implemented here are the CUT4-G and CUT6-G tables in that paper.
- Zanetti, R., and DeMars, K. J. (2013). [Joseph Formulation of Unscented and Quadrature Filters with Application to Consider States](https://doi.org/10.2514/1.59935). *Journal of Guidance, Control, and Dynamics*, 36(6), 1860–1863.

## Prediction

Let ``(\xi^{(i)}, w_i)`` be the standard-normal CUT points, and let ``L`` be the lower Cholesky factor ``P_{k-1:k-1} = LL^\top``. The filter points are

```math
\chi_{k-1:k-1}^{(i)} = x_{k-1:k-1} + L \xi^{(i)}.
```

Pass each point through the noise-free model ``f(t, x, \Delta t)``:

```math
x_{k:k-1} = \sum_i w_i\, f\bigl(t_{k-1}, \chi_{k-1:k-1}^{(i)}, \Delta t\bigr),
```

```math
P_{k:k-1} = \sum_i w_i
\big(f\bigl(t_{k-1}, \chi_{k-1:k-1}^{(i)}, \Delta t\bigr) - x_{k:k-1}\big)
\big(f\bigl(t_{k-1}, \chi_{k-1:k-1}^{(i)}, \Delta t\bigr) - x_{k:k-1}\big)^\top + Q.
```

``P_{k:k-1}`` is symmetrized, and time advances by ``\Delta t``. A zero weight adds nothing. For CUT4 with `n ≥ 3` the center weight is zero, and that point is omitted.

## Update

Sigma points are redrawn from the prior ``\bigl(x_{k:k-1}, P_{k:k-1}\bigr)`` and passed through ``h(t, x)``:

```math
\hat y_{k:k-1} = \sum_i w_i\, h\bigl(t_k, \chi_{k:k-1}^{(i)}\bigr),
```

```math
S_{y,k:k-1} = \sum_i w_i
\big(h\bigl(t_k, \chi_{k:k-1}^{(i)}\bigr) - \hat y_{k:k-1}\big)
\big(h\bigl(t_k, \chi_{k:k-1}^{(i)}\bigr) - \hat y_{k:k-1}\big)^\top + R,
```

```math
P_{xy,k:k-1} = \sum_i w_i
\big(\chi_{k:k-1}^{(i)} - x_{k:k-1}\big)
\big(h\bigl(t_k, \chi_{k:k-1}^{(i)}\bigr) - \hat y_{k:k-1}\big)^\top,
```

```math
K = P_{xy,k:k-1} S_{y,k:k-1}^{-1}, \qquad
x_{k:k} = x_{k:k-1} + K\bigl(y_{k} - \hat y_{k:k-1}\bigr),
```

```math
P_{k:k}
  = P_{k:k-1}
  - P_{xy,k:k-1} K^\top
  - K P_{xy,k:k-1}^\top
  + K S_{y,k:k-1} K^\top.
```

``S_{y,k:k-1}`` and ``P_{k:k}`` are symmetrized. Process and measurement noise are added after the cubature. The posterior is the Joseph formula of Zanetti and DeMars rather than ``P_{k:k-1} - K S_{y,k:k-1} K^\top``. As on the [UKF](@ref ukf-tutorial) page, that form stays symmetric and is less sensitive to roundoff in ``K``. It remains valid when the gain is not the optimal ``P_{xy} S_y^{-1}``. For the optimal gain the two formulas agree in exact arithmetic. No measurement Jacobian is required: ``P_{xy}`` takes the place of ``P H^\top``. The cubature only supplies the moments.

## Two-body demo

`demo/demo_cut.jl` uses the same 7000 km orbit, initial mean and covariance, range and range-rate measurements, and noise as the EKF demo. Process noise is an [`UnbiasedRandomProcess`](@ref) with scale `1e-6`. The filter order is 4. The figure is the estimation error (estimate minus truth) with bands ``\pm 3\sqrt{P_{ii}}``, written to `demo/output/twobody_cut_errors.png`.

```julia
using SequentialFilters

order = 4
filt = ConjugateUnscentedKalmanFilter(0.0, x_hat0, P0; order = order)
Qmodel = UnbiasedRandomProcess(1.0e-6)
for k in 1:n_meas
    predict!(filt, propagate, get_noise(Qmodel, Δt), Δt)
    update!(filt, range_rangerate, ys[k], R)
end
```

`demo/demo_srcut.jl` prints elapsed time, process CPU time, and allocations for this filter and the square-root CUT. Both two-body runs propagate the same CUT4 points. The linear kernel is 5000 steps of a 6-state model with ``f(x) = Fx`` and no ODE.

```bash
julia --project=demo demo/demo_cut.jl
```

## Same calls on a linear model

```@example cut-step
using LinearAlgebra
using SequentialFilters

F = [1.0 0.5; 0.0 1.0]
H = [1.0 0.0]
Q = [1.0e-3 0.0; 0.0 1.0e-3]
R = [0.05;;]
filt = ConjugateUnscentedKalmanFilter(0.0, [0.0, 1.0], Matrix{Float64}(I, 2, 2); order = 4)
predict!(filt, (t, x, Δt) -> F * x, Q, 1.0)
update!(filt, (t, x) -> H * x, [0.2], R)
filt.x
```
