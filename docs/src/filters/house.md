# [Higher-order unscented estimator (HOUSE)](@id house-tutorial)

`HigherOrderUnscentedEstimator` is the higher-order unscented estimator of Stojanovski and Savransky, with additive process and measurement noise. It stores the mean `x`, the covariance `P`, and the marginal skewness and kurtosis of the standardized state. No Jacobians are required.

Sigma points for one distribution come from [`house_sigma_points`](@ref). Defaults are `skewness = 0`, `kurtosis = 3`, and `delta = 0`. The two-body demo is `demo/twobody/demo_house.jl`.

## Motivation

The scaled [unscented transform](@ref ukf-tutorial) matches a Gaussian mean and covariance with a symmetric set of ``2n+1`` points. Odd moments beyond the mean then vanish because the points are symmetric, and the fourth moment is whatever `beta` implies. When the state or the noise is skewed, or has heavier tails than that choice, the predicted mean of a nonlinear map is biased.

HOUSE keeps the same ``2n+1`` evaluations of ``f`` or ``h`` and moves the points off the symmetric pair. Each direction of the standardized state gets its own scales ``\alpha_i`` and ``\beta_i``, chosen so the weighted points match that direction's skewness and kurtosis. Process noise and measurement noise are independent increments with their own marginal moments. Their sigma points sit on the current mean, so they do not add further evaluations of ``f`` or ``h``.

The cubature matches marginal moments. A mixed monomial with two or more nonzero powers, such as ``\tilde x_i^2 \tilde x_j^2``, is not reproduced. The kurtosis floor below can also raise a small kurtosis so that the center weight stays nonnegative. Both are the prices of a closed-form rule with ``2n+1`` function evaluations.

Use HOUSE when the marginal skewness or kurtosis is part of the model and a particle filter is more than the problem needs. For a Gaussian state and Gaussian noise the [conjugate unscented filter](@ref cut-tutorial) matches more of the mixed moments. HOUSE is the filter that carries the marginal third and fourth moments from one step to the next.

**Further reading.**

- Stojanovski, Z., and Savransky, D. (2021). [Higher-Order Unscented Estimator](https://doi.org/10.2514/1.G006109). *Journal of Guidance, Control, and Dynamics*, 44(12), 2186–2198. The scales, the kurtosis floor, and the predict and update steps implemented here are that paper's. The reference code is [SIOSlab/HOUSE](https://github.com/SIOSlab/HOUSE).
- Ponomareva, K., Date, P., and Wang, Z. (2010). A New Unscented Kalman Filter with Higher Order Moment-Matching. *Proc. 19th International Symposium on Mathematical Theory of Networks and Systems*, 1609–1613. HOUSE gives each coordinate its own scale; that filter uses one scale for the average marginal moments.
- Zanetti, R., and DeMars, K. J. (2013). [Joseph Formulation of Unscented and Quadrature Filters with Application to Consider States](https://doi.org/10.2514/1.59935). *Journal of Guidance, Control, and Dynamics*, 36(6), 1860–1863.

## Sigma points

Let ``\gamma`` and ``\kappa`` be the marginal skewness and kurtosis of the standardized variable ``L^{-1}(x - \bar x)``, with ``P = LL^\top`` the lower Cholesky factor. For an augmented dimension ``N`` and a center-weight floor ``\delta``,

```math
\kappa_{\min} = \frac{N}{1 - \delta}.
```

Any direction with ``\kappa_i - \gamma_i^2 < \kappa_{\min}`` is raised until the excess equals ``\kappa_{\min}``. That raised value is used only to place the points. With the possibly raised ``\kappa_i``,

```math
c_i = \sqrt{4\kappa_i - 3\gamma_i^2}, \qquad
\alpha_i = \frac{\gamma_i + c_i}{2}, \qquad
\beta_i = \frac{-\gamma_i + c_i}{2},
```

```math
w_i^+ = \frac{1}{\alpha_i c_i}, \qquad
w_i^- = \frac{1}{\beta_i c_i}, \qquad
w^{(0)} = 1 - \sum_i \left(w_i^+ + w_i^-\right).
```

`delta` (``\delta``) must lie in ``[0, 1)``. The default ``\delta = 0`` makes ``\kappa_{\min} = N``. A standard normal has excess ``3``, so the floor is active whenever the augmented dimension is greater than 3. ``\alpha_i`` and ``\beta_i`` have to be positive, which the floor guarantees for finite moments.

[`house_sigma_points`](@ref) builds the state block alone, so ``N = n``. Column `1` is the mean. Columns `2:n+1` are ``x + \alpha_i L_{:,i}`` and columns `n+2:2n+1` are ``x - \beta_i L_{:,i}``.

`predict!` and `update!` append a noise block of the same form. The augmented count is ``2(n + n_w) + 1``. Noise columns hold the state at its mean. State columns hold the noise at zero. For a positive definite `Q` or `R`, ``n_w`` is the dimension of that matrix. A zero matrix adds no noise columns.

## Prediction

`Q = get_noise(process_noise_model, Δt)` is the covariance of an independent zero-mean increment. `noise_skewness` and `noise_kurtosis` are its marginal moments. A `Real` is used for every component. A vector must have length `n`. The defaults are the Gaussian values `0` and `3`.

Pass the ``2n+1`` state points through the noise-free model ``f(t, x, \Delta t)``. On a noise column the propagated state is ``f`` at the mean, plus that column's noise value. The predicted mean and covariance are the weighted moments of this cloud, and the predicted skewness and kurtosis are the marginal moments after standardizing by the new covariance. ``P_{k|k-1}`` is symmetrized. Time advances by ``\Delta t``.

`predict!` takes a [`DynamicsModel`](@ref) and a [`ProcessNoiseModel`](@ref). The state points are propagated by [`EnsemblePropagation`](@ref). The default `ensemblealg` is [`EnsembleSerial`](@ref). Pass [`EnsembleThreads`](@ref), [`EnsembleDistributed`](@ref), or [`EnsembleSplitThreads`](@ref) to change the parallelism. The returned `ODESolution` stores `[x; vec(Φ)]`. Its cloud collapses every noise column onto the propagated center, so ``\Phi P_{k-1|k-1} \Phi^\top`` is the covariance before process noise. `filter.P` includes the noise offsets.

## Update

Sigma points are redrawn from the prior, including its skewness and kurtosis. `h(t, x)` is evaluated on the state points. Measurement-noise offsets are added to `h` at the mean. `noise_skewness` and `noise_kurtosis` are the marginal moments of that noise, with the same `Real` or vector convention and the same Gaussian defaults. The innovation covariance is the weighted measurement covariance, so it already includes `R`.

```math
K = P_{xy,k|k-1} S_{y,k|k-1}^{-1}, \qquad
x_{k|k} = x_{k|k-1} + K\bigl(y_k - \hat y_{k|k-1}\bigr).
```

The posterior covariance, skewness, and kurtosis are the moments of the linear error points

```math
\varepsilon^{(j)} = \chi^{(j)}_{k|k-1} - x_{k|k-1} - K\bigl(z^{(j)} - \hat y_{k|k-1}\bigr),
```

standardized with the Cholesky factor of the error covariance. ``P_{k|k}`` is symmetrized. That covariance is the Joseph formula

```math
P_{k|k}
  = P_{k|k-1}
  - P_{xy,k|k-1} K^\top
  - K P_{xy,k|k-1}^\top
  + K S_{y,k|k-1} K^\top.
```

As on the [UKF](@ref ukf-tutorial) page, the Joseph form stays symmetric and is less sensitive to roundoff in ``K``. For this optimal gain it agrees with ``P_{k|k-1} - K S_{y,k|k-1} K^\top``. The stored mean is the measurement update above. The mean of the error points is the prior mean, and it is not written back.

## Two-body demo

`demo/twobody/demo_house.jl` uses the same 7000 km orbit, initial mean and covariance, range and range-rate measurements, and noise as the EKF demo. Process noise is an [`UnbiasedRandomProcess`](@ref) with scale `1e-6`. The state moments start at the Gaussian defaults. The figure samples the returned `ODESolution` and draws the estimation error (estimate minus truth) with bands ``\pm 3\sqrt{P_{ii}}``, written to `demo/twobody/plots/twobody_house_errors.png`.

```julia
using SequentialFilters

filt = HigherOrderUnscentedEstimator(0.0, x_hat0, P0)
dynamics = NonlinearODEModel(twobody!; n = n, p = μ)
process_noise_model = UnbiasedRandomProcess(1.0e-6)
for k in 1:n_meas
    predict!(filt, dynamics, process_noise_model, Δt)
    update!(filt, range_rangerate, ys[k], R)
end
```

The default propagation is `ensemblealg = EnsembleSerial()`. Pass `EnsembleThreads()`, `EnsembleDistributed()`, or `EnsembleSplitThreads()` to run the state points in parallel. Pass `noise_skewness` and `noise_kurtosis` on `predict!` or `update!` when the increment is not Gaussian.

```bash
julia --project=demo demo/twobody/demo_house.jl
```

## Same calls on a linear model

```@example house-step
using LinearAlgebra
using SequentialFilters

F = [1.0 0.5; 0.0 1.0]
H = [1.0 0.0]
Q = [1.0e-3 0.0; 0.0 1.0e-3]
R = [0.05;;]
filt = HigherOrderUnscentedEstimator(0.0, [0.0, 1.0], Matrix{Float64}(I, 2, 2))
predict!(filt, DiscreteDynamicsModel((t, x, Δt) -> F * x; n = 2), AdditiveProcessNoise(Q), 1.0)
update!(filt, (t, x) -> H * x, [0.2], R)
filt.x
```
