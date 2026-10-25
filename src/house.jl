"""
    HigherOrderUnscentedEstimator(t, x, P; skewness=zeros(n), kurtosis=fill(3, n), delta=0)

Higher-order unscented estimator (HOUSE) of Stojanovski and Savransky, with
additive process and measurement noise.

The state `x` has dimension `n`. The filter stores the mean `x`, the
covariance `P`, and the marginal skewness and kurtosis of the standardized
state `P^{-1/2} (x - mean)`. `skewness` and `kurtosis` each have length `n`.
The defaults `skewness = 0` and `kurtosis = 3` are the standard-normal
moments. `delta` is the minimum center weight used to place sigma points,
with `0 ≤ delta < 1`. The reference examples use `delta = 0`.

Sigma points are the asymmetric set of [`house_sigma_points`](@ref). Process
noise and measurement noise use the same set on their own covariance, so a
predict draws `2(n + n_w) + 1` points and an update draws `2(n + n_y) + 1`.
For additive noise the noise points sit on the current mean: `f` and `h` are
evaluated on the `2n + 1` state points, and the noise offsets are added to
the image of the mean. A zero covariance contributes no noise points. Any
other noise covariance must be positive definite.

`predict!` propagates the state points through a [`DynamicsModel`](@ref) with
[`EnsemblePropagation`](@ref). `update!` takes a noise-free `h`. The
predictive covariance already includes `Q`, and the innovation covariance
already includes `R`. No Jacobians are required.

The kurtosis floor that keeps the center weight at least `delta` is applied
only while building points. It is not written back. `predict!` and `update!`
replace `skewness` and `kurtosis` with the marginal moments of the propagated
cloud and of the linear-minimum-mean-square-error points.

The constructor copies `x`, `P`, `skewness`, and `kurtosis`. Fields `t`, `x`,
and `P` are the current time, mean, and covariance. The element type is the
promotion of `eltype(x)` and `eltype(P)` (`Float64` when both are integers).
"""
mutable struct HigherOrderUnscentedEstimator{T<:AbstractFloat}
    t::T
    x::Vector{T}
    P::Matrix{T}
    skewness::Vector{T}
    kurtosis::Vector{T}
    nx::Int
    delta::T

    function HigherOrderUnscentedEstimator(
        t::Real,
        x::AbstractVector{<:Real},
        P::AbstractMatrix{<:Real};
        skewness::AbstractVector{<:Real} = zeros(length(x)),
        kurtosis::AbstractVector{<:Real} = fill(3.0, length(x)),
        delta::Real = 0.0,
    )
        nx = length(x)
        nx >= 1 || throw(ArgumentError("state dimension must be positive, got $nx"))
        if size(P) != (nx, nx)
            throw(DimensionMismatch("P is $(size(P)); expected ($nx, $nx)"))
        end
        _check_house_delta(delta)
        T = _filter_eltype(x, P)
        γ = _house_moment_vector(skewness, nx, T, "skewness")
        κ = _house_moment_vector(kurtosis, nx, T, "kurtosis")
        return new{T}(
            T(t),
            Vector{T}(x),
            Matrix{T}(P),
            γ,
            κ,
            nx,
            T(delta),
        )
    end
end


"""
    house_sigma_points(x, P, skewness, kurtosis; delta=0)

Asymmetric sigma points for one distribution with mean `x` and covariance `P`.

Returns `(X, w)`. `X` has size `(n, 2n + 1)` and `w` has length `2n + 1`.
Column `1` is the mean. Columns `2:n+1` are `x + αᵢ L[:, i]` and columns
`n+2:2n+1` are `x - βᵢ L[:, i]`, where `L` is the lower Cholesky factor of
`P` (`P = L Lᵀ`). The scales and weights match the marginal skewness `γ` and
kurtosis `κ` of the standardized state:

```
cᵢ = √(4κᵢ − 3γᵢ²)
αᵢ = (γᵢ + cᵢ) / 2
βᵢ = (−γᵢ + cᵢ) / 2
w⁺ᵢ = 1 / (αᵢ cᵢ)
w⁻ᵢ = 1 / (βᵢ cᵢ)
```

The center weight is one minus the sum of the other weights. Before those
scales are formed, any direction with `κᵢ − γᵢ²` below `n / (1 − delta)` is
raised to that floor so the center weight is at least `delta` when every
direction sits on the floor. The stored moments of a filter are not changed
by this call. `delta` must lie in `[0, 1)`.

This is the state block of the augmented rule. [`predict!`](@ref) and
[`update!`](@ref) add an independent noise block of the same form.
"""
function house_sigma_points(
    x::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real},
    skewness::AbstractVector{<:Real},
    kurtosis::AbstractVector{<:Real};
    delta::Real = 0.0,
)
    n = length(x)
    n >= 1 || throw(ArgumentError("state dimension must be positive, got $n"))
    if size(P) != (n, n)
        throw(DimensionMismatch("P is $(size(P)); expected ($n, $n) to match the state"))
    end
    _check_house_delta(delta)
    T = _filter_eltype(x, P)
    γ = _house_moment_vector(skewness, n, T, "skewness")
    κ = _house_moment_vector(kurtosis, n, T, "kurtosis")
    X, _, w = _house_sigma_cloud(
        Vector{T}(x),
        Matrix{T}(P),
        γ,
        κ,
        nothing,
        T(delta),
    )
    return X, w
end


"""
    predict!(filter::HigherOrderUnscentedEstimator, dynamics_model::DynamicsModel, process_noise_model::ProcessNoiseModel, Δt::Float64; noise_skewness=0, noise_kurtosis=3, save_everystep=false, saveat=(), ensemblealg=EnsembleSerial())

Propagate the HOUSE state points with [`EnsemblePropagation`](@ref).

`Q = get_noise(process_noise_model, Δt)` is the covariance of an independent
zero-mean increment. `noise_skewness` and `noise_kurtosis` are that
increment's marginal moments: a `Real` is used for every component, and a
vector must have length `n`. The defaults are the Gaussian values `0` and
`3`. A zero `Q` adds no noise points. Any other `Q` must be positive definite.

The returned `ODESolution` stores `[μ; vec(Φ)]`. `Φ P_start Φᵀ` is the
covariance of the noise-free cloud: noise columns are collapsed onto the
propagated center, and process noise is not included. `filter.P` is the
covariance of the full cloud, including the noise offsets. `filter.skewness`
and `filter.kurtosis` are the marginal moments of that full cloud.
`filter.t` advances by `Δt`.

The default `ensemblealg` is [`EnsembleSerial`](@ref). Pass
[`EnsembleThreads`](@ref), [`EnsembleDistributed`](@ref), or
[`EnsembleSplitThreads`](@ref) to run the state points in parallel.
`save_everystep` and `saveat` match [`ExtendedKalmanFilter`](@ref).
"""
function predict!(
    filter::HigherOrderUnscentedEstimator,
    dynamics_model::DynamicsModel,
    process_noise_model::ProcessNoiseModel,
    Δt::Float64;
    noise_skewness::Union{Real, AbstractVector{<:Real}} = 0,
    noise_kurtosis::Union{Real, AbstractVector{<:Real}} = 3,
    save_everystep::Bool = false,
    saveat = (),
    ensemblealg = EnsembleSerial(),
)
    _check_step(Δt)
    T = eltype(filter.x)
    nx = filter.nx
    Qm = _as_eltype(_square_matrix(get_noise(process_noise_model, Δt), nx, "Q"), T)
    noise = _house_noise(Qm, noise_skewness, noise_kurtosis)
    X, W, w = _house_sigma_cloud(
        filter.x,
        filter.P,
        filter.skewness,
        filter.kurtosis,
        noise,
        filter.delta,
    )
    n_state = 2 * nx + 1
    P_start = Matrix{Float64}(filter.P)
    series = EnsemblePropagation(
        dynamics_model,
        X[:, 1:n_state],
        Float64(filter.t),
        Δt;
        ensemblealg = ensemblealg,
        save_everystep = save_everystep,
        saveat = saveat,
    )
    if size(series.X[end], 2) != n_state
        error("propagated $(size(series.X[end], 2)) state points; expected $n_state")
    end
    Xfull = _attach_noise(series.X[end], W, n_state)
    μ, P, skew, kurt = _house_marginal_moments(Xfull, w)
    sol = _moment_solution(P_start, _noise_free_series(series, n_state, length(w)), w, w)
    filter.x .= μ
    filter.P .= P
    filter.skewness .= skew
    filter.kurtosis .= kurt
    filter.t = T(filter.t + Δt)
    return sol
end


"""
    update!(filter::HigherOrderUnscentedEstimator, h, y, R; noise_skewness=0, noise_kurtosis=3)

Measurement update with additive measurement noise.

`h(t, x)` is the noise-free measurement model. `y` is the measurement (a
vector, or a scalar when the measurement dimension is 1) and `R` is its
covariance. `noise_skewness` and `noise_kurtosis` are the marginal moments of
that noise. A `Real` is used for every component, and a vector must have
length equal to `y`. The defaults are the Gaussian values `0` and `3`.

Sigma points are redrawn from the current mean, covariance, skewness, and
kurtosis. `h` is evaluated on the state points, and the measurement-noise
offsets are added to `h` at the mean. The innovation covariance is the
weighted covariance of those measurement values, so it already includes `R`.

The posterior mean is `x + K (y − ŷ)`. The posterior covariance, skewness,
and kurtosis are the moments of the linear error points

```
εⱼ = χⱼ − x − K (zⱼ − ŷ),
```

standardized with the Cholesky factor of that error covariance. The
covariance is symmetrized. For the optimal gain it agrees with the Joseph
form `P − Pxy Kᵀ − K Pxyᵀ + K S Kᵀ`. No measurement Jacobian is required.
"""
function update!(
    filter::HigherOrderUnscentedEstimator,
    h,
    y,
    R;
    noise_skewness::Union{Real, AbstractVector{<:Real}} = 0,
    noise_kurtosis::Union{Real, AbstractVector{<:Real}} = 3,
)
    T = eltype(filter.x)
    yv = _measurement_vector(y, T)
    ny = length(yv)
    Rm = _as_eltype(_square_matrix(R, ny, "R"), T)
    noise = _house_noise(Rm, noise_skewness, noise_kurtosis)
    x_prior = filter.x
    X, W, w = _house_sigma_cloud(
        x_prior,
        filter.P,
        filter.skewness,
        filter.kurtosis,
        noise,
        filter.delta,
    )
    Y = _house_measurements(h, filter.t, X, W, filter.nx, ny)
    yhat = _weighted_column_mean(Y, w)
    S = _symmetrize(_deviation_covariance(Y, yhat, w))
    Pxy = _cross_covariance(X, x_prior, Y, yhat, w)
    K = Pxy / S
    Xu = Matrix{T}(undef, filter.nx, length(w))
    for j in axes(X, 2)
        innov = Y[:, j] .- yhat
        correction = K * innov
        for i in 1:filter.nx
            Xu[i, j] = X[i, j] - correction[i]
        end
    end
    _, P, skew, kurt = _house_marginal_moments(Xu, w)
    x_new = x_prior .+ (K * (yv .- yhat))
    filter.x .= x_new
    filter.P .= P
    filter.skewness .= skew
    filter.kurtosis .= kurt
    return nothing
end


function _check_house_delta(delta::Real)
    if !(isfinite(delta) && 0 <= delta < 1)
        throw(ArgumentError("delta must lie in [0, 1), got $delta"))
    end
    return nothing
end

function _house_moment_vector(
    value::Real,
    n::Int,
    ::Type{T},
    name::AbstractString,
) where {T<:AbstractFloat}
    v = T(value)
    isfinite(v) || throw(ArgumentError("$name must be finite, got $value"))
    return fill(v, n)
end

function _house_moment_vector(
    value::AbstractVector{<:Real},
    n::Int,
    ::Type{T},
    name::AbstractString,
) where {T<:AbstractFloat}
    if length(value) != n
        throw(DimensionMismatch("$name has length $(length(value)); expected $n"))
    end
    v = Vector{T}(value)
    all(isfinite, v) || throw(ArgumentError("$name must contain only finite values"))
    return v
end

"""`nothing` when `Q` is zero. Otherwise the noise covariance and its moments."""
function _house_noise(
    Q::AbstractMatrix{T},
    skewness::Union{Real, AbstractVector{<:Real}},
    kurtosis::Union{Real, AbstractVector{<:Real}},
) where {T<:AbstractFloat}
    if all(iszero, Q)
        return nothing
    end
    nw = size(Q, 1)
    return (;
        cov = Q,
        skewness = _house_moment_vector(skewness, nw, T, "noise_skewness"),
        kurtosis = _house_moment_vector(kurtosis, nw, T, "noise_kurtosis"),
    )
end

"""
Scales `α`, `β` and outer weights for one block of marginal moments.

`kurtosis` is copied. Directions whose excess `κ − γ²` is below `mmin` are
raised to that floor. The caller's stored moments are left unchanged.
"""
function _house_scales(
    skewness::AbstractVector{T},
    kurtosis::AbstractVector{T},
    mmin::T,
) where {T<:AbstractFloat}
    n = length(skewness)
    α = Vector{T}(undef, n)
    β = Vector{T}(undef, n)
    wplus = Vector{T}(undef, n)
    wminus = Vector{T}(undef, n)
    κ = copy(kurtosis)
    for i in 1:n
        excess = κ[i] - skewness[i]^2
        if excess < mmin
            κ[i] += mmin - excess
        end
        γ = skewness[i]
        disc = 4 * κ[i] - 3 * γ^2
        c = disc > 0 ? sqrt(disc) : T(NaN)
        αi = (γ + c) / 2
        βi = (-γ + c) / 2
        if !(isfinite(αi) && isfinite(βi) && αi > 0 && βi > 0)
            throw(
                ArgumentError(
                    "skewness $(skewness[i]) and kurtosis $(kurtosis[i]) in direction $i do not yield positive sigma-point scales",
                ),
            )
        end
        α[i] = αi
        β[i] = βi
        wplus[i] = 1 / (αi * c)
        wminus[i] = 1 / (βi * c)
    end
    return α, β, wplus, wminus
end

"""
State points `X`, noise values `W`, and weights `w` for the augmented rule.

`noise` is `nothing` or `(; cov, skewness, kurtosis)` with a zero mean.
Column 1 is the center. The next `2 nx` columns move the state and hold the
noise at zero. The last `2 nw` columns hold the state at its mean and move
the noise. `W` has size `(nw, 2(nx + nw) + 1)`.
"""
function _house_sigma_cloud(
    x::AbstractVector{T},
    P::AbstractMatrix{T},
    skewness::AbstractVector{T},
    kurtosis::AbstractVector{T},
    noise,
    delta::T,
) where {T<:AbstractFloat}
    nx = length(x)
    nw = noise === nothing ? 0 : length(noise.skewness)
    mmin = T(nx + nw) / (one(T) - delta)
    α, β, wplus, wminus = _house_scales(skewness, kurtosis, mmin)
    npts = 2 * (nx + nw) + 1
    X = Matrix{T}(undef, nx, npts)
    W = zeros(T, nw, npts)
    w = zeros(T, npts)
    for j in 1:npts
        for i in 1:nx
            X[i, j] = x[i]
        end
    end
    L = _lower_cholesky(P)
    for i in 1:nx
        step = L[:, i]
        plus = i + 1
        minus = nx + i + 1
        αi = α[i]
        βi = β[i]
        for j in 1:nx
            X[j, plus] = x[j] + αi * step[j]
            X[j, minus] = x[j] - βi * step[j]
        end
        w[plus] = wplus[i]
        w[minus] = wminus[i]
    end
    if nw > 0
        αn, βn, wplusn, wminusn = _house_scales(noise.skewness, noise.kurtosis, mmin)
        Ln = _lower_cholesky(noise.cov)
        base = 2 * nx + 1
        for i in 1:nw
            step = Ln[:, i]
            plus = base + i
            minus = base + nw + i
            αi = αn[i]
            βi = βn[i]
            for j in 1:nw
                W[j, plus] = αi * step[j]
                W[j, minus] = -βi * step[j]
            end
            w[plus] = wplusn[i]
            w[minus] = wminusn[i]
        end
    end
    w[1] = one(T) - sum(@view w[2:end])
    return X, W, w
end

"""Marginal mean, covariance, skewness, and kurtosis of the columns of `X`."""
function _house_marginal_moments(X::AbstractMatrix, w::AbstractVector)
    μ = _weighted_column_mean(X, w)
    P = _symmetrize(_deviation_covariance(X, μ, w))
    L = _lower_cholesky(P)
    n, m = size(X)
    T = promote_type(eltype(μ), eltype(w), eltype(X))
    skew = zeros(T, n)
    kurt = zeros(T, n)
    for j in 1:m
        d = L \ (X[:, j] .- μ)
        wj = w[j]
        for i in 1:n
            di = d[i]
            di2 = di * di
            skew[i] += wj * di2 * di
            kurt[i] += wj * di2 * di2
        end
    end
    return μ, P, skew, kurt
end

"""Propagated state points, with noise offsets added on the noise columns."""
function _attach_noise(Xprop::AbstractMatrix, W::AbstractMatrix, n_state::Int)
    n, nprop = size(Xprop)
    npts = size(W, 2)
    T = promote_type(eltype(Xprop), eltype(W))
    Xfull = Matrix{T}(undef, n, npts)
    for j in 1:nprop
        for i in 1:n
            Xfull[i, j] = Xprop[i, j]
        end
    end
    for j in (n_state + 1):npts
        for i in 1:n
            Xfull[i, j] = Xprop[i, 1] + W[i, j]
        end
    end
    return Xfull
end

"""Same time grid, with noise columns collapsed onto the propagated center."""
function _noise_free_series(series, n_state::Int, npts::Int)
    Xs = Vector{Matrix{Float64}}(undef, length(series.X))
    for k in eachindex(series.X)
        Xprop = series.X[k]
        n = size(Xprop, 1)
        Xfull = Matrix{Float64}(undef, n, npts)
        for j in 1:n_state
            for i in 1:n
                Xfull[i, j] = Xprop[i, j]
            end
        end
        for j in (n_state + 1):npts
            for i in 1:n
                Xfull[i, j] = Xprop[i, 1]
            end
        end
        Xs[k] = Xfull
    end
    return (; t = series.t, X = Xs)
end

function _house_measurements(h, t, X::AbstractMatrix{T}, W::AbstractMatrix{T}, nx::Int, ny::Int) where {T}
    n_state = 2 * nx + 1
    npts = size(X, 2)
    Y = Matrix{T}(undef, ny, npts)
    for j in 1:n_state
        yi = h(t, X[:, j])
        length(yi) == ny || throw(
            DimensionMismatch("h returned length $(length(yi)); expected $ny"),
        )
        for i in 1:ny
            Y[i, j] = yi[i]
        end
    end
    for j in (n_state + 1):npts
        for i in 1:ny
            Y[i, j] = Y[i, 1] + W[i, j]
        end
    end
    return Y
end
