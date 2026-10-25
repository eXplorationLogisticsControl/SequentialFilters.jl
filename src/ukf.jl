"""
    UnscentedKalmanFilter(t, x, P; alpha=1e-3, beta=2.0, kappa=0.0)

Scaled unscented Kalman filter (Wan and Van der Merwe) with additive process
and measurement noise.

The state `x` has dimension `n`. Sigma points use

```
λ = α² (n + κ) - n
```

with center and outer weights

```
Wm[1] = λ / (n + λ)
Wc[1] = λ / (n + λ) + (1 - α² + β)
Wm[i] = Wc[i] = 1 / (2 (n + λ))    for i = 2, …, 2n+1
```

`alpha` (`α`) sets the spread of the sigma points (a small positive value,
typically `1e-4 ≤ α ≤ 1`). `beta` (`β`) is the prior kurtosis parameter (`2`
for a Gaussian). `kappa` (`κ`) is the secondary scaling parameter (`0`, or
`3 - n`, are common). Defaults are `alpha=1e-3`, `beta=2`, `kappa=0`.
Small `alpha` keeps the points close to the mean and approaches a derivative
expansion; for models that are strongly nonlinear across the covariance, use
a larger `alpha` (up to about 1).

`f` and `h` passed to `predict!` and `update!` are noise-free.
`Q` and `R` are added after the unscented transform. No Jacobians are required.
`n + κ` must be positive so that `n + λ = α² (n + κ)` is positive.

The constructor copies `x` and `P`. Fields `t`, `x`, and `P` are the current
time, mean, and covariance.
"""
mutable struct UnscentedKalmanFilter
    t::Float64
    x::Vector{Float64}
    P::Matrix{Float64}
    nx::Int
    alpha::Float64
    beta::Float64
    kappa::Float64

    function UnscentedKalmanFilter(
        t::Real,
        x::AbstractVector{<:Real},
        P::AbstractMatrix{<:Real};
        alpha::Real = 1e-3,
        beta::Real = 2.0,
        kappa::Real = 0.0,
    )
        nx = length(x)
        nx >= 1 || throw(ArgumentError("state dimension must be positive, got $nx"))
        if size(P) != (nx, nx)
            throw(DimensionMismatch("P is $(size(P)); expected ($nx, $nx)"))
        end
        alpha64 = Float64(alpha)
        beta64 = Float64(beta)
        kappa64 = Float64(kappa)
        _check_ukf_parameters(nx, alpha64, beta64, kappa64)
        return new(
            Float64(t),
            Vector{Float64}(x),
            Matrix{Float64}(P),
            nx,
            alpha64,
            beta64,
            kappa64,
        )
    end
end


"""
    unscented_weights(n; alpha=1e-3, beta=2.0, kappa=0.0)

Mean and covariance weights for an `n`-dimensional scaled unscented transform.

Returns a named tuple `(; lambda, Wm, Wc)`. `Wm` and `Wc` have length `2n + 1`.
Index `1` is the center sigma point. Mean weights sum to `1`. Covariance
weights match the mean weights except

```
Wc[1] = Wm[1] + (1 - alpha^2 + beta)
```
"""
function unscented_weights(
    n::Integer;
    alpha::Real = 1e-3,
    beta::Real = 2.0,
    kappa::Real = 0.0,
)
    n = Int(n)
    n >= 1 || throw(ArgumentError("state dimension must be positive, got $n"))
    alpha64 = Float64(alpha)
    beta64 = Float64(beta)
    kappa64 = Float64(kappa)
    c = _check_ukf_parameters(n, alpha64, beta64, kappa64)
    lambda = c - n
    Wm = fill(0.5 / c, 2 * n + 1)
    Wc = copy(Wm)
    Wm[1] = lambda / c
    Wc[1] = Wm[1] + (1 - alpha64^2 + beta64)
    return (; lambda, Wm, Wc)
end


"""
    get_sigma_points(x, P; alpha=1e-3, beta=2.0, kappa=0.0)
    get_sigma_points(filter::UnscentedKalmanFilter)

Matrix of scaled sigma points with size `(n, 2n + 1)`.

Column `1` is the mean. Columns `2:n+1` are `x + √(n+λ) L[:, i]` and columns
`n+2:2n+1` are `x - √(n+λ) L[:, i]`, where `L` is the lower Cholesky factor of
`P` (`P = L Lᵀ`) and `λ` is the scaled unscented parameter. `beta` does not
move the points; it only enters the covariance weights.
"""
function get_sigma_points(
    x::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real};
    alpha::Real = 1e-3,
    beta::Real = 2.0,
    kappa::Real = 0.0,
)
    n = length(x)
    if size(P) != (n, n)
        throw(DimensionMismatch("P is $(size(P)); expected ($n, $n) to match the state"))
    end
    weights = unscented_weights(n; alpha = alpha, beta = beta, kappa = kappa)
    c = n + weights.lambda
    L = _lower_cholesky(P)
    scale = sqrt(c)
    X = Matrix{Float64}(undef, n, 2 * n + 1)
    @inbounds for j in 1:n
        X[j, 1] = x[j]
    end
    for i in 1:n
        for j in 1:n
            step = scale * L[j, i]
            X[j, i + 1] = x[j] + step
            X[j, n + i + 1] = x[j] - step
        end
    end
    return X
end

function get_sigma_points(filter::UnscentedKalmanFilter)
    return get_sigma_points(
        filter.x,
        filter.P;
        alpha = filter.alpha,
        beta = filter.beta,
        kappa = filter.kappa,
    )
end


"""
    predict!(filter::UnscentedKalmanFilter, f, Q, Δt)

Discrete-time prediction with additive process noise.

`f(t, x, Δt)` is the noise-free process model and must return a vector of
length `filter.nx`. Each current sigma point is passed through `f`. The
predicted mean and covariance are the weighted unscented moments, and `Q` is
added to the covariance. `filter.t` advances by `Δt`.

`Q` is an `nx × nx` covariance (`Matrix` or `Diagonal`). Derivatives of `f`
are not used.
"""
function predict!(filter::UnscentedKalmanFilter, f, Q, Δt::Real)
    nx = filter.nx
    Qm = _square_matrix(Q, nx, "Q")
    Δt64 = Float64(Δt)
    isfinite(Δt64) || throw(ArgumentError("Δt must be finite, got $Δt"))
    t = filter.t
    weights = _filter_weights(filter)
    X = get_sigma_points(filter)
    Xprop = Matrix{Float64}(undef, nx, size(X, 2))
    for i in axes(X, 2)
        xi = f(t, X[:, i], Δt64)
        length(xi) == nx || throw(
            DimensionMismatch("f returned length $(length(xi)); expected $nx"),
        )
        for j in 1:nx
            Xprop[j, i] = xi[j]
        end
    end
    μ = _weighted_column_mean(Xprop, weights.Wm)
    P = _deviation_covariance(Xprop, μ, weights.Wc) + Qm
    filter.x .= μ
    filter.P .= _symmetrize(P)
    filter.t = t + Δt64
    return nothing
end


"""
    update!(filter::UnscentedKalmanFilter, h, y, R)

Measurement update with additive measurement noise.

`h(t, x)` is the noise-free measurement model. `y` is the measurement (a
vector, or a scalar when the measurement dimension is 1) and `R` is its
covariance. Sigma points are redrawn from the current mean and covariance,
passed through `h`, and combined with the standard outer-product moments.
The Kalman gain uses those moments. No Jacobian is required.

Posterior covariance is the Joseph form
`P - Pxy Kᵀ - K Pxyᵀ + K S Kᵀ`, then symmetrized, where `S` is the
innovation covariance and `Pxy` is the state–measurement cross-covariance.
"""
function update!(filter::UnscentedKalmanFilter, h, y, R)
    yv = _measurement_vector(y)
    ny = length(yv)
    Rm = _square_matrix(R, ny, "R")
    t = filter.t
    weights = _filter_weights(filter)
    X = get_sigma_points(filter)
    Y = Matrix{Float64}(undef, ny, size(X, 2))
    for i in axes(X, 2)
        yi = h(t, X[:, i])
        length(yi) == ny || throw(
            DimensionMismatch("h returned length $(length(yi)); expected $ny"),
        )
        for j in 1:ny
            Y[j, i] = yi[j]
        end
    end
    yhat = _weighted_column_mean(Y, weights.Wm)
    S = _symmetrize(_deviation_covariance(Y, yhat, weights.Wc) + Rm)
    Pxy = _cross_covariance(X, filter.x, Y, yhat, weights.Wc)
    K = Pxy / S
    x_new = filter.x .+ (K * (yv .- yhat))
    P_new = filter.P - Pxy * K' - K * Pxy' + K * S * K'
    filter.x .= x_new
    filter.P .= _symmetrize(P_new)
    return nothing
end


function _filter_weights(filter::UnscentedKalmanFilter)
    return unscented_weights(
        filter.nx;
        alpha = filter.alpha,
        beta = filter.beta,
        kappa = filter.kappa,
    )
end

function _check_ukf_parameters(n::Int, alpha::Float64, beta::Float64, kappa::Float64)
    if !(isfinite(alpha) && alpha > 0)
        throw(ArgumentError("alpha must be finite and positive, got $alpha"))
    end
    isfinite(beta) || throw(ArgumentError("beta must be finite, got $beta"))
    isfinite(kappa) || throw(ArgumentError("kappa must be finite, got $kappa"))
    n_plus_kappa = n + kappa
    if !(n_plus_kappa > 0)
        throw(ArgumentError("n + kappa must be positive, got $n_plus_kappa"))
    end
    c = alpha^2 * n_plus_kappa
    c > 0 || throw(ArgumentError("n + lambda must be positive, got $c"))
    return c
end

function _square_matrix(A, n::Int, name::AbstractString)
    if size(A) != (n, n)
        throw(DimensionMismatch("$name is $(size(A)); expected ($n, $n)"))
    end
    M = Matrix{Float64}(A)
    all(isfinite, M) || throw(ArgumentError("$name must contain only finite values"))
    return M
end

function _measurement_vector(y)
    if y isa Real
        isfinite(Float64(y)) || throw(ArgumentError("measurement must be finite, got $y"))
        return Float64[y]
    end
    yv = collect(Float64, y)
    length(yv) >= 1 || throw(ArgumentError("measurement must be non-empty"))
    all(isfinite, yv) || throw(ArgumentError("measurement must contain only finite values"))
    return yv
end

function _lower_cholesky(P::AbstractMatrix)
    S = _symmetrize(P)
    factorization = cholesky(Hermitian(S); check = false)
    if !issuccess(factorization)
        throw(
            ArgumentError(
                "covariance is not positive definite; sigma points need a positive definite covariance",
            ),
        )
    end
    return factorization.L
end

function _symmetrize(A::AbstractMatrix)
    return (A + A') ./ 2
end

function _weighted_column_mean(X::AbstractMatrix, w::AbstractVector)
    n, m = size(X)
    μ = zeros(Float64, n)
    for i in 1:m
        wi = w[i]
        for j in 1:n
            μ[j] += wi * X[j, i]
        end
    end
    return μ
end

"""Covariance `Σ wᵢ (columnᵢ - μ) (columnᵢ - μ)ᵀ`."""
function _deviation_covariance(X::AbstractMatrix, μ::AbstractVector, w::AbstractVector)
    n, m = size(X)
    P = zeros(Float64, n, n)
    for i in 1:m
        d = X[:, i] .- μ
        P .+= w[i] .* (d * d')
    end
    return P
end

"""Cross-covariance `Σ wᵢ (xᵢ - x) (yᵢ - y)ᵀ`."""
function _cross_covariance(
    X::AbstractMatrix,
    x::AbstractVector,
    Y::AbstractMatrix,
    y::AbstractVector,
    w::AbstractVector,
)
    nx, m = size(X)
    ny = size(Y, 1)
    Pxy = zeros(Float64, nx, ny)
    for i in 1:m
        dx = X[:, i] .- x
        dy = Y[:, i] .- y
        Pxy .+= w[i] .* (dx * dy')
    end
    return Pxy
end
