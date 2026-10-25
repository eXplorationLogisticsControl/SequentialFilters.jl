"""
    SquareRootUnscentedKalmanFilter(t, x, P; alpha=1e-3, beta=2.0, kappa=0.0)

Van der Merwe and Wan square-root unscented Kalman filter with additive noise.

The covariance is a lower Cholesky factor `S` with `P = S * S'` and a positive
diagonal. Sigma points are the columns of `S` scaled by `√(n + λ)`. Weights
come from `unscented_weights` (the same scaled convention as
`UnscentedKalmanFilter`): defaults `alpha=1e-3`, `beta=2`, `kappa=0`.

The time update takes a thin QR of the weighted propagated sigma points that
have positive covariance weight, together with a process-noise factor, then a
Cholesky rank-one downdate when the center weight is negative. The measurement
update does the same for the predicted measurement, forms the gain from that
factor, and applies a Cholesky rank-one downdate for each column of `K * Sy`.
No Jacobians are required.
"""
mutable struct SquareRootUnscentedKalmanFilter
    t::Float64
    x::Vector{Float64}
    S::Matrix{Float64}
    nx::Int
    alpha::Float64
    beta::Float64
    kappa::Float64

    function SquareRootUnscentedKalmanFilter(
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
        unscented_weights(nx; alpha = alpha64, beta = beta64, kappa = kappa64)
        S = Matrix(LowerTriangular(_lower_cholesky(P)))
        return new(Float64(t), Vector{Float64}(x), S, nx, alpha64, beta64, kappa64)
    end
end


"""
    covariance(filter::SquareRootUnscentedKalmanFilter)

Reconstructed covariance `P = S * S'`, symmetrized.
"""
function covariance(filter::SquareRootUnscentedKalmanFilter)
    P = filter.S * filter.S'
    return (P + P') ./ 2
end


"""
    predict!(filter::SquareRootUnscentedKalmanFilter, f, Q, Δt)

`f(t, x, Δt)` is the noise-free process model. `Q` is the additive
process-noise covariance. `filter.t` advances by `Δt`.
"""
function predict!(filter::SquareRootUnscentedKalmanFilter, f, Q, Δt::Real)
    nx = filter.nx
    Qm = _square_matrix(Q, nx, "Q")
    Δt64 = Float64(Δt)
    isfinite(Δt64) || throw(ArgumentError("Δt must be finite, got $Δt"))
    t = filter.t
    weights = _srukf_weights(filter)
    X = _sigma_points_from_factor(filter.x, filter.S, sqrt(nx + weights.lambda))
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
    Snew = _sr_factor(_column_deviations(Xprop, μ), weights.Wc, _noise_factor(Qm))
    filter.x .= μ
    filter.S .= Snew
    filter.t = t + Δt64
    return nothing
end


"""
    update!(filter::SquareRootUnscentedKalmanFilter, h, y, R)

`h(t, x)` is the noise-free measurement model. `y` is the measurement and `R`
its additive covariance.
"""
function update!(filter::SquareRootUnscentedKalmanFilter, h, y, R)
    yv = _measurement_vector(y)
    ny = length(yv)
    Rm = _square_matrix(R, ny, "R")
    nx = filter.nx
    weights = _srukf_weights(filter)
    X = _sigma_points_from_factor(filter.x, filter.S, sqrt(nx + weights.lambda))
    Y = Matrix{Float64}(undef, ny, size(X, 2))
    for i in axes(X, 2)
        yi = h(filter.t, X[:, i])
        length(yi) == ny || throw(
            DimensionMismatch("h returned length $(length(yi)); expected $ny"),
        )
        for j in 1:ny
            Y[j, i] = yi[j]
        end
    end
    yhat = _weighted_column_mean(Y, weights.Wm)
    Sy = _sr_factor(_column_deviations(Y, yhat), weights.Wc, _noise_factor(Rm))
    Pxy = _cross_covariance(X, filter.x, Y, yhat, weights.Wc)
    Sl = LowerTriangular(Sy)
    K = (Pxy / Sl') / Sl
    xnew = filter.x .+ K * (yv .- yhat)
    U = K * Sy
    Snew = copy(filter.S)
    for j in 1:ny
        _cholesky_rank1!(Snew, U[:, j], -1)
    end
    filter.x .= xnew
    filter.S .= Snew
    return nothing
end


function _srukf_weights(filter::SquareRootUnscentedKalmanFilter)
    return unscented_weights(
        filter.nx;
        alpha = filter.alpha,
        beta = filter.beta,
        kappa = filter.kappa,
    )
end

function _sigma_points_from_factor(x, S, scale)
    n = length(x)
    X = Matrix{Float64}(undef, n, 2 * n + 1)
    for j in 1:n
        X[j, 1] = x[j]
    end
    for i in 1:n
        for j in 1:n
            step = scale * S[j, i]
            X[j, i + 1] = x[j] + step
            X[j, n + i + 1] = x[j] - step
        end
    end
    return X
end

function _column_deviations(X, μ)
    n, m = size(X)
    D = Matrix{Float64}(undef, n, m)
    for i in 1:m
        for j in 1:n
            D[j, i] = X[j, i] - μ[j]
        end
    end
    return D
end


"""
Square-root covariance from sigma-point deviations and an additive noise factor.

Columns `2:end` of `dev` use positive weights and enter the QR. A positive
center weight is included in that QR. A negative center weight is applied
afterwards as a Cholesky rank-one downdate.
"""
function _sr_factor(dev::AbstractMatrix, w::AbstractVector, noise::AbstractMatrix)
    n, m = size(dev)
    if length(w) != m
        throw(DimensionMismatch("expected $m weights, got $(length(w))"))
    end
    pieces = Matrix{Float64}[]
    for i in 2:m
        w[i] > 0 || throw(ArgumentError("outer unscented covariance weights must be positive"))
        push!(pieces, reshape(sqrt(w[i]) .* dev[:, i], n, 1))
    end
    if w[1] > 0
        push!(pieces, reshape(sqrt(w[1]) .* dev[:, 1], n, 1))
    end
    if size(noise, 2) > 0
        push!(pieces, Matrix{Float64}(noise))
    end
    isempty(pieces) && throw(ArgumentError("square-root covariance factor has no columns"))
    S = _qr_lower_factor(reduce(hcat, pieces)')
    if w[1] < 0
        _cholesky_rank1!(S, sqrt(-w[1]) .* dev[:, 1], -1)
    end
    return S
end


"""
In-place rank-one Cholesky update of a lower factor.

On exit, `L * L' ` equals the input `L * L' + sign * x * x'`, where `sign` is
`+1` or `-1`. `L` stays lower triangular with a positive diagonal.
"""
function _cholesky_rank1!(L::AbstractMatrix, x::AbstractVector, sign::Int)
    n = size(L, 1)
    length(x) == n || throw(DimensionMismatch("rank-one vector length $(length(x)); expected $n"))
    sign == 1 || sign == -1 || throw(ArgumentError("sign must be +1 or -1, got $sign"))
    w = collect(Float64, x)
    for k in 1:n
        Lkk = L[k, k]
        Lkk > 0 || throw(ArgumentError("Cholesky factor diagonal must be positive"))
        wkk = w[k]
        r2 = Lkk * Lkk + sign * wkk * wkk
        if r2 <= 0
            throw(ArgumentError("Cholesky downdate lost positive definiteness"))
        end
        r = sqrt(r2)
        c = r / Lkk
        s = wkk / Lkk
        L[k, k] = r
        for j in (k + 1):n
            Ljk = L[j, k]
            L[j, k] = (Ljk + sign * s * w[j]) / c
            w[j] = c * w[j] - s * L[j, k]
        end
    end
    return L
end
