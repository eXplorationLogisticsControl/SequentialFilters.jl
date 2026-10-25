"""
    SquareRootConjugateUnscentedKalmanFilter(t, x, P; order=4)

Square-root conjugate unscented Kalman filter with additive noise.

Sigma points and weights come from `cut_points`. `order` is the CUT order
that generator already supports: `4` (CUT4, every `n ≥ 1`) or `6` (CUT6,
`2 ≤ n ≤ 9`). The default is `4`. CUT8 is not implemented.

The covariance is a lower Cholesky factor `S` with `P = S * S'` and a positive
diagonal, the same convention as `SquareRootUnscentedKalmanFilter`. The
constructor factors `P` once. Predict and update rebuild `S` from the CUT
weights with a thin QR and, when a weight is negative, a Cholesky rank-one
downdate. They do not form `P` and call `cholesky` again.

CUT uses one weight for each sigma point, and that weight enters both the mean
and the covariance. A positive weight `w` contributes the column
`√w (χ − μ)` to the QR, together with a factor of the additive noise. A zero
weight is omitted. A negative weight is not placed in the QR; after the QR,
the factor is downdated by `√(−w) (χ − μ)` with the same rank-one Cholesky
downdate used by the square-root UKF (`σ = −1`). Supported CUT4 and CUT6
points have nonnegative weights, so those recursions only take the QR branch.
The downdate is the rule if a weight is negative. Several negative weights are
applied in column order, each to the factor left by the previous downdate.

No Jacobians are required.
"""
mutable struct SquareRootConjugateUnscentedKalmanFilter
    t::Float64
    x::Vector{Float64}
    S::Matrix{Float64}
    nx::Int
    order::Int

    function SquareRootConjugateUnscentedKalmanFilter(
        t::Real,
        x::AbstractVector{<:Real},
        P::AbstractMatrix{<:Real};
        order::Integer = 4,
    )
        nx = length(x)
        nx >= 1 || throw(ArgumentError("state dimension must be positive, got $nx"))
        if size(P) != (nx, nx)
            throw(DimensionMismatch("P is $(size(P)); expected ($nx, $nx)"))
        end
        order_int = Int(order)
        _check_cut_support(nx, order_int)
        S = Matrix(LowerTriangular(_lower_cholesky(P)))
        return new(Float64(t), Vector{Float64}(x), S, nx, order_int)
    end
end


"""
    covariance(filter::SquareRootConjugateUnscentedKalmanFilter)

Reconstructed covariance `P = S * S'`, symmetrized.
"""
function covariance(filter::SquareRootConjugateUnscentedKalmanFilter)
    P = filter.S * filter.S'
    return (P + P') ./ 2
end


"""
    predict!(filter::SquareRootConjugateUnscentedKalmanFilter, f, Q, Δt)

`f(t, x, Δt)` is the noise-free process model. `Q` is the additive
process-noise covariance. Sigma points are `x + S ξ`, where `ξ` and the
weights are `cut_points(n; order)`. `filter.t` advances by `Δt`.
"""
function predict!(filter::SquareRootConjugateUnscentedKalmanFilter, f, Q, Δt::Real)
    nx = filter.nx
    Qm = _square_matrix(Q, nx, "Q")
    Δt64 = Float64(Δt)
    isfinite(Δt64) || throw(ArgumentError("Δt must be finite, got $Δt"))
    t = filter.t
    X, w = _srcut_sigma_points(filter)
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
    μ = _weighted_column_mean(Xprop, w)
    Snew = _srcut_factor(_column_deviations(Xprop, μ), w, _noise_factor(Qm))
    filter.x .= μ
    filter.S .= Snew
    filter.t = t + Δt64
    return nothing
end


"""
    update!(filter::SquareRootConjugateUnscentedKalmanFilter, h, y, R)

`h(t, x)` is the noise-free measurement model. `y` is the measurement and `R`
its additive covariance. The innovation factor uses the same QR and negative-weight
downdate as `predict!`. The gain is `Pxy (Sy Sy')⁻¹`, with `Pxy` the CUT
cross-covariance, and the posterior factor is downdated by each column of `K Sy`.
"""
function update!(filter::SquareRootConjugateUnscentedKalmanFilter, h, y, R)
    yv = _measurement_vector(y)
    ny = length(yv)
    Rm = _square_matrix(R, ny, "R")
    nx = filter.nx
    X, w = _srcut_sigma_points(filter)
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
    yhat = _weighted_column_mean(Y, w)
    Sy = _srcut_factor(_column_deviations(Y, yhat), w, _noise_factor(Rm))
    Pxy = _cross_covariance(X, filter.x, Y, yhat, w)
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


function _srcut_sigma_points(filter::SquareRootConjugateUnscentedKalmanFilter)
    ξ, w = cut_points(filter.nx; order = filter.order)
    X = filter.S * ξ .+ reshape(filter.x, :, 1)
    return X, w
end


"""
Square-root covariance from CUT deviations and an additive noise factor.

Positive weights enter the thin QR. Each negative weight is then a Cholesky
rank-one downdate, in column order. Zero weights are skipped.
"""
function _srcut_factor(dev::AbstractMatrix, w::AbstractVector, noise::AbstractMatrix)
    n, m = size(dev)
    if length(w) != m
        throw(DimensionMismatch("expected $m weights, got $(length(w))"))
    end
    pieces = Matrix{Float64}[]
    negatives = Tuple{Int, Float64}[]
    for i in 1:m
        wi = w[i]
        if wi > 0
            push!(pieces, reshape(sqrt(wi) .* dev[:, i], n, 1))
        elseif wi < 0
            push!(negatives, (i, wi))
        elseif wi != 0
            throw(ArgumentError("CUT weight must be finite, got $wi"))
        end
    end
    if size(noise, 2) > 0
        push!(pieces, Matrix{Float64}(noise))
    end
    isempty(pieces) && throw(ArgumentError("square-root CUT factor has no positive columns"))
    S = _qr_lower_factor(reduce(hcat, pieces)')
    for (i, wi) in negatives
        _cholesky_rank1!(S, sqrt(-wi) .* dev[:, i], -1)
    end
    return S
end
