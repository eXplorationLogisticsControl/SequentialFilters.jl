"""
    cubature_points(n)

Third-degree spherical-radial cubature points for a standard normal vector
of dimension `n` (Arasaratnam and Haykin).

Returns `(ξ, w)`. `ξ` has size `(n, 2n)`. Column `i` is `√n e_i` and column
`n + i` is `-√n e_i`. Every weight equals `1 / (2n)`. The weights are
positive and sum to 1. The rule integrates polynomials through total degree 3
exactly, so a linear-Gaussian update is exact up to roundoff.
"""
function cubature_points(n::Integer)
    n_int = Int(n)
    n_int >= 1 || throw(ArgumentError("state dimension must be positive, got $n_int"))
    ξ = zeros(Float64, n_int, 2 * n_int)
    radius = sqrt(Float64(n_int))
    for i in 1:n_int
        ξ[i, i] = radius
        ξ[i, n_int + i] = -radius
    end
    w = fill(1 / (2 * n_int), 2 * n_int)
    return ξ, w
end


"""
    SquareRootCubatureKalmanFilter(t, x, P)

Square-root cubature Kalman filter with additive noise.

The covariance is a lower Cholesky factor `S` with `P = S * S'` and a positive
diagonal. There are `2n` cubature points from [`cubature_points`](@ref), all
with the positive weight `1 / (2n)`. The predicted factor and the innovation
factor are the thin QR of `[√w (χ − μ)  noise factor]`. No weight is negative,
so there is no covariance-weight downdate. The posterior factor is a Cholesky
rank-one downdate by each column of `K Sy`, the same measurement update as the
square-root UKF. The element type follows `eltype(x)` and `eltype(P)`.

No Jacobians are required.
"""
mutable struct SquareRootCubatureKalmanFilter{T<:AbstractFloat}
    t::T
    x::Vector{T}
    S::Matrix{T}
    nx::Int

    function SquareRootCubatureKalmanFilter(
        t::Real,
        x::AbstractVector{<:Real},
        P::AbstractMatrix{<:Real},
    )
        nx = length(x)
        nx >= 1 || throw(ArgumentError("state dimension must be positive, got $nx"))
        if size(P) != (nx, nx)
            throw(DimensionMismatch("P is $(size(P)); expected ($nx, $nx)"))
        end
        T = _filter_eltype(x, P)
        S = Matrix(LowerTriangular(_lower_cholesky(Matrix{T}(P))))
        return new{T}(T(t), Vector{T}(x), S, nx)
    end
end


"""
    covariance(filter::SquareRootCubatureKalmanFilter)

Reconstructed covariance `P = S * S'`, symmetrized.
"""
function covariance(filter::SquareRootCubatureKalmanFilter)
    P = filter.S * filter.S'
    return (P + P') ./ 2
end


"""
    predict!(filter::SquareRootCubatureKalmanFilter, f, Q, Δt)

`f(t, x, Δt)` is the noise-free process model. `Q` is the additive
process-noise covariance. `filter.t` advances by `Δt`.
"""
function predict!(filter::SquareRootCubatureKalmanFilter, f, Q, Δt::Real)
    T = eltype(filter.x)
    nx = filter.nx
    Qm = _as_eltype(_square_matrix(Q, nx, "Q"), T)
    ΔtT = T(Δt)
    isfinite(ΔtT) || throw(ArgumentError("Δt must be finite, got $Δt"))
    t = filter.t
    X, w = _srckf_sigma_points(filter)
    Xprop = Matrix{T}(undef, nx, size(X, 2))
    for i in axes(X, 2)
        xi = f(t, X[:, i], ΔtT)
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
    filter.t = t + ΔtT
    return nothing
end


"""
    update!(filter::SquareRootCubatureKalmanFilter, h, y, R)

`h(t, x)` is the noise-free measurement model. `y` is the measurement and `R`
its additive covariance.
"""
function update!(filter::SquareRootCubatureKalmanFilter, h, y, R)
    T = eltype(filter.x)
    yv = _measurement_vector(y, T)
    ny = length(yv)
    Rm = _as_eltype(_square_matrix(R, ny, "R"), T)
    X, w = _srckf_sigma_points(filter)
    Y = Matrix{T}(undef, ny, size(X, 2))
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


function _srckf_sigma_points(filter::SquareRootCubatureKalmanFilter{T}) where {T}
    ξ64, w64 = cubature_points(filter.nx)
    ξ = Matrix{T}(ξ64)
    w = Vector{T}(w64)
    X = filter.S * ξ .+ reshape(filter.x, :, 1)
    return X, w
end
