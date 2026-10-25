"""
    ConjugateUnscentedKalmanFilter(t, x, P; order=4)

Conjugate unscented Kalman filter with additive process and measurement noise.

The sigma points are the Gaussian conjugate unscented transform (CUT) of
Adurthi, Singla, and Singh (ASME J. Dyn. Sys., Meas., Control, 2018). `order`
selects the polynomial degree of the underlying cubature:

  - `order = 4` is CUT4, exact for standard-normal moments through total
    degree 5. It is specified for every state dimension `n ≥ 1`.
  - `order = 6` is CUT6, exact through total degree 7. It is specified for
    `2 ≤ n ≤ 9`.

Order 8 is tabulated in that paper with scaled-conjugate axes and is not
implemented. Passing `order = 8`, any other order, or a dimension outside the
range above throws `ArgumentError`.

Points for a covariance `P` are `x + L ξ`. `L` is the lower Cholesky factor
of `P` (`P = L Lᵀ`) and `ξ` are the standard-normal CUT points returned by
`cut_points`. The filter stores `P` itself. Each point carries one
cubature weight, and that same weight is used for the mean and the covariance.

`f` and `h` passed to `predict!` and `update!` are noise-free. `Q` and `R`
are added after the cubature. No Jacobians are required.

The constructor copies `x` and `P`. Fields `t`, `x`, and `P` are the current
time, mean, and covariance. `order` is `4` or `6`.
"""
mutable struct ConjugateUnscentedKalmanFilter
    t::Float64
    x::Vector{Float64}
    P::Matrix{Float64}
    nx::Int
    order::Int

    function ConjugateUnscentedKalmanFilter(
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
        return new(
            Float64(t),
            Vector{Float64}(x),
            Matrix{Float64}(P),
            nx,
            order_int,
        )
    end
end


"""
    cut_points(n; order=4)

Standard-normal conjugate unscented points and weights in dimension `n`.

Returns `(X, w)`. `X` has size `(n, N)` and `w` has length `N`. The weights
are the paper's cubature weights: they sum to 1, they are nonnegative on the
supported dimensions, and the same weight multiplies each point in the mean
and in the covariance.

Column layout, with `e_i` the `i`th unit vector:

CUT4, `n = 1`. Four Gauss–Hermite points and no center: `+r1`, `-r1`, `+r2`,
`-r2`.

CUT4, `n = 2`. Center, then principal points `±r1 e_i` for `i = 1, 2`, then
the conjugate points whose coordinates are `(±r2, ±r2)`. Nine points.

CUT4, `n ≥ 3`. The center weight is zero and the center is omitted. Principal
points `±r1 e_i` (`+r1` then `-r1` on each axis), then the `2^n` points of the
`n`th-conjugate set `r2 * (±1, …, ±1)`. Closed form

```
r1 = sqrt((n + 2) / 2),   w1 = 4 / (n + 2)^2
r2 = sqrt((n + 2) / (n - 2)),   w2 = (n - 2)^2 / (2^n (n + 2)^2)
```

CUT6, `2 ≤ n ≤ 6`. Center, principal points `±r1 e_i`, the `n`th-conjugate
set `r2 * (±1, …, ±1)`, then the second-conjugate set: every choice of two
coordinates equal to `±r3` and the rest zero. Radii and weights are the
ten-decimal appendix values. The center weight is the remainder that makes
the weights sum to 1.

CUT6, `7 ≤ n ≤ 9`. Center, principal points, the `n`th-conjugate set, then
the third-conjugate set (exactly three coordinates equal to `±r3`).

On a conjugate axis the selected coordinates are `±r`, so the Euclidean
distance from the origin is `r * sqrt(m)` for an `m`-conjugate point.
Principal points sit at Euclidean distance `r1`. Sign patterns on conjugate
axes walk the bit masks of the selected coordinates in lexicographic index
order: bit `k - 1` set means coordinate `k` of that selection is `+r`, and a
clear bit means `-r`.
"""
function cut_points(n::Integer; order::Integer = 4)
    n_int = Int(n)
    order_int = Int(order)
    n_int >= 1 || throw(ArgumentError("state dimension must be positive, got $n_int"))
    _check_cut_support(n_int, order_int)
    if order_int == 4
        return _cut4_points(n_int)
    end
    return _cut6_points(n_int)
end


"""
    predict!(filter::ConjugateUnscentedKalmanFilter, f, Q, Δt)

Discrete-time prediction with additive process noise.

`f(t, x, Δt)` is the noise-free process model and must return a vector of
length `filter.nx`. Each current sigma point is passed through `f`. The
predicted mean and covariance are the CUT cubature moments, and `Q` is added
to the covariance. `filter.t` advances by `Δt`.

`Q` is an `nx × nx` covariance (`Matrix` or `Diagonal`). Derivatives of `f`
are not used.
"""
function predict!(filter::ConjugateUnscentedKalmanFilter, f, Q, Δt::Real)
    nx = filter.nx
    Qm = _square_matrix(Q, nx, "Q")
    Δt64 = Float64(Δt)
    isfinite(Δt64) || throw(ArgumentError("Δt must be finite, got $Δt"))
    t = filter.t
    X, w = _cut_sigma_points(filter)
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
    P = _deviation_covariance(Xprop, μ, w) + Qm
    filter.x .= μ
    filter.P .= _symmetrize(P)
    filter.t = t + Δt64
    return nothing
end


"""
    update!(filter::ConjugateUnscentedKalmanFilter, h, y, R)

Measurement update with additive measurement noise.

`h(t, x)` is the noise-free measurement model. `y` is the measurement (a
vector, or a scalar when the measurement dimension is 1) and `R` is its
covariance. Sigma points are redrawn from the current mean and covariance,
passed through `h`, and combined with the CUT cubature moments. The Kalman
gain uses those moments. No Jacobian is required.

Posterior covariance is the Joseph form
`P - Pxy Kᵀ - K Pxyᵀ + K S Kᵀ`, then symmetrized, where `S` is the
innovation covariance and `Pxy` is the state–measurement cross-covariance.
"""
function update!(filter::ConjugateUnscentedKalmanFilter, h, y, R)
    yv = _measurement_vector(y)
    ny = length(yv)
    Rm = _square_matrix(R, ny, "R")
    t = filter.t
    x_prior = filter.x
    P_prior = filter.P
    X, w = _cut_sigma_points(filter)
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
    yhat = _weighted_column_mean(Y, w)
    S = _symmetrize(_deviation_covariance(Y, yhat, w) + Rm)
    Pxy = _cross_covariance(X, x_prior, Y, yhat, w)
    K = Pxy / S
    x_new = x_prior .+ (K * (yv .- yhat))
    P_new = P_prior - Pxy * K' - K * Pxy' + K * S * K'
    filter.x .= x_new
    filter.P .= _symmetrize(P_new)
    return nothing
end


function _cut_sigma_points(filter::ConjugateUnscentedKalmanFilter)
    ξ, w = cut_points(filter.nx; order = filter.order)
    L = _lower_cholesky(filter.P)
    X = L * ξ .+ reshape(filter.x, :, 1)
    return X, w
end

function _check_cut_support(n::Int, order::Int)
    if order == 4
        return nothing
    elseif order == 6
        if 2 <= n <= 9
            return nothing
        end
        throw(
            ArgumentError(
                "CUT6 is specified for 2 ≤ n ≤ 9, got n = $n. CUT4 supports n ≥ 1. CUT8 is not implemented.",
            ),
        )
    elseif order == 8
        throw(
            ArgumentError(
                "CUT8 is not implemented. Supported conjugate unscented orders are CUT4 for n ≥ 1 and CUT6 for 2 ≤ n ≤ 9.",
            ),
        )
    end
    throw(
        ArgumentError(
            "order must be 4 or 6, got $order. CUT8 is not implemented. CUT4 supports n ≥ 1 and CUT6 supports 2 ≤ n ≤ 9.",
        ),
    )
end

# Table 7 of Adurthi, Singla, and Singh: optimized CUT4-G solution.
# Columns are n = 1 and n = 2.
const _CUT4_TABLE_R1 = (0.7419637843027252, 2.6060099476935847)
const _CUT4_TABLE_R2 = (2.3344142183389796, 1.190556300661233)
const _CUT4_TABLE_W0 = (0.0, 0.41553535186548973)
const _CUT4_TABLE_W1 = (0.45412414523193145, 0.021681819434216532)
const _CUT4_TABLE_W2 = (0.04587585476806854, 0.12443434259941118)

# Table 27: CUT6-G radii and weights for n = 2, …, 9, printed to ten decimals
# (n = 5 has r3 = 3, n = 8 has r2 = 1 and r3 printed to nine decimals).
const _CUT6_R1 = (
    2.4494897427,
    2.3587090379,
    2.2520650012,
    2.1213203430,
    1.9488352799,
    2.5512003554,
    2.4494897427,
    2.3439073215,
)
const _CUT6_R2 = (
    1.1147379454,
    1.1198362859,
    1.1260325006,
    1.1338934189,
    1.1445968942,
    0.9642630979,
    1.0,
    1.0232622230,
)
const _CUT6_R3 = (
    3.2004125801,
    3.1421303838,
    3.0763780026,
    3.0,
    2.9068006056,
    2.3255766977,
    2.449489742,
    2.5342864499,
)
const _CUT6_W1 = (
    0.0277777777,
    0.0290351301,
    0.0306601632,
    0.0329218107,
    0.0365072564,
    0.0126940628,
    0.0138888888,
    0.0150763910,
)
const _CUT6_W2 = (
    0.1302876649,
    0.0633844605,
    0.0306601632,
    0.0147033607,
    0.0069487173,
    0.0048594459,
    0.00234375,
    0.0011342717,
)
const _CUT6_W3 = (
    0.0004653012,
    0.0005195469,
    0.0005898367,
    0.0006858710,
    0.0008288549,
    0.0003950899,
    0.0002314814,
    0.0001572731,
)

function _cut4_points(n::Int)
    if n <= 2
        r1 = _CUT4_TABLE_R1[n]
        r2 = _CUT4_TABLE_R2[n]
        w0 = _CUT4_TABLE_W0[n]
        w1 = _CUT4_TABLE_W1[n]
        w2 = _CUT4_TABLE_W2[n]
        if n == 1
            X = zeros(Float64, 1, 4)
            X[1, 1] = r1
            X[1, 2] = -r1
            X[1, 3] = r2
            X[1, 4] = -r2
            w = Float64[w1, w1, w2, w2]
            return X, w
        end
        # Center, 4 principal points, 4 conjugate points.
        X = zeros(Float64, 2, 9)
        w = zeros(Float64, 9)
        w[1] = w0
        col = _write_principal!(X, w, 1, 2, r1, w1)
        col = _write_conjugate!(X, w, col, 2, 2, r2, w2)
        col == 9 || error("CUT4 n = 2 produced $col columns")
        return X, w
    end
    r1 = sqrt((n + 2) / 2)
    r2 = sqrt((n + 2) / (n - 2))
    w1 = 4 / (n + 2)^2
    w2 = (n - 2)^2 / (2.0^n * (n + 2)^2)
    N = 2 * n + (1 << n)
    X = zeros(Float64, n, N)
    w = zeros(Float64, N)
    col = _write_principal!(X, w, 0, n, r1, w1)
    col = _write_conjugate!(X, w, col, n, n, r2, w2)
    col == N || error("CUT4 n = $n produced $col columns, expected $N")
    return X, w
end

function _cut6_points(n::Int)
    k = n - 1
    r1 = _CUT6_R1[k]
    r2 = _CUT6_R2[k]
    r3 = _CUT6_R3[k]
    w1 = _CUT6_W1[k]
    w2 = _CUT6_W2[k]
    w3 = _CUT6_W3[k]
    # n ≤ 6 uses the second-conjugate family; n ≥ 7 uses the third-conjugate family.
    m3 = n <= 6 ? 2 : 3
    n3 = _conjugate_count(n, m3)
    N = 1 + 2 * n + (1 << n) + n3
    X = zeros(Float64, n, N)
    w = zeros(Float64, N)
    col = _write_principal!(X, w, 1, n, r1, w1)
    col = _write_conjugate!(X, w, col, n, n, r2, w2)
    col = _write_conjugate!(X, w, col, n, m3, r3, w3)
    col == N || error("CUT6 n = $n produced $col columns, expected $N")
    w[1] = 1 - sum(@view w[2:end])
    w[1] > 0 || throw(
        ArgumentError("CUT6 center weight is $(w[1]) for n = $n; tabulated weights must leave a positive remainder"),
    )
    return X, w
end

function _conjugate_count(n::Int, m::Int)
    return binomial(n, m) * (1 << m)
end

"""Write `±radius` on each coordinate axis. `col` is the last column already filled."""
function _write_principal!(X, w, col, n, radius, weight)
    for i in 1:n
        col += 1
        X[i, col] = radius
        w[col] = weight
        col += 1
        X[i, col] = -radius
        w[col] = weight
    end
    return col
end

"""
Write every point with exactly `m` nonzero coordinates, each equal to `±radius`.

Index combinations are lexicographic. For a combination, bit `k - 1` of the
sign mask selects `+radius` on the `k`th selected coordinate.
"""
function _write_conjugate!(X, w, col, n, m, radius, weight)
    for indices in _combinations(n, m)
        nsigns = 1 << m
        for mask in 0:(nsigns - 1)
            col += 1
            w[col] = weight
            for k in 1:m
                sign = ((mask >> (k - 1)) & 1) == 1 ? 1.0 : -1.0
                X[indices[k], col] = sign * radius
            end
        end
    end
    return col
end

function _combinations(n::Int, m::Int)
    combos = Vector{Vector{Int}}()
    chosen = Int[]
    function rec(start)
        if length(chosen) == m
            push!(combos, copy(chosen))
            return nothing
        end
        remaining = m - length(chosen)
        for i in start:(n - remaining + 1)
            push!(chosen, i)
            rec(i + 1)
            pop!(chosen)
        end
        return nothing
    end
    rec(1)
    return combos
end
