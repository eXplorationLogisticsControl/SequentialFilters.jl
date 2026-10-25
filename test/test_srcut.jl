using LinearAlgebra
using Random
using Test

using SequentialFilters

# SR-CUT and full-covariance CUT are the same cubature written two ways.
# With nonnegative weights the predicted factor is the thin QR of
# [√wᵢ (χᵢ − x̄), Q^{1/2}], and that Gram matrix is the CUT covariance.
# CUT4 and CUT6 weights are nonnegative, so those orders never take the
# negative-weight downdate. The posterior downdate by the columns of K Sy
# is the same rank-ny update as P − K Sy Syᵀ Kᵀ. The two recursions are
# algebraically identical. They are not bit-identical: each QR and each
# rank-one downdate moves the result by a few units in the last place.
# On the linear-Gaussian systems below (P of order 1, condition number of
# order 10, 12 to 20 steps) the Euclidean difference in the state and in P
# measures below 10 ε(Float64). The tolerance is 100 ε(Float64) ≈ 2.2e-14.
const SRCUT_LINEAR_ATOL = 100 * eps(Float64)

function spd_lower_factor(S)
    @test istril(S)
    @test all(>(0), diag(S))
    P = S * S'
    @test norm(P - P') < 1e-8 * max(1.0, norm(P))
    Ps = (P + P') ./ 2
    @test all(>(0), eigvals(Symmetric(Ps)))
    return Ps
end

function compare_cut_srcut(; F, H, Q, R, x0, P0, nsteps, seed, order, y_atol)
    nx = length(x0)
    ny = size(H, 1)
    rng = MersenneTwister(seed)
    Qm = Matrix{Float64}(Q)
    Rm = Matrix{Float64}(R)
    Qchol = cholesky(Hermitian((Qm + Qm') ./ 2)).L
    Rchol = cholesky(Hermitian((Rm + Rm') ./ 2)).L
    x_true = copy(x0)
    ys = Vector{Float64}[]
    for _ in 1:nsteps
        x_true = F * x_true .+ Qchol * randn(rng, nx)
        push!(ys, H * x_true .+ Rchol * randn(rng, ny))
    end

    cut = ConjugateUnscentedKalmanFilter(0.0, x0, P0; order = order)
    sr = SquareRootConjugateUnscentedKalmanFilter(0.0, x0, P0; order = order)
    f = (t, x, Δt) -> F * x
    h = (t, x) -> H * x
    max_x = 0.0
    max_P = 0.0
    for k in 1:nsteps
        dynamics = DiscreteDynamicsModel(f; n = nx)
        noise = AdditiveProcessNoise(Qm)
        predict!(cut, dynamics, noise, 1.0)
        predict!(sr, dynamics, noise, 1.0)
        max_x = max(max_x, norm(sr.x .- cut.x))
        max_P = max(max_P, norm(covariance(sr) .- cut.P))
        @test sr.x ≈ cut.x atol = y_atol rtol = y_atol
        @test covariance(sr) ≈ cut.P atol = y_atol rtol = y_atol
        spd_lower_factor(sr.S)

        update!(cut, h, ys[k], Rm)
        update!(sr, h, ys[k], Rm)
        max_x = max(max_x, norm(sr.x .- cut.x))
        max_P = max(max_P, norm(covariance(sr) .- cut.P))
        @test sr.x ≈ cut.x atol = y_atol rtol = y_atol
        @test covariance(sr) ≈ cut.P atol = y_atol rtol = y_atol
        @test sr.t ≈ cut.t
        spd_lower_factor(sr.S)
    end
    return max_x, max_P
end

@testset "square-root CUT matches the CUT" begin
    P0 = [1.0 0.2; 0.2 0.8]
    sr = SquareRootConjugateUnscentedKalmanFilter(0.0, [1.0, -0.5], P0)
    @test sr.order == 4
    @test sr.S * sr.S' ≈ P0 atol = 1e-12
    spd_lower_factor(sr.S)

    for order in (4, 6)
        max_x, max_P = compare_cut_srcut(;
            F = [1.0 0.1; 0.0 1.0],
            H = [1.0 -0.2],
            Q = 0.01 * Matrix{Float64}(I, 2, 2),
            R = fill(0.05, 1, 1),
            x0 = [1.0, -0.5],
            P0 = P0,
            nsteps = 15,
            seed = 20 + order,
            order = order,
            y_atol = SRCUT_LINEAR_ATOL,
        )
        @test max_x < SRCUT_LINEAR_ATOL
        @test max_P < SRCUT_LINEAR_ATOL
    end

    max_x, max_P = compare_cut_srcut(;
        F = [1.0;;],
        H = [1.0;;],
        Q = [0.2;;],
        R = [0.5;;],
        x0 = [0.0],
        P0 = [1.0;;],
        nsteps = 20,
        seed = 4,
        order = 4,
        y_atol = SRCUT_LINEAR_ATOL,
    )
    @test max_x < SRCUT_LINEAR_ATOL
    @test max_P < SRCUT_LINEAR_ATOL

    max_x, max_P = compare_cut_srcut(;
        F = [0.9 0.1 0.0; 0.0 0.8 0.2; 0.0 0.0 0.95],
        H = [1.0 0.0 0.2; 0.0 1.0 -0.1],
        Q = Matrix(Diagonal([0.1, 0.1, 0.05])),
        R = [0.4 0.05; 0.05 0.3],
        x0 = [1.0, -0.5, 0.25],
        P0 = Matrix{Float64}(I, 3, 3),
        nsteps = 12,
        seed = 6,
        order = 6,
        y_atol = SRCUT_LINEAR_ATOL,
    )
    @test max_x < SRCUT_LINEAR_ATOL
    @test max_P < SRCUT_LINEAR_ATOL

    @test_throws ArgumentError SquareRootConjugateUnscentedKalmanFilter(
        0.0,
        zeros(1),
        fill(1.0, 1, 1);
        order = 6,
    )
    @test_throws ArgumentError SquareRootConjugateUnscentedKalmanFilter(
        0.0,
        zeros(2),
        Matrix{Float64}(I, 2, 2);
        order = 8,
    )
end

@testset "nonnegative CUT weights take the QR path" begin
    for order in (4, 6)
        dims = order == 4 ? (1, 2, 4, 6) : (2, 4, 6, 9)
        for n in dims
            _, w = cut_points(n; order = order)
            @test all(>=(0), w)
        end
    end

    rng = MersenneTwister(1)
    n = 4
    _, w = cut_points(n; order = 4)
    dev = randn(rng, n, length(w))
    Q = 0.1 * Matrix{Float64}(I, n, n) + fill(0.01, n, n)
    Nq = Matrix(cholesky(Hermitian(Q)).L)
    S = SequentialFilters._srcut_factor(dev, w, Nq)
    cols = Matrix{Float64}[]
    for i in eachindex(w)
        w[i] > 0 && push!(cols, reshape(sqrt(w[i]) .* dev[:, i], n, 1))
    end
    push!(cols, Nq)
    Sqr = SequentialFilters._qr_lower_factor(reduce(hcat, cols)')
    @test S == Sqr

    dev_neg = [1.0 0.0 0.2; 0.0 1.0 -0.3]
    w_neg = [0.5, 0.5, -0.05]
    Sneg = SequentialFilters._srcut_factor(dev_neg, w_neg, zeros(2, 0))
    positive = hcat(sqrt(0.5) .* dev_neg[:, 1], sqrt(0.5) .* dev_neg[:, 2])
    Smanual = SequentialFilters._qr_lower_factor(positive')
    SequentialFilters._cholesky_rank1!(Smanual, sqrt(0.05) .* dev_neg[:, 3], -1)
    @test Sneg ≈ Smanual atol = 0 rtol = 0
end

@testset "Float32 CUT and SR-CUT share one linear step" begin
    x = Float32[1.0, -0.5]
    P = Float32[1.0 0.2; 0.2 0.8]
    Q = Float32(0.01) * Matrix{Float32}(I, 2, 2)
    R = fill(Float32(0.05), 1, 1)
    F = Float32[1.0 0.1; 0.0 1.0]
    H = Float32[1.0 -0.2]
    y = Float32(0.7)
    for order in (4, 6)
        cut = ConjugateUnscentedKalmanFilter(0.0, x, P; order = order)
        sr = SquareRootConjugateUnscentedKalmanFilter(0.0, x, P; order = order)
        @test eltype(cut.x) == Float32
        @test eltype(cut.P) == Float32
        @test eltype(sr.x) == Float32
        @test eltype(sr.S) == Float32
        f = (t, z, dt) -> F * z
        h = (t, z) -> H * z
        dynamics = DiscreteDynamicsModel(f; n = 2)
        noise = AdditiveProcessNoise(Q)
        predict!(cut, dynamics, noise, 1.0)
        predict!(sr, dynamics, noise, 1.0)
        update!(cut, h, y, R)
        update!(sr, h, y, R)
        @test eltype(cut.P) == Float32
        @test eltype(covariance(sr)) == Float32
        @test sr.x ≈ cut.x atol = 1e-5
        @test covariance(sr) ≈ cut.P atol = 1e-5
    end
end
