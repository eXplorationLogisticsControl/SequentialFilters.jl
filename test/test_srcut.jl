using LinearAlgebra
using Random
using Test

using SequentialFilters

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
        predict!(cut, f, Qm, 1.0)
        predict!(sr, f, Qm, 1.0)
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
            y_atol = order == 4 ? 1e-8 : 1e-7,
        )
        @test max_x < (order == 4 ? 1e-8 : 1e-7)
        @test max_P < (order == 4 ? 1e-8 : 1e-7)
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
        y_atol = 1e-8,
    )
    @test max_x < 1e-8
    @test max_P < 1e-8

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
        y_atol = 1e-7,
    )
    @test max_x < 1e-7
    @test max_P < 1e-7

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
