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
    return nothing
end

@testset "square-root UKF square moment" begin
    for beta in (0.0, 2.0, 3.0)
        sr = SquareRootUnscentedKalmanFilter(
            0.0,
            [0.0],
            [1.0;;];
            alpha = 1.0,
            beta = beta,
            kappa = 0.0,
        )
        predict!(sr, (t, x, Δt) -> [x[1]^2], zeros(1, 1), 1.0)
        @test sr.x[1] ≈ 1.0 atol = 1e-10
        @test covariance(sr)[1, 1] ≈ beta atol = 1e-8
        if beta > 0
            spd_lower_factor(sr.S)
        end
    end

    sr = SquareRootUnscentedKalmanFilter(0.0, [0.0], [1.0;;])
    @test sr.alpha == 1e-3
    @test sr.beta == 2.0
    @test sr.kappa == 0.0
    predict!(sr, (t, x, Δt) -> [x[1]^2], zeros(1, 1), 1.0)
    @test sr.x[1] ≈ 1.0 atol = 1e-6
    @test covariance(sr)[1, 1] ≈ 2.0 atol = 1e-5
    spd_lower_factor(sr.S)
end

function compare_ukf_srukf(; F, H, Q, R, x0, P0, nsteps, seed, alpha = 1e-3)
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

    ukf = UnscentedKalmanFilter(0.0, x0, P0; alpha = alpha, beta = 2.0, kappa = 0.0)
    sr = SquareRootUnscentedKalmanFilter(0.0, x0, P0; alpha = alpha, beta = 2.0, kappa = 0.0)
    f = (t, x, Δt) -> F * x
    h = (t, x) -> H * x
    max_x = 0.0
    max_P = 0.0
    for k in 1:nsteps
        predict!(ukf, f, Qm, 1.0)
        predict!(sr, f, Qm, 1.0)
        max_x = max(max_x, norm(sr.x .- ukf.x))
        max_P = max(max_P, norm(covariance(sr) .- ukf.P))
        @test sr.x ≈ ukf.x atol = 1e-6 rtol = 1e-6
        @test covariance(sr) ≈ ukf.P atol = 1e-6 rtol = 1e-6
        spd_lower_factor(sr.S)

        update!(ukf, h, ys[k], Rm)
        update!(sr, h, ys[k], Rm)
        max_x = max(max_x, norm(sr.x .- ukf.x))
        max_P = max(max_P, norm(covariance(sr) .- ukf.P))
        @test sr.x ≈ ukf.x atol = 1e-6 rtol = 1e-6
        @test covariance(sr) ≈ ukf.P atol = 1e-6 rtol = 1e-6
        @test sr.t ≈ ukf.t
        spd_lower_factor(sr.S)
    end
    return max_x, max_P
end

@testset "square-root UKF matches the UKF" begin
    max_x, max_P = compare_ukf_srukf(;
        F = [1.0;;],
        H = [1.0;;],
        Q = [0.2;;],
        R = [0.5;;],
        x0 = [0.0],
        P0 = [1.0;;],
        nsteps = 30,
        seed = 1,
    )
    @test max_x < 1e-6
    @test max_P < 1e-6

    Δt = 0.5
    q = 0.05
    max_x, max_P = compare_ukf_srukf(;
        F = [1.0 Δt; 0.0 1.0],
        H = [1.0 0.0],
        Q = q * [Δt^3 / 3 Δt^2 / 2; Δt^2 / 2 Δt],
        R = [0.8;;],
        x0 = [0.0, 1.0],
        P0 = [1.0 0.0; 0.0 1.0],
        nsteps = 30,
        seed = 2,
    )
    @test max_x < 1e-6
    @test max_P < 1e-6

    max_x, max_P = compare_ukf_srukf(;
        F = [0.9 0.1 0.0; 0.0 0.8 0.2; 0.0 0.0 0.95],
        H = [1.0 0.0 0.2; 0.0 1.0 -0.1],
        Q = Diagonal([0.1, 0.1, 0.05]),
        R = [0.4 0.05; 0.05 0.3],
        x0 = [1.0, -0.5, 0.25],
        P0 = Matrix{Float64}(I, 3, 3),
        nsteps = 20,
        seed = 3,
        alpha = 1.0,
    )
    @test max_x < 1e-8
    @test max_P < 1e-8
end
