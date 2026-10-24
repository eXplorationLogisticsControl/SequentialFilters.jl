using LinearAlgebra
using Random
using Test

using SequentialFilters

@testset "UKF" begin

@testset "scaled unscented weights" begin
    cases = [
        (n = 1, alpha = 1e-3, beta = 2.0, kappa = 0.0),
        (n = 2, alpha = 1e-3, beta = 2.0, kappa = 0.0),
        (n = 4, alpha = 0.5, beta = 2.0, kappa = -1.0),
        (n = 5, alpha = 1.0, beta = 0.0, kappa = 3.0 - 5.0),
    ]
    for case in cases
        weights = unscented_weights(
            case.n;
            alpha = case.alpha,
            beta = case.beta,
            kappa = case.kappa,
        )
        @test length(weights.Wm) == 2 * case.n + 1
        @test length(weights.Wc) == 2 * case.n + 1
        @test sum(weights.Wm) ≈ 1.0 atol = 1e-9
        @test weights.Wc[2:end] ≈ weights.Wm[2:end]
        @test weights.Wc[1] ≈ weights.Wm[1] + (1 - case.alpha^2 + case.beta)
        @test weights.lambda ≈ case.alpha^2 * (case.n + case.kappa) - case.n
        @test weights.Wm[2] ≈ 0.5 / (case.n + weights.lambda)
        @test sum(weights.Wc) ≈ 1 + (1 - case.alpha^2 + case.beta) atol = 1e-9
    end

    ukf = UnscentedKalmanFilter(0.0, [0.0], [1.0;;])
    @test ukf.alpha == 1e-3
    @test ukf.beta == 2.0
    @test ukf.kappa == 0.0
    defaults = unscented_weights(1)
    @test defaults.lambda ≈ ukf.alpha^2 * (1 + ukf.kappa) - 1
    @test sum(defaults.Wm) ≈ 1.0 atol = 1e-9
end

@testset "sigma points" begin
    Random.seed!(11)
    for n in (1, 2, 6)
        x = randn(n)
        A = randn(n, n)
        P = A * A' + 0.3 * Matrix{Float64}(I, n, n)
        for alpha in (1e-3, 1.0)
            X = get_sigma_points(x, P; alpha = alpha, beta = 2.0, kappa = 0.0)
            weights = unscented_weights(n; alpha = alpha, beta = 2.0, kappa = 0.0)
            @test size(X) == (n, 2 * n + 1)
            @test X[:, 1] ≈ x
            for i in 1:n
                @test X[:, 1 + i] ≈ -X[:, 1 + n + i] .+ 2 .* x atol = 1e-12
            end
            μ = zeros(n)
            for i in axes(X, 2)
                μ .+= weights.Wm[i] .* X[:, i]
            end
            @test μ ≈ x atol = 1e-8 rtol = 1e-8
            Prec = zeros(n, n)
            for i in axes(X, 2)
                d = X[:, i] .- x
                Prec .+= weights.Wc[i] .* (d * d')
            end
            Prec .= (Prec + Prec') ./ 2
            @test Prec ≈ P atol = 1e-6 rtol = 1e-6
        end
    end

    x = zeros(2)
    P = Matrix{Float64}(I, 2, 2)
    Xsmall = get_sigma_points(x, P; alpha = 1e-3, kappa = 0.0)
    Xlarge = get_sigma_points(x, P; alpha = 1.0, kappa = 0.0)
    @test norm(Xlarge[:, 2]) / norm(Xsmall[:, 2]) ≈ 1e3 rtol = 1e-8

    stored = [1.0, -2.0]
    Pc = [2.0 0.3; 0.3 1.5]
    ukf = UnscentedKalmanFilter(1.5, stored, Pc)
    stored[1] = 0.0
    Pc[1, 1] = 0.0
    @test ukf.t == 1.5
    @test ukf.x == [1.0, -2.0]
    @test ukf.P[1, 1] == 2.0
    @test ukf.nx == 2
    @test get_sigma_points(ukf)[:, 1] ≈ ukf.x

    @test_throws ArgumentError UnscentedKalmanFilter(0.0, [0.0], [1.0;;]; alpha = 0.0)
    @test_throws ArgumentError UnscentedKalmanFilter(0.0, [0.0], [1.0;;]; alpha = -0.1)
    @test_throws ArgumentError UnscentedKalmanFilter(0.0, [0.0, 0.0], Matrix{Float64}(I, 2, 2); kappa = -2.0)
    @test_throws DimensionMismatch UnscentedKalmanFilter(0.0, [0.0, 0.0], [1.0;;])
    @test_throws ArgumentError get_sigma_points([0.0, 0.0], [1.0 0.0; 0.0 -1.0])
end

@testset "square moment through predict!" begin
    # x ~ N(0, 1), y = x^2. With α = 1, κ = 0 the center deviation is ±1 and
    # the predicted variance equals β. β = 2 recovers Var(x^2) = 2.
    for beta in (0.0, 2.0, 3.0)
        ukf = UnscentedKalmanFilter(0.0, [0.0], [1.0;;]; alpha = 1.0, beta = beta, kappa = 0.0)
        predict!(ukf, (t, x, Δt) -> [x[1]^2], zeros(1, 1), 1.0)
        @test ukf.x[1] ≈ 1.0 atol = 1e-12
        @test ukf.P[1, 1] ≈ beta atol = 1e-12
        @test ukf.t ≈ 1.0
    end

    ukf = UnscentedKalmanFilter(0.0, [0.0], [1.0;;])
    predict!(ukf, (t, x, Δt) -> [x[1]^2], zeros(1, 1), 1.0)
    @test ukf.alpha == 1e-3
    @test ukf.x[1] ≈ 1.0 atol = 1e-6
    @test ukf.P[1, 1] ≈ 2.0 atol = 1e-6
end

function kalman_vs_ukf_errors(; F, H, Q, R, x0, P0, nsteps, seed, alpha = 1e-3)
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

    x_kf = copy(x0)
    P_kf = Matrix{Float64}(P0)
    ukf = UnscentedKalmanFilter(0.0, x0, P0; alpha = alpha, beta = 2.0, kappa = 0.0)
    f = (t, x, Δt) -> F * x
    h = (t, x) -> H * x
    max_x = 0.0
    max_P = 0.0
    for k in 1:nsteps
        x_kf = F * x_kf
        P_kf = (F * P_kf * F' + Qm)
        P_kf = (P_kf + P_kf') ./ 2
        predict!(ukf, f, Qm, 1.0)
        max_x = max(max_x, norm(ukf.x .- x_kf))
        max_P = max(max_P, norm(ukf.P .- P_kf))

        y = ys[k]
        innov = y .- H * x_kf
        S = H * P_kf * H' + Rm
        S = (S + S') ./ 2
        K = (P_kf * H') / S
        x_kf = x_kf .+ K * innov
        P_kf = P_kf - K * S * K'
        P_kf = (P_kf + P_kf') ./ 2
        update!(ukf, h, y, Rm)
        max_x = max(max_x, norm(ukf.x .- x_kf))
        max_P = max(max_P, norm(ukf.P .- P_kf))
        @test issymmetric(ukf.P)
        @test minimum(eigvals(Symmetric(ukf.P))) > 0
    end
    return max_x, max_P
end

@testset "UKF matches the Kalman filter on linear-Gaussian models" begin
    max_x, max_P = kalman_vs_ukf_errors(;
        F = [1.0;;],
        H = [1.0;;],
        Q = [0.2;;],
        R = [0.5;;],
        x0 = [0.0],
        P0 = [1.0;;],
        nsteps = 40,
        seed = 1,
    )
    @test max_x < 1e-6
    @test max_P < 1e-6

    Δt = 0.5
    q = 0.05
    F = [1.0 Δt; 0.0 1.0]
    Q = q * [Δt^3 / 3 Δt^2 / 2; Δt^2 / 2 Δt]
    max_x, max_P = kalman_vs_ukf_errors(;
        F = F,
        H = [1.0 0.0],
        Q = Q,
        R = [0.8;;],
        x0 = [0.0, 1.0],
        P0 = [1.0 0.0; 0.0 1.0],
        nsteps = 40,
        seed = 2,
    )
    @test max_x < 1e-6
    @test max_P < 1e-6

    max_x, max_P = kalman_vs_ukf_errors(;
        F = [0.9 0.1 0.0; 0.0 0.8 0.2; 0.0 0.0 0.95],
        H = [1.0 0.0 0.2; 0.0 1.0 -0.1],
        Q = Diagonal([0.1, 0.1, 0.05]),
        R = [0.4 0.05; 0.05 0.3],
        x0 = [1.0, -0.5, 0.25],
        P0 = Matrix{Float64}(I, 3, 3),
        nsteps = 30,
        seed = 3,
        alpha = 1.0,
    )
    @test max_x < 1e-8
    @test max_P < 1e-8

    Qd = Diagonal([0.2, 0.05])
    Rd = Diagonal([0.3])
    F = [1.0 0.2; 0.0 1.0]
    H = [1.0 0.0]
    x0 = [0.4, -0.2]
    P0 = [1.0 0.1; 0.1 1.0]
    y = [0.5]
    dense = UnscentedKalmanFilter(0.0, x0, P0)
    diagf = UnscentedKalmanFilter(0.0, x0, P0)
    f = (t, x, Δt) -> F * x
    h = (t, x) -> H * x
    predict!(dense, f, Matrix(Qd), 0.2)
    predict!(diagf, f, Qd, 0.2)
    update!(dense, h, y, Matrix(Rd))
    update!(diagf, h, y, Rd)
    @test diagf.x ≈ dense.x atol = 1e-12
    @test diagf.P ≈ dense.P atol = 1e-12

    ukf = UnscentedKalmanFilter(0.0, [1.0, 2.0], [1.0 0.0; 0.0 1.0])
    x_before = copy(ukf.x)
    P_before = copy(ukf.P)
    @test_throws DimensionMismatch predict!(
        ukf,
        (t, x, Δt) -> [x[1]],
        [0.1 0.0; 0.0 0.1],
        0.1,
    )
    @test ukf.x == x_before
    @test ukf.P == P_before
    @test ukf.t == 0.0
    @test_throws DimensionMismatch update!(ukf, (t, x) -> [x[1]], [1.0, 2.0], [1.0;;])
    @test ukf.x == x_before
    @test ukf.t == 0.0
end

"""Discrete coordinated turn. State is `[x, y, vx, vy, ω]`."""
function coordinated_turn(_t, state, Δt)
    px, py, vx, vy, ω = state
    θ = ω * Δt
    if abs(θ) < 1e-9
        return [px + vx * Δt, py + vy * Δt, vx, vy, ω]
    end
    s, c = sincos(θ)
    s_over_ω = s / ω
    one_c_over_ω = (1 - c) / ω
    return [
        px + s_over_ω * vx - one_c_over_ω * vy,
        py + one_c_over_ω * vx + s_over_ω * vy,
        c * vx - s * vy,
        s * vx + c * vy,
        ω,
    ]
end

function range_bearing(_t, state)
    return [hypot(state[1], state[2]), atan(state[2], state[1])]
end

function posterior_is_spd(P)
    issymmetric(P) || return false
    return minimum(eigvals(Symmetric(P))) > 0
end

@testset "UKF tracks a coordinated turn" begin
    Δt = 0.5
    nsteps = 40
    x_true = [200.0, 50.0, 8.0, 0.0, 0.08]
    x_hat0 = [185.0, 60.0, 7.0, 1.5, 0.06]
    P0 = Diagonal([20.0, 20.0, 2.0, 2.0, 0.05] .^ 2)
    Q = Diagonal([0.15, 0.15, 0.08, 0.08, 0.002] .^ 2)
    R = Diagonal([2.0, deg2rad(1.0)] .^ 2)
    initial_pos_error = norm(x_hat0[1:2] .- x_true[1:2])

    rng = MersenneTwister(7)
    Qchol = cholesky(Hermitian(Matrix(Q))).L
    Rchol = cholesky(Hermitian(Matrix(R))).L
    truth = [copy(x_true)]
    ys = Vector{Float64}[]
    t = 0.0
    x = copy(x_true)
    for _ in 1:nsteps
        x = coordinated_turn(t, x, Δt) .+ Qchol * randn(rng, 5)
        t += Δt
        y = range_bearing(t, x) .+ Rchol * randn(rng, 2)
        push!(truth, copy(x))
        push!(ys, y)
    end
    @test all(s -> s[1] > 50, truth)
    @test all(y -> 0 < y[2] < π / 2, ys)

    ukf = UnscentedKalmanFilter(0.0, x_hat0, P0)
    @test ukf.alpha == 1e-3
    @test ukf.beta == 2.0
    @test ukf.kappa == 0.0
    pos_err = Float64[]
    vel_err = Float64[]
    for k in 1:nsteps
        predict!(ukf, coordinated_turn, Q, Δt)
        @test posterior_is_spd(ukf.P)
        update!(ukf, range_bearing, ys[k], R)
        @test posterior_is_spd(ukf.P)
        @test ukf.t ≈ k * Δt
        push!(pos_err, norm(ukf.x[1:2] .- truth[k + 1][1:2]))
        push!(vel_err, norm(ukf.x[3:4] .- truth[k + 1][3:4]))
    end

    half = (nsteps ÷ 2 + 1):nsteps
    pos_rmse = sqrt(sum(abs2, pos_err) / length(pos_err))
    vel_rmse = sqrt(sum(abs2, vel_err) / length(vel_err))
    pos_half_rmse = sqrt(sum(abs2, pos_err[half]) / length(half))
    vel_half_rmse = sqrt(sum(abs2, vel_err[half]) / length(half))
    @test pos_err[end] < 0.25 * initial_pos_error
    @test pos_err[end] < 2.0
    @test vel_err[end] < 0.5
    @test pos_rmse < 3.0
    @test vel_rmse < 1.2
    @test pos_half_rmse < 2.5
    @test vel_half_rmse < 0.8
    @test abs(ukf.x[5] - truth[end][5]) < 0.02
end

@testset "UKF tracks the univariate nonstationary growth model" begin
    # Wan and Van der Merwe benchmark:
    # xₖ = 0.5 xₖ₋₁ + 25 xₖ₋₁ / (1 + xₖ₋₁²) + 8 cos(1.2 k) + wₖ
    # yₖ = xₖ² / 20 + vₖ
    function ungm_f(t, x, Δt)
        xk = x[1]
        return [0.5 * xk + 25 * xk / (1 + xk * xk) + 8 * cos(1.2 * (t + Δt))]
    end
    ungm_h(_t, x) = [x[1]^2 / 20]

    Δt = 1.0
    nsteps = 50
    Q = [1.0;;]
    R = [1.0;;]
    rng = MersenneTwister(4)
    x = 0.1
    t = 0.0
    truth = Float64[]
    ys = Float64[]
    for _ in 1:nsteps
        x = ungm_f(t, [x], Δt)[1] + randn(rng)
        t += Δt
        push!(truth, x)
        push!(ys, x^2 / 20 + randn(rng))
    end

    # α = 1 (with β = 2, κ = 0) is the usual setting for this example. The
    # process slope near the origin is about 25, so the default α = 1e-3
    # collapses to a local linearization, inflates P, and diverges.
    ukf = UnscentedKalmanFilter(0.0, [0.0], [1.0;;]; alpha = 1.0, beta = 2.0, kappa = 0.0)
    errors = Float64[]
    for k in 1:nsteps
        predict!(ukf, ungm_f, Q, Δt)
        update!(ukf, ungm_h, ys[k], R)
        @test posterior_is_spd(ukf.P)
        push!(errors, abs(ukf.x[1] - truth[k]))
    end
    rmse = sqrt(sum(abs2, errors) / length(errors))
    @test rmse < 10.0
    @test maximum(errors) < 25.0
    @test errors[end] < 5.0
end

end
