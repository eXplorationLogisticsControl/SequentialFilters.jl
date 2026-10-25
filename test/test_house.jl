using LinearAlgebra
using Random
using Test

using SequentialFilters

function house_sample_moments(X, w)
    μ = zeros(size(X, 1))
    for j in axes(X, 2)
        μ .+= w[j] .* X[:, j]
    end
    P = zeros(size(X, 1), size(X, 1))
    for j in axes(X, 2)
        d = X[:, j] .- μ
        P .+= w[j] .* (d * d')
    end
    P .= (P + P') ./ 2
    L = cholesky(Hermitian(P)).L
    skew = zeros(size(X, 1))
    kurt = zeros(size(X, 1))
    for j in axes(X, 2)
        d = L \ (X[:, j] .- μ)
        for i in eachindex(d)
            di2 = d[i]^2
            skew[i] += w[j] * di2 * d[i]
            kurt[i] += w[j] * di2 * di2
        end
    end
    return μ, P, skew, kurt
end

function posterior_is_spd(P)
    issymmetric(P) || return false
    return minimum(eigvals(Symmetric(P))) > 0
end

@testset "HOUSE" begin

@testset "closed-form sigma points" begin
    X, w = house_sigma_points([0.0], [1.0;;], [0.0], [3.0])
    @test size(X) == (1, 3)
    @test X[1, :] ≈ [0.0, sqrt(3.0), -sqrt(3.0)] atol = 1e-12
    @test w ≈ [2 / 3, 1 / 6, 1 / 6] atol = 1e-12
    @test sum(w) ≈ 1 atol = 1e-12

    X, w = house_sigma_points([0.0], [1.0;;], [1.0], [3.0])
    @test X[1, :] ≈ [0.0, 2.0, -1.0] atol = 1e-12
    @test w ≈ [1 / 2, 1 / 6, 1 / 3] atol = 1e-12
    @test sum(w) ≈ 1 atol = 1e-12

    Random.seed!(3)
    n = 2
    x = [1.0, -2.0]
    A = [1.2 0.3; -0.4 0.8]
    P = A * A' + 0.3 * Matrix{Float64}(I, n, n)
    γ = [0.5, -0.4]
    κ = [3.5, 4.0]
    X, w = house_sigma_points(x, P, γ, κ)
    @test size(X) == (n, 2 * n + 1)
    @test length(w) == 2 * n + 1
    @test sum(w) ≈ 1 atol = 1e-12
    @test all(w[2:end] .> 0)
    @test w[1] > 0
    μ, Prec, skew, kurt = house_sample_moments(X, w)
    @test μ ≈ x atol = 1e-10
    @test Prec ≈ P atol = 1e-10
    @test skew ≈ γ atol = 1e-10
    @test kurt ≈ κ atol = 1e-10
end

@testset "kurtosis floor" begin
    n = 4
    x = zeros(n)
    P = Matrix{Float64}(I, n, n)
    γ = zeros(n)
    κ = fill(3.0, n)
    X, w = house_sigma_points(x, P, γ, κ; delta = 0.0)
    _, Prec, skew, kurt = house_sample_moments(X, w)
    @test sum(w) ≈ 1 atol = 1e-12
    @test w[1] ≈ 0 atol = 1e-12
    @test w[1] >= -1e-12
    @test Prec ≈ P atol = 1e-10
    @test skew ≈ γ atol = 1e-10
    @test kurt ≈ fill(4.0, n) atol = 1e-10

    X, w = house_sigma_points(x, P, γ, κ; delta = 0.2)
    _, _, _, kurt = house_sample_moments(X, w)
    @test w[1] ≈ 0.2 atol = 1e-12
    @test w[1] >= 0.2 - 1e-12
    @test kurt ≈ fill(5.0, n) atol = 1e-10

    stored = fill(3.0, n)
    filt = HigherOrderUnscentedEstimator(0.0, x, P; skewness = γ, kurtosis = stored, delta = 0.2)
    house_sigma_points(filt.x, filt.P, filt.skewness, filt.kurtosis; delta = filt.delta)
    @test filt.kurtosis == stored
    @test filt.skewness == γ
end

@testset "constructor" begin
    stored = [1.0, -2.0]
    Pc = [2.0 0.3; 0.3 1.5]
    γ = [0.1, -0.2]
    κ = [3.0, 4.0]
    filt = HigherOrderUnscentedEstimator(1.5, stored, Pc; skewness = γ, kurtosis = κ, delta = 0.1)
    stored[1] = 0.0
    Pc[1, 1] = 0.0
    γ[1] = 0.0
    κ[1] = 0.0
    @test filt.t == 1.5
    @test filt.x == [1.0, -2.0]
    @test filt.P[1, 1] == 2.0
    @test filt.skewness == [0.1, -0.2]
    @test filt.kurtosis == [3.0, 4.0]
    @test filt.nx == 2
    @test filt.delta == 0.1

    defaults = HigherOrderUnscentedEstimator(0, [0, 1], [1 0; 0 1])
    @test eltype(defaults.x) == Float64
    @test defaults.skewness == [0.0, 0.0]
    @test defaults.kurtosis == [3.0, 3.0]
    @test defaults.delta == 0.0
    @test defaults.P ≈ Matrix{Float64}(I, 2, 2)

    @test_throws ArgumentError HigherOrderUnscentedEstimator(0.0, Float64[], zeros(0, 0))
    @test_throws DimensionMismatch HigherOrderUnscentedEstimator(0.0, [0.0, 0.0], [1.0;;])
    @test_throws DimensionMismatch HigherOrderUnscentedEstimator(0.0, [0.0], [1.0;;]; skewness = [0.0, 1.0])
    @test_throws DimensionMismatch HigherOrderUnscentedEstimator(0.0, [0.0], [1.0;;]; kurtosis = [3.0, 3.0])
    @test_throws ArgumentError HigherOrderUnscentedEstimator(0.0, [0.0], [1.0;;]; kurtosis = [NaN])
    @test_throws ArgumentError HigherOrderUnscentedEstimator(0.0, [0.0], [1.0;;]; delta = -0.01)
    @test_throws ArgumentError HigherOrderUnscentedEstimator(0.0, [0.0], [1.0;;]; delta = 1)
    @test_throws ArgumentError HigherOrderUnscentedEstimator(0.0, [0.0], [1.0;;]; delta = 1.5)
    @test_throws ArgumentError house_sigma_points([0.0], [-1.0;;], [0.0], [3.0])
    @test_throws ArgumentError house_sigma_points([0.0], [1.0;;], [0.0], [Inf])
    @test_throws ArgumentError house_sigma_points([0.0], [1.0;;], [0.0], [3.0]; delta = -0.2)
    @test_throws DimensionMismatch house_sigma_points([0.0, 0.0], [1.0;;], [0.0, 0.0], [3.0, 3.0])
end

@testset "square and cube moments" begin
    square = HigherOrderUnscentedEstimator(0.0, [0.0], [1.0;;])
    sol = predict!(
        square,
        DiscreteDynamicsModel((t, x, Δt) -> [x[1]^2]; n = 1),
        AdditiveProcessNoise(zeros(1, 1)),
        1.0,
    )
    @test square.x[1] ≈ 1 atol = 1e-12
    @test square.P[1, 1] ≈ 2 atol = 1e-12
    @test square.t ≈ 1
    @test sol.t[end] ≈ 1
    @test length(sol.u[end]) == 2

    symmetric = HigherOrderUnscentedEstimator(0.0, [0.0], [1.0;;])
    skewed = HigherOrderUnscentedEstimator(0.0, [0.0], [1.0;;]; skewness = [1.0], kurtosis = [3.0])
    cube = DiscreteDynamicsModel((t, x, Δt) -> [x[1]^3]; n = 1)
    zero_noise = AdditiveProcessNoise(zeros(1, 1))
    predict!(symmetric, cube, zero_noise, 1.0)
    predict!(skewed, cube, zero_noise, 1.0)
    @test symmetric.x[1] ≈ 0 atol = 1e-12
    @test skewed.x[1] ≈ 1 atol = 1e-12

    low = HigherOrderUnscentedEstimator(0.0, [0.0], [1.0;;]; delta = 0.0)
    high = HigherOrderUnscentedEstimator(0.0, [0.0], [1.0;;]; delta = 0.9)
    @test low.kurtosis == [3.0]
    @test high.kurtosis == [3.0]
    fourth = DiscreteDynamicsModel((t, x, Δt) -> [x[1]^4]; n = 1)
    predict!(low, fourth, zero_noise, 1.0)
    predict!(high, fourth, zero_noise, 1.0)
    @test low.x[1] ≈ 3 atol = 1e-10
    @test high.x[1] ≈ 10 atol = 1e-10
end

function kalman_vs_house(; F, H, Q, R, x0, P0, nsteps, seed)
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
    house = HigherOrderUnscentedEstimator(0.0, x0, P0)
    dynamics = DiscreteDynamicsModel((t, x, Δt) -> F * x; n = nx)
    noise = AdditiveProcessNoise(Qm)
    h = (t, x) -> H * x
    max_x = 0.0
    max_P = 0.0
    max_skew = 0.0
    for k in 1:nsteps
        x_kf = F * x_kf
        P_kf = (F * P_kf * F' + Qm)
        P_kf = (P_kf + P_kf') ./ 2
        predict!(house, dynamics, noise, 1.0)
        max_x = max(max_x, norm(house.x .- x_kf))
        max_P = max(max_P, norm(house.P .- P_kf))
        max_skew = max(max_skew, norm(house.skewness))

        y = ys[k]
        innov = y .- H * x_kf
        S = H * P_kf * H' + Rm
        S = (S + S') ./ 2
        K = (P_kf * H') / S
        x_kf = x_kf .+ K * innov
        P_kf = P_kf - K * S * K'
        P_kf = (P_kf + P_kf') ./ 2
        update!(house, h, y, Rm)
        max_x = max(max_x, norm(house.x .- x_kf))
        max_P = max(max_P, norm(house.P .- P_kf))
        max_skew = max(max_skew, norm(house.skewness))
        @test posterior_is_spd(house.P)
    end
    return max_x, max_P, max_skew
end

@testset "linear-Gaussian moments match the Kalman filter" begin
    max_x, max_P, max_skew = kalman_vs_house(;
        F = [1.0;;],
        H = [1.0;;],
        Q = [0.2;;],
        R = [0.5;;],
        x0 = [0.0],
        P0 = [1.0;;],
        nsteps = 40,
        seed = 1,
    )
    @test max_x < 1e-8
    @test max_P < 1e-8
    @test max_skew < 1e-8

    Δt = 0.5
    q = 0.05
    F = [1.0 Δt; 0.0 1.0]
    Q = q * [Δt^3 / 3 Δt^2 / 2; Δt^2 / 2 Δt]
    max_x, max_P, max_skew = kalman_vs_house(;
        F = F,
        H = [1.0 0.0],
        Q = Q,
        R = [0.8;;],
        x0 = [0.0, 1.0],
        P0 = [1.0 0.0; 0.0 1.0],
        nsteps = 30,
        seed = 2,
    )
    @test max_x < 1e-8
    @test max_P < 1e-8
    @test max_skew < 1e-8

    scalar = HigherOrderUnscentedEstimator(0.0, [0.4, -0.2], [1.0 0.1; 0.1 1.0])
    vector = HigherOrderUnscentedEstimator(0.0, [0.4, -0.2], [1.0 0.1; 0.1 1.0])
    F = [1.0 0.2; 0.0 1.0]
    Qd = Diagonal([0.2, 0.05])
    dynamics = DiscreteDynamicsModel((t, x, Δt) -> F * x; n = 2)
    predict!(scalar, dynamics, AdditiveProcessNoise(Qd), 0.2; noise_skewness = 0, noise_kurtosis = 3)
    predict!(
        vector,
        dynamics,
        AdditiveProcessNoise(Matrix(Qd)),
        0.2;
        noise_skewness = [0.0, 0.0],
        noise_kurtosis = [3.0, 3.0],
    )
    @test vector.x ≈ scalar.x atol = 1e-12
    @test vector.P ≈ scalar.P atol = 1e-12
    @test vector.skewness ≈ scalar.skewness atol = 1e-12
    @test vector.kurtosis ≈ scalar.kurtosis atol = 1e-12
end

@testset "skewed measurement noise" begin
    h = (t, x) -> [x[1]]
    R = [1.0;;]
    y = [0.2]
    gauss = HigherOrderUnscentedEstimator(0.0, [0.0], [1.0;;])
    skewed = HigherOrderUnscentedEstimator(0.0, [0.0], [1.0;;])
    update!(gauss, h, y, R)
    update!(skewed, h, y, R; noise_skewness = -1.0, noise_kurtosis = 20.0)
    @test gauss.x ≈ skewed.x atol = 1e-8
    @test gauss.P ≈ skewed.P atol = 1e-8
    @test abs(gauss.skewness[1] - skewed.skewness[1]) > 0.05
    @test posterior_is_spd(gauss.P)
    @test posterior_is_spd(skewed.P)
end

@testset "failed predict and update leave the filter unchanged" begin
    house = HigherOrderUnscentedEstimator(0.0, [1.0, 2.0], Matrix{Float64}(I, 2, 2))
    x_before = copy(house.x)
    P_before = copy(house.P)
    γ_before = copy(house.skewness)
    κ_before = copy(house.kurtosis)
    @test_throws DimensionMismatch predict!(
        house,
        DiscreteDynamicsModel((t, x, Δt) -> [x[1]]; n = 1),
        AdditiveProcessNoise([0.1 0.0; 0.0 0.1]),
        0.1,
    )
    @test_throws ArgumentError predict!(
        house,
        DiscreteDynamicsModel((t, x, Δt) -> x; n = 2),
        AdditiveProcessNoise([1.0 0.0; 0.0 0.0]),
        0.1,
    )
    @test_throws DimensionMismatch predict!(
        house,
        DiscreteDynamicsModel((t, x, Δt) -> x; n = 2),
        AdditiveProcessNoise(Matrix{Float64}(I, 2, 2)),
        0.1;
        noise_skewness = [0.0],
    )
    @test_throws DimensionMismatch update!(house, (t, x) -> [x[1]], [1.0, 2.0], [1.0;;])
    @test house.x == x_before
    @test house.P == P_before
    @test house.skewness == γ_before
    @test house.kurtosis == κ_before
    @test house.t == 0.0
end

@testset "threaded predict matches serial" begin
    dynamics = DiscreteDynamicsModel((t, x, Δt) -> [x[1]^2 - x[2], x[1] * x[2]]; n = 2)
    noise = AdditiveProcessNoise(zeros(2, 2))
    P0 = [1.0 0.2; 0.2 0.5]
    serial = HigherOrderUnscentedEstimator(0.0, [0.3, -0.4], P0; skewness = [0.2, -0.1], kurtosis = [4.0, 5.0])
    threaded = HigherOrderUnscentedEstimator(0.0, [0.3, -0.4], P0; skewness = [0.2, -0.1], kurtosis = [4.0, 5.0])
    predict!(serial, dynamics, noise, 1.0)
    predict!(threaded, dynamics, noise, 1.0; ensemblealg = EnsembleThreads())
    @test threaded.x ≈ serial.x atol = 1e-12
    @test threaded.P ≈ serial.P atol = 1e-12
    @test threaded.skewness ≈ serial.skewness atol = 1e-12
    @test threaded.kurtosis ≈ serial.kurtosis atol = 1e-12
end

@testset "short ODE predict" begin
    function oscillator!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
        return nothing
    end
    filt = HigherOrderUnscentedEstimator(0.0, [1.0, 0.0], Matrix(0.1 * I(2)))
    sol = predict!(
        filt,
        NonlinearODEModel(oscillator!; n = 2),
        AdditiveProcessNoise(1.0e-4 * Matrix{Float64}(I, 2, 2)),
        0.1;
        saveat = range(0.0, 0.1; length = 3),
    )
    @test sol.t[end] ≈ 0.1
    @test length(sol.u[end]) == 2 + 4
    @test filt.t ≈ 0.1
    @test posterior_is_spd(filt.P)
    @test all(isfinite, filt.skewness)
    @test all(isfinite, filt.kurtosis)
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

@testset "HOUSE tracks a coordinated turn" begin
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

    house = HigherOrderUnscentedEstimator(0.0, x_hat0, P0)
    pos_err = Float64[]
    vel_err = Float64[]
    for k in 1:nsteps
        predict!(
            house,
            DiscreteDynamicsModel(coordinated_turn; n = 5),
            AdditiveProcessNoise(Q),
            Δt,
        )
        @test posterior_is_spd(house.P)
        update!(house, range_bearing, ys[k], R)
        @test posterior_is_spd(house.P)
        @test house.t ≈ k * Δt
        push!(pos_err, norm(house.x[1:2] .- truth[k + 1][1:2]))
        push!(vel_err, norm(house.x[3:4] .- truth[k + 1][3:4]))
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
    @test abs(house.x[5] - truth[end][5]) < 0.02
end

end
