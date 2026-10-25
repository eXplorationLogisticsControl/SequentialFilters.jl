using LinearAlgebra
using OrdinaryDiffEq
using Random
using Test

using SequentialFilters

function constant_velocity!(du, u, _p, _t)
    du[1] = u[4]
    du[2] = u[5]
    du[3] = u[6]
    du[4] = 0.0
    du[5] = 0.0
    du[6] = 0.0
    return nothing
end

function range_measurement(_t, x)
    return [hypot(x[1], x[2], x[3])]
end

function range_jacobian(_t, x)
    r = hypot(x[1], x[2], x[3])
    H = zeros(1, 6)
    H[1, 1] = x[1] / r
    H[1, 2] = x[2] / r
    H[1, 3] = x[3] / r
    return H
end

function spd_lower_factor(S)
    @test istril(S)
    @test all(>(0), diag(S))
    P = S * S'
    @test norm(P - P') < 1e-8 * max(1.0, norm(P))
    Ps = (P + P') ./ 2
    @test all(>(0), eigvals(Symmetric(Ps)))
    return Ps
end

@testset "square-root EKF matches the EKF" begin
    Δt = 2.0
    nsteps = 6
    x0 = [100.0, 40.0, 20.0, 0.2, -0.1, 0.05]
    P0 = Matrix(Diagonal([10.0, 10.0, 10.0, 0.5, 0.5, 0.5] .^ 2))
    noise = UnbiasedRandomProcess(1.0e-3)
    dynamics = NonlinearODEModel(constant_velocity!; n = 6)
    Q = get_noise(noise, Δt)
    R = [0.25;;]
    measurements = NonlinearMeasurementModel((t, x) -> (range_measurement(t, x), range_jacobian(t, x)))

    ekf = ExtendedKalmanFilter(0.0, x0, P0)
    srekf = SquareRootExtendedKalmanFilter(0.0, x0, P0)
    @test srekf.S * srekf.S' ≈ P0 atol = 1e-10
    spd_lower_factor(srekf.S)

    rng = MersenneTwister(11)
    max_x = 0.0
    max_P = 0.0
    for _ in 1:nsteps
        P_before = covariance(srekf)
        x_before = copy(srekf.x)
        sol = predict!(ekf, dynamics, noise, Δt)
        Φ = reshape(sol.u[end][7:end], 6, 6)
        predict!(srekf, (t, x, dt) -> Φ * x, Φ, Q, Δt)
        P_formula = Φ * P_before * Φ' + Q
        P_formula = (P_formula + P_formula') ./ 2
        @test srekf.x ≈ Φ * x_before atol = 1e-10 rtol = 1e-10
        @test covariance(srekf) ≈ P_formula atol = 1e-8 rtol = 1e-8
        @test srekf.x ≈ ekf.x atol = 1e-6 rtol = 1e-6
        @test covariance(srekf) ≈ ekf.P atol = 1e-6 rtol = 1e-6
        spd_lower_factor(srekf.S)
        max_x = max(max_x, norm(srekf.x .- ekf.x))
        max_P = max(max_P, norm(covariance(srekf) .- ekf.P))

        y = range_measurement(ekf.t, ekf.x) .+ 0.05 * randn(rng, 1)
        update!(ekf, measurements, y, R)
        update!(srekf, range_measurement, range_jacobian, y, R)
        @test srekf.x ≈ ekf.x atol = 1e-6 rtol = 1e-6
        @test covariance(srekf) ≈ ekf.P atol = 1e-6 rtol = 1e-6
        @test srekf.t ≈ ekf.t
        spd_lower_factor(srekf.S)
        max_x = max(max_x, norm(srekf.x .- ekf.x))
        max_P = max(max_P, norm(covariance(srekf) .- ekf.P))
    end
    @test max_x < 1e-6
    @test max_P < 1e-6

    fresh = SquareRootExtendedKalmanFilter(0.0, x0, P0)
    x_before = copy(fresh.x)
    S_before = copy(fresh.S)
    @test_throws DimensionMismatch predict!(
        fresh,
        (t, x, dt) -> x[1:5],
        Matrix{Float64}(I, 6, 6),
        Q,
        Δt,
    )
    @test fresh.x == x_before
    @test fresh.S == S_before
    @test fresh.t == 0.0
end
