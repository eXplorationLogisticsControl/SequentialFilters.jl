using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq
using Random
using Test

using SequentialFilters

@testset "two-body EKF" begin
    μ = 3.986004418e5
    R_E = 6378.137
    ω_E = 7.292115146706979e-5
    a = 7000.0
    inc = deg2rad(51.6)
    lat = deg2rad(28.5)
    lon0 = 0.0

    period = 2π * sqrt(a^3 / μ)
    Δt = period / 10
    n_orbits = 2
    n_meas = 10 * n_orbits

    function twobody!(du, u, p, t)
        μp = p
        r2 = u[1] * u[1] + u[2] * u[2] + u[3] * u[3]
        μ_inv_r3 = -μp / (r2 * sqrt(r2))
        du[1] = u[4]
        du[2] = u[5]
        du[3] = u[6]
        du[4] = μ_inv_r3 * u[1]
        du[5] = μ_inv_r3 * u[2]
        du[6] = μ_inv_r3 * u[3]
        return nothing
    end

    speed = sqrt(μ / a)
    x0 = [a, 0.0, 0.0, 0.0, speed * cos(inc), speed * sin(inc)]

    truth = solve(
        ODEProblem(twobody!, x0, (0.0, n_meas * Δt), μ),
        Tsit5();
        reltol = 1e-12,
        abstol = 1e-12,
        saveat = Δt,
    )
    @test truth.t[1] ≈ 0
    @test truth.t[end] ≈ n_meas * Δt

    function station_state(t)
        cφ = cos(lat)
        sφ = sin(lat)
        λ = lon0 + ω_E * t
        cλ = cos(λ)
        sλ = sin(λ)
        rs = R_E .* (cφ * cλ, cφ * sλ, sφ)
        vs = (ω_E * R_E) .* (-cφ * sλ, cφ * cλ, 0.0)
        return rs, vs
    end

    function range_rangerate(t, x)
        rs, vs = station_state(t)
        ρx = x[1] - rs[1]
        ρy = x[2] - rs[2]
        ρz = x[3] - rs[3]
        ρ = sqrt(ρx * ρx + ρy * ρy + ρz * ρz)
        ρdot = (ρx * (x[4] - vs[1]) + ρy * (x[5] - vs[2]) + ρz * (x[6] - vs[3])) / ρ
        return [ρ, ρdot]
    end

    σ_ρ = 0.01       # 10 m
    σ_ρdot = 1.0e-5  # 1 cm/s
    R = Diagonal([σ_ρ^2, σ_ρdot^2])

    Random.seed!(1)
    meas_times = Δt .* (1:n_meas)
    ys = map(meas_times) do t
        y = range_rangerate(t, truth(t))
        return y .+ [σ_ρ, σ_ρdot] .* randn(2)
    end

    σ_r = 1.0
    σ_v = 1.0e-3
    x_hat0 = x0 .+ [1.0, -0.5, 0.8, 1.0e-3, -0.5e-3, 0.7e-3]
    P0 = Matrix(Diagonal([fill(σ_r^2, 3); fill(σ_v^2, 3)]))

    ekf = ExtendedKalmanFilter(0.0, x_hat0, P0)
    dynamics = NonlinearODEModel(twobody!; n = length(x0), p = μ)
    process_noise_model = UnbiasedRandomProcess(1.0e-6)
    function get_h_H(t, x)
        yhat = range_rangerate(t, x)
        H = ForwardDiff.jacobian(z -> range_rangerate(t, z), x)
        return yhat, H
    end
    measurements = NonlinearMeasurementModel(get_h_H)

    ekf_hist = ExtendedKalmanFilter(0.0, x_hat0, P0)
    P_start = copy(ekf_hist.P)
    sol = predict!(
        ekf_hist,
        dynamics,
        process_noise_model,
        Δt;
        saveat = range(0.0, Δt; length = 5),
    )
    n = length(x0)
    Φend = reshape(sol.u[end][(n + 1):end], n, n)
    mapped = Φend * P_start * Φend' + get_noise(process_noise_model, Δt)
    @test length(sol.t) >= 5
    @test sol.t[1] ≈ 0
    @test sol.t[end] ≈ Δt
    @test sol.u[end][1:n] ≈ ekf_hist.x
    @test ekf_hist.P ≈ (mapped + mapped') ./ 2

    for k in 1:n_meas
        predict!(ekf, dynamics, process_noise_model, Δt)
        update!(ekf, measurements, ys[k], R)
    end

    x_true = truth(n_meas * Δt)
    pos_err = norm(ekf.x[1:3] .- x_true[1:3])
    vel_err = norm(ekf.x[4:6] .- x_true[4:6])
    @test ekf.t ≈ n_meas * Δt
    @test pos_err < 0.5
    @test vel_err < 0.5e-3
    @test issymmetric(ekf.P)
    @test all(>(0), eigvals(Symmetric(ekf.P)))
end
