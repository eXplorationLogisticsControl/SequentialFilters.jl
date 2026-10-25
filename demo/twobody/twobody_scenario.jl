"""
Two-body demo case shared by the UKF and square-root demos.

Constants, dynamics, sensor, noise, truth, and measurements match
`demo/twobody/demo_ekf.jl`.
"""

using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq
using Random

using SequentialFilters

function load_twobody_case()
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
    n = length(x0)

    truth = solve(
        ODEProblem(twobody!, x0, (0.0, n_meas * Δt), μ),
        Tsit5();
        reltol = 1e-12,
        abstol = 1e-12,
        saveat = Δt,
    )

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
    ys = map(Δt .* (1:n_meas)) do t
        y = range_rangerate(t, truth(t))
        return y .+ [σ_ρ, σ_ρdot] .* randn(2)
    end

    σ_r = 1.0
    σ_v = 1.0e-3
    x_hat0 = x0 .+ [1.0, -0.5, 0.8, 1.0e-3, -0.5e-3, 0.7e-3]
    P0 = Matrix(Diagonal([fill(σ_r^2, 3); fill(σ_v^2, 3)]))

    dynamics = NonlinearODEModel(twobody!; n = n, p = μ)
    process_noise = UnbiasedRandomProcess(1.0e-6)
    Q = get_noise(process_noise, Δt)

    cache = Ref{NamedTuple}((; valid = false))
    function step_dynamics(t, x, Δt_step)
        x64 = collect(Float64, x)
        t64 = Float64(t)
        dt64 = Float64(Δt_step)
        hit = cache[]
        if hit.valid && hit.t == t64 && hit.Δt == dt64 && hit.x == x64
            # One predict calls both `f` and `F`. Consume the slot so a later
            # run with the same state integrates again.
            cache[] = (; valid = false)
            return copy(hit.xnext), copy(hit.Φ)
        end
        xwork = copy(x64)
        Pwork = Matrix{Float64}(I, n, n)
        _, sol = propagate!(dynamics, process_noise, t64, dt64, xwork, Pwork)
        Φ = reshape(copy(sol.u[end][(n + 1):end]), n, n)
        xnext = copy(xwork)
        cache[] = (; valid = true, t = t64, Δt = dt64, x = x64, xnext = xnext, Φ = Φ)
        return copy(xnext), copy(Φ)
    end

    f(t, x, dt) = step_dynamics(t, x, dt)[1]
    F(t, x, dt) = step_dynamics(t, x, dt)[2]
    h(t, x) = range_rangerate(t, x)
    H(t, x) = ForwardDiff.jacobian(z -> range_rangerate(t, z), x)
    measurements = NonlinearMeasurementModel((t, x) -> (h(t, x), H(t, x)))

    labels = [
        "δx (km)",
        "δy (km)",
        "δz (km)",
        "δvx (km/s)",
        "δvy (km/s)",
        "δvz (km/s)",
    ]

    return (;
        x0, x_hat0, P0, R, Q, Δt, n_meas, truth, ys, f, F, h, H, labels,
        dynamics, process_noise, measurements,
    )
end
