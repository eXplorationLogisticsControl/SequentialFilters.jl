"""
Two-body case shared by the filter demos.

Include this file to define the orbit, ground station, range and range-rate
measurements, noise, truth, and the names the demos pass to `predict!` and
`update!`. `load_twobody_case()` returns that same case. It does not draw a
new measurement sequence or integrate a new truth.
"""

using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq
using Random

using SequentialFilters

const μ = 3.986004418e5
const R_E = 6378.137
const ω_E = 7.292115146706979e-5
const a = 7000.0
const inc = deg2rad(51.6)
const lat = deg2rad(28.5)
const lon0 = 0.0

function twobody!(du, u, p, t)
    du[1:3] = u[4:6]
    du[4:6] = -μ * u[1:3] / norm(u[1:3])^3
    return nothing
end

const period = 2π * sqrt(a^3 / μ)
const Δt = period / 10
const n_orbits = 2
const n_meas = 10 * n_orbits
const n_save = 9

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

function get_h_H(t, x)
    yhat = range_rangerate(t, x)
    jac = ForwardDiff.jacobian(z -> range_rangerate(t, z), x)
    return yhat, jac
end

const speed = sqrt(μ / a)
const x0 = [a, 0.0, 0.0, 0.0, speed * cos(inc), speed * sin(inc)]
const n = length(x0)

const truth = solve(
    ODEProblem(twobody!, x0, (0.0, n_meas * Δt), μ),
    Tsit5();
    reltol = 1e-12,
    abstol = 1e-12,
    dense = true,
)

const σ_ρ = 0.01       # 10 m
const σ_ρdot = 1.0e-5  # 1 cm/s
const R = Diagonal([σ_ρ^2, σ_ρdot^2])

Random.seed!(1)
const ys = map(Δt .* (1:n_meas)) do t
    y = range_rangerate(t, truth(t))
    return y .+ [σ_ρ, σ_ρdot] .* randn(2)
end

const σ_r = 1.0
const σ_v = 1.0e-3
const x_hat0 = x0 .+ [1.0, -0.5, 0.8, 1.0e-3, -0.5e-3, 0.7e-3]
const P0 = Matrix(Diagonal([fill(σ_r^2, 3); fill(σ_v^2, 3)]))

const dynamics = NonlinearODEModel(twobody!; n = n, p = μ)
const process_noise_model = UnbiasedRandomProcess(1.0e-6)
const Q = get_noise(process_noise_model, Δt)

const h = range_rangerate
H(t, x) = get_h_H(t, x)[2]
const measurements = NonlinearMeasurementModel(get_h_H)

const labels = [
    "δx (km)",
    "δy (km)",
    "δz (km)",
    "δvx (km/s)",
    "δvy (km/s)",
    "δvz (km/s)",
]

"""Covariance `Φ P_start Φᵀ + Q(Δt_elapsed)` stored in an augmented state `[x; vec(Φ)]`.

`Q` is `get_noise(process_noise_model, Δt_elapsed)` from the start of this predict step.
"""
function propagated_covariance(u, P_start, n, process_noise_model, Δt_elapsed)
    Φ = reshape(@view(u[(n + 1):end]), n, n)
    Q = get_noise(process_noise_model, Float64(Δt_elapsed))
    mapped = Φ * P_start * Φ' + Q
    return (mapped + mapped') ./ 2
end

"""Return the two-body case built when this file was included."""
function load_twobody_case()
    return (;
        x0, x_hat0, P0, R, Q, Δt, n_meas, truth, ys, h, H, labels,
        dynamics, process_noise = process_noise_model, measurements,
    )
end
