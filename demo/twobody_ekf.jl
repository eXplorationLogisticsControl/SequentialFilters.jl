# Two-body range and range-rate EKF, same scenario as test/test_ekf.jl.
#
#   julia --startup-file=no --project=demo -e 'using Pkg; Pkg.instantiate()'
#   julia --startup-file=no --project=demo demo/twobody_ekf.jl

import Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq
using Random

using SequentialFilters

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
n_save = 9

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
dynamics = NonlinearODEModel(twobody!; n = n, p = μ)
process_noise_model = UnbiasedRandomProcess(1.0e-6)
function get_h_H(t, x)
    yhat = range_rangerate(t, x)
    H = ForwardDiff.jacobian(z -> range_rangerate(t, z), x)
    return yhat, H
end
measurements = NonlinearMeasurementModel(get_h_H)

"""Covariance `Φ P_start Φᵀ` stored in an augmented state `[x; vec(Φ)]`."""
function propagated_covariance(u, P_start, n)
    Φ = reshape(@view(u[(n + 1):end]), n, n)
    mapped = Φ * P_start * Φ'
    return (mapped + mapped') ./ 2
end

t_hist = Float64[]
err_hist = Vector{Float64}[]
sig_hist = Vector{Float64}[]

function record!(t, xhat, P)
    push!(t_hist, t)
    push!(err_hist, xhat .- truth(t))
    push!(sig_hist, sqrt.(max.(diag(P), 0.0)))
    return nothing
end

record!(ekf.t, ekf.x, ekf.P)
for k in 1:n_meas
    t0 = ekf.t
    P_start = copy(ekf.P)
    sol = predict!(
        ekf,
        dynamics,
        process_noise_model,
        Δt;
        saveat = range(t0, t0 + Δt; length = n_save),
    )
    for i in eachindex(sol.t)
        t = sol.t[i]
        (t ≈ t0 || t ≈ t0 + Δt) && continue
        record!(t, sol.u[i][1:n], propagated_covariance(sol.u[i], P_start, n))
    end
    record!(ekf.t, ekf.x, ekf.P)
    update!(ekf, measurements, ys[k], R)
    record!(ekf.t, ekf.x, ekf.P)
end

err = reduce(hcat, err_hist)'
sig = reduce(hcat, sig_hist)'
t_min = t_hist ./ 60

labels = [
    "δx (km)",
    "δy (km)",
    "δz (km)",
    "δvx (km/s)",
    "δvy (km/s)",
    "δvz (km/s)",
]
fig = Figure(size = (1000, 640))
Label(fig[0, 1:3], "Two-body EKF estimation error"; fontsize = 16)
axes = Matrix{Axis}(undef, 2, 3)
for i in 1:6
    row, col = fldmod1(i, 3)
    axes[row, col] = Axis(
        fig[row, col];
        ylabel = labels[i],
        xlabel = row == 2 ? "time (min)" : "",
    )
    σ3 = 3 .* sig[:, i]
    band!(axes[row, col], t_min, -σ3, σ3; color = (:steelblue, 0.35))
    lines!(axes[row, col], t_min, err[:, i]; color = :black, linewidth = 1.2)
end
linkxaxes!(axes...)
hidexdecorations!.(axes[1, :]; grid = false, ticks = false)
Legend(
    fig[3, 1:3],
    [PolyElement(color = (:steelblue, 0.35)), LineElement(color = :black, linewidth = 1.2)],
    ["±3σ", "error"];
    orientation = :horizontal,
    tellheight = true,
    framevisible = false,
)

outfile = joinpath(@__DIR__, "twobody_ekf_errors.png")
save(outfile, fig)
println("saved $(outfile)")
x_true = truth(n_meas * Δt)
println("final position error $(norm(ekf.x[1:3] .- x_true[1:3])) km")
println("final velocity error $(norm(ekf.x[4:6] .- x_true[4:6])) km/s")
