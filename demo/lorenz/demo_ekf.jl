"""
Lorenz-63 demo using the extended Kalman filter.

The vector field, step, noise, and measurements live in `demo/lorenz/lorenz63.jl`.
The script saves the error figure and then prints elapsed time, process CPU
time, and allocations for this scenario.

    julia --project=demo demo/lorenz/demo_ekf.jl
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using LinearAlgebra
using OrdinaryDiffEq
using Random

using SequentialFilters

import SequentialFilters: predict!, update!

include(joinpath(@__DIR__, "lorenz63.jl"))

case = load_lorenz_case()
Δt = case.Δt
n_steps = case.n_steps
n_save = 9
n = 3
truth = case.truth
ys = case.ys
R = case.R
dynamics = case.dynamics
process_noise_model = case.process_noise
measurements = case.measurements


println("Lorenz-63: σ = 10, ρ = 28, β = 8/3, Δt = $(Δt), steps = $(n_steps)")
println("x0 = $(case.x0) is off the attractor; nonzero equilibria are near (±8.49, ±8.49, 27)")
println("measurement is (x, z) with σ = $(case.σ_meas), seed 1; additive process noise Q = 1e-3 I")


filt = ExtendedKalmanFilter(0.0, case.x_hat0, case.P0)

"""Covariance `Φ P_start Φᵀ + Q(Δt_elapsed)` stored in an augmented state `[x; vec(Φ)]`.

`Q` is `get_noise(process_noise_model, Δt_elapsed)` from the start of this predict step.
"""
function propagated_covariance(u, P_start, n, process_noise_model, Δt_elapsed)
    Φ = reshape(@view(u[(n + 1):end]), n, n)
    Q = get_noise(process_noise_model, Float64(Δt_elapsed))
    mapped = Φ * P_start * Φ' + Q
    return (mapped + mapped') ./ 2
end

t_hist = Float64[]
err_hist = Vector{Float64}[]
sig_hist = Vector{Float64}[]
post_errors = Vector{Float64}[]

function record!(t, xhat, P)
    push!(t_hist, t)
    push!(err_hist, xhat .- truth(t))
    push!(sig_hist, sqrt.(max.(diag(P), 0.0)))
    return nothing
end

record!(filt.t, filt.x, filt.P)
for k in 1:n_steps
    t0 = filt.t
    P_start = copy(filt.P)
    sol = predict!(
        filt,
        dynamics,
        process_noise_model,
        Δt;
        saveat = range(t0, t0 + Δt; length = n_save),
    )
    for i in eachindex(sol.t)
        t = sol.t[i]
        (t ≈ t0 || t ≈ t0 + Δt) && continue
        record!(t, sol.u[i][1:n], propagated_covariance(sol.u[i], P_start, n, process_noise_model, t - t0))
    end
    record!(filt.t, filt.x, filt.P)
    update!(filt, measurements, ys[k], R)
    record!(filt.t, filt.x, filt.P)
    push!(post_errors, err_hist[end])
end

err = reduce(hcat, err_hist)'
sig = reduce(hcat, sig_hist)'

labels = ["δx", "δy", "δz"]
fig = Figure(size = (1000, 380))
Label(fig[0, 1:3], "Lorenz-63 EKF estimation error"; fontsize = 16)
axes = Matrix{Axis}(undef, 1, 3)
for i in 1:3
    axes[1, i] = Axis(fig[1, i]; ylabel = labels[i], xlabel = "time")
    σ3 = 3 .* sig[:, i]
    band!(axes[1, i], t_hist, -σ3, σ3; color = (:steelblue, 0.35))
    lines!(axes[1, i], t_hist, err[:, i]; color = :black, linewidth = 1.2)
end
linkxaxes!(axes...)
Legend(
    fig[2, 1:3],
    [PolyElement(color = (:steelblue, 0.35)), LineElement(color = :black, linewidth = 1.2)],
    ["±3σ", "error"];
    orientation = :horizontal,
    tellheight = true,
    framevisible = false,
)

outfile = joinpath(@__DIR__, "plots", "lorenz_ekf_errors.png")
mkpath(dirname(outfile))
save(outfile, fig)
println("saved $(outfile)")
report_lorenz_errors(post_errors)

include(joinpath(@__DIR__, "..", "benchmark_filters.jl"))
include(joinpath(@__DIR__, "lorenz_benchmark.jl"))
print_lorenz_costs(case, :ekf)
