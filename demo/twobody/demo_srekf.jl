"""
Demo with two-body dynamics using the square-root extended Kalman filter.

Dynamics, measurements, and noise live in `twobody.jl`.
The script saves the error figure and then prints elapsed time, process CPU
time, and allocations for this scenario.
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using LinearAlgebra

using SequentialFilters

import SequentialFilters: predict!, update!, covariance

include(joinpath(@__DIR__, "twobody.jl"))

filt = SquareRootExtendedKalmanFilter(0.0, x_hat0, P0)

t_hist = Float64[]
err_hist = Vector{Float64}[]
sig_hist = Vector{Float64}[]

function record!(t, xhat, P)
    push!(t_hist, t)
    push!(err_hist, xhat .- truth(t))
    push!(sig_hist, sqrt.(max.(diag(P), 0.0)))
    return nothing
end

record!(filt.t, filt.x, covariance(filt))
for k in 1:n_meas
    t0 = filt.t
    P_start = copy(covariance(filt))
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
    record!(filt.t, filt.x, covariance(filt))
    update!(filt, measurements, ys[k], R)
    record!(filt.t, filt.x, covariance(filt))
end

err = reduce(hcat, err_hist)'
sig = reduce(hcat, sig_hist)'
t_min = t_hist ./ 60

fig = Figure(size = (1000, 640))
Label(fig[0, 1:3], "Two-body square-root EKF estimation error"; fontsize = 16)
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

outfile = joinpath(@__DIR__, "plots", "twobody_srekf_errors.png")
mkpath(dirname(outfile))
save(outfile, fig)
println("saved $(outfile)")
x_true = truth(n_meas * Δt)
println("final position error $(norm(filt.x[1:3] .- x_true[1:3])) km")
println("final velocity error $(norm(filt.x[4:6] .- x_true[4:6])) km/s")

include(joinpath(@__DIR__, "..", "benchmark_filters.jl"))
print_filter_costs(load_twobody_case(), :ekf)
