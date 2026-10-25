"""
Two-body orbit tracking with the scaled unscented Kalman filter.

Dynamics, initial conditions, sensor, and noise match `demo/demo_twobody_ekf.jl`.
The figure shows error = estimate − truth and ±3√P[i,i].

    julia --project=demo demo/demo_ukf.jl
"""

import Pkg
Pkg.activate(@__DIR__)

ENV["GKSwstype"] = "nul"

using LinearAlgebra
using Printf

using SequentialFilters

include(joinpath(@__DIR__, "twobody_scenario.jl"))
include(joinpath(@__DIR__, "plot_errors.jl"))
import SequentialFilters: predict!, update!

function run_demo()
    case = load_twobody_case()
    ukf = UnscentedKalmanFilter(0.0, case.x_hat0, case.P0)
    t_hist = Float64[]
    err_hist = Vector{Float64}[]
    sig_hist = Vector{Float64}[]
    push_error!(t_hist, err_hist, sig_hist, ukf.t, ukf.x, case.truth(ukf.t), ukf.P)
    for k in 1:case.n_meas
        predict!(ukf, case.f, case.Q, case.Δt)
        push_error!(t_hist, err_hist, sig_hist, ukf.t, ukf.x, case.truth(ukf.t), ukf.P)
        update!(ukf, case.h, case.ys[k], case.R)
        push_error!(t_hist, err_hist, sig_hist, ukf.t, ukf.x, case.truth(ukf.t), ukf.P)
    end

    err = reduce(hcat, err_hist)'
    sig = reduce(hcat, sig_hist)'
    outfile = joinpath(@__DIR__, "ukf_twobody_errors.png")
    save_error_figure(
        outfile,
        t_hist ./ 60,
        err,
        sig,
        case.labels,
        "Two-body UKF\nerror = estimate − truth,  bands = ±3 √P[i,i]",
    )
    print_twobody_summary("UKF two-body demo", err_hist, outfile)
    return nothing
end

run_demo()
