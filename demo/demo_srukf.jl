"""
Two-body orbit tracking with the square-root unscented Kalman filter.

Dynamics, initial conditions, sensor, and noise match `demo/demo_twobody_ekf.jl`.
The figure shows error = estimate − truth and ±3√P[i,i], with P = S*S'.

    julia --project=demo demo/demo_srukf.jl
"""

import Pkg
Pkg.activate(@__DIR__)

ENV["GKSwstype"] = "nul"

using LinearAlgebra
using Printf

using SequentialFilters

include(joinpath(@__DIR__, "twobody_scenario.jl"))
include(joinpath(@__DIR__, "plot_errors.jl"))
import SequentialFilters: predict!, update!, covariance

function run_demo()
    case = load_twobody_case()
    filt = SquareRootUnscentedKalmanFilter(0.0, case.x_hat0, case.P0)
    t_hist = Float64[]
    err_hist = Vector{Float64}[]
    sig_hist = Vector{Float64}[]
    push_error!(t_hist, err_hist, sig_hist, filt.t, filt.x, case.truth(filt.t), covariance(filt))
    for k in 1:case.n_meas
        predict!(filt, case.f, case.Q, case.Δt)
        push_error!(
            t_hist,
            err_hist,
            sig_hist,
            filt.t,
            filt.x,
            case.truth(filt.t),
            covariance(filt),
        )
        update!(filt, case.h, case.ys[k], case.R)
        push_error!(
            t_hist,
            err_hist,
            sig_hist,
            filt.t,
            filt.x,
            case.truth(filt.t),
            covariance(filt),
        )
    end

    err = reduce(hcat, err_hist)'
    sig = reduce(hcat, sig_hist)'
    outfile = joinpath(@__DIR__, "srukf_twobody_errors.png")
    save_error_figure(
        outfile,
        t_hist ./ 60,
        err,
        sig,
        case.labels,
        "Two-body square-root UKF\nerror = estimate − truth,  bands = ±3 √P[i,i]",
    )
    print_twobody_summary("square-root UKF two-body demo", err_hist, outfile)
    return nothing
end

run_demo()
