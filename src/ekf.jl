"""
    ExtendedKalmanFilter(t, x, P)

Continuous-discrete extended Kalman filter. `x` is the mean and `P` the covariance.
`predict!` integrates the dynamics and state-transition matrix. `update!` applies
the Joseph form.
"""
mutable struct ExtendedKalmanFilter
    t::Float64
    x::Vector
    P::Matrix

    nx::Int

    function ExtendedKalmanFilter(t::Float64, x::Vector, P::Matrix)
        nx = length(x)
        @assert size(P) == (nx, nx) "P must be a square matrix of size $nx"
        return new(Float64(t), copy(x), copy(P), nx)
    end
end


"""
Predict the state and covariance of the filter using the dynamics model.

Returns the ODE solution of the augmented state `[x; vec(Φ)]`, where `Φ` is the
state-transition matrix from the start of this predict interval. Covariance of
the continuous propagation along that solution is `P(t) = Φ(t) P_start Φ(t)ᵀ`.
Copy `filter.P` before this call to keep `P_start`; `predict!` overwrites
`filter.P`, adding process noise once at the end of the step.

`save_everystep` and `saveat` are forwarded to the integrator. The default
stores only the interval endpoints; pass `saveat` or `save_everystep=true` to
record the path used to plot `P(t)`.
"""
function predict!(
    filter::ExtendedKalmanFilter,
    dynamics_model::DynamicsModel,
    process_noise_model::ProcessNoiseModel,
    Δt::Float64;
    save_everystep::Bool = false,
    saveat = (),
)
    filter.t, sol = propagate!(
        dynamics_model,
        process_noise_model,
        filter.t,
        Float64(Δt),
        filter.x,
        filter.P;
        save_everystep = save_everystep,
        saveat = saveat,
    )
    return sol
end


"""
Update the state and covariance of the filter with a measurement.
"""
function update!(
    filter::ExtendedKalmanFilter,
    measurement_model::MeasurementModel,
    y,
    R,
)
    x = filter.x
    yhat, H = measurement_model.get_h_H(filter.t, x)
    innov = y .- yhat
    P = filter.P
    S = H * P * H' + R
    K = (P * H') / S
    x .+= K * innov
    IKH = I(filter.nx) - K * H
    updated = IKH * P * IKH' + K * R * K'
    P .= (updated + updated') ./ 2
    return
end
