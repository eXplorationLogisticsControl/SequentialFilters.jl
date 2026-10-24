"""Unscented Kalman filter struct"""

mutable struct UnscentedKalmanFilter
    t::Float64
    x::Vector
    P::Matrix

    nx::Int

    function UnscentedKalmanFilter(t::Float64, x::Vector, P::Matrix)
        nx = length(x)
        @assert size(P) == (nx, nx) "P must be a square matrix of size $nx"
        return new(Float64(t), copy(x), copy(P), nx)
    end
end


function get_sigma_points(x::Vector, P::Matrix)
end


function predict!(
    filter::UnscentedKalmanFilter,
    dynamics_model::DynamicsModel,
    process_noise_model::ProcessNoiseModel,
    Δt::Float64;
    save_everystep::Bool = false,
    saveat = (),
)
end


function update!(
    filter::ExtendedKalmanFilter,
    measurement_model::MeasurementModel,
    y,
    R,
)
end