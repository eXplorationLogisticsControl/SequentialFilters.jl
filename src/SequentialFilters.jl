module SequentialFilters

using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq

include("dynamics.jl")
include("measurement.jl")
include("ekf.jl")
include("ukf.jl")

export ProcessNoiseModel, UnbiasedRandomProcess, get_noise
export DynamicsModel, NonlinearODEModel, propagate!
export MeasurementModel, NonlinearMeasurementModel
export ExtendedKalmanFilter, predict!, update!
export UnscentedKalmanFilter, get_sigma_points, unscented_weights

end # module SequentialFilters
