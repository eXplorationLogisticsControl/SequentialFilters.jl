module SequentialFilters

using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq

include("dynamics.jl")
include("measurement.jl")
include("ekf.jl")

export ProcessNoiseModel, UnbiasedRandomProcess, get_noise
export DynamicsModel, NonlinearODEModel, propagate!
export MeasurementModel, NonlinearMeasurementModel
export ExtendedKalmanFilter, predict!, update!

end # module SequentialFilters
