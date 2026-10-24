"""
Measurement model.

[`NonlinearMeasurementModel`](@ref) returns the predicted measurement and its Jacobian.
"""
abstract type MeasurementModel end

"""
Nonlinear measurement model.

`get_h_H(t, x)` returns the predicted measurement `yhat` and its Jacobian `H`.
"""
struct NonlinearMeasurementModel <: MeasurementModel
    get_h_H::Function
end
