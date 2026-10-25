# API

## Dynamics and measurements

```@docs
ProcessNoiseModel
UnbiasedRandomProcess
get_noise
DynamicsModel
NonlinearODEModel
propagate!
MeasurementModel
NonlinearMeasurementModel
```

## Extended Kalman filter

```@docs
ExtendedKalmanFilter
```

## Unscented Kalman filter

```@docs
UnscentedKalmanFilter
unscented_weights
get_sigma_points
```

## Square-root filters

```@docs
SquareRootExtendedKalmanFilter
SquareRootUnscentedKalmanFilter
covariance
```

```@docs
predict!
update!
```
