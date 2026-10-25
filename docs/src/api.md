# API

## Dynamics and measurements

```@docs
ProcessNoiseModel
UnbiasedRandomProcess
AdditiveProcessNoise
get_noise
DynamicsModel
NonlinearODEModel
DiscreteDynamicsModel
propagate!
EnsemblePropagation
EnsembleSerial
EnsembleThreads
EnsembleDistributed
EnsembleSplitThreads
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

## Conjugate unscented filters

```@docs
ConjugateUnscentedKalmanFilter
cut_points
SquareRootConjugateUnscentedKalmanFilter
```

## Square-root filters

```@docs
SquareRootExtendedKalmanFilter
SquareRootUnscentedKalmanFilter
SquareRootCubatureKalmanFilter
cubature_points
covariance
```

## Particle filter

```@docs
ParticleFilter
resample!
```

```@docs
predict!
update!
```

## Diagnosis

```@docs
estimation_error
rmse
block_rmse
time_averaged_rmse
nees
averaged_nees
nis
averaged_nis
innovation_autocorrelation
whiteness_bound
is_diverged
divergence_rate
mahalanobis
```
