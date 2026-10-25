# Square-root CUT benchmarks

Monte Carlo size M = 200. Seed = 20261025.
Every filter in a setting uses the same draws. Float32 uses those Float64 draws and rounds inside the filter.
The posterior CRLB is the Tichavský recursion on each Monte Carlo truth, averaged as a covariance.
Factor failure means the covariance update is not positive definite, or a square-root QR or downdate threw.
Divergence is a factor failure or a position error above the threshold (1e5 km, or 100 times the position prior, on the ill-conditioned case; 5000 km on the long arc).
Reacquisition means the run finished and the final position error is under 25 km or under half the error at the end of the coast.
Runtime is wall time per scheduled filter step after one untimed warmup, averaged over runs.
RMSE and NEES skip non-finite samples, so a run that fails mid-arc still contributes the epochs before the failure. The failure rate is the companion column.
The linear-Gaussian unit test matches SR-CUT to full-covariance CUT within 100 ε(Float64).

## Ill-conditioned case

State `[x, y, vx, vy, β]` in km, km/s, and 1/km. Two ranges, each with σ = 1 cm.
Ratio is the position prior standard deviation divided by that range standard deviation.
CUT and SR-CUT of orders 4 and 6, and SR-CKF, run in Float64 and Float32. UKF and SR-UKF are Float64 baselines.
The time-history figure uses ratio 100000.0, the setting with the largest CUT minus SR-CUT factor-failure gap over orders 4 and 6 and both precisions (gap = 0.015).

| ratio | precision | filter | n_σ | pos. RMSE (km) | β RMSE | NEES | NEES in band | factor fail | diverge | s/step | mean log10 cond | RMSE/PCRLB |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e+03 | Float64 | CUT-4 | 42 | 1.49852e-05 | 0.0284251 | 4.97913 | 1 | 0.000 | 0.000 | 5.559e-05 | 5.90573 | 1.00793 |
| 1e+03 | Float64 | SR-CUT-4 | 42 | 1.49852e-05 | 0.0284251 | 4.97913 | 1 | 0.000 | 0.000 | 5.223e-05 | 5.90573 | 1.00793 |
| 1e+03 | Float64 | CUT-6 | 83 | 1.49852e-05 | 0.0284252 | 4.97794 | 1 | 0.000 | 0.000 | 5.398e-05 | 5.90533 | 1.00793 |
| 1e+03 | Float64 | SR-CUT-6 | 83 | 1.49852e-05 | 0.0284252 | 4.97794 | 1 | 0.000 | 0.000 | 5.119e-05 | 5.90533 | 1.00793 |
| 1e+03 | Float64 | SR-CKF | 10 | 1.49852e-05 | 0.0284347 | 5.0568 | 1 | 0.000 | 0.000 | 2.077e-05 | 5.88934 | 1.00793 |
| 1e+03 | Float32 | CUT-4 | 42 | 1.8971e-05 | 0.0264796 | 5.51564 | 0.85 | 0.930 | 0.930 | 7.997e-06 | 5.77883 | 1.27602 |
| 1e+03 | Float32 | SR-CUT-4 | 42 | 1.97861e-05 | 0.0289672 | 5.60306 | 0.9 | 0.935 | 0.935 | 3.645e-06 | 5.78386 | 1.33085 |
| 1e+03 | Float32 | CUT-6 | 83 | 2.35853e-05 | 0.0253358 | 6.10444 | 0.95 | 0.990 | 0.990 | 3.022e-06 | 5.76034 | 1.58639 |
| 1e+03 | Float32 | SR-CUT-6 | 83 | 2.39464e-05 | 0.0286503 | 7.39833 | 0.85 | 0.990 | 0.990 | 2.982e-06 | 5.75169 | 1.61068 |
| 1e+03 | Float32 | SR-CKF | 10 | 1.54127e-05 | 0.0210762 | 5.0331 | 0.8 | 0.975 | 0.975 | 1.414e-06 | 5.85092 | 1.03668 |
| 1e+03 | Float64 | UKF | 11 | 1.49852e-05 | 0.028434 | 5.06439 | 1 | 0.000 | 0.000 | 3.111e-05 | 5.89263 | 1.00793 |
| 1e+03 | Float64 | SR-UKF | 11 | 1.49852e-05 | 0.028434 | 5.0644 | 1 | 0.000 | 0.000 | 2.397e-05 | 5.89263 | 1.00793 |
| 1e+04 | Float64 | CUT-4 | 42 | 4.15506e-05 | 0.0301943 | 4.96688 | 0.85 | 0.000 | 0.000 | 3.759e-05 | 5.90146 | 2.79477 |
| 1e+04 | Float64 | SR-CUT-4 | 42 | 4.15506e-05 | 0.0301943 | 4.96688 | 0.85 | 0.000 | 0.000 | 3.069e-05 | 5.90146 | 2.79477 |
| 1e+04 | Float64 | CUT-6 | 83 | 4.15506e-05 | 0.0301943 | 4.96643 | 0.85 | 0.000 | 0.000 | 5.229e-05 | 5.90139 | 2.79477 |
| 1e+04 | Float64 | SR-CUT-6 | 83 | 4.15506e-05 | 0.0301943 | 4.96643 | 0.85 | 0.000 | 0.000 | 4.997e-05 | 5.90139 | 2.79477 |
| 1e+04 | Float64 | SR-CKF | 10 | 4.15312e-05 | 0.0303725 | 4.95126 | 0.85 | 0.000 | 0.000 | 1.634e-05 | 5.90962 | 2.79346 |
| 1e+04 | Float32 | CUT-4 | 42 | 4.53512e-05 | 0.0304925 | 5.36289 | 0.8 | 0.845 | 0.845 | 5.612e-06 | 5.80805 | 3.05041 |
| 1e+04 | Float32 | SR-CUT-4 | 42 | 4.50206e-05 | 0.032607 | 5.26395 | 0.75 | 0.855 | 0.855 | 8.028e-06 | 5.80926 | 3.02817 |
| 1e+04 | Float32 | CUT-6 | 83 | 4.93099e-05 | 0.0282884 | 5.62833 | 0.75 | 0.885 | 0.885 | 1.093e-05 | 5.78738 | 3.31667 |
| 1e+04 | Float32 | SR-CUT-6 | 83 | 5.06694e-05 | 0.0326693 | 5.73145 | 0.7 | 0.900 | 0.900 | 9.366e-06 | 5.80165 | 3.40812 |
| 1e+04 | Float32 | SR-CKF | 10 | 4.173e-05 | 0.0297105 | 5.22657 | 0.8 | 0.945 | 0.945 | 2.131e-06 | 5.89263 | 2.80684 |
| 1e+04 | Float64 | UKF | 11 | 4.15294e-05 | 0.036018 | 60.3018 | 0 | 0.000 | 0.000 | 1.499e-05 | 5.9698 | 2.79335 |
| 1e+04 | Float64 | SR-UKF | 11 | 4.15294e-05 | 0.0360179 | 60.3018 | 0 | 0.000 | 0.000 | 1.600e-05 | 5.9698 | 2.79334 |
| 1e+05 | Float64 | CUT-4 | 42 | 0.00310557 | 0.0409713 | 5.41773 | 0.8 | 0.000 | 0.000 | 2.821e-05 | 5.81747 | 208.886 |
| 1e+05 | Float64 | SR-CUT-4 | 42 | 0.00310557 | 0.0409713 | 5.41773 | 0.8 | 0.000 | 0.000 | 3.013e-05 | 5.81747 | 208.886 |
| 1e+05 | Float64 | CUT-6 | 83 | 0.00310526 | 0.0409639 | 5.41118 | 0.8 | 0.000 | 0.000 | 5.199e-05 | 5.81751 | 208.865 |
| 1e+05 | Float64 | SR-CUT-6 | 83 | 0.00310526 | 0.0409639 | 5.41118 | 0.8 | 0.000 | 0.000 | 4.966e-05 | 5.81751 | 208.865 |
| 1e+05 | Float64 | SR-CKF | 10 | 0.00310208 | 0.0410433 | 5.40523 | 0.8 | 0.000 | 0.000 | 1.645e-05 | 5.8308 | 208.651 |
| 1e+05 | Float32 | CUT-4 | 42 | 0.00310717 | 0.0374266 | 7.32766 | 0.9 | 0.915 | 0.915 | 4.472e-06 | 5.72865 | 208.994 |
| 1e+05 | Float32 | SR-CUT-4 | 42 | 0.00311201 | 0.0369907 | 7.20912 | 0.85 | 0.900 | 0.900 | 7.210e-06 | 5.6928 | 209.319 |
| 1e+05 | Float32 | CUT-6 | 83 | 0.00311276 | 0.0358645 | 5.40551 | 0.85 | 0.935 | 0.935 | 9.695e-06 | 5.67948 | 209.37 |
| 1e+05 | Float32 | SR-CUT-6 | 83 | 0.00311479 | 0.039488 | 4.63913 | 0.7 | 0.935 | 0.935 | 8.784e-06 | 5.69209 | 209.507 |
| 1e+05 | Float32 | SR-CKF | 10 | 0.00310397 | 0.0478416 | 8.60335 | 0.1 | 0.920 | 0.920 | 2.606e-06 | 5.80331 | 208.779 |
| 1e+05 | Float64 | UKF | 11 | 0.00314375 | 5.12129 | 1.56956e+06 | 0 | 0.000 | 0.000 | 1.431e-05 | 5.9835 | 211.454 |
| 1e+05 | Float64 | SR-UKF | 11 | 0.00314375 | 5.12133 | 1.56973e+06 | 0 | 0.000 | 0.000 | 1.610e-05 | 5.9835 | 211.454 |
| 1e+06 | Float64 | CUT-4 | 42 | 11.6836 | 0.68298 | 2.07695e+11 | 0 | 0.005 | 0.005 | 2.797e-05 | 6.00754 | 785858 |
| 1e+06 | Float64 | SR-CUT-4 | 42 | 11.6836 | 0.68298 | 2.07695e+11 | 0 | 0.005 | 0.005 | 3.004e-05 | 6.00754 | 785858 |
| 1e+06 | Float64 | CUT-6 | 83 | 1.94289e+06 | 0.235623 | 2.08096e+11 | 0 | 0.005 | 0.005 | 5.170e-05 | 5.9843 | 1.30683e+11 |
| 1e+06 | Float64 | SR-CUT-6 | 83 | 722.404 | 0.235659 | 2.07998e+11 | 0 | 0.005 | 0.005 | 5.007e-05 | 5.98424 | 4.85902e+07 |
| 1e+06 | Float64 | SR-CKF | 10 | 24354 | 22.2835 | 3.85578e+11 | 0 | 0.195 | 0.195 | 1.496e-05 | 6.14771 | 1.63809e+09 |
| 1e+06 | Float32 | CUT-4 | 42 | 6.93236 | 0.37956 | 4.39377e+09 | 0 | 0.950 | 0.950 | 7.432e-06 | 5.76671 | 466283 |
| 1e+06 | Float32 | SR-CUT-4 | 42 | 11.2804 | 0.207765 | 5.00963e+11 | 0 | 0.950 | 0.950 | 1.038e-05 | 5.78911 | 758740 |
| 1e+06 | Float32 | CUT-6 | 83 | 8.06273 | 0.125318 | 8.02775e+08 | 0 | 0.960 | 0.960 | 1.654e-05 | 5.63405 | 542313 |
| 1e+06 | Float32 | SR-CUT-6 | 83 | 10.6701 | 0.141677 | 2.49925e+11 | 0 | 0.960 | 0.960 | 1.331e-05 | 5.72251 | 717690 |
| 1e+06 | Float32 | SR-CKF | 10 | 5.71008 | 2.88454 | 7.52058e+09 | 0 | 0.985 | 0.985 | 5.496e-06 | 6.08055 | 384070 |
| 1e+06 | Float64 | UKF | 11 | 8896.14 | 189.335 | 5.41662e+11 | 0 | 0.705 | 0.705 | 6.369e-06 | 5.58669 | 5.98371e+08 |
| 1e+06 | Float64 | SR-UKF | 11 | 25332.9 | 192.935 | 5.28399e+11 | 0 | 0.665 | 0.665 | 7.210e-06 | 5.6001 | 1.70393e+09 |
| 1e+07 | Float64 | CUT-4 | 42 | 4777.8 | 65.6033 | 5.05639e+11 | 0 | 0.565 | 0.565 | 1.950e-05 | 7.01553 | 3.21363e+08 |
| 1e+07 | Float64 | SR-CUT-4 | 42 | 40660.2 | 70.3047 | 5.38588e+11 | 0 | 0.560 | 0.565 | 2.008e-05 | 7.01586 | 2.73488e+09 |
| 1e+07 | Float64 | CUT-6 | 83 | 458124 | 78.609 | 3.00156e+11 | 0 | 0.370 | 0.370 | 4.614e-05 | 7.67352 | 3.08143e+10 |
| 1e+07 | Float64 | SR-CUT-6 | 83 | 158603 | 180.277 | 3.37924e+11 | 0 | 0.370 | 0.370 | 5.580e-05 | 7.67258 | 1.06679e+10 |
| 1e+07 | Float64 | SR-CKF | 10 | 35384.8 | 1227.32 | 1.5259e+12 | 0 | 0.905 | 0.905 | 5.204e-06 | 6.16293 | 2.38005e+09 |
| 1e+07 | Float32 | CUT-4 | 42 | 55.4021 | 1.73388 | 3.96148e+10 | 0 | 0.885 | 0.885 | 1.411e-05 | 7.33432 | 3.72645e+06 |
| 1e+07 | Float32 | SR-CUT-4 | 42 | 55.6022 | 16.2727 | 4.96119e+10 | 0 | 0.870 | 0.870 | 1.488e-05 | 7.29977 | 3.7399e+06 |
| 1e+07 | Float32 | CUT-6 | 83 | 29.6216 | 15.4342 | 1.87675e+10 | 0 | 0.530 | 0.530 | 3.260e-05 | 7.75248 | 1.9924e+06 |
| 1e+07 | Float32 | SR-CUT-6 | 83 | 29.6658 | 12.1852 | 1.92424e+10 | 0 | 0.525 | 0.525 | 2.725e-05 | 7.7527 | 1.99537e+06 |
| 1e+07 | Float32 | SR-CKF | 10 | 401.219 | 534.454 | 4.05328e+10 | 0 | 1.000 | 1.000 | 2.873e-06 | 6.79209 | 2.69868e+07 |
| 1e+07 | Float64 | UKF | 11 | 18029.9 | 153.064 | 1.59799e+12 | 0 | 0.980 | 0.980 | 2.357e-06 | 5.76485 | 1.21272e+09 |
| 1e+07 | Float64 | SR-UKF | 11 | 8210.21 | 235.173 | 1.69911e+12 | 0 | 0.965 | 0.965 | 6.023e-06 | 5.07835 | 5.52233e+08 |
| 1e+08 | Float64 | CUT-4 | 42 | 18629 | 1252.64 | 8.43126e+11 | 0 | 0.840 | 0.840 | 1.249e-05 | 8.29481 | 1.25302e+09 |
| 1e+08 | Float64 | SR-CUT-4 | 42 | 64225.1 | 1246.21 | 8.67641e+11 | 0 | 0.835 | 0.835 | 1.476e-05 | 8.29278 | 4.3199e+09 |
| 1e+08 | Float64 | CUT-6 | 83 | 1.88258e+07 | 865.649 | 3.66571e+11 | 0 | 0.625 | 0.625 | 3.219e-05 | 8.47861 | 1.26626e+12 |
| 1e+08 | Float64 | SR-CUT-6 | 83 | 147211 | 862.926 | 4.00156e+11 | 0 | 0.625 | 0.625 | 3.121e-05 | 8.47726 | 9.90172e+09 |
| 1e+08 | Float64 | SR-CKF | 10 | 73500.8 | 2063.67 | 1.50027e+12 | 0 | 0.990 | 0.990 | 4.140e-06 | 5.6301 | 4.9438e+09 |
| 1e+08 | Float32 | CUT-4 | 42 | 652.394 | 39.2476 | 3.68375e+09 | 0 | 0.905 | 0.905 | 1.007e-05 | 8.4832 | 4.38813e+07 |
| 1e+08 | Float32 | SR-CUT-4 | 42 | 631.158 | 130.18 | 3.81752e+09 | 0 | 0.895 | 0.895 | 1.084e-05 | 8.52999 | 4.24528e+07 |
| 1e+08 | Float32 | CUT-6 | 83 | 395.86 | 20.9415 | 4.26562e+08 | 0 | 0.670 | 0.670 | 2.586e-05 | 8.56768 | 2.66263e+07 |
| 1e+08 | Float32 | SR-CUT-6 | 83 | 391.822 | 31.5125 | 6.72046e+09 | 0 | 0.665 | 0.665 | 2.222e-05 | 8.59469 | 2.63546e+07 |
| 1e+08 | Float32 | SR-CKF | 10 | 6595.25 | 628.904 | 7.31241e+10 | 0 | 1.000 | 1.000 | 4.732e-06 | 6.85599 | 4.43609e+08 |
| 1e+08 | Float64 | UKF | 11 | 1.566e+09 | 1958.57 | 1.43574e+12 | 0 | 1.000 | 1.000 | 1.490e-06 | 6.90045 | 1.05332e+14 |
| 1e+08 | Float64 | SR-UKF | 11 | 411934 | 505.203 | 7.44255e+11 | 0 | 0.985 | 0.985 | 3.013e-06 | 4.74991 | 2.77074e+10 |

## Long-arc angles-only case

LEO is a 400 km circular orbit with J2 and drag (B = 0.022 m²/kg).
HEO is a 500 km by 25000 km orbit with the same force model (B = 0.015 m²/kg).
The discrete map is fixed-step RK4 with a 60 s step, shared by the truth and every filter. The posterior CRLB differentiates that same map along each Monte Carlo truth.
Measurements are the east and north components of the line-of-sight unit vector, σ = 0.0002 rad.
The coast is the next visibility window at least as long as the requested gap. CUT-6 is not in this sweep: a 6-state CUT6 set has 137 points and the closed-form ill-conditioned sweep already compares order 6. This sweep uses CUT-4 (76 points), SR-CUT-4, SR-CKF (12 points), UKF, and SR-UKF.
The RMSE figure caps the axis at 1e6 km when a series exceeds that value, so one divergent sample does not hide the coast and the PCRLB.

| variant | gap req. (h) | gap actual (h) | filter | n_σ | pos. RMSE | vel. RMSE | gap-end RMSE | final RMSE | NEES | in band | factor fail | diverge | reacquire | s/step | RMSE/PCRLB |
|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LEO | 2.0 | 8.04 | CUT-4 | 76 | 12.8387 | 0.0167644 | 169.447 | 0.338757 | 5.86799 | 1 | 0.000 | 0.000 | 1.000 | 6.574e-04 | 0.990809 |
| LEO | 2.0 | 8.04 | SR-CUT-4 | 76 | 12.8387 | 0.0167644 | 169.447 | 0.338757 | 5.86799 | 1 | 0.000 | 0.000 | 1.000 | 3.350e-04 | 0.990809 |
| LEO | 2.0 | 8.04 | SR-CKF | 12 | 13.2228 | 0.0172081 | 169.45 | 0.361714 | 8.98859 | 0.466667 | 0.000 | 0.000 | 1.000 | 5.688e-05 | 1.02045 |
| LEO | 2.0 | 8.04 | UKF | 13 | 12.9058 | 0.0168126 | 169.444 | 0.338261 | 7.17501 | 0.466667 | 0.000 | 0.000 | 1.000 | 7.016e-05 | 0.995986 |
| LEO | 2.0 | 8.04 | SR-UKF | 13 | 12.9058 | 0.0168126 | 169.444 | 0.338261 | 7.17501 | 0.466667 | 0.000 | 0.000 | 1.000 | 7.416e-05 | 0.995986 |
| LEO | 6.0 | 8.04 | CUT-4 | 76 | 13.3163 | 0.017279 | 173.299 | 0.332397 | 6.37795 | 0.8 | 0.000 | 0.000 | 1.000 | 2.935e-04 | 1.02767 |
| LEO | 6.0 | 8.04 | SR-CUT-4 | 76 | 13.3163 | 0.017279 | 173.299 | 0.332397 | 6.37795 | 0.8 | 0.000 | 0.000 | 1.000 | 2.875e-04 | 1.02767 |
| LEO | 6.0 | 8.04 | SR-CKF | 12 | 13.5817 | 0.0176055 | 173.297 | 0.347199 | 9.02701 | 0.466667 | 0.000 | 0.000 | 1.000 | 5.714e-05 | 1.04815 |
| LEO | 6.0 | 8.04 | UKF | 13 | 13.4584 | 0.0173434 | 173.299 | 0.331347 | 10.0625 | 0.4 | 0.000 | 0.000 | 1.000 | 5.644e-05 | 1.03864 |
| LEO | 6.0 | 8.04 | SR-UKF | 13 | 13.4584 | 0.0173434 | 173.299 | 0.331347 | 10.0625 | 0.4 | 0.000 | 0.000 | 1.000 | 5.999e-05 | 1.03864 |
| LEO | 12.0 | 23.00 | CUT-4 | 76 | 43.831 | 0.0510464 | 544.942 | 0.273164 | 8.15375 | 0.466667 | 0.000 | 0.000 | 1.000 | 7.000e-04 | 1.20601 |
| LEO | 12.0 | 23.00 | SR-CUT-4 | 76 | 43.831 | 0.0510464 | 544.942 | 0.273164 | 8.15375 | 0.466667 | 0.000 | 0.000 | 1.000 | 7.008e-04 | 1.20601 |
| LEO | 12.0 | 23.00 | SR-CKF | 12 | 52.7058 | 0.0642912 | 544.942 | 1.32201 | 472.113 | 0.4 | 0.000 | 0.000 | 1.000 | 1.239e-04 | 1.4502 |
| LEO | 12.0 | 23.00 | UKF | 13 | 128.517 | 0.25606 | 544.94 | 178.88 | 560326 | 0.4 | 0.000 | 0.000 | 0.945 | 1.290e-04 | 3.53616 |
| LEO | 12.0 | 23.00 | SR-UKF | 13 | 128.517 | 0.25606 | 544.94 | 178.88 | 560330 | 0.4 | 0.000 | 0.000 | 0.945 | 1.327e-04 | 3.53616 |
| LEO | 24.0 | 24.62 | CUT-4 | 76 | 67.2037 | 0.0781618 | 561.668 | 0.918772 | 252.085 | 0.466667 | 0.005 | 0.005 | 0.995 | 7.466e-04 | 1.74039 |
| LEO | 24.0 | 24.62 | SR-CUT-4 | 76 | 67.2037 | 0.0781618 | 561.668 | 0.918772 | 252.085 | 0.466667 | 0.005 | 0.005 | 0.995 | 7.493e-04 | 1.74039 |
| LEO | 24.0 | 24.62 | SR-CKF | 12 | 90.7178 | 0.107739 | 561.684 | 17.6967 | 4729.61 | 0.4 | 0.005 | 0.005 | 0.995 | 1.302e-04 | 2.34934 |
| LEO | 24.0 | 24.62 | UKF | 13 | 198.907 | 0.240929 | 561.662 | 157.947 | 52718.4 | 0.4 | 0.005 | 0.005 | 0.920 | 1.350e-04 | 5.15113 |
| LEO | 24.0 | 24.62 | SR-UKF | 13 | 198.907 | 0.240929 | 561.662 | 157.947 | 52718.8 | 0.4 | 0.005 | 0.005 | 0.920 | 1.384e-04 | 5.15114 |
| LEO | 48.0 | 54.26 | CUT-4 | 76 | 1294.75 | 2.66293 | 1374.01 | 3276.93 | 49958.2 | 0.545455 | 0.020 | 0.030 | 0.970 | 2.172e-03 | 11.2607 |
| LEO | 48.0 | 54.26 | SR-CUT-4 | 76 | 1294.75 | 2.66293 | 1374.01 | 3276.93 | 49958.2 | 0.545455 | 0.020 | 0.030 | 0.970 | 2.163e-03 | 11.2607 |
| LEO | 48.0 | 54.26 | SR-CKF | 12 | 2.2817e+70 | 1.52113e+69 | 1374.04 | 2.50987e+71 | 6.44314e+06 | 0.545455 | 0.155 | 0.275 | 0.255 | 3.572e-04 | 1.98445e+68 |
| LEO | 48.0 | 54.26 | UKF | 13 | 8.07027e+43 | 2.6633e+42 | 1374 | 6.72428e+44 | 1.60436e+06 | 0.545455 | 0.400 | 0.710 | 0.220 | 3.777e-04 | 7.01891e+41 |
| LEO | 48.0 | 54.26 | SR-UKF | 13 | 1.72568e+09 | 1.15046e+08 | 1374 | 1.55691e+09 | 3.68605e+07 | 0.545455 | 0.395 | 0.710 | 0.220 | 3.855e-04 | 1.50087e+07 |
| HEO | 12.0 | 29.43 | CUT-4 | 76 | 4.24947e+07 | 2.83164e+06 | 5835.16 | 43196 | 185652 | 0.4 | 0.275 | 0.680 | 0.715 | 8.707e-04 | 98951.2 |
| HEO | 12.0 | 29.43 | SR-CUT-4 | 76 | 3.76452e+15 | 1.42366e+14 | 5835.16 | 6.21455e+12 | 262832 | 0.4 | 0.240 | 0.680 | 0.715 | 8.716e-04 | 8.76588e+12 |
| HEO | 12.0 | 29.43 | SR-CKF | 12 | 1.12263e+14 | 1.30033e+12 | 5839.86 | 6.25382e+14 | 1.83962e+06 | 0.4 | 0.210 | 0.920 | 0.620 | 1.508e-04 | 2.6141e+11 |
| HEO | 12.0 | 29.43 | UKF | 13 | 42437.3 | 44.1159 | 5844.73 | 93024 | 4.86185e+07 | 0 | 0.330 | 0.670 | 0.110 | 1.568e-04 | 98.8174 |
| HEO | 12.0 | 29.43 | SR-UKF | 13 | 2.6693e+09 | 1.25415e+08 | 5844.73 | 4.79027e+09 | 4.83042e+07 | 0 | 0.320 | 0.670 | 0.110 | 1.590e-04 | 6.2156e+06 |
| HEO | 24.0 | 29.43 | CUT-4 | 76 | 4.09711e+29 | 2.73141e+28 | 5828.3 | 2.75517e+09 | 257518 | 0.466667 | 0.285 | 0.725 | 0.675 | 8.713e-04 | 9.54034e+26 |
| HEO | 24.0 | 29.43 | SR-CUT-4 | 76 | 1.08561e+17 | 2.96893e+15 | 5828.3 | 2.08386e+14 | 320306 | 0.466667 | 0.250 | 0.725 | 0.675 | 8.696e-04 | 2.5279e+14 |
| HEO | 24.0 | 29.43 | SR-CKF | 12 | 3.28287e+14 | 4.62719e+12 | 5821.25 | 2.23591e+15 | 2.48525e+06 | 0.266667 | 0.160 | 0.900 | 0.630 | 1.495e-04 | 7.64432e+11 |
| HEO | 24.0 | 29.43 | UKF | 13 | 4.89922e+83 | 4.25781e+81 | 5815.6 | 2.50881e+84 | 4.96916e+06 | 0 | 0.340 | 0.685 | 0.100 | 1.569e-04 | 1.14081e+81 |
| HEO | 24.0 | 29.43 | SR-UKF | 13 | 5.40543e+07 | 687202 | 5815.6 | 3.07927e+08 | 8.02695e+06 | 0 | 0.330 | 0.685 | 0.100 | 1.606e-04 | 125868 |
| HEO | 48.0 | 58.77 | CUT-4 | 76 | 7.81449e+90 | 5.20966e+89 | 10098.5 | 1.53297e+06 | 136932 | 0.4 | 0.500 | 0.950 | 0.485 | 1.675e-03 | 1.01179e+88 |
| HEO | 48.0 | 58.77 | SR-CUT-4 | 76 | 1.93925e+11 | 1.85782e+09 | 10098.5 | 7.59887e+11 | 138353 | 0.4 | 0.475 | 0.950 | 0.485 | 1.668e-03 | 2.51087e+08 |
| HEO | 48.0 | 58.77 | SR-CKF | 12 | 7.53029e+14 | 9.14937e+12 | 10076.6 | 3.6503e+15 | 3.78952e+06 | 0.4 | 0.185 | 1.000 | 0.520 | 2.829e-04 | 9.74994e+11 |
| HEO | 48.0 | 58.77 | UKF | 13 | 1.54113e+21 | 1.69853e+19 | 10430 | 1.12384e+22 | 1.50873e+09 | 0 | 0.485 | 0.905 | 0.040 | 2.921e-04 | 1.9954e+18 |
| HEO | 48.0 | 58.77 | SR-UKF | 13 | 5.79664e+08 | 3.65914e+15 | 10430 | 2.53589e+09 | 1.49109e+09 | 0 | 0.480 | 0.905 | 0.045 | 2.996e-04 | 750527 |
| HEO | 96.0 | 102.48 | CUT-4 | 76 | 5.24411e+12 | 3.48888e+11 | 14978.1 | 1.23047e+11 | 8.14405e+06 | 0.4 | 0.310 | 1.000 | 0.040 | 2.878e-03 | 3.52433e+09 |
| HEO | 96.0 | 102.48 | SR-CUT-4 | 76 | 2.5028e+15 | 4.34633e+13 | 14978.1 | 4.45111e+15 | 7.99163e+06 | 0.4 | 0.290 | 1.000 | 0.040 | 2.876e-03 | 1.68202e+12 |
| HEO | 96.0 | 102.48 | SR-CKF | 12 | 5.00294e+73 | 3.33526e+72 | 14910.2 | 7.49236e+74 | 6.74268e+06 | 0.333333 | 0.115 | 1.000 | 0.145 | 4.729e-04 | 3.36225e+70 |
| HEO | 96.0 | 102.48 | UKF | 13 | 659692 | 914.653 | 16247.6 | 2.19237e+06 | 1.62255e+09 | 0 | 0.180 | 1.000 | 0.090 | 5.025e-04 | 443.349 |
| HEO | 96.0 | 102.48 | SR-UKF | 13 | 2.83106e+08 | 2.44645e+06 | 16454.7 | 1.1351e+09 | 1.46853e+09 | 0 | 0.100 | 1.000 | 0.090 | 5.078e-04 | 190263 |

## Decision

M = 200 on every setting. The same draws are shared across filters. A failure-rate gap is called clear only when full-covariance CUT fails at least 0.05 more often than SR-CUT of the same order and the SR-CUT rate is the lower one.

H1 is not supported. The largest CUT minus SR-CUT factor-failure gap is 0.015 (Float32, order 4, ratio 1e5: 0.915 versus 0.900). No order and precision reaches a gap of 0.05. At ratio 1e3 in Float32, SR-CUT-4 fails slightly more often than CUT-4 (0.935 versus 0.930), and at ratio 1e4 both orders fail more often in the square-root form. In Float64, ratios 1e3 through 1e5 have zero factor failures for CUT and SR-CUT, and the position RMSE and NEES match to the printed digits. At ratios 1e6, 1e7, and 1e8 the two forms fail together (order 4: 0.005, 0.560–0.565, and 0.835–0.840; order 6: 0.005, 0.370, and 0.625). Float32 fails for both forms at every ratio, including 0.845 or higher at ratio 1e3. UKF and SR-UKF lose consistency earlier (NEES about 60 at ratio 1e4, factor failure 0.665–0.705 at ratio 1e6) while CUT is still running. That is a CUT versus UKF difference, and the square-root CUT shares it with the covariance form. Mean position RMSE at ratio 1e6 and above is not a failure-rate result: NEES is 1e11 or larger and the in-band fraction is 0, and a late failure still contributes the earlier epochs.

H2 is not supported as a square-root result. On the LEO coasts of 23.00 h, 24.62 h, and 54.26 h, CUT-4 and SR-CUT-4 match to the printed digits in position RMSE, NEES, factor failure, and reacquisition. Factor failures stay at or below 0.020. Both have a smaller |NEES − 6| than the UKF and the SR-UKF (8.15 versus about 5.6e5 at 23 h; 5.0e4 versus 1.6e6 and 3.7e7 at 54 h), and both reacquire after the 54 h coast on 0.970 of runs while the UKF reacquires on 0.220. The square-root form does not fail less often than the covariance form on these gaps. The requested 2 h and 6 h LEO gaps both land on the same 8.04 h coast, because that is the next visible pass; they are separate Monte Carlo batches of that coast. The HEO 12 h and 24 h requests likewise share a 29.43 h coast.

On HEO, SR-CUT-4 factor failures are 0.020 to 0.035 below CUT-4. That is under the 0.05 gap, and both forms still diverge (0.680 to 1.000). Mean position RMSE is dominated by divergent runs and is not similar between the two forms. At the 102.48 h coast every filter diverges. SR-CUT does not clearly reduce failures or improve consistency relative to full-covariance CUT, so this repository does not add a paper outline.

