
# Sensors-and-Controls

Repository for Integrated Project

## Part A modelling

See the [modelling and validation study](simulation/PART_A_MODELLING.md) and
[observability analysis](simulation/PART_A_OBSERVABILITY.md).

## Sensor calibration

The reusable calibration APIs are exported from `sensing`:

```python
from sensing import calibrate_camera, calibrate_lidar

camera = calibrate_camera(
    checkerboard_frames, checkerboard_size=(9, 6), square_size_m=0.024
)
camera_matrix = camera.camera_matrix
distortion_coefficients = camera.dist_coeffs

lidar = calibrate_lidar(
    measured_ranges_m,
    reference_ranges_m,
    measured_angles=measured_angles_rad,
    reference_angles=reference_angles_rad,
)
corrected_ranges_m = lidar.apply_ranges(measured_ranges_m)
corrected_angles_rad = lidar.apply_angles(measured_angles_rad)
```

Camera calibration requires at least three same-resolution grayscale or
OpenCV-style BGR images showing a checkerboard; `checkerboard_size` counts
inner corners as `(columns, rows)`. LiDAR range calibration fits a scale and
bias from at least two distinct measured distances paired with known distances.
The optional angle arrays fit a constant angular offset. Both calibration
results report residual error so the main environment can check calibration
quality before using the fitted parameters.
