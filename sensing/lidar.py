from dataclasses import dataclass
from typing import Protocol

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from machinevisiontoolbox import Image, CentralCamera
from numpy.typing import NDArray
from spatialmath import SE3


@dataclass(frozen=True)
class LidarCalibration:
    """Linear range correction and optional constant angular correction."""

    range_scale: float
    range_offset_m: float
    range_rmse_m: float
    angle_offset_rad: float | None
    angle_rmse_rad: float | None
    sample_count: int

    def apply_ranges(self, ranges: NDArray[np.floating]) -> NDArray[np.float64]:
        """Apply the fitted scale and bias to a scan."""
        scan = np.asarray(ranges, dtype=np.float64)
        if scan.ndim != 1:
            raise ValueError("ranges must be a one-dimensional array")
        corrected = self.range_scale * scan + self.range_offset_m
        corrected[~np.isfinite(scan)] = np.nan
        return corrected

    def apply_angles(self, angles: NDArray[np.floating]) -> NDArray[np.float64]:
        """Apply the fitted angular offset, preserving wrapped angles."""
        if self.angle_offset_rad is None:
            raise RuntimeError("angle calibration was not included")
        values = np.asarray(angles, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("angles must contain only finite values")
        return np.arctan2(
            np.sin(values + self.angle_offset_rad),
            np.cos(values + self.angle_offset_rad),
        )


def calibrate_lidar(
    measured_ranges: NDArray[np.floating],
    reference_ranges: NDArray[np.floating],
    measured_angles: NDArray[np.floating] | None = None,
    reference_angles: NDArray[np.floating] | None = None,
) -> LidarCalibration:
    """Fit LiDAR range scale/bias and, optionally, a constant angle offset.

    Each measured range must be paired with its known reference distance in
    meters. To calibrate angular alignment, also provide matching arrays of
    measured and reference angles in radians.
    """
    measured = np.asarray(measured_ranges, dtype=np.float64)
    reference = np.asarray(reference_ranges, dtype=np.float64)
    if measured.ndim != 1 or reference.ndim != 1:
        raise ValueError("range observations must be one-dimensional arrays")
    if measured.shape != reference.shape or measured.size < 2:
        raise ValueError("at least two paired range observations are required")
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(reference)):
        raise ValueError("range observations must contain only finite values")
    if np.any(measured <= 0) or np.any(reference <= 0):
        raise ValueError("range observations must be positive")

    design = np.column_stack((measured, np.ones_like(measured)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, reference, rcond=None)
    if rank < 2:
        raise ValueError("measured ranges must include at least two distinct values")
    range_scale, range_offset = (float(value) for value in coefficients)
    if not np.isfinite(range_scale) or range_scale <= 0:
        raise ValueError("calibration produced an invalid range scale")
    range_residual = range_scale * measured + range_offset - reference
    range_rmse = float(np.sqrt(np.mean(range_residual**2)))

    if (measured_angles is None) != (reference_angles is None):
        raise ValueError(
            "measured_angles and reference_angles must be provided together"
        )
    angle_offset: float | None = None
    angle_rmse: float | None = None
    if measured_angles is not None and reference_angles is not None:
        measured_angle_values = np.asarray(measured_angles, dtype=np.float64)
        reference_angle_values = np.asarray(reference_angles, dtype=np.float64)
        if (
            measured_angle_values.ndim != 1
            or reference_angle_values.ndim != 1
            or measured_angle_values.shape != reference_angle_values.shape
            or measured_angle_values.size == 0
        ):
            raise ValueError("angle observations must be non-empty paired 1D arrays")
        if not np.all(np.isfinite(measured_angle_values)) or not np.all(
            np.isfinite(reference_angle_values)
        ):
            raise ValueError("angle observations must contain only finite values")
        differences = np.arctan2(
            np.sin(reference_angle_values - measured_angle_values),
            np.cos(reference_angle_values - measured_angle_values),
        )
        mean_sine = float(np.mean(np.sin(differences)))
        mean_cosine = float(np.mean(np.cos(differences)))
        if np.hypot(mean_sine, mean_cosine) < 1e-12:
            raise ValueError("angle observations do not define a unique offset")
        angle_offset = float(np.arctan2(mean_sine, mean_cosine))
        angular_residual = np.arctan2(
            np.sin(differences - angle_offset),
            np.cos(differences - angle_offset),
        )
        angle_rmse = float(np.sqrt(np.mean(angular_residual**2)))

    return LidarCalibration(
        range_scale=range_scale,
        range_offset_m=range_offset,
        range_rmse_m=range_rmse,
        angle_offset_rad=angle_offset,
        angle_rmse_rad=angle_rmse,
        sample_count=int(measured.size),
    )


class _Sensor(Protocol):
    adr: NDArray[np.integer]
    dim: NDArray[np.integer]


class _ModelWithSensors(Protocol):
    def sensor(self, name: str) -> _Sensor: ...


class _DataWithSensorReadings(Protocol):
    sensordata: NDArray[np.floating]


def get_raw_lidar_data(
    model: _ModelWithSensors,
    data: _DataWithSensorReadings,
    sensor_name: str,
) -> NDArray[np.float64]:
    
    # Return a copy of the named MuJoCo sensor's readings.
    sensor = model.sensor(sensor_name)
    address = int(np.asarray(sensor.adr).reshape(-1)[0])
    dimension = int(np.asarray(sensor.dim).reshape(-1)[0])
    if address < 0 or dimension <= 0:
        raise ValueError(f"MuJoCo sensor {sensor_name!r} has invalid address or dimension")

    sensor_data = np.asarray(data.sensordata, dtype=np.float64)
    end = address + dimension
    if end > sensor_data.size:
        raise ValueError(
            f"MuJoCo sensor {sensor_name!r} extends beyond data.sensordata"
        )
    return sensor_data[address:end].copy()


def filter_lidar_ranges(
    ranges: NDArray[np.floating],
    min_ranges: float = 0.12,
    max_range: float = 3.5,
) -> NDArray[np.float64]:
    
    # Replace invalid, too-close, and max-range-clipped readings with NaN.
    scan = np.asarray(ranges, dtype=np.float64)
    if scan.ndim != 1:
        raise ValueError("ranges must be a one-dimensional array")
    if not np.isfinite(min_ranges) or min_ranges < 0:
        raise ValueError("min_ranges must be finite and non-negative")
    if not np.isfinite(max_range) or max_range <= min_ranges:
        raise ValueError("max_range must be finite and greater than min_ranges")

    filtered = scan.copy()
    invalid = (
        ~np.isfinite(filtered)
        | (filtered < min_ranges)
        | (filtered >= max_range)
    )
    filtered[invalid] = np.nan
    return filtered


def convert_to_cartesian(
    ranges: NDArray[np.floating],
    angles_increment: float,
    angle_min: float = 0.0,
) -> NDArray[np.float64]:
    
    # Convert a scan to robot-frame ``(x forward, y left)`` points.

    # Invalid readings produce ``[nan, nan]`` rows, retaining each ray's index.
    # The default angle convention starts forward at zero and increases left.
    
    scan = np.asarray(ranges, dtype=np.float64)
    if scan.ndim != 1:
        raise ValueError("ranges must be a one-dimensional array")
    if not np.isfinite(angles_increment) or angles_increment <= 0:
        raise ValueError("angles_increment must be finite and positive")
    if not np.isfinite(angle_min):
        raise ValueError("angle_min must be finite")

    angles = angle_min + np.arange(scan.size) * angles_increment
    points = np.full((scan.size, 2), np.nan, dtype=np.float64)
    valid = np.isfinite(scan) & (scan > 0)
    points[valid, 0] = scan[valid] * np.cos(angles[valid])
    points[valid, 1] = scan[valid] * np.sin(angles[valid])
    return points


def get_closest_obstacle(
    ranges: NDArray[np.floating],
    angle_increment: float,
    critical_threshold: float = 0.5,
    min_range: float = 0.12,
    max_range: float = 3.5,
) -> tuple[float, float, bool]:
    
    # Return the nearest valid range, its ray angle in radians, and safety alert.

    # Angles start at zero (forward) and increase counter-clockwise. If no valid
    # obstacle readings remain, returns ``(nan, nan, False)``.
    
    if not np.isfinite(angle_increment) or angle_increment <= 0:
        raise ValueError("angle_increment must be finite and positive")
    if not np.isfinite(critical_threshold) or critical_threshold < 0:
        raise ValueError("critical_threshold must be finite and non-negative")

    filtered = filter_lidar_ranges(ranges, min_range, max_range)
    valid_indices = np.flatnonzero(np.isfinite(filtered))
    if valid_indices.size == 0:
        return float("nan"), float("nan"), False

    closest_index = int(valid_indices[np.argmin(filtered[valid_indices])])
    closest_range = float(filtered[closest_index])
    closest_angle = closest_index * angle_increment
    return closest_range, closest_angle, closest_range <= critical_threshold


def show_cloud(points, colours=None, title='', frame='robot', ax=None,
               elev=22, azim=-75, size=1, nmax=20000, seed=0):
    
    # Scatter plot of a point cloud, in the robot frame or the camera frame.
    P = np.asarray(points)
    C = colours
    per_point = (C is not None and not isinstance(C, str)
                 and np.asarray(C).ndim >= 1 and np.asarray(C).shape[0] == P.shape[0])
    if per_point:
        C = np.asarray(C)
    if len(P) > nmax:
        k = np.random.default_rng(seed).choice(len(P), nmax, replace=False)
        P = P[k]
        if per_point:
            C = C[k]

    if frame == 'robot':          # x forward, y left, z up
        A, B, Cc = P[:, 0], P[:, 1], P[:, 2]
        labels = ('x fwd [m]', 'y left [m]', 'z up [m]')
    else:                         # camera: x right, y down, z forward
        A, B, Cc = P[:, 0], P[:, 2], -P[:, 1]
        labels = ('X right [m]', 'Z forward [m]', '-Y up [m]')

    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection='3d')
    ax.scatter(A, B, Cc, c=C, s=size, marker='.', linewidths=0)
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_zlabel(labels[2])
    ax.set_box_aspect(np.maximum([np.ptp(A), np.ptp(B), np.ptp(Cc)], 1e-3))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    return ax

def cloud_stats(name, pcd):
    print(f'{name:24s} {len(pcd.points):8d} points')