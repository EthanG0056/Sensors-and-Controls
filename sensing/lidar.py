from typing import Protocol

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from machinevisiontoolbox import Image, CentralCamera
from numpy.typing import NDArray
from spatialmath import SE3


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
    """Return a copy of the named MuJoCo sensor's readings."""
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
    """Replace invalid, too-close, and max-range-clipped readings with NaN."""
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
    """Convert a scan to robot-frame ``(x forward, y left)`` points.

    Invalid readings produce ``[nan, nan]`` rows, retaining each ray's index.
    The default angle convention starts forward at zero and increases left.
    """
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
    """Return the nearest valid range, its ray angle in radians, and safety alert.

    Angles start at zero (forward) and increase counter-clockwise. If no valid
    obstacle readings remain, returns ``(nan, nan, False)``.
    """
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
    """Scatter plot of a point cloud, in the robot frame or the camera frame."""
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