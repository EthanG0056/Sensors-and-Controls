# ArUco marker detection and range/bearing measurements for MuJoCo.

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
_marker_size_m: float | None = None
_renderer: _Renderer | None = None
_renderer_model: object | None = None
_renderer_dimensions: tuple[int, int] | None = None


class _Geometry(Protocol):
    size: NDArray[np.float64]


class _ModelWithGeometries(Protocol):
    def geom(self, name: str) -> _Geometry: ...


class _Renderer(Protocol):
    def update_scene(self, data: object, camera: str) -> None: ...

    def render(self) -> NDArray[np.uint8]: ...

    def close(self) -> None: ...


def configure_marker_size(marker_size_m: float) -> None:

    # Set the marker's physical side length in meters.
    global _marker_size_m

    if not np.isfinite(marker_size_m) or marker_size_m <= 0:
        raise ValueError("marker_size_m must be a finite positive value")
    _marker_size_m = float(marker_size_m)


def configure_marker_size_from_geom(
        
    model: _ModelWithGeometries,
    geom_name: str,
    face_axes: tuple[int, int],
) -> None:
    
    # Read a square marker's side length from a named MuJoCo geom.

    # MuJoCo geom sizes are half-extents. ``face_axes`` selects the two local
    # axes spanning the marker face, for example ``(0, 2)`` for an XZ face.
    
    geom = model.geom(geom_name)
    half_extents = np.asarray(geom.size, dtype=float).reshape(-1)
    if half_extents.shape != (3,):
        raise ValueError(f"Geometry {geom_name!r} must have three size dimensions")
    if len(face_axes) != 2 or face_axes[0] == face_axes[1]:
        raise ValueError("face_axes must contain two distinct geometry axes")
    if any(axis not in (0, 1, 2) for axis in face_axes):
        raise ValueError("face_axes entries must be 0, 1, or 2")
    face_half_extents = half_extents[list(face_axes)]
    if not np.all(np.isfinite(face_half_extents)) or np.any(face_half_extents <= 0):
        raise ValueError(f"Geometry {geom_name!r} has invalid marker face dimensions")
    if not np.isclose(face_half_extents[0], face_half_extents[1], rtol=0.01):
        raise ValueError(f"Geometry {geom_name!r} does not have a square face")
    configure_marker_size(float(2 * face_half_extents[0]))


def close_camera_renderer() -> None:

    # Release the cached MuJoCo renderer after the simulation loop ends.
    global _renderer, _renderer_model, _renderer_dimensions

    if _renderer is not None:
        try:
            _renderer.close()
        finally:
            _renderer = None
            _renderer_model = None
            _renderer_dimensions = None


def detect_aruco_marker(
    image: NDArray[np.uint8],
) -> tuple[int, NDArray[np.float32]] | None:
    
    # Return the first detected marker ID and its four (x, y) pixel corners.

    # MuJoCo renderer images are RGB. Two-dimensional grayscale input is also
    # accepted. Returns ``None`` if no marker from ``DICT_4X4_50`` is visible.
    
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("image must be a grayscale or RGB image")

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    detector = cv2.aruco.ArucoDetector(
        dictionary,
        cv2.aruco.DetectorParameters(),
    )
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None

    return int(ids.flat[0]), np.asarray(corners[0], dtype=np.float32).reshape(4, 2)


def estimate_marker_pose(
    corners: NDArray[np.floating],
    camera_matrix: NDArray[np.floating],
    dist_coeffs: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    
    # Estimate and return ``(tvec, rvec)`` for a detected marker.

    # Configure the marker side length with ``configure_marker_size`` or
    # ``configure_marker_size_from_geom`` before calling this function.
    
    if _marker_size_m is None:
        raise RuntimeError(
            "Marker size is not configured; call configure_marker_size() "
            "or configure_marker_size_from_geom() after creating the model."
        )

    image_corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    intrinsics = np.asarray(camera_matrix, dtype=np.float64)
    distortion = np.asarray(dist_coeffs, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("camera_matrix must have shape (3, 3)")

    half_size = _marker_size_m / 2
    object_corners = np.array(
        [
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0],
        ],
        dtype=np.float32,
    )
    success, rvec, tvec = cv2.solvePnP(
        object_corners,
        image_corners,
        intrinsics,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("OpenCV could not estimate the ArUco marker pose")

    return np.asarray(tvec, dtype=np.float64), np.asarray(rvec, dtype=np.float64)


def get_range_and_bearing(tvec: NDArray[np.floating]) -> NDArray[np.float64]:

    # Convert camera-frame translation to ``[[range], [bearing]]``.

    # OpenCV's camera frame looks along positive Z; X is horizontal. Range is
    # the XZ-plane distance and bearing is measured from the camera's forward
    # axis, positive to the left.
    
    translation = np.asarray(tvec, dtype=float).reshape(-1)
    if translation.size != 3 or not np.all(np.isfinite(translation)):
        raise ValueError("tvec must contain three finite translation values")
    x, _, z = translation
    if z <= 0:
        raise ValueError("tvec must be in front of the camera (positive Z)")

    return np.array(
        [[np.hypot(x, z)], [-np.arctan2(x, z)]],
        dtype=np.float64,
    )


def get_camera_measurement(
    model: object,
    data: object,
    camera_name: str,
) -> NDArray[np.float64] | None:
    
    # Render a MuJoCo camera and return its ``[[range], [bearing]]`` marker measurement.

    # Returns ``None`` when the marker is not visible. The caller must configure
    # the marker size after creating its MuJoCo model.
    
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError(
            "get_camera_measurement requires the 'mujoco' Python package"
        ) from exc

    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name,
    )
    if camera_id < 0:
        raise ValueError(f"MuJoCo camera {camera_name!r} was not found")

    width = int(model.vis.global_.offwidth)
    height = int(model.vis.global_.offheight)
    if width <= 0 or height <= 0:
        raise ValueError("MuJoCo offscreen render dimensions must be positive")

    vertical_fov = np.deg2rad(float(model.cam_fovy[camera_id]))
    if not np.isfinite(vertical_fov) or not 0 < vertical_fov < np.pi:
        raise ValueError("MuJoCo camera must have a valid perspective field of view")

    focal_length = height / (2 * np.tan(vertical_fov / 2))
    camera_matrix = np.array(
        [
            [focal_length, 0, (width - 1) / 2],
            [0, focal_length, (height - 1) / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros(5, dtype=np.float64)

    global _renderer, _renderer_model, _renderer_dimensions
    dimensions = (width, height)
    if _renderer is None or _renderer_model is not model or _renderer_dimensions != dimensions:
        close_camera_renderer()
        _renderer = mujoco.Renderer(model, height=height, width=width)
        _renderer_model = model
        _renderer_dimensions = dimensions

    _renderer.update_scene(data, camera=camera_name)
    image = _renderer.render()

    detection = detect_aruco_marker(image)
    if detection is None:
        return None

    _, corners = detection
    tvec, _ = estimate_marker_pose(corners, camera_matrix, dist_coeffs)
    return get_range_and_bearing(tvec)
