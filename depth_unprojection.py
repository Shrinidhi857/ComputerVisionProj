"""
3D Spatial Unprojection & Depth Processing Module.
Converts 2D YOLO detections and Monocular Depth Maps into metric 3D Camera/World Coordinates.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


class CameraModel:
    """
    Pinhole Camera Model with Extrinsics transformation into Metropolis World Frame.
    """
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        R_cam_to_world: Optional[np.ndarray] = None,
        t_cam_to_world: Optional[np.ndarray] = None
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)

        # Extrinsics (Default: Camera is at origin aligned with world)
        self.R = R_cam_to_world if R_cam_to_world is not None else np.eye(3, dtype=np.float64)
        self.t = t_cam_to_world if t_cam_to_world is not None else np.zeros((3, 1), dtype=np.float64)

    def unproject_pixel_to_camera(self, u: float, v: float, depth_z: float) -> np.ndarray:
        """
        Back-projects image pixel (u, v) and metric depth Z into 3D Camera Frame (X_c, Y_c, Z_c).
        """
        x_c = (u - self.cx) * depth_z / self.fx
        y_c = (v - self.cy) * depth_z / self.fy
        z_c = depth_z
        return np.array([x_c, y_c, z_c], dtype=np.float64)

    def camera_to_world(self, p_cam: np.ndarray) -> np.ndarray:
        """Transforms 3D camera coordinates to 3D metropolis world coordinates: P_w = R * P_c + t."""
        return (self.R @ p_cam.reshape(3, 1) + self.t).flatten()


class MonocularDepthCalibrator:
    """
    Converts MiDaS / DPT relative inverse depth output d_rel to metric depth Z (meters).
    Uses affine calibration model: Z_metric = scale / (d_rel + eps) + shift,
    or reference object size priors (e.g. known vehicle bounding volume).
    """
    def __init__(self, scale: float = 1000.0, shift: float = 0.0, min_depth: float = 1.0, max_depth: float = 250.0):
        self.scale = scale
        self.shift = shift
        self.min_depth = min_depth
        self.max_depth = max_depth

    def to_metric_depth(self, relative_depth_map: np.ndarray) -> np.ndarray:
        """
        MiDaS outputs inverse disparity/depth d. Metric depth Z ~= scale / (d + eps)
        """
        eps = 1e-4
        metric_depth = self.scale / (relative_depth_map + eps) + self.shift
        return np.clip(metric_depth, self.min_depth, self.max_depth)

    @staticmethod
    def extract_robust_roi_depth(depth_map: np.ndarray, bbox: Tuple[int, int, int, int], crop_fraction: float = 0.3) -> float:
        """
        Extracts robust metric depth from central kernel of 2D bounding box [x1, y1, x2, y2].
        Using median of central 30% ROI removes background bleed / edge silhouettes.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = depth_map.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return 10.0

        # Central sub-window
        pad_w = int(bw * (1.0 - crop_fraction) / 2.0)
        pad_h = int(bh * (1.0 - crop_fraction) / 2.0)

        cx1, cx2 = x1 + pad_w, x2 - pad_w
        cy1, cy2 = y1 + pad_h, y2 - pad_h

        roi = depth_map[cy1:cx2 if cx2 > cx1 else cx1 + 1, cx1:cx2 if cx2 > cx1 else cx1 + 1]
        if roi.size == 0:
            roi = depth_map[y1:y2, x1:x2]

        return float(np.median(roi))
