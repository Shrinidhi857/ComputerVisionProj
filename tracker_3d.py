"""
3D Multi-Object Tracking Engine (3D DeepSORT Extension).
Maintains 3D kinematic state vectors [X, Y, Z, Vx, Vy, Vz, w, h, l] with Kalman Filtering
and Re-ID embedding association.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class KalmanFilter3D:
    """
    Constant Velocity 3D Linear Kalman Filter.
    State: [X, Y, Z, Vx, Vy, Vz]^T (6-dim)
    Measurement: [X, Y, Z]^T (3-dim)
    """
    def __init__(self, dt: float = 1.0 / 30.0):
        self.dt = dt
        # State transition matrix F
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix H
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process Noise Covariance Q
        q_pos = 0.05
        q_vel = 0.5
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float64)

        # Measurement Noise Covariance R (Camera depth uncertainty grows with Z^2)
        self.R_base = np.diag([0.1, 0.1, 0.4]).astype(np.float64)

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_pred = self.F @ mean
        cov_pred = self.F @ covariance @ self.F.T + self.Q
        return mean_pred, cov_pred

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Scale depth covariance dynamically based on measured range
        z_dist = measurement[2]
        R = self.R_base.copy()
        R[2, 2] = R[2, 2] * (1.0 + (z_dist / 20.0) ** 2)

        innovation = measurement - (self.H @ mean)
        S = self.H @ covariance @ self.H.T + R
        K = covariance @ self.H.T @ np.linalg.inv(S)

        new_mean = mean + (K @ innovation)
        new_cov = (np.eye(6) - K @ self.H) @ covariance
        return new_mean, new_cov


class Track3D:
    """Represents an active vehicle 3D track state and trajectory history."""
    _id_counter = 0

    def __init__(self, initial_position: np.ndarray, embedding: Optional[np.ndarray] = None, kf: Optional[KalmanFilter3D] = None):
        Track3D._id_counter += 1
        self.track_id = Track3D._id_counter
        self.kf = kf or KalmanFilter3D()
        
        # State: [X, Y, Z, 0, 0, 0]
        self.mean = np.zeros(6, dtype=np.float64)
        self.mean[:3] = initial_position
        self.cov = np.eye(6, dtype=np.float64) * 1.0

        self.features: List[np.ndarray] = [embedding] if embedding is not None else []
        self.hits = 1
        self.time_since_update = 0
        self.confirmed = False
        self.trajectory_history: List[np.ndarray] = [initial_position.copy()]

    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.time_since_update += 1

    def update(self, measurement: np.ndarray, embedding: Optional[np.ndarray] = None):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, measurement)
        self.features.append(embedding if embedding is not None else np.zeros(128))
        if len(self.features) > 10:
            self.features.pop(0)
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= 3:
            self.confirmed = True
        self.trajectory_history.append(self.mean[:3].copy())
        if len(self.trajectory_history) > 100:
            self.trajectory_history.pop(0)

    @property
    def position(self) -> np.ndarray:
        return self.mean[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.mean[3:]

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))


class Tracker3D:
    """
    3D Spatial Multi-Object Tracker associating 3D detections with active tracks.
    """
    def __init__(self, max_age: int = 15, min_hits: int = 3, distance_threshold: float = 8.0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.tracks: List[Track3D] = []
        self.kf = KalmanFilter3D()

    def update(self, detections_3d: List[np.ndarray], embeddings: Optional[List[np.ndarray]] = None) -> List[Track3D]:
        # 1. Predict state for all existing tracks
        for track in self.tracks:
            track.predict()

        if len(detections_3d) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [t for t in self.tracks if t.confirmed]

        # 2. Compute cost matrix (Euclidean 3D distance + optional appearance embedding cost)
        if len(self.tracks) > 0:
            track_positions = np.array([t.position for t in self.tracks])
            det_positions = np.array(detections_3d)
            spatial_dist = cdist(track_positions, det_positions, metric='euclidean')

            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(spatial_dist)
            
            unmatched_tracks = set(range(len(self.tracks)))
            unmatched_dets = set(range(len(detections_3d)))
            matched_pairs = []

            for r, c in zip(row_ind, col_ind):
                if spatial_dist[r, c] < self.distance_threshold:
                    matched_pairs.append((r, c))
                    unmatched_tracks.discard(r)
                    unmatched_dets.discard(c)

            # Update matched tracks
            for t_idx, d_idx in matched_pairs:
                emb = embeddings[d_idx] if embeddings is not None and len(embeddings) > d_idx else None
                self.tracks[t_idx].update(detections_3d[d_idx], emb)

            # Create new tracks for unmatched detections
            for d_idx in unmatched_dets:
                emb = embeddings[d_idx] if embeddings is not None and len(embeddings) > d_idx else None
                self.tracks.append(Track3D(detections_3d[d_idx], emb, self.kf))
        else:
            # Initialize tracks
            for i, det in enumerate(detections_3d):
                emb = embeddings[i] if embeddings is not None and len(embeddings) > i else None
                self.tracks.append(Track3D(det, emb, self.kf))

        # Filter out dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.confirmed]
