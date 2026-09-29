"""
3D Computational Geometry Module for Volumetric Flight Corridors & Boundary Enforcements.
Implements:
  1. Holographic Boundary Planes (Ray-Plane / Segment-Plane intersection + Polygon containment)
  2. Volumetric Flight Corridors / Restricted Airspaces (Convex Polyhedra / Half-space testing, OBBs)
  3. Altitude Ceiling and Floor checks
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class HolographicPlane:
    """
    Represents an oriented planar boundary in 3D space defined by a normal vector n,
    an offset d (n . x + d = 0), and an optional convex 3D polygon perimeter.
    """
    def __init__(self, normal: np.ndarray, point_on_plane: np.ndarray, polygon_vertices: Optional[np.ndarray] = None):
        self.normal = normal / np.linalg.norm(normal)
        self.point = np.array(point_on_plane, dtype=np.float64)
        # Plane equation: n . x + d = 0 => d = - (n . point)
        self.d = -float(np.dot(self.normal, self.point))
        # Optional polygon vertices on the plane (N x 3)
        self.polygon_vertices = polygon_vertices if polygon_vertices is not None else None

    def signed_distance(self, point: np.ndarray) -> float:
        """Returns signed distance from point to plane (>0 on normal side, <0 on opposite)."""
        return float(np.dot(self.normal, point) + self.d)

    def segment_intersection(self, p1: np.ndarray, p2: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Tests if trajectory line segment p1 -> p2 crosses the plane.
        Returns intersection coordinates and normalized parameter t in [0, 1].
        """
        d1 = self.signed_distance(p1)
        d2 = self.signed_distance(p2)

        # Check if endpoints lie on opposite sides or if one lies exactly on plane
        if (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0):
            return None  # No crossing

        # Vector along segment
        u = p2 - p1
        denom = np.dot(self.normal, u)

        if np.isclose(denom, 0.0):
            return None  # Segment is parallel to plane

        t = -(np.dot(self.normal, p1) + self.d) / denom

        if 0.0 <= t <= 1.0:
            intersection_pt = p1 + t * u
            
            # If a bounded polygon is defined on the plane, check point-in-polygon in 2D projected space
            if self.polygon_vertices is not None:
                if not self._point_in_3d_polygon(intersection_pt, self.polygon_vertices, self.normal):
                    return None

            return {
                "intersection_point": intersection_pt,
                "t": float(t),
                "direction": "outward" if d1 < 0 and d2 >= 0 else "inward",
                "normal": self.normal
            }
        return None

    @staticmethod
    def _point_in_3d_polygon(pt: np.ndarray, vertices: np.ndarray, normal: np.ndarray) -> bool:
        """
        Projects 3D point and vertices onto dominant coordinate plane and performs 2D Ray Casting.
        """
        # Find dominant axis of plane normal to drop for 2D projection
        abs_norm = np.abs(normal)
        drop_axis = int(np.argmax(abs_norm))
        axes = [i for i in range(3) if i != drop_axis]

        pt_2d = pt[axes]
        poly_2d = vertices[:, axes]

        # Standard 2D ray casting algorithm
        inside = False
        n = len(poly_2d)
        p1x, p1y = poly_2d[0]
        for i in range(n + 1):
            p2x, p2y = poly_2d[i % n]
            if pt_2d[1] > min(p1y, p2y):
                if pt_2d[1] <= max(p1y, p2y):
                    if pt_2d[0] <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (pt_2d[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or pt_2d[0] <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside


class ConvexPolyhedronCorridor:
    """
    Represents a 3D convex flight corridor or restricted volume defined by
    a system of linear half-space inequalities: A * x <= b.
    """
    def __init__(self, planes_normal_and_points: List[Tuple[np.ndarray, np.ndarray]], corridor_id: str, is_restricted: bool = False):
        """
        planes_normal_and_points: List of (outward_normal, point_on_face)
        A point x is INSIDE if dot(n_i, (x - p_i)) <= 0 for all faces.
        """
        self.corridor_id = corridor_id
        self.is_restricted = is_restricted  # True if violation occurs upon entering, False if violation occurs upon exiting
        self.normals = []
        self.b = []
        
        for n, p in planes_normal_and_points:
            n_norm = n / np.linalg.norm(n)
            self.normals.append(n_norm)
            # dot(n, x) <= dot(n, p) => b_i = dot(n, p)
            self.b.append(float(np.dot(n_norm, p)))

        self.A = np.array(self.normals, dtype=np.float64)  # (M, 3)
        self.b = np.array(self.b, dtype=np.float64)        # (M,)

    def contains(self, point: np.ndarray) -> bool:
        """Returns True if point is inside the convex volume."""
        return bool(np.all(np.dot(self.A, point) <= self.b + 1e-6))

    def evaluate_trajectory_violation(self, p_prev: np.ndarray, p_curr: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Checks if vehicle trajectory (p_prev -> p_curr) violates corridor boundaries.
        - For restricted zones: entering is a violation.
        - For designated corridors: exiting is a violation.
        """
        inside_prev = self.contains(p_prev)
        inside_curr = self.contains(p_curr)

        if self.is_restricted:
            # Restricted Airspace: Violation on Entry or Continuous Presence
            if inside_curr:
                return {
                    "corridor_id": self.corridor_id,
                    "violation_type": "RESTRICTED_AIRSPACE_BREACH",
                    "point": p_curr.tolist(),
                    "status": "ENTERED" if not inside_prev else "PRESENT"
                }
        else:
            # Authorized Corridor: Violation on Exit
            if not inside_curr:
                return {
                    "corridor_id": self.corridor_id,
                    "violation_type": "CORRIDOR_DEVIATION_BREACH",
                    "point": p_curr.tolist(),
                    "status": "EXITED" if inside_prev else "OUT_OF_BOUNDS"
                }
        return None


class AltitudeEnforcer:
    """
    Dedicated high-throughput altitude ceiling & floor enforcement layer.
    """
    def __init__(self, min_altitude_m: float = 10.0, max_altitude_m: float = 150.0):
        self.min_altitude = min_altitude_m
        self.max_altitude = max_altitude_m

    def check_altitude_violation(self, point: np.ndarray, velocity_z: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        In world coordinates, assuming Z is height above datum.
        """
        z = point[2]
        if z > self.max_altitude:
            return {
                "violation_type": "ALTITUDE_CEILING_EXCEEDED",
                "current_altitude": float(z),
                "threshold": self.max_altitude,
                "velocity_z": float(velocity_z),
                "delta": float(z - self.max_altitude)
            }
        elif z < self.min_altitude:
            return {
                "violation_type": "ALTITUDE_FLOOR_BREACH",
                "current_altitude": float(z),
                "threshold": self.min_altitude,
                "velocity_z": float(velocity_z),
                "delta": float(self.min_altitude - z)
            }
        return None
