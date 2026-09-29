"""
Machine Learning Traffic Intelligence System for Road Markings & Direction Flow.

Features:
  1. RoadMarkingDetector: Discovers lane markings, dividers, and road line orientations
     using adaptive computer vision & morphological edge analysis.
  2. SpatialFlowGridLearner: Online statistical ML engine that continuously learns
     dominant directional flow vectors, velocities, and travel transit density maps.
  3. WrongDirectionDetector: Evaluates vehicle velocity vs. learned lane flow vector,
     flagging wrong-way driving violations.
  4. TransitZoneParkingDetector: Detects vehicles stopping or parking in active travel
     corridors where vehicles normally flow.
  5. TrafficIntelligenceEngine: Unified orchestrator for real-time video governance HUD.
"""

import time
import math
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple


class RoadMarkingDetector:
    """
    Extracts road markings, lane boundaries, and dominant line orientations
    from video frames using adaptive thresholding, color filtering, and Hough transform.
    """
    def __init__(
        self,
        roi_polygon: Optional[np.ndarray] = None,
        min_line_length: int = 40,
        max_line_gap: int = 25
    ):
        self.roi_polygon = roi_polygon
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.detected_markings_cache: List[Tuple[int, int, int, int]] = []
        self.dominant_orientations: List[float] = []

    def detect_markings(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Detects road marking lines and returns marking coordinates and dominant orientations.
        """
        h, w = frame_bgr.shape[:2]
        
        # Convert to HLS and Grayscale for white & yellow marking extraction
        hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # White markings: high lightness
        white_mask = cv2.inRange(hls, np.array([0, 180, 0]), np.array([255, 255, 255]))
        
        # Yellow markings: hue in yellow range with moderate saturation
        yellow_mask = cv2.inRange(hls, np.array([15, 30, 100]), np.array([35, 204, 255]))
        
        # Adaptive thresholding for local contrast (handles shadows / varying illumination)
        adaptive_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -10
        )
        
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        combined_mask = cv2.bitwise_or(combined_mask, adaptive_mask)

        # Apply ROI mask if specified, or default to lower 65% of frame (road area)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        if self.roi_polygon is not None:
            cv2.fillPoly(roi_mask, [self.roi_polygon.astype(np.int32)], 255)
        else:
            # Default road horizon trapezoid
            pts = np.array([
                [int(w * 0.05), h],
                [int(w * 0.35), int(h * 0.40)],
                [int(w * 0.65), int(h * 0.40)],
                [int(w * 0.95), h]
            ], dtype=np.int32)
            cv2.fillPoly(roi_mask, [pts], 255)

        masked = cv2.bitwise_and(combined_mask, roi_mask)

        # Edge detection & morphological clean-up
        edges = cv2.Canny(masked, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        detected_lines = []
        orientations = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length >= self.min_line_length:
                    angle_rad = math.atan2(dy, dx)
                    detected_lines.append((x1, y1, x2, y2))
                    orientations.append(angle_rad)

        self.detected_markings_cache = detected_lines
        self.dominant_orientations = orientations

        return {
            "marking_lines": detected_lines,
            "orientations": orientations,
            "marking_mask": masked
        }


class SpatialFlowGridLearner:
    """
    Online ML Vector Field & Transit Density Model.
    Discretizes the 2D spatial roadway into a grid of cells.
    Learns:
      1. Dominant Unit Direction Vector [dx, dy] per cell (Bayesian / EMA accumulation).
      2. Directional Confidence / Kappa (concentration of flow).
      3. Average Velocity / Speed in each cell.
      4. Travel Transit Density (frequency of vehicle traffic -> identifying active flow corridors).
    """
    def __init__(
        self,
        grid_width: int = 1280,
        grid_height: int = 720,
        cell_size: int = 40,
        learning_rate: float = 0.08,
        min_speed_threshold: float = 0.5,
        min_samples_for_confidence: int = 3
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.lr = learning_rate
        self.min_speed_threshold = min_speed_threshold
        self.min_samples_for_confidence = min_samples_for_confidence

        self.cols = int(math.ceil(grid_width / cell_size))
        self.rows = int(math.ceil(grid_height / cell_size))

        # Grid state matrices
        # Direction unit vectors: (rows, cols, 2)
        self.flow_vectors = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        # Confidence score in [0.0, 1.0]: (rows, cols)
        self.confidence = np.zeros((self.rows, self.cols), dtype=np.float32)
        # Average speed (pixels/frame or m/s): (rows, cols)
        self.mean_speed = np.zeros((self.rows, self.cols), dtype=np.float32)
        # Sample observation count: (rows, cols)
        self.sample_counts = np.zeros((self.rows, self.cols), dtype=np.int32)
        # Transit density / Motion occurrence heatmap: (rows, cols)
        self.transit_density = np.zeros((self.rows, self.cols), dtype=np.float32)

    def _get_cell_coords(self, x: float, y: float) -> Tuple[int, int]:
        c = int(np.clip(x // self.cell_size, 0, self.cols - 1))
        r = int(np.clip(y // self.cell_size, 0, self.rows - 1))
        return r, c

    def update_from_trajectory(
        self,
        pos_prev: Tuple[float, float],
        pos_curr: Tuple[float, float],
        speed: Optional[float] = None
    ):
        """
        Online incremental update of direction flow field and transit density from a moving vehicle.
        """
        x1, y1 = pos_prev
        x2, y2 = pos_curr
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)

        if dist < 0.8:
            # Vehicle is nearly stationary; do not update direction vector
            # But update cell observation location
            r, c = self._get_cell_coords(x2, y2)
            return

        unit_dx = dx / dist
        unit_dy = dy / dist
        curr_vec = np.array([unit_dx, unit_dy], dtype=np.float32)

        # Center point
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        r, c = self._get_cell_coords(mid_x, mid_y)

        # Update sample count & transit density
        self.sample_counts[r, c] += 1
        self.transit_density[r, c] = min(1.0, self.transit_density[r, c] + 0.05)

        # Update directional vector using Exponential Moving Average / Vector Average
        current_flow = self.flow_vectors[r, c]
        if np.linalg.norm(current_flow) < 1e-4:
            self.flow_vectors[r, c] = curr_vec
            self.confidence[r, c] = 0.2
            self.mean_speed[r, c] = float(dist if speed is None else speed)
        else:
            # Check alignment with existing flow
            dot_prod = float(np.dot(current_flow, curr_vec))
            
            # Blend new observation
            updated_flow = (1.0 - self.lr) * current_flow + self.lr * curr_vec
            norm = np.linalg.norm(updated_flow)
            if norm > 1e-4:
                self.flow_vectors[r, c] = updated_flow / norm

            # Confidence increases when motions are consistently aligned
            if dot_prod > 0.5:
                self.confidence[r, c] = min(1.0, self.confidence[r, c] + 0.08)
            elif dot_prod < -0.3:
                # Opposing motions in same cell decrease confidence slightly
                self.confidence[r, c] = max(0.1, self.confidence[r, c] - 0.05)

            # Update speed EMA
            observed_speed = float(dist if speed is None else speed)
            self.mean_speed[r, c] = (1.0 - self.lr) * self.mean_speed[r, c] + self.lr * observed_speed

    def get_flow_at(self, x: float, y: float) -> Tuple[np.ndarray, float, float, float]:
        """
        Queries learned flow vector, confidence, mean speed, and transit density at (x, y).
        Returns: (flow_unit_vector, confidence, mean_speed, transit_density)
        """
        r, c = self._get_cell_coords(x, y)
        return (
            self.flow_vectors[r, c],
            float(self.confidence[r, c]),
            float(self.mean_speed[r, c]),
            float(self.transit_density[r, c])
        )

    def is_active_transit_zone(self, x: float, y: float, density_threshold: float = 0.15) -> bool:
        """
        Returns True if the location is identified as an active travel lane/transit corridor.
        """
        r, c = self._get_cell_coords(x, y)
        return bool(self.transit_density[r, c] >= density_threshold or self.sample_counts[r, c] >= 2)


class WrongDirectionDetector:
    """
    Evaluates vehicle kinematic vectors against learned lane directional flow vectors.
    Flags wrong-way traffic if vehicle travels in opposite/divergent direction with high confidence.
    """
    def __init__(
        self,
        flow_learner: SpatialFlowGridLearner,
        angle_threshold_deg: float = 100.0,
        min_confidence: float = 0.25,
        min_speed: float = 1.0
    ):
        self.flow_learner = flow_learner
        self.angle_threshold_deg = angle_threshold_deg
        self.min_confidence = min_confidence
        self.min_speed = min_speed

    def evaluate_track(
        self,
        track_id: int,
        position_curr: Tuple[float, float],
        velocity_curr: Tuple[float, float],
        speed: float
    ) -> Optional[Dict[str, Any]]:
        """
        Checks if vehicle motion violates the learned lane direction.
        """
        if speed < self.min_speed:
            return None  # Vehicle is too slow/stopped for meaningful direction evaluation

        vx, vy = velocity_curr
        v_norm = math.hypot(vx, vy)
        if v_norm < 1e-4:
            return None

        u_veh = np.array([vx / v_norm, vy / v_norm], dtype=np.float32)

        flow_vec, conf, mean_spd, transit_dens = self.flow_learner.get_flow_at(position_curr[0], position_curr[1])

        if conf < self.min_confidence or np.linalg.norm(flow_vec) < 1e-4:
            return None  # Not enough learned data for this cell yet

        # Compute dot product and angle between vehicle direction and learned lane direction
        dot_product = float(np.clip(np.dot(u_veh, flow_vec), -1.0, 1.0))
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)

        if angle_deg >= self.angle_threshold_deg:
            # Violation: Vehicle is driving against the dominant flow
            severity = "CRITICAL" if angle_deg >= 140.0 else "WARNING"
            return {
                "violation_type": "WRONG_DIRECTION_IN_LANE",
                "track_id": track_id,
                "angle_deviation_deg": round(angle_deg, 1),
                "dot_product": round(dot_product, 3),
                "lane_confidence": round(conf, 3),
                "vehicle_heading": [round(float(u_veh[0]), 3), round(float(u_veh[1]), 3)],
                "learned_lane_flow": [round(float(flow_vec[0]), 3), round(float(flow_vec[1]), 3)],
                "severity": severity,
                "position_2d": [round(position_curr[0], 1), round(position_curr[1], 1)]
            }

        return None


class TransitZoneParkingDetector:
    """
    Monitors vehicle dwell time in active travel/transit corridors.
    Flags stopped or parked vehicles that obstruct normal traffic flow.
    """
    def __init__(
        self,
        flow_learner: SpatialFlowGridLearner,
        stationary_speed_threshold: float = 0.8,
        stop_time_threshold_seconds: float = 3.0,
        fps: float = 30.0
    ):
        self.flow_learner = flow_learner
        self.stationary_speed_threshold = stationary_speed_threshold
        self.stop_frames_threshold = int(stop_time_threshold_seconds * fps)
        self.fps = fps

        # Track ID -> { "stationary_frames": int, "initial_pos": (x,y), "start_time": float }
        self.stationary_track_records: Dict[int, Dict[str, Any]] = {}

    def update_and_evaluate(
        self,
        track_id: int,
        position_curr: Tuple[float, float],
        speed: float
    ) -> Optional[Dict[str, Any]]:
        """
        Updates stationary counter and checks for illegal stopping/parking in transit lanes.
        """
        x, y = position_curr
        is_stopped = speed < self.stationary_speed_threshold
        is_transit_zone = self.flow_learner.is_active_transit_zone(x, y)

        if is_stopped:
            if track_id not in self.stationary_track_records:
                self.stationary_track_records[track_id] = {
                    "stationary_frames": 1,
                    "pos": position_curr,
                    "first_stopped_time": time.time()
                }
            else:
                self.stationary_track_records[track_id]["stationary_frames"] += 1
                self.stationary_track_records[track_id]["pos"] = position_curr

            record = self.stationary_track_records[track_id]
            frames_stopped = record["stationary_frames"]
            duration_sec = frames_stopped / self.fps

            if frames_stopped >= self.stop_frames_threshold and is_transit_zone:
                violation_type = (
                    "VEHICLE_PARKED_IN_TRAVEL_LANE" if duration_sec >= 6.0
                    else "ILLEGAL_STOPPING_IN_TRANSIT_LANE"
                )
                return {
                    "violation_type": violation_type,
                    "track_id": track_id,
                    "stopped_duration_sec": round(duration_sec, 2),
                    "stationary_frames": frames_stopped,
                    "speed": round(speed, 2),
                    "position_2d": [round(x, 1), round(y, 1)],
                    "is_active_transit_zone": True
                }
        else:
            # Vehicle moved again; reset or decrement stationary record
            if track_id in self.stationary_track_records:
                self.stationary_track_records[track_id]["stationary_frames"] = max(
                    0, self.stationary_track_records[track_id]["stationary_frames"] - 2
                )
                if self.stationary_track_records[track_id]["stationary_frames"] == 0:
                    del self.stationary_track_records[track_id]

        return None

    def purge_inactive_tracks(self, active_track_ids: List[int]):
        """Cleans up internal records for lost tracks."""
        active_set = set(active_track_ids)
        to_remove = [tid for tid in self.stationary_track_records if tid not in active_set]
        for tid in to_remove:
            del self.stationary_track_records[tid]


class TrafficIntelligenceEngine:
    """
    Unified Orchestrator combining:
      - Road marking discovery
      - Spatial vector field flow learning
      - Wrong-direction traffic evaluation
      - Illegal stopping/parking detection in active travel corridors
      - Real-time HUD visual overlay generation
    """
    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        cell_size: int = 40,
        wrong_way_angle_deg: float = 100.0,
        stop_time_sec: float = 3.0,
        fps: float = 30.0
    ):
        self.marking_detector = RoadMarkingDetector()
        self.flow_learner = SpatialFlowGridLearner(
            grid_width=frame_width,
            grid_height=frame_height,
            cell_size=cell_size
        )
        self.wrong_way_detector = WrongDirectionDetector(
            flow_learner=self.flow_learner,
            angle_threshold_deg=wrong_way_angle_deg,
            min_confidence=0.20
        )
        self.parking_detector = TransitZoneParkingDetector(
            flow_learner=self.flow_learner,
            stop_time_threshold_seconds=stop_time_sec,
            fps=fps
        )

        self.last_marking_update_frame = -100
        self.cached_markings: Dict[str, Any] = {"marking_lines": [], "orientations": []}

    def process_frame_tracks(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        tracks: List[Any],
        pixel_projector_func: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Processes one video frame:
          1. Periodically refreshes road markings.
          2. Continuously trains flow vectors from moving tracks.
          3. Evaluates wrong-direction and illegal stopping violations.
        """
        # 1. Update road markings every 30 frames (1 Hz at 30 fps)
        if frame_idx - self.last_marking_update_frame >= 30:
            self.cached_markings = self.marking_detector.detect_markings(frame_bgr)
            self.last_marking_update_frame = frame_idx

        violations = []
        track_evaluations = {}
        active_track_ids = []

        # 2. Process active tracks
        for t in tracks:
            active_track_ids.append(t.track_id)
            
            # Determine 2D pixel position
            if pixel_projector_func is not None:
                p2d = pixel_projector_func(t.position)
            else:
                p2d = (int(t.position[0]), int(t.position[1]))

            if p2d is None:
                continue

            px, py = float(p2d[0]), float(p2d[1])
            speed = getattr(t, 'speed', 0.0)

            # Compute 2D velocity from trajectory history
            if len(t.trajectory_history) >= 2:
                if pixel_projector_func is not None:
                    p_prev_2d = pixel_projector_func(t.trajectory_history[-2])
                else:
                    p_prev_2d = (t.trajectory_history[-2][0], t.trajectory_history[-2][1])
                
                if p_prev_2d is not None:
                    vx_2d = px - p_prev_2d[0]
                    vy_2d = py - p_prev_2d[1]
                    # Update online flow learner
                    self.flow_learner.update_from_trajectory(p_prev_2d, (px, py), speed)
                else:
                    vx_2d, vy_2d = 0.0, 0.0
            else:
                vx_2d, vy_2d = 0.0, 0.0

            # 3. Evaluate Wrong Direction Violation
            wrong_way_viol = self.wrong_way_detector.evaluate_track(
                track_id=t.track_id,
                position_curr=(px, py),
                velocity_curr=(vx_2d, vy_2d),
                speed=speed
            )
            if wrong_way_viol:
                violations.append(wrong_way_viol)

            # 4. Evaluate Illegal Stopping / Parking in Active Transit Corridor
            parking_viol = self.parking_detector.update_and_evaluate(
                track_id=t.track_id,
                position_curr=(px, py),
                speed=speed
            )
            if parking_viol:
                violations.append(parking_viol)

            track_evaluations[t.track_id] = {
                "wrong_way": wrong_way_viol,
                "parking": parking_viol,
                "pos_2d": (int(px), int(py))
            }

        # Purge inactive tracks from parking monitor
        self.parking_detector.purge_inactive_tracks(active_track_ids)

        return {
            "violations": violations,
            "track_evaluations": track_evaluations,
            "markings": self.cached_markings
        }

    def render_overlay(
        self,
        vis: np.ndarray,
        track_evaluations: Dict[int, Any],
        show_flow_field: bool = True,
        show_markings: bool = True
    ) -> np.ndarray:
        """
        Renders futuristic learned flow arrows, marking highlights, and violation indicators.
        """
        h, w = vis.shape[:2]
        cell_size = self.flow_learner.cell_size

        # 1. Render Detected Road Markings
        if show_markings and self.cached_markings.get("marking_lines"):
            for x1, y1, x2, y2 in self.cached_markings["marking_lines"]:
                cv2.line(vis, (x1, y1), (x2, y2), (255, 230, 0), 1, cv2.LINE_AA)

        # 2. Render Learned Directional Flow Vector Grid
        if show_flow_field:
            for r in range(self.flow_learner.rows):
                for c in range(self.flow_learner.cols):
                    conf = self.flow_learner.confidence[r, c]
                    dens = self.flow_learner.transit_density[r, c]
                    vec = self.flow_learner.flow_vectors[r, c]

                    if conf > 0.15 and (abs(vec[0]) > 0.01 or abs(vec[1]) > 0.01):
                        cx = int((c + 0.5) * cell_size)
                        cy = int((r + 0.5) * cell_size)

                        arrow_len = int(12 + 10 * min(1.0, dens))
                        tip_x = int(cx + vec[0] * arrow_len)
                        tip_y = int(cy + vec[1] * arrow_len)

                        # Color based on confidence (Teal/Cyan for high confidence flow)
                        alpha_c = int(conf * 255)
                        color = (int(255 * conf), int(220 * conf), 50)
                        
                        cv2.arrowedLine(vis, (cx, cy), (tip_x, tip_y), color, 1, tipLength=0.35)

        # 3. Render Vehicle Specific Alerts & Target Badges
        for tid, eval_data in track_evaluations.items():
            pos_2d = eval_data.get("pos_2d")
            if not pos_2d:
                continue

            cx, cy = pos_2d
            ww = eval_data.get("wrong_way")
            pk = eval_data.get("parking")

            if ww:
                # Flashing Red Wrong Way Warning Box
                box_sz = 30
                cv2.rectangle(vis, (cx - box_sz, cy - box_sz), (cx + box_sz, cy + box_sz), (0, 0, 255), 3)
                cv2.putText(vis, "! WRONG WAY !", (cx - 48, cy - box_sz - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(vis, f"DEV: {ww['angle_deviation_deg']} deg", (cx - 45, cy + box_sz + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 255), 1, cv2.LINE_AA)

            elif pk:
                # Flashing Amber/Orange Stopped/Parked Obstruction Warning Box
                box_sz = 28
                cv2.rectangle(vis, (cx - box_sz, cy - box_sz), (cx + box_sz, cy + box_sz), (0, 140, 255), 2)
                lbl = "PARKED IN LANE" if pk['violation_type'] == "VEHICLE_PARKED_IN_TRAVEL_LANE" else "STOPPED IN LANE"
                cv2.putText(vis, f"! {lbl} !", (cx - 56, cy - box_sz - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 255), 2, cv2.LINE_AA)
                cv2.putText(vis, f"DWELL: {pk['stopped_duration_sec']}s", (cx - 42, cy + box_sz + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

        return vis
