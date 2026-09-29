"""
Optimized 3D Traffic Governance Pipeline with ML Road Markings & Direction Learning.
Integrates:
  - Video Input (File / Webcam / RTSP / Simulated Synthetic Stream)
  - YOLOv8 / Fast Adaptive 2D/3D Vehicle Detection & Tracking (Kalman Filter)
  - ML Road Marking & Lane Boundary Discovery (Hough / Adaptive Morphology)
  - Online Spatial Vector Field Learning (Dominant Lane Direction & Velocity)
  - Real-Time Wrong-Direction Traffic Violation Detection
  - Illegal Stopping / Parked Vehicle Detection in Active Transit Corridors
  - Cryptographic Tamper-Evident Ed25519 / SHA-256 Audit Trail Chaining
  - Futuristic Live Visual HUD Preview with Directional Flow Vectors & Alerts
"""

import time
import argparse
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple

from geometry import HolographicPlane, ConvexPolyhedronCorridor, AltitudeEnforcer
from depth_unprojection import CameraModel, MonocularDepthCalibrator
from tracker_3d import Tracker3D, Track3D
from governance import CryptographicViolationSigner
from traffic_learning_system import TrafficIntelligenceEngine


class EdgeGovernancePipeline:
    """
    High-throughput 3D Traffic Governance Engine with Machine Learning Direction & Markings Intelligence.
    """
    def __init__(
        self,
        camera_model: CameraModel,
        depth_calibrator: MonocularDepthCalibrator,
        altitude_enforcer: AltitudeEnforcer,
        flight_corridors: List[ConvexPolyhedronCorridor],
        holographic_planes: List[HolographicPlane],
        signer: CryptographicViolationSigner,
        traffic_ml_engine: Optional[TrafficIntelligenceEngine] = None,
        depth_stride: int = 3,
        show_flow_field: bool = True,
        show_markings: bool = True
    ):
        self.camera = camera_model
        self.depth_calibrator = depth_calibrator
        self.altitude_enforcer = altitude_enforcer
        self.corridors = flight_corridors
        self.planes = holographic_planes
        self.signer = signer
        self.depth_stride = depth_stride
        self.show_flow_field = show_flow_field
        self.show_markings = show_markings

        self.traffic_ml = traffic_ml_engine or TrafficIntelligenceEngine(
            frame_width=1280,
            frame_height=720,
            cell_size=40,
            wrong_way_angle_deg=100.0,
            stop_time_sec=3.0
        )

        self.tracker = Tracker3D(max_age=15, min_hits=3, distance_threshold=8.0)
        self.frame_count = 0
        self.cached_depth_map: Optional[np.ndarray] = None
        self.reported_violations = set()
        self.recent_alerts: List[Dict[str, Any]] = []

        # Initialize YOLO if available
        self.yolo_model = None
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8s.pt')
            print("[INFO] YOLOv8s initialized successfully.")
        except Exception:
            print("[INFO] Ultralytics not found. Using high-performance background subtraction & geometry detector.")
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)

    def detect_vehicles_2d(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Detects 2D bounding boxes using YOLOv8 or dynamic visual segmentation."""
        if self.yolo_model is not None:
            results = self.yolo_model(frame, verbose=False, classes=[2, 3, 5, 7]) # car, motorcycle, bus, truck
            boxes = []
            for r in results:
                for b in r.boxes.xyxy.cpu().numpy():
                    boxes.append(tuple(b))
            return boxes

        # Fast built-in detector fallback
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) > 600:
                x, y, w, h = cv2.boundingRect(c)
                boxes.append((float(x), float(y), float(x + w), float(y + h)))
        return boxes

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimates monocular depth or generates geometric inverse-perspective depth."""
        h, w = frame.shape[:2]
        y_indices = np.linspace(50.0, 5.0, h, dtype=np.float32)
        depth_map = np.tile(y_indices[:, None], (1, w))
        return depth_map

    def project_world_to_pixel(self, p_world: np.ndarray) -> Optional[Tuple[int, int]]:
        """Projects a 3D world coordinate point back to 2D image coordinates."""
        p_cam = np.linalg.inv(self.camera.R) @ (p_world.reshape(3, 1) - self.camera.t)
        xc, yc, zc = p_cam.flatten()
        if zc <= 0.1:
            return None
        u = int(self.camera.fx * (xc / zc) + self.camera.cx)
        v = int(self.camera.fy * (yc / zc) + self.camera.cy)
        return (u, v)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes one video frame, evaluates 3D spatial boundaries, ML direction/parking violations,
        generates cryptographic violations, and renders a real-time HUD preview overlay.
        """
        self.frame_count += 1
        t_start = time.perf_counter()
        h, w = frame_bgr.shape[:2]

        # 1. Depth Estimation (temporal striding)
        if self.frame_count % self.depth_stride == 0 or self.cached_depth_map is None:
            self.cached_depth_map = self.estimate_depth(frame_bgr)

        # 2. 2D Vehicle Detection
        boxes_2d = self.detect_vehicles_2d(frame_bgr)

        # 3. 3D Spatial Unprojection
        detections_3d: List[np.ndarray] = []
        for box in boxes_2d:
            x1, y1, x2, y2 = box
            u_c = (x1 + x2) / 2.0
            v_c = (y1 + y2) / 2.0
            z_metric = self.depth_calibrator.extract_robust_roi_depth(self.cached_depth_map, (int(x1), int(y1), int(x2), int(y2)))
            p_cam = self.camera.unproject_pixel_to_camera(u_c, v_c, z_metric)
            p_world = self.camera.camera_to_world(p_cam)
            detections_3d.append(p_world)

        # 4. 3D Tracking
        active_tracks: List[Track3D] = self.tracker.update(detections_3d)

        # 5. ML Traffic Intelligence (Markings, Direction Field, Wrong-Way & Parking)
        ml_results = self.traffic_ml.process_frame_tracks(
            frame_bgr=frame_bgr,
            frame_idx=self.frame_count,
            tracks=active_tracks,
            pixel_projector_func=self.project_world_to_pixel
        )
        ml_violations = ml_results["violations"]
        track_evaluations = ml_results["track_evaluations"]

        frame_violations = []

        # Register ML Direction & Parking Violations
        for ml_v in ml_violations:
            tid = ml_v["track_id"]
            vtype = ml_v["violation_type"]
            viol_key = f"{tid}_{vtype}_{self.frame_count // 25}"
            if viol_key not in self.reported_violations:
                self.reported_violations.add(viol_key)
                
                # Match track object for 3D state
                matched_track = next((t for t in active_tracks if t.track_id == tid), None)
                pos_3d = matched_track.position if matched_track else np.array([ml_v.get("position_2d", [0, 0])[0], ml_v.get("position_2d", [0, 0])[1], 0.0])
                vel_3d = matched_track.velocity if matched_track else np.zeros(3)
                traj_3d = matched_track.trajectory_history if matched_track else [pos_3d]

                payload = self.signer.generate_payload(
                    vehicle_track_id=tid,
                    violation_type=vtype,
                    position_3d=pos_3d,
                    velocity_3d=vel_3d,
                    trajectory=traj_3d,
                    boundary_info=ml_v
                )
                frame_violations.append(payload)
                self.recent_alerts.append(payload)

        # 6. Computational Geometry Boundary Enforcement
        for track in active_tracks:
            if len(track.trajectory_history) < 2:
                continue

            p_prev = track.trajectory_history[-2]
            p_curr = track.trajectory_history[-1]
            v_z = track.velocity[2]

            # Altitude Violations
            alt_viol = self.altitude_enforcer.check_altitude_violation(p_curr, velocity_z=v_z)
            if alt_viol:
                viol_key = f"{track.track_id}_ALT_{self.frame_count // 20}"
                if viol_key not in self.reported_violations:
                    self.reported_violations.add(viol_key)
                    payload = self.signer.generate_payload(
                        vehicle_track_id=track.track_id,
                        violation_type=alt_viol["violation_type"],
                        position_3d=p_curr,
                        velocity_3d=track.velocity,
                        trajectory=track.trajectory_history,
                        boundary_info=alt_viol
                    )
                    frame_violations.append(payload)
                    self.recent_alerts.append(payload)

            # Holographic Plane Breaches
            for idx, plane in enumerate(self.planes):
                intersection = plane.segment_intersection(p_prev, p_curr)
                if intersection:
                    viol_key = f"{track.track_id}_PLANE_{idx}_{self.frame_count // 20}"
                    if viol_key not in self.reported_violations:
                        self.reported_violations.add(viol_key)
                        payload = self.signer.generate_payload(
                            vehicle_track_id=track.track_id,
                            violation_type="HOLOGRAPHIC_BOUNDARY_PENETRATION",
                            position_3d=intersection["intersection_point"],
                            velocity_3d=track.velocity,
                            trajectory=track.trajectory_history,
                            boundary_info={"plane_idx": idx, "t": intersection["t"]}
                        )
                        frame_violations.append(payload)
                        self.recent_alerts.append(payload)

            # Volumetric Corridor Violations
            for corridor in self.corridors:
                corr_viol = corridor.evaluate_trajectory_violation(p_prev, p_curr)
                if corr_viol:
                    viol_key = f"{track.track_id}_CORR_{corridor.corridor_id}_{self.frame_count // 20}"
                    if viol_key not in self.reported_violations:
                        self.reported_violations.add(viol_key)
                        payload = self.signer.generate_payload(
                            vehicle_track_id=track.track_id,
                            violation_type=corr_viol["violation_type"],
                            position_3d=p_curr,
                            velocity_3d=track.velocity,
                            trajectory=track.trajectory_history,
                            boundary_info=corr_viol
                        )
                        frame_violations.append(payload)
                        self.recent_alerts.append(payload)

        # Keep last 4 alerts for HUD display
        if len(self.recent_alerts) > 4:
            self.recent_alerts = self.recent_alerts[-4:]

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # 7. Render HUD and Visual Overlays on Frame
        annotated_frame = self.render_hud(
            frame_bgr, active_tracks, track_evaluations, frame_violations, latency_ms
        )

        metadata = {
            "frame_id": self.frame_count,
            "active_tracks_count": len(active_tracks),
            "new_violations": frame_violations,
            "latency_ms": latency_ms
        }
        return annotated_frame, metadata

    def render_hud(
        self,
        frame: np.ndarray,
        tracks: List[Track3D],
        track_evaluations: Dict[int, Any],
        violations: List[Dict[str, Any]],
        latency_ms: float
    ) -> np.ndarray:
        """Renders rich 3D holographic HUD, ML flow fields, trajectory ribbons, coordinate tags, and audit alerts."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 1. Render ML Learned Flow Overlay and Road Markings
        vis = self.traffic_ml.render_overlay(
            vis,
            track_evaluations,
            show_flow_field=self.show_flow_field,
            show_markings=self.show_markings
        )

        # Draw Holographic Horizon Grid
        cv2.line(vis, (0, int(h * 0.45)), (w, int(h * 0.45)), (255, 140, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, "ALTITUDE CEILING (100.0m)", (w - 260, int(h * 0.45) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 50), 1, cv2.LINE_AA)

        # Render Active 3D Vehicle Tracks
        for t in tracks:
            pos = t.position
            vel = t.velocity
            speed = t.speed
            eval_info = track_evaluations.get(t.track_id, {})
            is_wrong_way = eval_info.get("wrong_way") is not None
            is_parked = eval_info.get("parking") is not None

            # Color scheme: Red if wrong way, Amber if parked, Cyan/Teal if compliant
            if is_wrong_way:
                theme_color = (0, 0, 255)
            elif is_parked:
                theme_color = (0, 140, 255)
            else:
                theme_color = (0, 255, 200)

            # Draw 3D Trajectory History Ribbon
            if len(t.trajectory_history) >= 2:
                pts_2d = []
                for p3d in t.trajectory_history[-15:]:
                    p2d = self.project_world_to_pixel(p3d)
                    if p2d and 0 <= p2d[0] < w and 0 <= p2d[1] < h:
                        pts_2d.append(p2d)
                for i in range(len(pts_2d) - 1):
                    alpha = (i + 1) / len(pts_2d)
                    r_c = int(theme_color[2] * alpha)
                    g_c = int(theme_color[1] * alpha)
                    b_c = int(theme_color[0] * alpha)
                    cv2.line(vis, pts_2d[i], pts_2d[i + 1], (b_c, g_c, r_c), 2, cv2.LINE_AA)

            # Project Current 3D Centroid
            proj_center = self.project_world_to_pixel(pos)
            if proj_center and 0 <= proj_center[0] < w and 0 <= proj_center[1] < h:
                cx, cy = proj_center
                
                # Draw Futuristic Target Marker
                r = 16
                cv2.circle(vis, (cx, cy), r, theme_color, 2, cv2.LINE_AA)
                cv2.circle(vis, (cx, cy), 3, (0, 0, 255) if is_wrong_way else (255, 255, 255), -1)
                cv2.line(vis, (cx - r - 4, cy), (cx - 4, cy), theme_color, 1)
                cv2.line(vis, (cx + 4, cy), (cx + r + 4, cy), theme_color, 1)
                cv2.line(vis, (cx, cy - r - 4), (cx, cy - 4), theme_color, 1)
                cv2.line(vis, (cx, cy + 4), (cx, cy + r + 4), theme_color, 1)

                # Coordinate & Status Tag Card
                tag_x, tag_y = cx + 22, cy - 10
                status_text = "STATUS: WRONG-WAY" if is_wrong_way else ("STATUS: PARKED/STOPPED" if is_parked else "STATUS: COMPLIANT")
                info_lines = [
                    f"ID: #{t.track_id:03d} | {status_text}",
                    f"POS: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]m",
                    f"SPD: {speed * 3.6:.1f} km/h"
                ]
                
                # Semi-transparent background for label
                card_w, card_h = 190, 54
                overlay = vis.copy()
                cv2.rectangle(overlay, (tag_x - 4, tag_y - 14), (tag_x + card_w, tag_y + card_h - 14), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
                cv2.rectangle(vis, (tag_x - 4, tag_y - 14), (tag_x + card_w, tag_y + card_h - 14), theme_color, 1)

                for idx, line in enumerate(info_lines):
                    text_color = (100, 100, 255) if (is_wrong_way and idx == 0) else (255, 255, 255)
                    cv2.putText(vis, line, (tag_x, tag_y + (idx * 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, text_color, 1, cv2.LINE_AA)

        # Header HUD Panel
        hud_h = 52
        hud_overlay = vis.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (10, 15, 25), -1)
        cv2.addWeighted(hud_overlay, 0.85, vis, 0.15, 0, vis)
        cv2.line(vis, (0, hud_h), (w, hud_h), (0, 200, 255), 2)

        cv2.putText(vis, "ANTIGRAVITY 3D TRAFFIC GOVERNANCE & ML LANE INTELLIGENCE ENGINE", (16, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"FRAME: {self.frame_count:05d} | TRACKS: {len(tracks)} | ML FLOW FIELD: ACTIVE | WRONG-WAY SENSOR: ARMED | LATENCY: {latency_ms:.1f}ms",
                    (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 230, 255), 1, cv2.LINE_AA)

        # Depth Heatmap Picture-In-Picture (PIP) in Top-Right
        if self.cached_depth_map is not None:
            d_norm = cv2.normalize(self.cached_depth_map, None, 0, 255, cv2.NORM_MINMAX)
            d_color = cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_TURBO)
            pip_w, pip_h = 160, 90
            d_pip = cv2.resize(d_color, (pip_w, pip_h))
            
            px, py = w - pip_w - 15, hud_h + 10
            vis[py:py+pip_h, px:px+pip_w] = d_pip
            cv2.rectangle(vis, (px, py), (px + pip_w, py + pip_h), (0, 255, 255), 1)
            cv2.putText(vis, "DEPTH MAP (PIP)", (px + 4, py + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Violation Alerts & Cryptographic Hash Feed in Bottom Left
        if self.recent_alerts:
            alert_y = h - 20
            for alert in reversed(self.recent_alerts):
                card_w = 480
                card_h = 42
                overlay = vis.copy()
                cv2.rectangle(overlay, (15, alert_y - card_h + 8), (15 + card_w, alert_y + 8), (10, 10, 50), -1)
                cv2.addWeighted(overlay, 0.8, vis, 0.2, 0, vis)
                cv2.rectangle(vis, (15, alert_y - card_h + 8), (15 + card_w, alert_y + 8), (0, 50, 255), 2)

                cv2.putText(vis, f"! AUDIT VIOLATION: {alert['violation_type']} [VEHICLE #{alert['vehicle_track_id']}]",
                            (22, alert_y - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 120, 255), 1, cv2.LINE_AA)
                cv2.putText(vis, f"HASH: {alert['record_hash'][:24]}... | SIG: {alert['device_signature'][:16]}...",
                            (22, alert_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1, cv2.LINE_AA)
                alert_y -= 48

        return vis


def run_video_pipeline(
    video_source: Optional[str] = None,
    output_path: Optional[str] = None,
    show_preview: bool = True,
    max_frames: Optional[int] = None,
    wrong_way_angle_deg: float = 100.0,
    stop_time_sec: float = 3.0,
    show_flow: bool = True,
    show_markings: bool = True
):
    """
    Runs the full 3D Traffic Governance Pipeline with ML lane learning on a video file or live camera stream.
    """
    # 1. Camera Model & Calibrator
    cam = CameraModel(fx=800.0, fy=800.0, cx=640.0, cy=360.0)
    depth_calib = MonocularDepthCalibrator(scale=400.0, min_depth=2.0, max_depth=200.0)
    alt_enf = AltitudeEnforcer(min_altitude_m=5.0, max_altitude_m=80.0)

    # 2. Virtual Holographic Sky-Wall Plane at X = 35.0m
    sky_wall = HolographicPlane(normal=np.array([1.0, 0.0, 0.0]), point_on_plane=np.array([35.0, 0.0, 0.0]))

    # 3. Volumetric Flight Corridor Box: X in [-25, 25], Y in [-25, 25], Z in [0, 80]
    box_faces = [
        (np.array([1.0, 0.0, 0.0]), np.array([25.0, 0.0, 0.0])),
        (np.array([-1.0, 0.0, 0.0]), np.array([-25.0, 0.0, 0.0])),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, 25.0, 0.0])),
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, -25.0, 0.0])),
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 80.0])),
        (np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 0.0]))
    ]
    corridor = ConvexPolyhedronCorridor(box_faces, corridor_id="CORRIDOR_ALPHA_PRIME", is_restricted=False)

    # 4. Cryptographic Signer for Sensor Node
    signer = CryptographicViolationSigner(sensor_id="EDGE_NODE_ORBITAL_7", private_device_key="tpm_secret_key_8841")

    # 5. ML Traffic Intelligence Engine
    traffic_ml = TrafficIntelligenceEngine(
        frame_width=1280,
        frame_height=720,
        cell_size=40,
        wrong_way_angle_deg=wrong_way_angle_deg,
        stop_time_sec=stop_time_sec
    )

    # 6. Initialize Pipeline Engine
    pipeline = EdgeGovernancePipeline(
        camera_model=cam,
        depth_calibrator=depth_calib,
        altitude_enforcer=alt_enf,
        flight_corridors=[corridor],
        holographic_planes=[sky_wall],
        signer=signer,
        traffic_ml_engine=traffic_ml,
        depth_stride=3,
        show_flow_field=show_flow,
        show_markings=show_markings
    )

    # Open Video Source
    is_synthetic = False
    if video_source is None or video_source == "" or video_source == "synthetic":
        print("[INFO] No input video file specified. Launching high-fidelity synthetic 3D flight stream simulation...")
        is_synthetic = True
        cap = None
        width, height, fps = 1280, 720, 30
    else:
        if video_source.isdigit():
            video_source = int(video_source)
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {video_source}. Falling back to synthetic simulation.")
            is_synthetic = True
            cap = None
            width, height, fps = 1280, 720, 30
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            print(f"[INFO] Opened Video: {video_source} ({width}x{height} @ {fps:.1f} FPS)")

    # Video Writer if requested
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[INFO] Recording output preview to: {output_path}")

    frame_idx = 0
    sim_t = 0.0

    try:
        while True:
            if max_frames and frame_idx >= max_frames:
                break

            if is_synthetic:
                # Realistic synthetic skyway traffic frame with road markings and multi-craft scenarios
                frame = np.full((height, width, 3), 20, dtype=np.uint8)
                # Skyline gradient
                for y_i in range(height):
                    frame[y_i, :] = [int(30 - y_i*0.02), int(25 + y_i*0.04), int(45 + y_i*0.06)]

                # Draw road lanes and markings on bottom half
                cv2.rectangle(frame, (0, 400), (width, height), (40, 42, 45), -1)
                # White lane dividers
                for lx in range(0, width, 100):
                    cv2.line(frame, (lx, 550), (lx + 50, 550), (240, 240, 240), 3)
                # Yellow solid boundary
                cv2.line(frame, (0, 420), (width, 420), (0, 220, 255), 3)

                # Synthetic vehicle 1: Normal compliant traffic flowing Eastbound (+X)
                sim_t += 0.05
                v1_x = int((frame_idx * 12) % (width + 120) - 60)
                v1_y = int(480 + 10 * np.sin(sim_t))
                cv2.rectangle(frame, (v1_x - 35, v1_y - 20), (v1_x + 35, v1_y + 20), (220, 220, 220), -1)
                cv2.circle(frame, (v1_x - 20, v1_y + 15), 5, (0, 255, 255), -1)
                cv2.circle(frame, (v1_x + 20, v1_y + 15), 5, (0, 255, 255), -1)

                # Synthetic vehicle 2: Wrong-way traffic driving Westbound (-X) in the same eastbound lane
                if frame_idx > 30: # Begins after lane direction is established
                    v2_x = int(width - ((frame_idx - 30) * 10) % (width + 100))
                    v2_y = 480
                    cv2.rectangle(frame, (v2_x - 35, v2_y - 20), (v2_x + 35, v2_y + 20), (180, 180, 255), -1)
                    cv2.circle(frame, (v2_x - 20, v2_y + 15), 5, (0, 0, 255), -1)
                    cv2.circle(frame, (v2_x + 20, v2_y + 15), 5, (0, 0, 255), -1)

                # Synthetic vehicle 3: Vehicle stopped/parked in the middle of the active travel lane
                if frame_idx > 20:
                    v3_x = 600
                    v3_y = 560
                    cv2.rectangle(frame, (v3_x - 35, v3_y - 20), (v3_x + 35, v3_y + 20), (160, 220, 160), -1)
                    cv2.circle(frame, (v3_x - 20, v3_y + 15), 5, (0, 165, 255), -1)
                    cv2.circle(frame, (v3_x + 20, v3_y + 15), 5, (0, 165, 255), -1)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] Video stream completed.")
                    break

            # Execute pipeline
            annotated_frame, meta = pipeline.process_frame(frame)

            if writer:
                writer.write(annotated_frame)

            if show_preview:
                cv2.imshow("Antigravity 3D Traffic Governance - Live Preview", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] User terminated live preview.")
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)

            frame_idx += 1

    finally:
        if cap:
            cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        print(f"[COMPLETE] Processed {frame_idx} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Antigravity 3D Traffic Governance with ML Lane Intelligence")
    parser.add_argument("--video", type=str, default="", help="Path to input video file (or camera index like 0, or 'synthetic')")
    parser.add_argument("--output", type=str, default="", help="Path to save annotated output video (.mp4)")
    parser.add_argument("--no-preview", action="store_true", help="Disable live cv2 window preview")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--wrong-way-angle", type=float, default=100.0, help="Angle deviation threshold in degrees for wrong-way detection")
    parser.add_argument("--stop-time-sec", type=float, default=3.0, help="Stationary dwell time threshold in seconds in travel lane")
    parser.add_argument("--hide-flow", action="store_true", help="Hide learned directional flow vector arrows")
    parser.add_argument("--hide-markings", action="store_true", help="Hide road markings overlay")

    args = parser.parse_args()

    run_video_pipeline(
        video_source=args.video if args.video else None,
        output_path=args.output if args.output else None,
        show_preview=not args.no_preview,
        max_frames=args.max_frames,
        wrong_way_angle_deg=args.wrong_way_angle,
        stop_time_sec=args.stop_time_sec,
        show_flow=not args.hide_flow,
        show_markings=not args.hide_markings
    )
