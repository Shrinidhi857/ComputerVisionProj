"""
Unit and Integration Tests for ML Traffic Direction & Road Markings Learning System.
"""

import unittest
import numpy as np
import cv2
from traffic_learning_system import (
    RoadMarkingDetector,
    SpatialFlowGridLearner,
    WrongDirectionDetector,
    TransitZoneParkingDetector,
    TrafficIntelligenceEngine
)
from governance import CryptographicViolationSigner


class MockTrack:
    """Mock Track object mimicking Track3D interface."""
    def __init__(self, track_id: int, position: np.ndarray, velocity: np.ndarray, speed: float):
        self.track_id = track_id
        self.position = position
        self.velocity = velocity
        self.speed = speed
        self.trajectory_history = [
            position - velocity * 2.0,
            position - velocity,
            position.copy()
        ]


class TestTrafficLearningSystem(unittest.TestCase):

    def setUp(self):
        self.frame_w = 640
        self.frame_h = 480
        self.learner = SpatialFlowGridLearner(
            grid_width=self.frame_w,
            grid_height=self.frame_h,
            cell_size=40,
            learning_rate=0.20
        )
        self.wrong_way_detector = WrongDirectionDetector(
            flow_learner=self.learner,
            angle_threshold_deg=100.0,
            min_confidence=0.20,
            min_speed=0.5
        )
        self.parking_detector = TransitZoneParkingDetector(
            flow_learner=self.learner,
            stationary_speed_threshold=0.5,
            stop_time_threshold_seconds=1.0, # 30 frames at 30 fps
            fps=30.0
        )

    def test_road_marking_detector(self):
        """Test marking detector on a synthetic frame with white and yellow road lines."""
        frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        # Draw white dashed line
        for y in range(200, 450, 40):
            cv2.line(frame, (300, y), (300, y + 20), (255, 255, 255), 3)
        # Draw yellow solid lane divider
        cv2.line(frame, (400, 200), (400, 460), (0, 255, 255), 4)

        detector = RoadMarkingDetector()
        res = detector.detect_markings(frame)
        
        self.assertIn("marking_lines", res)
        self.assertIn("orientations", res)
        self.assertGreater(len(res["marking_lines"]), 0)

    def test_spatial_flow_learning_convergence(self):
        """Test that vehicle trajectories going in (+X, 0) direction update the flow vector."""
        # Simulate vehicles traveling eastward across row cell at y=240
        for i in range(10):
            x1, y1 = 100.0 + i * 20, 240.0
            x2, y2 = 120.0 + i * 20, 240.0
            self.learner.update_from_trajectory((x1, y1), (x2, y2), speed=20.0)

        vec, conf, mean_spd, dens = self.learner.get_flow_at(200.0, 240.0)
        self.assertGreater(conf, 0.20)
        # Direction vector should be pointed approximately [1.0, 0.0]
        self.assertGreater(vec[0], 0.90)
        self.assertAlmostEqual(vec[1], 0.0, delta=0.2)
        self.assertTrue(self.learner.is_active_transit_zone(200.0, 240.0))

    def test_wrong_direction_detection(self):
        """Test detection of vehicle driving opposite to learned flow."""
        # Train lane flow going downward (+Y)
        for i in range(15):
            self.learner.update_from_trajectory((200.0, 100.0 + i * 15), (200.0, 120.0 + i * 15), speed=20.0)

        # Compliant vehicle moving downward
        viol_compliant = self.wrong_way_detector.evaluate_track(
            track_id=1,
            position_curr=(200.0, 200.0),
            velocity_curr=(0.0, 15.0),
            speed=15.0
        )
        self.assertIsNone(viol_compliant)

        # Wrong-way vehicle moving upward (-Y)
        viol_wrong = self.wrong_way_detector.evaluate_track(
            track_id=2,
            position_curr=(200.0, 200.0),
            velocity_curr=(0.0, -15.0),
            speed=15.0
        )
        self.assertIsNotNone(viol_wrong)
        self.assertEqual(viol_wrong["violation_type"], "WRONG_DIRECTION_IN_LANE")
        self.assertGreater(viol_wrong["angle_deviation_deg"], 150.0)

    def test_illegal_stopping_and_parking_detection(self):
        """Test that vehicle stopping in active transit lane triggers violation after threshold."""
        # First establish active transit zone
        for i in range(10):
            self.learner.update_from_trajectory((300.0, 200.0), (320.0, 200.0), speed=15.0)

        # Vehicle stops at (310, 200)
        viol = None
        for frame in range(40): # > 30 frames (1.0 sec threshold)
            viol = self.parking_detector.update_and_evaluate(
                track_id=101,
                position_curr=(310.0, 200.0),
                speed=0.1
            )
            if frame < 29:
                self.assertIsNone(viol)

        self.assertIsNotNone(viol)
        self.assertIn("STOPPING", viol["violation_type"])

    def test_cryptographic_violation_signing(self):
        """Test cryptographic signing of wrong-direction and parked violations."""
        signer = CryptographicViolationSigner("TEST_NODE_1", "test_secret_key_99")
        
        payload = signer.generate_payload(
            vehicle_track_id=42,
            violation_type="WRONG_DIRECTION_IN_LANE",
            position_3d=np.array([12.0, 4.0, 0.0]),
            velocity_3d=np.array([-5.0, 0.0, 0.0]),
            trajectory=[[10.0, 4.0, 0.0], [12.0, 4.0, 0.0]],
            boundary_info={"angle_deviation_deg": 178.5, "severity": "CRITICAL"}
        )

        self.assertIn("record_hash", payload)
        self.assertIn("device_signature", payload)
        self.assertEqual(payload["violation_type"], "WRONG_DIRECTION_IN_LANE")


if __name__ == '__main__':
    unittest.main()
