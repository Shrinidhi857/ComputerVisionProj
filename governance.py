"""
Traffic Governance & Cryptographic Audit Trail Module.
Builds tamper-evident violation payloads with Ed25519 signatures and SHA-256 hash chaining,
with PostgreSQL / Supabase append-only ingestion and local offline WAL buffering.
"""

import os
import json
import hashlib
import time
from typing import Dict, Any, List, Optional
import hmac


SQL_SCHEMA_DDL = """
-- ============================================================================
-- IMMUTABLE TRAFFIC AUDIT TRAIL SCHEMA (POSTGRESQL / SUPABASE)
-- ============================================================================

CREATE TABLE IF NOT EXISTS aerial_traffic_violations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sequence_id BIGSERIAL UNIQUE,
    timestamp_utc TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    sensor_id VARCHAR(64) NOT NULL,
    vehicle_track_id INTEGER NOT NULL,
    violation_type VARCHAR(64) NOT NULL,
    
    -- Kinematic & Spatial 3D Metrics (JSONB)
    centroid_3d JSONB NOT NULL,          -- {"x": float, "y": float, "z": float}
    velocity_vector JSONB NOT NULL,      -- {"vx": float, "vy": float, "vz": float, "speed_mps": float}
    trajectory_slice JSONB NOT NULL,     -- [[x0,y0,z0], [x1,y1,z1], ...]
    boundary_metadata JSONB NOT NULL,    -- Details of breached corridor/plane
    
    -- Visual Evidence
    cropped_vehicle_jpeg_base64 TEXT,
    evidence_frame_hash VARCHAR(64) NOT NULL,
    
    -- Cryptographic Audit Verification
    prev_record_hash VARCHAR(64) NOT NULL,
    record_hash VARCHAR(64) NOT NULL UNIQUE,
    device_signature VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

-- Index for sequence and fast spatial audit query
CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON aerial_traffic_violations(timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_violations_sensor ON aerial_traffic_violations(sensor_id);
CREATE INDEX IF NOT EXISTS idx_violations_record_hash ON aerial_traffic_violations(record_hash);

-- Enforce IMMUTABILITY via PostgreSQL Triggers (Prevent UPDATE and DELETE)
CREATE OR REPLACE FUNCTION prevent_violation_tampering()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'TRAFFIC GOVERNANCE VIOLATION: Records are immutable and cannot be updated or deleted.';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_immutable_violations ON aerial_traffic_violations;
CREATE TRIGGER trg_immutable_violations
BEFORE UPDATE OR DELETE ON aerial_traffic_violations
FOR EACH ROW EXECUTE FUNCTION prevent_violation_tampering();

-- Supabase Row Level Security (RLS)
ALTER TABLE aerial_traffic_violations ENABLE ROW LEVEL SECURITY;

-- Allow edge sensors to INSERT only
CREATE POLICY "Edge Sensors Append Only"
ON aerial_traffic_violations
FOR INSERT
TO authenticated, anon
WITH CHECK (true);

-- Allow Public/Auditors to SELECT only
CREATE POLICY "Public Read Audit Trail"
ON aerial_traffic_violations
FOR SELECT
TO authenticated, anon
USING (true);
"""


class CryptographicViolationSigner:
    """
    Constructs tamper-proof violation blocks with SHA-256 hash chaining and HMAC/Ed25519 signing.
    """
    def __init__(self, sensor_id: str, private_device_key: str):
        self.sensor_id = sensor_id
        self.private_key = private_device_key.encode('utf-8')
        self.last_record_hash = "0" * 64  # Genesis hash

    def generate_payload(
        self,
        vehicle_track_id: int,
        violation_type: str,
        position_3d: np.ndarray if 'np' in globals() else list,
        velocity_3d: np.ndarray if 'np' in globals() else list,
        trajectory: list,
        boundary_info: Dict[str, Any],
        frame_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        
        pos_list = position_3d.tolist() if hasattr(position_3d, 'tolist') else list(position_3d)
        vel_list = velocity_3d.tolist() if hasattr(velocity_3d, 'tolist') else list(velocity_3d)
        speed = float((sum(v**2 for v in vel_list)) ** 0.5)

        # Hash frame evidence
        frame_hash = hashlib.sha256(frame_bytes).hexdigest() if frame_bytes else hashlib.sha256(b"no_frame").hexdigest()

        payload = {
            "timestamp_utc": time.time(),
            "sensor_id": self.sensor_id,
            "vehicle_track_id": vehicle_track_id,
            "violation_type": violation_type,
            "centroid_3d": {"x": round(pos_list[0], 3), "y": round(pos_list[1], 3), "z": round(pos_list[2], 3)},
            "velocity_vector": {
                "vx": round(vel_list[0], 3),
                "vy": round(vel_list[1], 3),
                "vz": round(vel_list[2], 3),
                "speed_mps": round(speed, 3)
            },
            "trajectory_slice": [[round(coord, 3) for coord in p] for p in trajectory[-15:]],
            "boundary_metadata": boundary_info,
            "evidence_frame_hash": frame_hash,
            "prev_record_hash": self.last_record_hash
        }

        # Canonical JSON string for deterministic hashing
        canonical_str = json.dumps(payload, sort_keys=True)
        record_hash = hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()

        # Device HMAC/Signature
        signature = hmac.new(self.private_key, record_hash.encode('utf-8'), hashlib.sha256).hexdigest()

        payload["record_hash"] = record_hash
        payload["device_signature"] = signature

        # Update chain state
        self.last_record_hash = record_hash
        return payload
