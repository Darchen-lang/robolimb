"""
ROBOLIMB - Complete Pipeline
Flow: Webcam → YOLOv8 Detection → 3D Coordinates → Inverse Kinematics → Pick Object
"""

import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from collections import deque
from typing import Optional, Tuple, Dict, List

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import math
import time
import serial  # pip install pyserial  (for Arduino/servo communication)
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════

# Camera settings
CAMERA_INDEX       = 2
FRAME_WIDTH        = 640
FRAME_HEIGHT       = 480
FOCAL_LENGTH_PX    = 800          # calibrate with checkerboard for best results
KNOWN_OBJECT_WIDTH = 0.07         # meters — average object width (e.g. small cup ~7cm)

# Workspace (meters) — adjust to your arm's reach
WORKSPACE_X_RANGE  = (-0.3, 0.3)
WORKSPACE_Y_RANGE  = (0.0,  0.5)
WORKSPACE_Z_RANGE  = (0.0,  0.4)

# Robotic arm link lengths (meters) — CHANGE to your robot's actual values
L1 = 0.15    # base to shoulder
L2 = 0.15    # shoulder to elbow
L3 = 0.10    # elbow to wrist
L4 = 0.06    # wrist to gripper tip

# Serial port for Arduino/servo controller (set to None to run in simulation mode)
SERIAL_PORT = None          # e.g. "COM5" on Windows, "/dev/ttyUSB0" on Linux
SERIAL_BAUD = 115200

# Detection confidence threshold
CONF_THRESHOLD = 0.55

# Precision & filtering settings
TEMPORAL_HISTORY_SIZE = 5       # Number of frames to track for smoothing
KALMAN_Q = 0.01                 # Process noise (lower = smoother tracking)
KALMAN_R = 0.1                  # Measurement noise (lower = trust measurements more)
MIN_TEMPORAL_CONFIDENCE = 0.70   # Min stability score (0-1) for valid coordinate

# Gripper servo values
GRIPPER_OPEN   = 0
GRIPPER_CLOSED = 90


# ═══════════════════════════════════════════════════════
#  CAMERA CALIBRATION  (replace with your values)
# ═══════════════════════════════════════════════════════

camera_matrix = np.array([
    [FOCAL_LENGTH_PX, 0,               FRAME_WIDTH  / 2],
    [0,               FOCAL_LENGTH_PX, FRAME_HEIGHT / 2],
    [0,               0,               1               ]
], dtype=np.float64)

# Lens distortion coefficients (k1, k2, p1, p2, k3)
# For best results, calibrate with checkerboard pattern. Use zeros for approximate/webcam
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

# Precompute undistortion maps for faster processing
map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, 
                                            (FRAME_WIDTH, FRAME_HEIGHT), cv2.CV_32F)


# ═══════════════════════════════════════════════════════
#  TEMPORAL FILTERING (Kalman Filter for coordinate smoothing)
# ═══════════════════════════════════════════════════════

class KalmanFilter1D:
    """Simple 1D Kalman filter for smoothing noisy measurements."""
    def __init__(self, q: float = 0.01, r: float = 0.1):
        self.q = q              # process noise
        self.r = r              # measurement noise
        self.x = 0.0            # state (position)
        self.p = 1.0            # error covariance
        self.initialized = False

    def update(self, measurement: float) -> float:
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return measurement

        # Predict
        p_pred = self.p + self.q

        # Update
        k = p_pred / (p_pred + self.r)  # Kalman gain
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * p_pred

        return self.x


class CoordinateTracker:
    """Track object coordinates with temporal filtering and confidence scoring."""
    def __init__(self, history_size: int = 5):
        self.history_size = history_size
        self.position_history: Dict[int, deque] = {}  # obj_id -> deque of (x,y,z)
        self.kalman_filters: Dict[int, List[KalmanFilter1D]] = {}  # obj_id -> [kf_x, kf_y, kf_z]
        self.confidence_history: Dict[int, deque] = {}  # obj_id -> deque of confidence scores
        self.frame_count = 0

    def track(self, obj_id: int, raw_pos: Tuple[float, float, float], 
              confidence: float) -> Tuple[Tuple[float, float, float], float]:
        """
        Track object and return smoothed position + temporal confidence score.
        obj_id: unique identifier for tracking
        raw_pos: (x, y, z) raw 3D position from detector
        confidence: detection confidence (0-1)
        Returns: (smoothed_pos, temporal_confidence_score)
        """
        self.frame_count += 1

        # Initialize tracking for new object
        if obj_id not in self.position_history:
            self.position_history[obj_id] = deque(maxlen=self.history_size)
            self.kalman_filters[obj_id] = [
                KalmanFilter1D(q=KALMAN_Q, r=KALMAN_R),
                KalmanFilter1D(q=KALMAN_Q, r=KALMAN_R),
                KalmanFilter1D(q=KALMAN_Q, r=KALMAN_R),
            ]
            self.confidence_history[obj_id] = deque(maxlen=self.history_size)

        # Apply Kalman filtering to each coordinate
        x, y, z = raw_pos
        kf_x, kf_y, kf_z = self.kalman_filters[obj_id]
        smoothed_x = kf_x.update(x)
        smoothed_y = kf_y.update(y)
        smoothed_z = kf_z.update(z)
        smoothed_pos = (smoothed_x, smoothed_y, smoothed_z)

        # Track history for temporal consistency check
        self.position_history[obj_id].append(smoothed_pos)
        self.confidence_history[obj_id].append(confidence)

        # Compute temporal stability score (how consistent is position over time)
        temporal_confidence = self._compute_temporal_confidence(obj_id)

        return smoothed_pos, temporal_confidence

    def _compute_temporal_confidence(self, obj_id: int) -> float:
        """
        Compute stability score: 1.0 if position is stable, 0.0 if jittery.
        Based on variance of recent detections.
        """
        history = self.position_history[obj_id]
        if len(history) < 2:
            return float(np.mean(self.confidence_history[obj_id]))

        positions = np.array(list(history))
        variances = np.var(positions, axis=0)
        spatial_stability = 1.0 / (1.0 + np.mean(variances))  # closer to 1 = more stable
        detection_confidence = float(np.mean(self.confidence_history[obj_id]))

        # Combined score
        combined = 0.6 * spatial_stability + 0.4 * detection_confidence
        return min(1.0, max(0.0, combined))

    def cleanup(self, active_obj_ids: set):
        """Remove tracking for objects no longer detected."""
        for obj_id in list(self.position_history.keys()):
            if obj_id not in active_obj_ids:
                del self.position_history[obj_id]
                del self.kalman_filters[obj_id]
                del self.confidence_history[obj_id]


class SharedStream:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_jpeg = None

    def update(self, frame_bgr):
        success, encoded = cv2.imencode(".jpg", frame_bgr)
        if not success:
            return
        with self.lock:
            self.latest_jpeg = encoded.tobytes()

    def read(self):
        with self.lock:
            return self.latest_jpeg


class MJPEGHandler(BaseHTTPRequestHandler):
    stream = None

    def do_GET(self):
        if self.path == "/":
            html = (
                "<html><head><title>ROBOLIMB</title></head>"
                "<body style='background:#111;color:#eee;font-family:sans-serif;'>"
                "<h3>ROBOLIMB live preview</h3>"
                "<img src='/stream.mjpg' style='max-width:100%;height:auto;border:1px solid #444;'/>"
                "</body></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                frame = self.stream.read() if self.stream else None
                if frame is None:
                    time.sleep(0.02)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format, *args):
        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ═══════════════════════════════════════════════════════
#  SERIAL COMMUNICATION  (Arduino / servo board)
# ═══════════════════════════════════════════════════════

class SerialController:
    def __init__(self, port, baud):
        self.connected = False
        if port:
            try:
                self.ser = serial.Serial(port, baud, timeout=1)
                time.sleep(2)
                self.connected = True
                print(f"[SERIAL] Connected to {port}")
            except Exception as e:
                print(f"[SERIAL] Could not connect: {e} — running in simulation mode")
        else:
            print("[SERIAL] No port specified — simulation mode")

    def send_angles(self, angles: dict):
        """angles = {'base': θ, 'shoulder': θ, 'elbow': θ, 'wrist': θ, 'gripper': θ}"""
        if self.connected:
            cmd = "MOVE"
            for joint, angle in angles.items():
                cmd += f",{joint}:{int(angle)}"
            cmd += "\n"
            self.ser.write(cmd.encode())
            print(f"[SERIAL] Sent → {cmd.strip()}")
        else:
            print(f"[SIM]    Angles → {angles}")

    def close(self):
        if self.connected:
            self.ser.close()


# ═══════════════════════════════════════════════════════
#  INVERSE KINEMATICS  (4-DOF planar + base rotation)
# ═══════════════════════════════════════════════════════

class InverseKinematics:
    def __init__(self, l1, l2, l3, l4):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4

    def solve(self, x: float, y: float, z: float, pitch_deg: float = -90.0):
        """
        Solve IK for target position (x, y, z) in meters.
        Returns dict of joint angles in degrees, or None if unreachable.

        Convention:
          x — left/right from robot base
          y — forward from robot base
          z — height above base
          pitch_deg — desired end-effector pitch (default -90 = pointing down)
        """
        # ── Base rotation (yaw) ──────────────────────────────
        base_angle = math.degrees(math.atan2(x, y))

        # ── Planar reach in the arm's vertical plane ─────────
        r_horiz = math.sqrt(x**2 + y**2)    # horizontal distance from base
        dz      = z - 0                      # height relative to base joint

        # Account for gripper offset (end-effector points downward by default)
        pitch_rad = math.radians(pitch_deg)
        wx = r_horiz - self.l4 * math.cos(pitch_rad)
        wz = dz      - self.l4 * math.sin(pitch_rad)

        # Effective reach to wrist (l2 + l3 arm)
        dist = math.sqrt(wx**2 + wz**2)
        arm_reach = self.l2 + self.l3

        if dist > arm_reach:
            print(f"[IK] Target unreachable  dist={dist:.3f}m  max={arm_reach:.3f}m")
            return None

        # ── Elbow angle (cosine rule) ────────────────────────
        cos_elbow = (dist**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)
        cos_elbow = max(-1.0, min(1.0, cos_elbow))        # clamp for numerical safety
        elbow_angle = math.degrees(math.acos(cos_elbow))  # elbow-up solution

        # ── Shoulder angle ───────────────────────────────────
        alpha = math.atan2(wz, wx)
        beta  = math.atan2(
            self.l3 * math.sin(math.radians(elbow_angle)),
            self.l2 + self.l3 * math.cos(math.radians(elbow_angle))
        )
        shoulder_angle = math.degrees(alpha - beta)

        # ── Wrist pitch to maintain desired end-effector pitch ─
        wrist_angle = pitch_deg - shoulder_angle - elbow_angle

        return {
            "base": round(base_angle, 2),
            "shoulder": round(shoulder_angle, 2),
            "elbow": round(elbow_angle, 2),
            "wrist": round(wrist_angle, 2),
            "gripper": GRIPPER_OPEN,
        }

    def clamp_angles(self, angles: dict) -> dict:
        """Clamp to safe servo limits — adjust per your robot."""
        limits = {
            "base":     (-90, 90),
            "shoulder": (-30, 150),
            "elbow":    (0,   150),
            "wrist":    (-90, 90),
            "gripper":  (0,   90)
        }
        for joint, (lo, hi) in limits.items():
            if joint in angles:
                angles[joint] = max(lo, min(hi, angles[joint]))
        return angles


# ═══════════════════════════════════════════════════════
#  3D COORDINATE ESTIMATION  (monocular webcam)
# ═══════════════════════════════════════════════════════

def estimate_3d_position_refined(bbox: Tuple, frame_w: int, frame_h: int, 
                                  known_width_m: float, focal_px: float,
                                  frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Refined 3D coordinate estimation with multiple techniques:
    1. Subpixel centroid detection using image moments
    2. Multi-scale depth estimation (width & height)
    3. Better handling of camera frame conversion
    
    Returns (x, y, z) in robot frame or None if invalid.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    obj_px_width = x2 - x1
    obj_px_height = y2 - y1

    if obj_px_width < 5 or obj_px_height < 5:
        return None

    # ── Subpixel centroid refinement ─────────────────────────────
    # Extract object region and find centroid with subpixel accuracy
    roi = frame[y1:y2+1, x1:x2+1]
    if roi.size == 0:
        return None

    # Convert to grayscale and compute weighted centroid
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    
    # Compute moments for subpixel localization
    moments = cv2.moments(roi_gray)
    if moments['m00'] > 0:
        local_cx = moments['m10'] / moments['m00']
        local_cy = moments['m01'] / moments['m00']
    else:
        local_cx = obj_px_width / 2.0
        local_cy = obj_px_height / 2.0

    # Convert local to image coordinates
    cx = x1 + local_cx
    cy = y1 + local_cy

    # ── Multi-scale depth estimation ─────────────────────────────
    # Use both width and height to reduce errors from aspect ratio variation
    z_from_width = (known_width_m * focal_px) / obj_px_width
    z_from_height = (known_width_m * focal_px) / obj_px_height  # assuming similar dimensions
    
    # Weighted average: prefer width estimate as primary
    Z = 0.7 * z_from_width + 0.3 * z_from_height

    # ── Lateral position calculation ──────────────────────────────
    X = (cx - frame_w / 2.0) * Z / focal_px
    Y = (cy - frame_h / 2.0) * Z / focal_px

    # ── Frame conversion: camera → robot ─────────────────────────
    # Camera convention: +X right, +Y down, +Z forward
    # Robot convention: +X right, +Y forward, +Z up
    robot_x = X
    robot_y = Z
    robot_z = -Y + 0.30  # height offset

    return (robot_x, robot_y, robot_z)


def estimate_3d_position(bbox, frame_w, frame_h, known_width_m, focal_px):
    """
    Legacy function for backward compatibility.
    Use estimate_3d_position_refined for better precision.
    """
    x1, y1, x2, y2 = bbox
    obj_px_width = x2 - x1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if obj_px_width < 5:
        return None

    Z = (known_width_m * focal_px) / obj_px_width
    X = (cx - frame_w / 2) * Z / focal_px
    Y = (cy - frame_h / 2) * Z / focal_px

    robot_x = X
    robot_y = Z
    robot_z = -Y + 0.30

    return (robot_x, robot_y, robot_z)


# ═══════════════════════════════════════════════════════
#  PICK SEQUENCE
# ═══════════════════════════════════════════════════════

def pick_object(controller: SerialController, ik: InverseKinematics,
                target_xyz: tuple, label: str):
    """Full pick routine: move above → descend → grip → lift."""

    x, y, z = target_xyz
    print(f"\n[PICK] Target '{label}'  x={x:.3f}  y={y:.3f}  z={z:.3f}")

    steps = [
        ("HOME",        (0.0, 0.20, 0.25), GRIPPER_OPEN),    # safe home position
        ("ABOVE",       (x,   y,    z + 0.08), GRIPPER_OPEN),  # 8 cm above object
        ("DESCEND",     (x,   y,    z + 0.01), GRIPPER_OPEN),  # just above object
        ("GRIP",        (x,   y,    z + 0.01), GRIPPER_CLOSED),# close gripper
        ("LIFT",        (x,   y,    z + 0.12), GRIPPER_CLOSED),# lift object
        ("DELIVER",     (0.0, 0.15, 0.20),     GRIPPER_CLOSED),# carry to drop zone
        ("RELEASE",     (0.0, 0.15, 0.20),     GRIPPER_OPEN),  # open gripper
        ("HOME",        (0.0, 0.20, 0.25), GRIPPER_OPEN),    # return home
    ]

    for step_name, pos, gripper in steps:
        px, py, pz = pos
        angles = ik.solve(px, py, pz)
        if angles is None:
            print(f"[PICK] Step '{step_name}' unreachable — aborting pick.")
            return False

        angles = ik.clamp_angles(angles)
        angles["gripper"] = gripper
        print(f"  [{step_name}]  {angles}")
        controller.send_angles(angles)
        time.sleep(0.8)     # wait for servos to reach position

    print(f"[PICK] '{label}' picked and delivered ✓\n")
    return True


# ═══════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("       ROBOLIMB — Object Detection + Pick System")
    print("=" * 55)

    shared_stream = SharedStream()
    MJPEGHandler.stream = shared_stream
    server = ThreadedHTTPServer(("127.0.0.1", 8000), MJPEGHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    webbrowser.open("http://127.0.0.1:8000")
    print("[INFO] Browser preview opened at http://127.0.0.1:8000")

    # Load YOLO model
    print("[INIT] Loading YOLOv8 model...")
    try:
        model = YOLO("yolov8n.pt")          # downloads automatically on first run
        print("[INIT] Model loaded.")
    except Exception as exc:
        print(f"[ERROR] Failed to load YOLO model: {exc}")
        server.shutdown()
        server.server_close()
        return

    # Init subsystems
    controller = SerialController(SERIAL_PORT, SERIAL_BAUD)
    ik         = InverseKinematics(L1, L2, L3, L4)
    coord_tracker = CoordinateTracker(history_size=TEMPORAL_HISTORY_SIZE)

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("[INFO] Press Ctrl+C to quit\n")

    try:
        while True:
            if not cap.isOpened():
                print("[WARN] Cannot open webcam yet; retrying...")
                shared_stream.update(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
                time.sleep(1.0)
                cap.open(CAMERA_INDEX, cv2.CAP_V4L2)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            ret, raw_frame = cap.read()
            if not ret:
                print("[ERROR] Frame grab failed.")
                shared_stream.update(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
                break

            # Apply undistortion for better coordinate accuracy
            display_frame = cv2.remap(raw_frame, map_x, map_y, cv2.INTER_LINEAR)

            # ── Run YOLO ────────────────────────────────────────
            results = model(display_frame, conf=CONF_THRESHOLD, verbose=False)
            detections = []
            h, w = display_frame.shape[:2]
            active_ids = set()
            obj_counter = 0

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                obj_id = obj_counter  # Assign unique ID per detection this frame
                active_ids.add(obj_id)
                obj_counter += 1

                # Use refined coordinate estimation with subpixel detection
                pos3d = estimate_3d_position_refined(
                    (x1, y1, x2, y2), w, h, KNOWN_OBJECT_WIDTH, FOCAL_LENGTH_PX, display_frame
                )
                if pos3d is None:
                    continue

                # Apply temporal filtering for smoother tracking
                smoothed_pos, temporal_conf = coord_tracker.track(obj_id, pos3d, conf)
                
                # Only use detection if temporal confidence is high enough
                if temporal_conf < MIN_TEMPORAL_CONFIDENCE:
                    confidence_indicator = "⚠️ "
                else:
                    confidence_indicator = "✓ "

                detections.append({
                    "label": label,
                    "conf": conf,
                    "temporal_conf": temporal_conf,
                    "bbox": (x1, y1, x2, y2),
                    "pos3d_raw": pos3d,
                    "pos3d": smoothed_pos,  # Use smoothed coordinates
                })

                rx, ry, rz = smoothed_pos
                # Color indicates confidence: green = high, yellow = medium, orange = low
                confidence = temporal_conf
                if confidence >= 0.8:
                    color = (0, 255, 80)  # green
                elif confidence >= 0.6:
                    color = (0, 255, 255)  # yellow
                else:
                    color = (0, 165, 255)  # orange

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{confidence_indicator}{label}  {conf:.2f}",
                            (x1, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(display_frame,
                            f"X:{rx:.2f}m Y:{ry:.2f}m Z:{rz:.2f}m [stab:{confidence:.2f}]",
                            (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 0), 1)

            # Cleanup stale tracks
            coord_tracker.cleanup(active_ids)

            status = f"Objects: {len(detections)}"
            cv2.putText(display_frame, status, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            shared_stream.update(display_frame)

    except KeyboardInterrupt:
        print("[INFO] Quit.")

    cap.release()
    server.shutdown()
    server.server_close()
    controller.close()


if __name__ == "__main__":
    main()