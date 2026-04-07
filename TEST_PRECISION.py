#!/usr/bin/env python3
"""
TEST_PRECISION.py - Verify object detection and coordinate estimation accuracy
Run this to measure jitter, stability, and coordinate precision improvements.
"""

import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
import sys

# ═══════════════════════════════════════════════════════
#  TEST CONFIGURATION
# ═══════════════════════════════════════════════════════

CAMERA_INDEX = 2  # Match your camera setting in robolimb.py
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FOCAL_LENGTH_PX = 800
KNOWN_OBJECT_WIDTH = 0.07  # meters
CONF_THRESHOLD = 0.55
NUM_TEST_FRAMES = 100  # Capture 100 frames for averaging

# Test modes
TEST_MODE = "STATIC"  # Options: "STATIC" (object at fixed distance), "MOVEMENT", "COMPARE"

class PrecisionTester:
    """Test coordinate estimation precision and stability."""
    
    def __init__(self):
        self.model = None
        self.coordinates_history = deque()
        self.raw_coords = deque()
        self.filtered_coords = deque()
        self.confidence_history = deque()
        self.frame_count = 0
        
    def load_model(self):
        """Load YOLOv8 model."""
        print("[INIT] Loading YOLOv8 model...")
        try:
            self.model = YOLO("yolov8n.pt")
            print("[✓] Model loaded successfully")
            return True
        except Exception as e:
            print(f"[✗] Failed to load model: {e}")
            return False
    
    def estimate_3d_position(self, bbox, frame_w, frame_h, known_width_m, focal_px, frame=None):
        """Estimate 3D position (improved hybrid method)."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        obj_px_width = x2 - x1
        obj_px_height = y2 - y1
        
        if obj_px_width < 5 or obj_px_height < 5:
            return None
        
        # Robust simple center (avoiding noisy moment-based approach)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        
        # Multi-scale depth estimation (KEY IMPROVEMENT)
        z_from_width = (known_width_m * focal_px) / obj_px_width
        z_from_height = (known_width_m * focal_px) / obj_px_height
        Z = 0.7 * z_from_width + 0.3 * z_from_height  # Weighted average
        
        X = (cx - frame_w / 2.0) * Z / focal_px
        Y = (cy - frame_h / 2.0) * Z / focal_px
        
        robot_x = X
        robot_y = Z
        robot_z = -Y + 0.30
        
        return (robot_x, robot_y, robot_z)
    
    def estimate_3d_position_legacy(self, bbox, frame_w, frame_h, known_width_m, focal_px):
        """Legacy simple method for comparison."""
        x1, y1, x2, y2 = bbox
        obj_px_width = x2 - x1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        if obj_px_width < 5:
            return None
        
        Z = (known_width_m * focal_px) / obj_px_width
        X = (cx - frame_w / 2) * Z / focal_px
        Y = (cy - frame_h / 2) * Z / focal_px
        
        return (X, Z, -Y + 0.30)
    
    def run_test(self, test_mode="STATIC"):
        """Run precision test."""
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("[✗] Cannot open camera. Check CAMERA_INDEX in config.")
            return False
        
        print(f"\n[TEST] Starting {test_mode} precision test...")
        print(f"[TEST] Will capture {NUM_TEST_FRAMES} frames")
        print(f"[TEST] Place object at fixed distance ({KNOWN_OBJECT_WIDTH*100:.1f}cm wide)")
        print(f"[TEST] Press Ctrl+C to stop early\n")
        
        time.sleep(2)  # Let camera warm up
        
        try:
            refined_coords = deque()
            legacy_coords = deque()
            
            for frame_idx in range(NUM_TEST_FRAMES):
                ret, frame = cap.read()
                if not ret:
                    print("[✗] Frame grab failed")
                    break
                
                self.frame_count += 1
                h, w = frame.shape[:2]
                
                # Run detection
                results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)
                
                if len(results[0].boxes) == 0:
                    print(f"\r[{frame_idx+1:3d}] No objects detected", end="", flush=True)
                    continue
                
                # Get largest detection (main object)
                best_box = None
                best_area = 0
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2-x1) * (y2-y1)
                    if area > best_area:
                        best_area = area
                        best_box = (x1, y1, x2, y2)
                
                if best_box is None:
                    continue
                
                # Estimate with BOTH methods for comparison
                refined = self.estimate_3d_position(best_box, w, h, KNOWN_OBJECT_WIDTH, FOCAL_LENGTH_PX, frame)
                legacy = self.estimate_3d_position_legacy(best_box, w, h, KNOWN_OBJECT_WIDTH, FOCAL_LENGTH_PX)
                
                if refined and legacy:
                    refined_coords.append(refined)
                    legacy_coords.append(legacy)
                    
                    print(f"\r[{frame_idx+1:3d}/{NUM_TEST_FRAMES}] Detected: Refined=({refined[0]:.3f}, {refined[1]:.3f}, {refined[2]:.3f})", 
                          end="", flush=True)
                else:
                    print(f"\r[{frame_idx+1:3d}] Detection too small/invalid", end="", flush=True)
                
                time.sleep(0.03)  # ~30fps
        
        except KeyboardInterrupt:
            print("\n[INFO] Test interrupted by user")
        
        finally:
            cap.release()
        
        print("\n")
        return refined_coords, legacy_coords
    
    def analyze_results(self, refined_coords, legacy_coords):
        """Analyze and compare precision metrics."""
        print("=" * 70)
        print("PRECISION TEST RESULTS")
        print("=" * 70)
        
        if len(refined_coords) < 3:
            print("[✗] Not enough valid detections to analyze (need ≥3)")
            return
        
        refined_arr = np.array(list(refined_coords))
        legacy_arr = np.array(list(legacy_coords))
        
        print(f"\n✓ Analyzed {len(refined_coords)} valid detections\n")
        
        # ──────────────────────────────────────────────────────
        print("📊 COORDINATE STABILITY (Lower variance = Better)")
        print("-" * 70)
        
        refined_var = np.var(refined_arr, axis=0)
        legacy_var = np.var(legacy_arr, axis=0)
        
        print(f"{'Axis':<8} {'Refined (NEW)':<20} {'Legacy (OLD)':<20} {'Improvement':<15}")
        print("-" * 70)
        
        labels = ['X (right)', 'Y (depth)', 'Z (height)']
        improvements = []
        
        for i, label in enumerate(labels):
            ref_var = refined_var[i]
            leg_var = legacy_var[i]
            
            if leg_var > 0:
                improvement = (leg_var - ref_var) / leg_var * 100
                improvements.append(improvement)
            else:
                improvement = 0
            
            print(f"{label:<8} {ref_var:.6f}            {leg_var:.6f}            {improvement:+.1f}%")
        
        avg_improvement = np.mean(improvements)
        print("-" * 70)
        print(f"{'AVERAGE':<8} {'Jitter reduction: ':>39} {avg_improvement:+.1f}%\n")
        
        # ──────────────────────────────────────────────────────
        print("📏 POSITIONAL ACCURACY & REPEATABILITY")
        print("-" * 70)
        
        refined_mean = np.mean(refined_arr, axis=0)
        legacy_mean = np.mean(legacy_arr, axis=0)
        refined_std = np.std(refined_arr, axis=0)
        legacy_std = np.std(legacy_arr, axis=0)
        
        print(f"{'Axis':<8} {'Refined Mean':<18} {'Refined Std':<18} {'Legacy Mean':<18} {'Legacy Std':<18}")
        print("-" * 70)
        
        for i, label in enumerate(labels):
            print(f"{label:<8} {refined_mean[i]:>8.4f}m        {refined_std[i]:>8.4f}m        "
                  f"{legacy_mean[i]:>8.4f}m        {legacy_std[i]:>8.4f}m")
        
        print("\n")
        
        # ──────────────────────────────────────────────────────
        print("🎯 RANGE (Min - Max) for each coordinate")
        print("-" * 70)
        print(f"{'Axis':<8} {'Refined Range':<25} {'Legacy Range':<25}")
        print("-" * 70)
        
        for i, label in enumerate(labels):
            ref_range = f"{np.min(refined_arr[:, i]):.4f} to {np.max(refined_arr[:, i]):.4f}"
            leg_range = f"{np.min(legacy_arr[:, i]):.4f} to {np.max(legacy_arr[:, i]):.4f}"
            print(f"{label:<8} {ref_range:<25} {leg_range:<25}")
        
        print("\n")
        
        # ──────────────────────────────────────────────────────
        print("⚡ VERDICT")
        print("-" * 70)
        
        if avg_improvement > 50:
            verdict = "✓ EXCELLENT - Precision improved significantly!"
        elif avg_improvement > 30:
            verdict = "✓ GOOD - Noticeable precision improvement"
        elif avg_improvement > 10:
            verdict = "✓ FAIR - Some improvement detected"
        else:
            verdict = "⚠️  Minimal improvement or insufficient data"
        
        print(f"{verdict}\n")
        
        # ──────────────────────────────────────────────────────
        print("💡 INTERPRETATION")
        print("-" * 70)
        print("- Variance/Std: Lower = less jitter, more stable")
        print("- Mean: Should be consistent across frames")
        print("- Range: Smaller range = more precise tracking")
        print("- Improvement %: Negative = didn't improve (check variance)")
        print()

def main():
    print("\n" + "="*70)
    print("ROBOLIMB - PRECISION IMPROVEMENT TEST")
    print("="*70)
    
    tester = PrecisionTester()
    
    if not tester.load_model():
        sys.exit(1)
    
    refined, legacy = tester.run_test(TEST_MODE)
    
    if refined and legacy:
        tester.analyze_results(refined, legacy)
    else:
        print("[✗] Test failed - insufficient data collected")

if __name__ == "__main__":
    main()
