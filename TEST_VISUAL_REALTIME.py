#!/usr/bin/env python3
"""
TEST_VISUAL_REALTIME.py - Real-time visualization of coordinate stability
Shows refined vs legacy method side-by-side with live metrics.
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import time

# Match robolimb.py settings
CAMERA_INDEX = 2
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FOCAL_LENGTH_PX = 800
KNOWN_OBJECT_WIDTH = 0.07
CONF_THRESHOLD = 0.55

class RealtimeVisualTest:
    """Real-time visualization of precision improvements."""
    
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.coord_history_refined = deque(maxlen=30)  # Last 30 detections
        self.coord_history_legacy = deque(maxlen=30)
        self.frame_count = 0
    
    def estimate_refined(self, bbox, frame_w, frame_h, frame):
        """Refined method with multi-scale depth (hybrid approach)."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        obj_px_width = x2 - x1
        obj_px_height = y2 - y1
        
        if obj_px_width < 5 or obj_px_height < 5:
            return None
        
        # Robust simple center (more stable than moment-based)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        
        # Multi-scale depth estimation (KEY IMPROVEMENT)
        z_from_width = (KNOWN_OBJECT_WIDTH * FOCAL_LENGTH_PX) / obj_px_width
        z_from_height = (KNOWN_OBJECT_WIDTH * FOCAL_LENGTH_PX) / obj_px_height
        Z = 0.7 * z_from_width + 0.3 * z_from_height
        
        X = (cx - frame_w / 2.0) * Z / FOCAL_LENGTH_PX
        Y = (cy - frame_h / 2.0) * Z / FOCAL_LENGTH_PX
        
        return (X, Z, -Y + 0.30)
    
    def estimate_legacy(self, bbox, frame_w, frame_h):
        """Legacy simple method."""
        x1, y1, x2, y2 = bbox
        obj_px_width = x2 - x1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        if obj_px_width < 5:
            return None
        
        Z = (KNOWN_OBJECT_WIDTH * FOCAL_LENGTH_PX) / obj_px_width
        X = (cx - frame_w / 2) * Z / FOCAL_LENGTH_PX
        Y = (cy - frame_h / 2) * Z / FOCAL_LENGTH_PX
        
        return (X, Z, -Y + 0.30)
    
    def run_visual_test(self):
        """Run real-time visual comparison."""
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("[✗] Cannot open camera")
            return
        
        print("[✓] Camera opened. Press 'q' to quit, 's' to save stats")
        print("[TEST] Place object in frame and keep it STILL for best results\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            h, w = frame.shape[:2]
            display = frame.copy()
            
            # Detect
            results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)
            
            if len(results[0].boxes) > 0:
                # Get largest detection
                best_box = None
                best_area = 0
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2-x1) * (y2-y1)
                    if area > best_area:
                        best_area = area
                        best_box = (x1, y1, x2, y2)
                
                if best_box:
                    x1, y1, x2, y2 = [int(v) for v in best_box]
                    
                    # Estimate both methods
                    refined = self.estimate_refined(best_box, w, h, frame)
                    legacy = self.estimate_legacy(best_box, w, h)
                    
                    if refined:
                        self.coord_history_refined.append(refined)
                    if legacy:
                        self.coord_history_legacy.append(legacy)
                    
                    # Draw detection box
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 80), 2)
                    
                    # Draw center point (refined method)
                    roi = frame[y1:y2+1, x1:x2+1]
                    if roi.size > 0:
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
                        moments = cv2.moments(roi_gray)
                        if moments['m00'] > 0:
                            local_cx = moments['m10'] / moments['m00']
                            local_cy = moments['m01'] / moments['m00']
                            cx = int(x1 + local_cx)
                            cy = int(y1 + local_cy)
                            cv2.circle(display, (cx, cy), 4, (0, 255, 255), -1)  # Refined center
                    
                    # Draw simple center (legacy method)
                    cx_legacy = (x1 + x2) // 2
                    cy_legacy = (y1 + y2) // 2
                    cv2.circle(display, (cx_legacy, cy_legacy), 3, (255, 100, 100), -1)  # Legacy center
                    
                    # Compute statistics
                    if len(self.coord_history_refined) >= 2:
                        coords_ref = np.array(list(self.coord_history_refined))
                        coords_leg = np.array(list(self.coord_history_legacy))
                        
                        var_ref = np.var(coords_ref, axis=0)
                        var_leg = np.var(coords_leg, axis=0)
                        mean_ref = np.mean(coords_ref, axis=0)
                        std_ref = np.std(coords_ref, axis=0)
                        std_leg = np.std(coords_leg, axis=0)
                        
                        # Draw statistics on screen
                        y_offset = 30
                        
                        # Title
                        cv2.putText(display, "REFINED (Yellow dot) vs LEGACY (Blue dot)", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        y_offset += 25
                        
                        # Refined stats
                        cv2.putText(display, f"REFINED - Mean: X={mean_ref[0]:.3f}m  Y={mean_ref[1]:.3f}m  Z={mean_ref[2]:.3f}m", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                        y_offset += 20
                        cv2.putText(display, f"          Std:  X={std_ref[0]:.4f}m  Y={std_ref[1]:.4f}m  Z={std_ref[2]:.4f}m", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                        y_offset += 20
                        
                        # Legacy stats
                        cv2.putText(display, f"LEGACY   - Mean: X={mean_ref[0]:.3f}m  Y={mean_ref[1]:.3f}m  Z={mean_ref[2]:.3f}m", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)
                        y_offset += 20
                        cv2.putText(display, f"          Std:  X={std_leg[0]:.4f}m  Y={std_leg[1]:.4f}m  Z={std_leg[2]:.4f}m", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)
                        y_offset += 20
                        
                        # Improvement percentage
                        improvement_x = (std_leg[0] - std_ref[0]) / std_leg[0] * 100 if std_leg[0] > 0 else 0
                        improvement_y = (std_leg[1] - std_ref[1]) / std_leg[1] * 100 if std_leg[1] > 0 else 0
                        improvement_z = (std_leg[2] - std_ref[2]) / std_leg[2] * 100 if std_leg[2] > 0 else 0
                        
                        cv2.putText(display, f"IMPROVEMENT: X={improvement_x:+.1f}%  Y={improvement_y:+.1f}%  Z={improvement_z:+.1f}%", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                        
                        # Sample count
                        cv2.putText(display, f"Samples: {len(self.coord_history_refined)}", 
                                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Display frame
            cv2.imshow("ROBOLIMB Precision Test", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.print_summary()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def print_summary(self):
        """Print summary statistics."""
        if len(self.coord_history_refined) < 2:
            print("\n[INFO] Not enough samples yet")
            return
        
        print("\n" + "="*60)
        print("REAL-TIME TEST SUMMARY")
        print("="*60)
        
        coords_ref = np.array(list(self.coord_history_refined))
        coords_leg = np.array(list(self.coord_history_legacy))
        
        std_ref = np.std(coords_ref, axis=0)
        std_leg = np.std(coords_leg, axis=0)
        
        labels = ['X (right)', 'Y (depth)', 'Z (height)']
        improvements = []
        
        print(f"\n{'Axis':<12} {'Refined Std':<15} {'Legacy Std':<15} {'Improvement':<15}")
        print("-"*60)
        
        for i, label in enumerate(labels):
            if std_leg[i] > 0:
                improvement = (std_leg[i] - std_ref[i]) / std_leg[i] * 100
                improvements.append(improvement)
            else:
                improvement = 0
            
            print(f"{label:<12} {std_ref[i]:.6f}m      {std_leg[i]:.6f}m      {improvement:+.1f}%")
        
        avg_improvement = np.mean(improvements)
        print("-"*60)
        print(f"{'AVERAGE':<12} {'':<15} {'':<15} {avg_improvement:+.1f}%\n")

def main():
    print("\n" + "="*70)
    print("ROBOLIMB - REAL-TIME PRECISION VISUALIZATION TEST")
    print("="*70 + "\n")
    
    tester = RealtimeVisualTest()
    tester.run_visual_test()

if __name__ == "__main__":
    main()
