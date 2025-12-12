#!/usr/bin/env python3

import cv2
import numpy as np
import json
import os
from datetime import datetime


def find_red_marker(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    red_lower_1 = np.array([0, 100, 100])
    red_upper_1 = np.array([10, 255, 255])
    red_lower_2 = np.array([160, 100, 100])
    red_upper_2 = np.array([180, 255, 255])
    
    m1 = cv2.inRange(hsv_img, red_lower_1, red_upper_1)
    m2 = cv2.inRange(hsv_img, red_lower_2, red_upper_2)
    combined_mask = cv2.bitwise_or(m1, m2)
    
    k = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, k)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, k)
    
    cnts, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        return None, None, 0
    
    biggest = max(cnts, key=cv2.contourArea)
    a = cv2.contourArea(biggest)
    
    if a <= 50:
        return None, None, 0
    
    moments = cv2.moments(biggest)
    if moments["m00"] == 0:
        return None, None, 0
    
    x_center = int(moments["m10"] / moments["m00"])
    y_center = int(moments["m01"] / moments["m00"])
    
    return (x_center, y_center), biggest, a


def render_overlay(img, pos, cnt):
    output = img.copy()
    
    if pos is None:
        return output
    
    cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
    cv2.circle(output, pos, 5, (0, 255, 0), -1)
    cv2.circle(output, pos, 15, (0, 255, 0), 2)
    
    cv2.line(output, (pos[0] - 20, pos[1]), (pos[0] + 20, pos[1]), (0, 255, 0), 1)
    cv2.line(output, (pos[0], pos[1] - 20), (pos[0], pos[1] + 20), (0, 255, 0), 1)
    
    label = f"Pixel: ({pos[0]}, {pos[1]})"
    cv2.putText(output, label, (pos[0] + 20, pos[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return output


class RobotCalibrator:
    def __init__(self, cam_id=2):
        self.cam_id = cam_id
        self.data_points = []
        self.active_frame = None
        self.marker_position = None
    
    def record_point(self, pix, robot):
        entry = {
            'pixel': pix,
            'dobot': robot,
            'timestamp': datetime.now().isoformat()
        }
        self.data_points.append(entry)
        print(f"\nPoint #{len(self.data_points)} recorded:")
        print(f"  Pixel: {pix}")
        print(f"  Dobot: {robot}")
    
    def write_json(self, fname="calibration_points.json"):
        output_data = {
            'calibration_points': self.data_points,
            'total_points': len(self.data_points),
            'created': datetime.now().isoformat()
        }
        
        with open(fname, 'w') as file:
            json.dump(output_data, file, indent=2)
        
        print(f"\nCalibration saved to {fname}")
        return fname
    
    def write_python(self, fname="calibration_export.py"):
        with open(fname, 'w') as file:
            file.write(f"# Generated: {datetime.now().isoformat()}\n")
            file.write(f"# Total points: {len(self.data_points)}\n\n")
            file.write("CALIBRATION_POINTS = [\n")
            
            for entry in self.data_points:
                p = entry['pixel']
                d = entry['dobot']
                file.write(f"    {{'pixel': {p}, 'dobot': {d}}},\n")
            
            file.write("]\n")
        
        print(f"Python code exported to {fname}")
        return fname
    
    def execute_session(self):
        print("\n" + "="*70)
        print("DOBOT CALIBRATION HELPER - RED DOT DETECTION")
        print("="*70)
        print("\nInstructions:")
        print("1. Position Dobot at a location")
        print("2. Place RED marker under the Dobot end-effector")
        print("3. Press SPACE when red dot is detected to record coordinates")
        print("4. Enter Dobot X, Y coordinates manually")
        print("5. Repeat for at least 5-8 points across the workspace")
        print("6. Press 'q' to finish and save calibration")
        print("\nRecommended calibration point distribution:")
        print("  - 4 corners of workspace")
        print("  - 2-4 points in the middle")
        print("  - Spread out as much as possible")
        print("="*70)
        
        input("\nPress ENTER to start camera...")
        
        camera = cv2.VideoCapture(self.cam_id)
        
        if not camera.isOpened():
            print("Error: Could not open camera")
            return False
        
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\nCamera opened successfully")
        print(f"\nCalibration points collected: {len(self.data_points)}")
        
        while True:
            success, img = camera.read()
            if not success:
                print("Error: Could not read frame")
                break
            
            self.active_frame = img
            pos, cnt, area = find_red_marker(img)
            self.marker_position = pos
            
            display_img = render_overlay(img, pos, cnt)
            
            y_offset = 30
            
            if pos is not None:
                msg = f"RED DOT DETECTED - Area: {int(area)} px"
                cv2.putText(display_img, msg, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_img, "Press SPACE to record this point", (10, y_offset + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                msg = "NO RED DOT DETECTED"
                cv2.putText(display_img, msg, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            progress = f"Points: {len(self.data_points)}/8 recommended"
            cv2.putText(display_img, progress, (10, y_offset + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(display_img, "SPACE: Record | Q: Finish & Save", (10, display_img.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Calibration Helper - Red Dot Detection", display_img)
            
            pressed = cv2.waitKey(1) & 0xFF
            
            if pressed == ord(' '):
                if pos is not None:
                    pixel_loc = pos
                    
                    print("\n" + "-"*70)
                    print(f"Red dot detected at pixel: {pixel_loc}")
                    print("Enter the current Dobot coordinates:")
                    
                    try:
                        x_val = float(input("  Dobot X (mm): "))
                        y_val = float(input("  Dobot Y (mm): "))
                        
                        robot_loc = (x_val, y_val)
                        self.record_point(pixel_loc, robot_loc)
                        
                        print(f"\nTotal points collected: {len(self.data_points)}")
                        
                        if len(self.data_points) >= 5:
                            print("Minimum points reached! You can add more or press 'q' to finish.")
                        else:
                            print(f"Need {5 - len(self.data_points)} more points (minimum 5)")
                        
                        print("-"*70)
                        
                    except ValueError:
                        print("Invalid input. Point not added.")
                else:
                    print("\nNo red dot detected! Position red marker and try again.")
            
            elif pressed == ord('q'):
                if len(self.data_points) >= 5:
                    print("\n\nFinishing calibration...")
                    break
                else:
                    print(f"\nWarning: Only {len(self.data_points)} points collected.")
                    print("  Recommended: at least 5 points")
                    resp = input("  Save anyway? (y/n): ")
                    if resp.lower() == 'y':
                        break
        
        camera.release()
        cv2.destroyAllWindows()
        
        if len(self.data_points) > 0:
            print("\n" + "="*70)
            print("CALIBRATION COMPLETE")
            print("="*70)
            
            print(f"\nTotal points collected: {len(self.data_points)}")
            print("\nCalibration Points:")
            for idx, entry in enumerate(self.data_points, 1):
                print(f"  {idx}. Pixel{entry['pixel']} -> Dobot{entry['dobot']}")
            
            json_path = self.write_json()
            py_path = self.write_python()
            
            print(f"\nFiles saved:")
            print(f"  - {json_path} (JSON format)")
            print(f"  - {py_path} (Python code)")
            
            print("\n" + "="*70)
            print("Next steps:")
            print("1. Copy the calibration points from calibration_export.py")
            print("2. Replace CALIBRATION_POINTS in your main program")
            print("3. Run the improved calibration program")
            print("="*70)
            
            return True
        else:
            print("\nNo calibration points collected")
            return False


def run_detection_test():
    print("\n" + "="*70)
    print("RED DETECTION TEST")
    print("="*70)
    print("\nThis will show you what the camera sees.")
    print("Place a red marker in view to test detection.")
    print("Press 'q' to exit test mode.")
    print("="*70)
    
    input("\nPress ENTER to start...")
    
    camera = cv2.VideoCapture(2)
    
    if not camera.isOpened():
        print("Error: Could not open camera")
        return
    
    while True:
        success, img = camera.read()
        if not success:
            break
        
        pos, cnt, area = find_red_marker(img)
        display_img = render_overlay(img, pos, cnt)
        
        if pos is not None:
            cv2.putText(display_img, f"DETECTED - Area: {int(area)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_img, "NO RED DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Red Detection Test - Press Q to exit", display_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    cv2.destroyAllWindows()
    print("\nTest complete")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║        DOBOT CALIBRATION HELPER                          ║
║        Red Dot Detection System                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    print("\nSelect mode:")
    print("1. Test red detection (recommended first)")
    print("2. Start calibration session")
    
    user_choice = input("\nEnter choice (1 or 2): ").strip()
    
    if user_choice == '1':
        run_detection_test()
    elif user_choice == '2':
        calibrator = RobotCalibrator()
        calibrator.execute_session()
    else:
        print("Invalid choice")