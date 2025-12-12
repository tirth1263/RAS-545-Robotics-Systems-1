#!/usr/bin/env python3
"""
CAMERA FINDER - Find the correct robot camera index
"""

import cv2
import time
import subprocess

print("\n" + "="*70)
print("CAMERA DETECTOR - Finding all available cameras")
print("="*70)

# First check video devices in system
print("\nChecking system video devices:")
print("-" * 70)
try:
    result = subprocess.run(['ls', '-l', '/dev/video*'], capture_output=True, text=True)
    print(result.stdout)
except:
    print("Could not list /dev/video* devices")

print("\n" + "="*70)
print("Testing camera indices 0-10...")
print("="*70)

working_cameras = []

for index in range(11):
    print(f"\nTesting camera index {index}...", end=" ")
    
    try:
        cap = cv2.VideoCapture(index)
        time.sleep(0.3)  # Wait for camera to initialize
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                
                # Get camera name if possible
                camera_name = "Unknown"
                try:
                    backend_name = cap.getBackendName()
                    camera_name = f"{backend_name}"
                except:
                    pass
                
                print(f"✓ WORKS! Resolution: {width}x{height}, Backend: {camera_name}")
                working_cameras.append({
                    'index': index,
                    'resolution': f"{width}x{height}",
                    'backend': camera_name
                })
                
                # Show preview for 2 seconds
                cv2.putText(frame, f"Camera {index} - {width}x{height}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Is this the ROBOT camera?", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(f"Camera {index} Preview", frame)
                print(f"   Showing preview for 2 seconds...")
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
            else:
                print("✗ Opens but can't read frames")
            cap.release()
        else:
            print("✗ Can't open")
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "="*70)
print("RESULTS:")
print("="*70)

if working_cameras:
    print(f"\n✓ Found {len(working_cameras)} working camera(s):\n")
    for cam in working_cameras:
        print(f"   Camera Index {cam['index']}:")
        print(f"      Resolution: {cam['resolution']}")
        print(f"      Backend: {cam['backend']}")
        print()
    
    print("="*70)
    print("IMPORTANT: Which camera showed the ROBOT's view?")
    print("="*70)
    
    if len(working_cameras) == 1:
        print(f"\nOnly one camera found, using index {working_cameras[0]['index']}")
        print(f"\nUpdate your llm.py:")
        print(f"   CAMERA_INDEX = {working_cameras[0]['index']}")
    else:
        print("\nMultiple cameras found. You need to identify which is the robot camera.")
        print("\nTo update llm.py, change this line:")
        for cam in working_cameras:
            print(f"   CAMERA_INDEX = {cam['index']}  # Use this if camera {cam['index']} is the robot")
    
    print("\n" + "="*70)
else:
    print("\n✗ No working cameras found!")
    print("\nTroubleshooting:")
    print("  1. Check if camera is connected (USB)")
    print("  2. Check permissions:")
    print("     sudo usermod -a -G video $USER")
    print("     (then logout and login)")
    print("  3. Check if camera is being used by another program:")
    print("     lsof /dev/video*")
    print("  4. Try unplugging and replugging the USB camera")
    print("="*70)
