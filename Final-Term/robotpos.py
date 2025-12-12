from pydobot import Dobot
import time

PORT = '/dev/ttyACM0'


def get_current_position(robot):
    pos_x, pos_y, pos_z, pos_r, j1, j2, j3, j4 = robot.pose()
    print(f"X: {pos_x:.1f} mm, Y: {pos_y:.1f} mm, Z: {pos_z:.1f} mm, R: {pos_r:.1f}°")
    return pos_x, pos_y, pos_z, pos_r


def display_menu():
    print("Press [Enter] to capture, [m] to move, [h] for home, [q] to quit.")


def handle_capture(coords):
    coord_x, coord_y, coord_z, coord_r = coords
    print(f"Captured Pose -> X:{coord_x:.1f} Y:{coord_y:.1f} Z:{coord_z:.1f} R:{coord_r:.1f}\n")


def handle_manual_move(robot, current_coords):
    cx, cy, cz, cr = current_coords
    try:
        target_x = float(input("Enter X (mm): ") or cx)
        target_y = float(input("Enter Y (mm): ") or cy)
        target_z = float(input("Enter Z (mm): ") or cz)
        target_r = float(input("Enter R (deg): ") or cr)
        robot.move_to(target_x, target_y, target_z, target_r)
        print(f"Moving to -> X:{target_x} Y:{target_y} Z:{target_z} R:{target_r}\n")
    except ValueError:
        print("Invalid input. Try again.\n")


def handle_home_position(robot):
    print("Moving to home position (250, 0, 150, 0)...")
    robot.move_to(250, 0, 150, 0)
    print("Home position reached.\n")


def run_control_loop(robot):
    print("\nControls:")
    print("  [Enter] = Capture current coordinates")
    print("  [m]     = Enter manual move (X,Y,Z,R)") 
    print("  [h]     = Move to home position (250,0,150,0)")
    print("  [q]     = Quit program\n")
    
    while True:
        current_pos = get_current_position(robot)
        display_menu()
        cmd = input().strip().lower()
        
        if cmd == "":
            handle_capture(current_pos)
        elif cmd == "m":
            handle_manual_move(robot, current_pos)
        elif cmd == "h":
            handle_home_position(robot)
        elif cmd == "q":
            print("Quitting program...")
            break
        else:
            print("Unknown option. Press Enter to capture, m to move, h for home, q to quit.\n")


def main():
    robot_device = None
    try:
        robot_device = Dobot(port=PORT)
        time.sleep(1)
        print("Connected to Dobot on", PORT)
        run_control_loop(robot_device)
    except KeyboardInterrupt:
        print("Stopped by user")
    except Exception as error:
        print("Error:", error)
    finally:
        if robot_device is not None:
            robot_device.close()
            print("Connection closed.")


if __name__ == "__main__":
    main()