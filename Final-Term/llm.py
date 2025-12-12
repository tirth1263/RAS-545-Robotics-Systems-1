#!/usr/bin/env python3

import os
import sys
import json
import time
import ast
import cv2
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime
from pathlib import Path

API_KEY_GROQ = ""
os.environ["GROQ_API_KEY"] = API_KEY_GROQ

USB_PORT = '/dev/ttyACM0'
CAM_IDX = 2

STATE_PATH = Path("/home/tirth_1263/Final-Term RS-1/dobot_state.json")

DEFAULT_POS = {'x': 250.0, 'y': 0, 'z': 150.0, 'r': 0.0}

Z_PICK = -54.2
Z_SAFE = 50.0
Z_PLACE = -54.2
BLOCK_H = 17.9

CALIB_DATA = [
    {'pixel': (304, 207), 'dobot': (298.7, 18.4)},
    {'pixel': (211, 205), 'dobot': (297.9, 60.0)},
    {'pixel': (170, 345), 'dobot': (234.5, 74.9)},
    {'pixel': (374, 321), 'dobot': (246.0, -12.5)},
    {'pixel': (380, 204), 'dobot': (298.9, -14.8)},
]


class StateManager:
    
    def __init__(self, filepath=STATE_PATH):
        self.filepath = filepath
        self.gripper_state = "EMPTY"
        self.held_block = None
        self.block_data = {}
        self.cmd_history = []
        self.location_stacks = {}
        self.previous_placement = None
        
    def _get_location_key(self, px, py, thresh=30):
        bucket_x = round(px / thresh) * thresh
        bucket_y = round(py / thresh) * thresh
        return f"{bucket_x},{bucket_y}"
    
    def setup_blocks(self, block_list):
        self.block_data = {}
        self.location_stacks = {}
        
        for item in block_list:
            bid = item['global_id']
            self.block_data[bid] = {
                'color': item['color'],
                'id': item['id'],
                'pixel_x': item['pixel_x'],
                'pixel_y': item['pixel_y'],
                'original_x': item['pixel_x'],
                'original_y': item['pixel_y'],
                'current_position': {'x': item['pixel_x'], 'y': item['pixel_y'], 'z': 0},
                'stack_level': 0,
                'blocks_below': []
            }
            
            loc = self._get_location_key(item['pixel_x'], item['pixel_y'])
            if loc not in self.location_stacks:
                self.location_stacks[loc] = []
            self.location_stacks[loc].append(bid)
        
        print(f"[State] Initialized {len(self.block_data)} blocks")
        self.persist_state()
    
    def set_gripper_state(self, bid=None):
        if bid is None:
            self.gripper_state = "EMPTY"
            self.held_block = None
        else:
            self.gripper_state = "HOLDING"
            self.held_block = bid
        print(f"[State] Gripper: {self.gripper_state} {bid or ''}")
        self.persist_state()
    
    def set_block_location(self, bid, px, py, z=0, below_blocks=None):
        if bid in self.block_data:
            old_loc = self._get_location_key(
                self.block_data[bid]['pixel_x'],
                self.block_data[bid]['pixel_y']
            )
            
            self.block_data[bid]['pixel_x'] = px
            self.block_data[bid]['pixel_y'] = py
            self.block_data[bid]['current_position'] = {'x': px, 'y': py, 'z': z}
            
            level = round(z / BLOCK_H) if z > 0 else 0
            self.block_data[bid]['stack_level'] = level
            
            if below_blocks:
                self.block_data[bid]['blocks_below'] = below_blocks
            else:
                self.block_data[bid]['blocks_below'] = []
            
            new_loc = self._get_location_key(px, py)
            
            if old_loc in self.location_stacks and bid in self.location_stacks[old_loc]:
                self.location_stacks[old_loc].remove(bid)
                if not self.location_stacks[old_loc]:
                    del self.location_stacks[old_loc]
            
            if new_loc not in self.location_stacks:
                self.location_stacks[new_loc] = []
            if bid not in self.location_stacks[new_loc]:
                self.location_stacks[new_loc].append(bid)
            
            self.location_stacks[new_loc].sort(
                key=lambda b: self.block_data[b]['stack_level']
            )
            
            self.previous_placement = {'x': px, 'y': py, 'z': z}
            
            print(f"[State] {bid} moved to ({px}, {py}, z={z}, level={level})")
            self.persist_state()
    
    def fetch_stack_at(self, px, py):
        loc = self._get_location_key(px, py)
        return self.location_stacks.get(loc, [])
    
    def fetch_stack_count(self, px, py):
        stack = self.fetch_stack_at(px, py)
        return len(stack)
    
    def fetch_block_info(self, bid):
        return self.block_data.get(bid)
    
    def fetch_all_blocks(self):
        return [
            {
                'id': bid,
                'color': info['color'],
                'pixel_x': info['pixel_x'],
                'pixel_y': info['pixel_y'],
                'stack_level': info['stack_level']
            }
            for bid, info in self.block_data.items()
        ]
    
    def fetch_held_block(self):
        return self.held_block
    
    def append_history(self, cmd, result, extra=None):
        record = {
            'prompt': cmd,
            'action': result,
            'timestamp': datetime.now().isoformat(),
            'details': extra or {}
        }
        self.cmd_history.append(record)
        
        if len(self.cmd_history) > 20:
            self.cmd_history = self.cmd_history[-20:]
        
        self.persist_state()
    
    def build_summary(self):
        text = f"""CURRENT SYSTEM STATE:
- Gripper Status: {self.gripper_state}
- Holding Block: {self.held_block or 'None'}

AVAILABLE BLOCKS (with stacking info):
"""
        for bid, info in self.block_data.items():
            stack_desc = ""
            if info['stack_level'] > 0:
                stack_desc = f" [STACKED - Level {info['stack_level']}, on: {', '.join(info['blocks_below'])}]"
            text += f"  - {bid}: {info['color']} at pixel ({info['pixel_x']}, {info['pixel_y']}){stack_desc}\n"
        
        if self.location_stacks:
            text += "\nDETECTED STACKS:\n"
            for loc, blocks in self.location_stacks.items():
                if len(blocks) > 1:
                    text += f"  - Location {loc}: {' -> '.join(blocks)} ({len(blocks)} blocks high)\n"
        
        if self.previous_placement:
            text += f"\nLAST PLACE LOCATION: pixel ({self.previous_placement['x']}, {self.previous_placement['y']}, z={self.previous_placement['z']})\n"
        
        if len(self.cmd_history) > 0:
            text += f"\nRECENT ACTIONS (last {min(5, len(self.cmd_history))}):\n"
            for record in self.cmd_history[-5:]:
                details_part = ""
                if record.get('details'):
                    details_part = f" | {record['details']}"
                text += f"  - '{record['prompt']}' -> {record['action']}{details_part}\n"
        
        return text
    
    def persist_state(self):
        try:
            data = {
                'gripper_status': self.gripper_state,
                'holding_block': self.held_block,
                'blocks': self.block_data,
                'conversation_history': self.cmd_history,
                'stack_map': self.location_stacks,
                'last_place_location': self.previous_placement,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"[State] Saved to {self.filepath}")
            
        except Exception as err:
            print(f"[State] Save failed: {err}")
    
    def restore_state(self):
        try:
            if not self.filepath.exists():
                print(f"[State] No saved state found")
                return False
            
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            
            self.gripper_state = data.get('gripper_status', 'EMPTY')
            self.held_block = data.get('holding_block')
            self.block_data = data.get('blocks', {})
            self.cmd_history = data.get('conversation_history', [])
            self.location_stacks = data.get('stack_map', {})
            self.previous_placement = data.get('last_place_location')
            
            print(f"[State] Loaded state from {self.filepath}")
            print(f"[State] Restored: {len(self.block_data)} blocks, {len(self.cmd_history)} history entries")
            
            return True
            
        except Exception as err:
            print(f"[State] Load failed: {err}")
            return False


class TransformationMatrix:
    
    def __init__(self):
        self.calib_pts = CALIB_DATA
        self.rotation_avg = 14.25
        self.compute_matrix()
        print("[Coordinates] Calibration loaded with 5 points")
        
    def compute_matrix(self):
        pix_matrix = np.array([
            [pt['pixel'][0], pt['pixel'][1], 1] for pt in self.calib_pts
        ])
        
        robo_x = np.array([pt['dobot'][0] for pt in self.calib_pts])
        robo_y = np.array([pt['dobot'][1] for pt in self.calib_pts])
        
        x_coeffs, _, _, _ = np.linalg.lstsq(pix_matrix, robo_x, rcond=None)
        self.coeff_x1, self.coeff_x2, self.offset_x = x_coeffs
        
        y_coeffs, _, _, _ = np.linalg.lstsq(pix_matrix, robo_y, rcond=None)
        self.coeff_y1, self.coeff_y2, self.offset_y = y_coeffs
        
        print(f"[Coordinates] Transformation calculated")
        
    def convert_coordinates(self, px, py):
        rx = self.coeff_x1 * px + self.coeff_x2 * py + self.offset_x
        ry = self.coeff_y1 * px + self.coeff_y2 * py + self.offset_y
        
        print(f"[Coord] Pixel ({px}, {py}) -> Dobot ({rx:.1f}, {ry:.1f})")
        return rx, ry


class RobotArm:
    
    def __init__(self):
        self.arm = None
        self.transformer = TransformationMatrix()
        
    def establish_connection(self, port=USB_PORT):
        try:
            from pydobot import Dobot
            print(f"\n[Dobot] Connecting to {port}...")
            self.arm = Dobot(port=port)
            time.sleep(1)
            
            if hasattr(self.arm, 'set_speed'):
                self.arm.set_speed(velocity=150, acceleration=150)
            
            print("[Dobot] Connected successfully")
            return True
            
        except Exception as err:
            print(f"[Dobot] ERROR: {err}")
            return False
    
    def navigate_to(self, x, y, z, r=0.0, wait=True):
        if self.arm:
            print(f"[Dobot] Moving to X:{x:.1f} Y:{y:.1f} Z:{z:.1f}")
            self.arm.move_to(x, y, z, r, wait=wait)
            if wait:
                time.sleep(0.2)
    
    def return_home(self):
        print("[Dobot] Moving to home position")
        self.navigate_to(DEFAULT_POS['x'], DEFAULT_POS['y'], DEFAULT_POS['z'], DEFAULT_POS['r'])
        print("[Dobot] At home position")
    
    def activate_suction(self):
        if self.arm:
            self.arm.suck(True)
            print("[Dobot] Suction: ON")
        time.sleep(0.8)
    
    def deactivate_suction(self):
        if self.arm:
            self.arm.suck(False)
            print("[Dobot] Suction: OFF")
        time.sleep(0.5)
    
    def grab_block(self, px, py):
        print(f"\n[Dobot] Picking block at pixel ({px}, {py})")
        rx, ry = self.transformer.convert_coordinates(px, py)
        self.navigate_to(rx, ry, Z_SAFE)
        self.navigate_to(rx, ry, Z_PICK)
        self.activate_suction()
        self.navigate_to(rx, ry, Z_SAFE)
        print(f"[Dobot] Block picked")
    
    def release_block(self, px, py, z_adj=0):
        print(f"\n[Dobot] Placing block at pixel ({px}, {py}), z_offset={z_adj}mm")
        rx, ry = self.transformer.convert_coordinates(px, py)
        target_z = Z_PLACE + z_adj
        print(f"[Dobot] Target Z height: {target_z:.1f}mm")
        self.navigate_to(rx, ry, Z_SAFE)
        self.navigate_to(rx, ry, target_z)
        self.deactivate_suction()
        self.navigate_to(rx, ry, Z_SAFE)
        print(f"[Dobot] Block placed")
    
    def disconnect(self):
        if self.arm:
            try:
                self.arm.close()
                print("[Dobot] Connection closed")
            except:
                pass


class VisionSystem:
    
    def __init__(self, cam_idx=CAM_IDX):
        self.cam_idx = cam_idx
        self.cam_device = None
        
        self.color_hsv = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))
            ],
            'blue': [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
            'green': [(np.array([40, 100, 100]), np.array([80, 255, 255]))],
            'yellow': [(np.array([20, 100, 100]), np.array([35, 255, 255]))]
        }
    
    def find_colored_objects(self, img, color_name):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        combined_mask = None
        for low, high in self.color_hsv[color_name]:
            if combined_mask is None:
                combined_mask = cv2.inRange(hsv_img, low, high)
            else:
                combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(hsv_img, low, high))
        
        morph_kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, morph_kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, morph_kernel)
        
        contour_list, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for cnt in contour_list:
            size = cv2.contourArea(cnt)
            if 200 < size < 10000:
                moments = cv2.moments(cnt)
                if moments["m00"] != 0:
                    center_x = int(moments["m10"] / moments["m00"])
                    center_y = int(moments["m01"] / moments["m00"])
                    results.append({'center': (center_x, center_y), 'contour': cnt, 'area': size})
        
        return results
    
    def scan_all_blocks_interactive(self):
        print(f"\n[Camera] Opening camera {self.cam_idx}")
        
        try:
            self.cam_device = cv2.VideoCapture(self.cam_idx)
            time.sleep(0.5)
            
            if not self.cam_device.isOpened():
                print(f"[Camera] Trying V4L2 backend")
                self.cam_device = cv2.VideoCapture(self.cam_idx, cv2.CAP_V4L2)
                time.sleep(0.5)
                
                if not self.cam_device.isOpened():
                    print(f"[Camera] Failed to open camera")
                    return None
            
            print(f"[Camera] Camera opened")
            
        except Exception as err:
            print(f"[Camera] ERROR: {err}")
            return None
        
        self.cam_device.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam_device.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        ok, test_img = self.cam_device.read()
        if not ok:
            print(f"[Camera] Cannot read frames")
            self.cam_device.release()
            return None
        
        print("[Camera] Press SPACE to capture | Q to quit")
        
        captured_items = []
        
        while True:
            ok, img = self.cam_device.read()
            if not ok:
                print("[Camera] Frame read failed")
                break
            
            overlay = img.copy()
            all_found = []
            
            for col in ['red', 'blue', 'green', 'yellow']:
                objects = self.find_colored_objects(img, col)
                for obj in objects:
                    all_found.append({'color': col, 'center': obj['center'], 'contour': obj['contour']})
            
            all_found.sort(key=lambda item: (item['center'][1] // 50, item['center'][0]))
            
            display_colors = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0), 'yellow': (0, 255, 255)}
            temp_items = []
            color_idx = {'red': 1, 'blue': 1, 'green': 1, 'yellow': 1}
            
            for item in all_found:
                cx, cy = item['center']
                col = item['color']
                bgr_col = display_colors[col]
                num = color_idx[col]
                color_idx[col] += 1
                
                cv2.drawContours(overlay, [item['contour']], -1, bgr_col, 2)
                cv2.circle(overlay, (cx, cy), 5, bgr_col, -1)
                
                tag = f"{col}_{num}"
                cv2.putText(overlay, tag, (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(overlay, f"({cx},{cy})", (cx - 30, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                temp_items.append({'color': col, 'id': num, 'center': (cx, cy)})
            
            cv2.putText(overlay, f"Blocks: {len(all_found)} | Camera: {self.cam_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Block Detection", overlay)
            
            pressed = cv2.waitKey(1) & 0xFF
            
            if pressed == ord(' '):
                if len(temp_items) > 0:
                    captured_items = []
                    for item in temp_items:
                        global_id = f"{item['color']}_{item['id']}"
                        captured_items.append({
                            'global_id': global_id,
                            'color': item['color'],
                            'id': item['id'],
                            'pixel_x': item['center'][0],
                            'pixel_y': item['center'][1]
                        })
                    print(f"\n[Camera] Captured {len(captured_items)} blocks")
                    break
            elif pressed == ord('q'):
                print("[Camera] Detection cancelled")
                captured_items = None
                break
        
        self.cam_device.release()
        cv2.destroyAllWindows()
        return captured_items
    
    def scan_blocks_auto(self, duration=2.0):
        print(f"\n[Camera] Auto-capturing blocks")
        
        try:
            cam = cv2.VideoCapture(self.cam_idx)
            time.sleep(0.5)
            
            if not cam.isOpened():
                cam = cv2.VideoCapture(self.cam_idx, cv2.CAP_V4L2)
                time.sleep(0.5)
                
                if not cam.isOpened():
                    print(f"[Camera] Failed to open")
                    return None
            
            print(f"[Camera] Camera opened")
            
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            ok, test_img = cam.read()
            if not ok:
                print(f"[Camera] Cannot read frames")
                cam.release()
                return None
            
            print(f"[Camera] Starting capture")
            
            display_colors = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0), 'yellow': (0, 255, 255)}
            start = time.time()
            recent_items = []
            
            while True:
                ok, img = cam.read()
                if not ok:
                    break
                
                overlay = img.copy()
                all_found = []
                
                for col in ['red', 'blue', 'green', 'yellow']:
                    objects = self.find_colored_objects(img, col)
                    for obj in objects:
                        all_found.append({'color': col, 'center': obj['center'], 'contour': obj['contour']})
                
                all_found.sort(key=lambda item: (item['center'][1] // 50, item['center'][0]))
                
                color_idx = {'red': 1, 'blue': 1, 'green': 1, 'yellow': 1}
                temp_items = []
                
                for item in all_found:
                    cx, cy = item['center']
                    col = item['color']
                    bgr_col = display_colors[col]
                    num = color_idx[col]
                    color_idx[col] += 1
                    
                    cv2.drawContours(overlay, [item['contour']], -1, bgr_col, 2)
                    cv2.circle(overlay, (cx, cy), 5, bgr_col, -1)
                    
                    tag = f"{col}_{num}"
                    cv2.putText(overlay, tag, (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    temp_items.append({'global_id': tag, 'color': col, 'id': num, 'pixel_x': cx, 'pixel_y': cy})
                
                recent_items = temp_items
                
                elapsed = time.time() - start
                
                cv2.putText(overlay, f"AUTO-CAPTURE | Blocks: {len(all_found)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Auto Block Capture", overlay)
                cv2.waitKey(30)
                
                if elapsed >= duration:
                    break
            
            cam.release()
            cv2.destroyAllWindows()
            time.sleep(0.2)
            
            if len(recent_items) > 0:
                print(f"[Camera] Captured {len(recent_items)} blocks")
            else:
                print(f"[Camera] No blocks detected")
            
            return recent_items
            
        except Exception as err:
            print(f"[Camera] ERROR: {err}")
            return None


class AICodeGenerator:
    
    def __init__(self, api_key):
        self.model = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile", api_key=api_key)
        
        self.sys_template = """You are a Python code generator for a Dobot robotic arm.

{state_summary}

ALLOWED FUNCTIONS:
- dobot.pick_block(pixel_x, pixel_y)
- dobot.place_block(pixel_x, pixel_y, z_offset=0)
- system_state.update_gripper(block_id or None)
- system_state.update_block_position(block_id, pixel_x, pixel_y, z, blocks_below=[...])
- system_state.get_block(block_id)
- system_state.get_blocks()
- system_state.get_gripper()
- system_state.get_stack_at_location(pixel_x, pixel_y)
- system_state.get_stack_height(pixel_x, pixel_y)
- time.sleep(seconds)
- print()

EXAMPLE - Pick red block:
USER: "pick up red block"
STATE: Gripper EMPTY
CODE:
red_block = system_state.get_block('red_1')
dobot.pick_block(red_block['pixel_x'], red_block['pixel_y'])
system_state.update_gripper('red_1')

EXAMPLE - Place at position:
USER: "place it at 300, 50"
STATE: Holding red_1
CODE:
target_x = 300
target_y = 50
stack_height = system_state.get_stack_height(target_x, target_y)
z_offset = stack_height * 17.9
dobot.place_block(target_x, target_y, z_offset)
stack_below = system_state.get_stack_at_location(target_x, target_y)
system_state.update_block_position('red_1', target_x, target_y, z_offset, blocks_below=stack_below)
system_state.update_gripper(None)

Generate executable Python code for:
"""
    
    def produce_code(self, cmd, state_mgr):
        summary = state_mgr.build_summary()
        sys_prompt = self.sys_template.format(state_summary=summary)
        
        msg_list = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=cmd)
        ]
        
        try:
            print(f"\n[LLM] Analyzing: '{cmd}'")
            print(f"[LLM] Current state: {state_mgr.gripper_state}")
            
            reply = self.model.invoke(msg_list)
            code_output = reply.content.strip()
            
            if "```python" in code_output:
                code_output = code_output.split("```python")[1].split("```")[0].strip()
            elif "```" in code_output:
                code_output = code_output.split("```")[1].split("```")[0].strip()
            
            print(f"[LLM] Code generated")
            return code_output
            
        except Exception as err:
            print(f"[LLM] ERROR: {err}")
            return None


class SafeExecutor:
    
    def __init__(self):
        self.permitted_funcs = [
            'dobot.pick_block',
            'dobot.place_block',
            'system_state.update_gripper',
            'system_state.update_block_position',
            'system_state.get_block',
            'system_state.get_blocks',
            'system_state.get_gripper',
            'system_state.get_stack_at_location',
            'system_state.get_stack_height',
            'time.sleep',
            'print'
        ]
    
    def verify_code(self, code_str):
        try:
            tree = ast.parse(code_str)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        func_name = f"{ast.unparse(node.func.value)}.{node.func.attr}"
                        
                        if node.func.attr in ['startswith', 'endswith', 'upper', 'lower', 'strip', 'split']:
                            continue
                        
                        permitted = False
                        for allowed in self.permitted_funcs:
                            if func_name == allowed:
                                permitted = True
                                break
                        
                        if not permitted:
                            print(f"[Validator] Blocked: {func_name}")
                            return False
                
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mod = node.names[0].name if hasattr(node, 'names') else node.module
                    if mod not in ['time']:
                        print(f"[Validator] Blocked import: {mod}")
                        return False
            
            print(f"[Validator] Code validated")
            return True
            
        except SyntaxError as err:
            print(f"[Validator] Syntax error: {err}")
            return False
        except Exception as err:
            print(f"[Validator] Validation error: {err}")
            return False
    
    def run_code(self, code_str, robot, state_mgr):
        
        if not self.verify_code(code_str):
            print("[Executor] Code validation failed")
            return False
        
        safe_env = {
            'dobot': robot,
            'system_state': state_mgr,
            'time': time,
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'int': int,
                'float': float,
                'str': str,
                'round': round,
            }
        }
        
        try:
            print("\n[Executor] Executing code")
            print("-" * 70)
            exec(code_str, safe_env)
            print("-" * 70)
            print("[Executor] Execution complete")
            return True
            
        except Exception as err:
            print(f"\n[Executor] EXECUTION ERROR: {err}")
            import traceback
            traceback.print_exc()
            return False


class SystemOrchestrator:
    
    def __init__(self):
        self.robot = RobotArm()
        self.vision = VisionSystem()
        self.state_mgr = StateManager()
        self.ai_gen = AICodeGenerator(API_KEY_GROQ)
        self.exec_engine = SafeExecutor()
    
    def refresh_block_positions(self):
        print("\n" + "="*70)
        print("AUTO-UPDATING BLOCK POSITIONS")
        print("="*70)
        
        self.robot.return_home()
        time.sleep(0.5)
        
        scanned = self.vision.scan_blocks_auto(display_duration=2.0)
        
        if scanned and len(scanned) > 0:
            print(f"\n[Camera] Detected {len(scanned)} blocks")
            
            for item in scanned:
                bid = item['global_id']
                if bid in self.state_mgr.block_data:
                    self.state_mgr.block_data[bid]['pixel_x'] = item['pixel_x']
                    self.state_mgr.block_data[bid]['pixel_y'] = item['pixel_y']
                else:
                    self.state_mgr.block_data[bid] = {
                        'color': item['color'],
                        'id': item['id'],
                        'pixel_x': item['pixel_x'],
                        'pixel_y': item['pixel_y'],
                        'original_x': item['pixel_x'],
                        'original_y': item['pixel_y'],
                        'current_position': {'x': item['pixel_x'], 'y': item['pixel_y'], 'z': 0},
                        'stack_level': 0,
                        'blocks_below': []
                    }
            
            self.state_mgr.persist_state()
            print(f"\n[System] Block positions updated")
        else:
            print(f"\n[Camera] No blocks detected")
    
    def setup_system(self):
        print("\n" + "="*70)
        print("SYSTEM INITIALIZATION")
        print("="*70)
        
        print("\n[STEP 1/3] LOADING STATE")
        print("-" * 70)
        if self.state_mgr.restore_state():
            print("\nPrevious state loaded")
            print(f"Blocks: {len(self.state_mgr.block_data)}")
            print(f"History: {len(self.state_mgr.cmd_history)}")
            
            choice = input("\nUse saved state? (y/n): ").strip().lower()
            if choice == 'y':
                print("[System] Using saved state")
                
                print("\n[STEP 2/2] DOBOT CONNECTION")
                print("-" * 70)
                if not self.robot.establish_connection():
                    print("\nFATAL: Dobot connection failed")
                    sys.exit(1)
                
                self.robot.return_home()
                
                print("\n" + "="*70)
                print("INITIALIZATION COMPLETE")
                print("="*70)
                return
        
        print("\n[STEP 2/3] DOBOT CONNECTION")
        print("-" * 70)
        if not self.robot.establish_connection():
            print("\nFATAL: Dobot connection failed")
            sys.exit(1)
        
        print("\n[STEP 3/3] BLOCK DETECTION")
        print("-" * 70)
        self.robot.return_home()
        time.sleep(1)
        
        print(f"[Camera] Using camera index: {CAM_IDX}")
        found_blocks = self.vision.scan_all_blocks_interactive()
        
        if found_blocks is None or len(found_blocks) == 0:
            print("\nFATAL: No blocks detected")
            self.robot.disconnect()
            sys.exit(1)
        
        self.state_mgr.setup_blocks(found_blocks)
        
        print("\n" + "="*70)
        print("INITIALIZATION COMPLETE")
        print("="*70)
    
    def execute_interactive_session(self):
        print("\n" + "="*70)
        print("READY FOR COMMANDS")
        print("="*70)
        print(f"\nCamera: Index {CAM_IDX}")
        print("\nExample commands:")
        print("  - pick up red block")
        print("  - place it at 300, 50")
        print("  - stack red on blue")
        print("\nSpecial commands:")
        print("  - status")
        print("  - history")
        print("  - home")
        print("  - open camera")
        print("  - quit")
        print("="*70 + "\n")
        
        while True:
            try:
                cmd_input = input("\n[You] Enter command: ").strip()
                
                if not cmd_input:
                    continue
                
                if cmd_input.lower() in ['quit', 'exit', 'q']:
                    print("\n[System] Shutting down")
                    break
                
                elif cmd_input.lower() == 'status':
                    print("\n" + "="*70)
                    print(self.state_mgr.build_summary())
                    print("="*70)
                    continue
                
                elif cmd_input.lower() == 'history':
                    print("\n" + "="*70)
                    print("COMMAND HISTORY:")
                    print("-" * 70)
                    for idx, record in enumerate(self.state_mgr.cmd_history, 1):
                        print(f"{idx}. '{record['prompt']}'")
                        print(f"   -> {record['action']} at {record['timestamp']}")
                    print("="*70)
                    continue
                
                elif cmd_input.lower() == 'home':
                    self.robot.return_home()
                    continue
                
                elif cmd_input.lower() == 'open camera':
                    self.robot.return_home()
                    time.sleep(0.5)
                    scanned = self.vision.scan_all_blocks_interactive()
                    if scanned:
                        for item in scanned:
                            bid = item['global_id']
                            if bid in self.state_mgr.block_data:
                                self.state_mgr.block_data[bid]['pixel_x'] = item['pixel_x']
                                self.state_mgr.block_data[bid]['pixel_y'] = item['pixel_y']
                        self.state_mgr.persist_state()
                        print(f"\nUpdated {len(scanned)} blocks")
                    continue
                
                code_gen = self.ai_gen.produce_code(cmd_input, self.state_mgr)
                
                if code_gen is None:
                    print("[System] Failed to generate code")
                    continue
                
                print("\n" + "="*70)
                print("GENERATED CODE:")
                print("-" * 70)
                print(code_gen)
                print("="*70)
                
                proceed = input("\nExecute this code? (y/n): ").strip().lower()
                
                if proceed != 'y':
                    print("[System] Execution cancelled")
                    continue
                
                result = self.exec_engine.run_code(code_gen, self.robot, self.state_mgr)
                
                if result:
                    self.state_mgr.append_history(
                        cmd_input,
                        "Success",
                        extra={'code_length': len(code_gen)}
                    )
                    print("\nCommand completed successfully")
                    
                    self.refresh_block_positions()
                else:
                    self.state_mgr.append_history(cmd_input, "Failed")
                    print("\nCommand failed")
                    self.robot.return_home()
                
            except KeyboardInterrupt:
                print("\n\n[System] Interrupted by user")
                break
            except Exception as err:
                print(f"\n[System] ERROR: {err}")
                import traceback
                traceback.print_exc()
        
        print("\n[System] Cleanup")
        try:
            self.robot.return_home()
        except:
            pass
        
        self.robot.disconnect()
        print("\nSystem shutdown complete")
        print(f"State saved to {STATE_PATH}")
        print("="*70)
    
    def start(self):
        try:
            self.setup_system()
            self.execute_interactive_session()
        except Exception as err:
            print(f"\nFATAL ERROR: {err}")
            import traceback
            traceback.print_exc()
            try:
                self.robot.disconnect()
            except:
                pass


if __name__ == "__main__":
    print("""
LLM-POWERED DOBOT PICK & PLACE SYSTEM v3.0
    """)
    
    print(f"\nState File: {STATE_PATH}")
    if STATE_PATH.exists():
        print(f"Found existing state file")
    else:
        print(f"No saved state")
    
    print("\nConfiguration:")
    print(f"   Dobot: {USB_PORT}")
    print(f"   Camera: Index {CAM_IDX}")
    print(f"   Block Height: {BLOCK_H}mm")
    
    confirm = input("\nStart system? (y/n): ").strip().lower()
    
    if confirm == 'y':
        orchestrator = SystemOrchestrator()
        orchestrator.start()
    else:
        print("\nExiting")
