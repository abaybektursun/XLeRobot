#!/usr/bin/env python3
"""
Kinesthetic Teaching for XLeRobot

Move the robot arms by hand while recording joint positions and camera frames.
The motors are set to passive mode (torque disabled) so you can freely move them.

Controls:
- SPACE: Start/stop recording episode
- ENTER: Save current episode and start new one
- R: Reset episode (discard current recording)
- Q: Quit and save all data

Usage:
    python examples/9_kinesthetic_teaching.py
"""

import os
import sys
import time
import queue
import threading
import numpy as np
import select
import termios
import tty

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.cameras.opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.xlerobot import XLerobot, XLerobotConfig

# ============== CONFIGURATION ==============
# Camera indices (from camera_config.json)
HEAD_CAMERA_INDEX = 6
LEFT_WRIST_CAMERA_INDEX = 4
RIGHT_WRIST_CAMERA_INDEX = 2

# Dataset settings
DATASET_REPO = "abayb/xlerobot_towel_pickup"
TASK = "Pick up the towel and place it in the basket"
FPS = 30
MAX_EPISODE_LENGTH = 300  # 10 seconds max at 30fps
MIN_EPISODE_LENGTH = 60   # 2 seconds minimum

# Which arms to use (set False to disable)
USE_LEFT_ARM = False
USE_RIGHT_ARM = True

# Wrist cameras (set False to disable if having camera issues)
USE_WRIST_CAMERAS = False  # Set True once cameras are working
# ============================================


class KinestheticTeacher:
    def __init__(self):
        self.robot = None
        self.cameras = {}
        self.dataset = None

        # Recording state
        self.is_recording = False
        self.current_episode_frames = []
        self.episode_count = 0
        self.frame_count = 0

        # Threading
        self.shutdown_event = threading.Event()
        self.frame_queue = queue.Queue()

        # Keyboard state
        self.key_pressed = None

    def init_dataset(self):
        """Initialize LeRobot dataset with appropriate features."""
        # Build feature dict based on which arms are used
        state_names = []
        action_names = []

        if USE_LEFT_ARM:
            left_joints = [
                "left_arm_shoulder_pan", "left_arm_shoulder_lift", "left_arm_elbow_flex",
                "left_arm_wrist_flex", "left_arm_wrist_roll", "left_arm_gripper"
            ]
            state_names.extend(left_joints)
            action_names.extend(left_joints)

        if USE_RIGHT_ARM:
            right_joints = [
                "right_arm_shoulder_pan", "right_arm_shoulder_lift", "right_arm_elbow_flex",
                "right_arm_wrist_flex", "right_arm_wrist_roll", "right_arm_gripper"
            ]
            state_names.extend(right_joints)
            action_names.extend(right_joints)

        features = {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
            "observation.images.head": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
        }

        if USE_WRIST_CAMERAS:
            if USE_LEFT_ARM:
                features["observation.images.left_wrist"] = {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channel"],
                }

            if USE_RIGHT_ARM:
                features["observation.images.right_wrist"] = {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channel"],
                }

        dataset = LeRobotDataset.create(
            repo_id=DATASET_REPO,
            root=f"dataset_{int(time.time())}",
            features=features,
            fps=FPS,
            image_writer_processes=0,
            image_writer_threads=4,
        )
        # Smaller video chunks to avoid issues
        dataset.meta.update_chunk_settings(video_files_size_in_mb=0.001)
        return dataset

    def connect_robot(self):
        """Connect to robot and disable arm torque for manual movement."""
        print("Connecting to robot...")
        config = XLerobotConfig()
        self.robot = XLerobot(config)
        self.robot.connect()
        print("Robot connected!")

        # Interactive head positioning
        print("\n" + "="*50)
        print("HEAD CAMERA POSITIONING")
        print("="*50)
        print("Use these keys to adjust head position:")
        print("  W/S - Tilt up/down")
        print("  A/D - Pan left/right")
        print("  ENTER - Confirm position and continue")
        print("="*50)

        # Read CURRENT head position (don't reset!)
        obs = self.robot.get_observation()
        head_pan = obs.get("head_motor_1.pos", 0.0)
        head_tilt = obs.get("head_motor_2.pos", 0.0)
        step = 5.0  # degrees per keypress
        print(f"Current head position: pan={head_pan:.1f}, tilt={head_tilt:.1f}")

        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        try:
            while True:
                # Show current position
                print(f"\rHead: pan={head_pan:.1f}, tilt={head_tilt:.1f} (WASD to adjust, ENTER to confirm)", end="", flush=True)

                # Move head
                head_action = {
                    "head_motor_1.pos": head_pan,
                    "head_motor_2.pos": head_tilt,
                }
                self.robot.send_action(head_action)

                # Wait for key
                key = sys.stdin.read(1)

                if key == 'w' or key == 'W':
                    head_tilt += step
                elif key == 's' or key == 'S':
                    head_tilt -= step
                elif key == 'a' or key == 'A':
                    head_pan -= step
                elif key == 'd' or key == 'D':
                    head_pan += step
                elif key == '\n' or key == '\r':
                    print(f"\n  Head position confirmed: pan={head_pan:.1f}, tilt={head_tilt:.1f}")
                    break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        print("  NOTE: Head will stay fixed during recording")

        # Get the motor lists
        left_arm_motors = ["left_arm_shoulder_pan", "left_arm_shoulder_lift",
                          "left_arm_elbow_flex", "left_arm_wrist_flex",
                          "left_arm_wrist_roll", "left_arm_gripper"]
        right_arm_motors = ["right_arm_shoulder_pan", "right_arm_shoulder_lift",
                           "right_arm_elbow_flex", "right_arm_wrist_flex",
                           "right_arm_wrist_roll", "right_arm_gripper"]

        # Disable torque on arm motors only (head stays active)
        print("Disabling arm torque for kinesthetic teaching...")
        if USE_LEFT_ARM:
            for motor in left_arm_motors:
                self.robot.bus1.write("Torque_Enable", motor, 0)
                self.robot.bus1.write("Lock", motor, 0)
            print("  Left arm: torque disabled (can move freely)")
        if USE_RIGHT_ARM:
            for motor in right_arm_motors:
                self.robot.bus2.write("Torque_Enable", motor, 0)
                self.robot.bus2.write("Lock", motor, 0)
            print("  Right arm: torque disabled (can move freely)")

        # Verify torque is disabled
        time.sleep(0.5)
        print("\n*** TRY MOVING THE ARM NOW ***")
        print("If arm is still stiff, check motor connections.")
        print("\nYou can now move the robot arm by hand!")

    def connect_cameras(self):
        """Connect to cameras."""
        print("Connecting cameras...")

        # Head camera (always used)
        config_head = OpenCVCameraConfig(
            index_or_path=HEAD_CAMERA_INDEX,
            width=640, height=480, fps=FPS
        )
        self.cameras['head'] = OpenCVCamera(config_head)
        self.cameras['head'].connect()
        print(f"  Head camera (index {HEAD_CAMERA_INDEX}): connected")

        if USE_WRIST_CAMERAS:
            if USE_LEFT_ARM:
                config_left = OpenCVCameraConfig(
                    index_or_path=LEFT_WRIST_CAMERA_INDEX,
                    width=640, height=480, fps=FPS
                )
                self.cameras['left_wrist'] = OpenCVCamera(config_left)
                self.cameras['left_wrist'].connect()
                print(f"  Left wrist camera (index {LEFT_WRIST_CAMERA_INDEX}): connected")

            if USE_RIGHT_ARM:
                config_right = OpenCVCameraConfig(
                    index_or_path=RIGHT_WRIST_CAMERA_INDEX,
                    width=640, height=480, fps=FPS
                )
                self.cameras['right_wrist'] = OpenCVCamera(config_right)
                self.cameras['right_wrist'].connect()
                print(f"  Right wrist camera (index {RIGHT_WRIST_CAMERA_INDEX}): connected")

    def get_observation(self):
        """Get current robot state and camera images."""
        obs = self.robot.get_observation()

        # Extract joint positions
        state = []
        if USE_LEFT_ARM:
            state.extend([
                obs["left_arm_shoulder_pan.pos"],
                obs["left_arm_shoulder_lift.pos"],
                obs["left_arm_elbow_flex.pos"],
                obs["left_arm_wrist_flex.pos"],
                obs["left_arm_wrist_roll.pos"],
                obs["left_arm_gripper.pos"],
            ])
        if USE_RIGHT_ARM:
            state.extend([
                obs["right_arm_shoulder_pan.pos"],
                obs["right_arm_shoulder_lift.pos"],
                obs["right_arm_elbow_flex.pos"],
                obs["right_arm_wrist_flex.pos"],
                obs["right_arm_wrist_roll.pos"],
                obs["right_arm_gripper.pos"],
            ])

        # Get camera images
        images = {
            'head': self.cameras['head'].read()
        }
        if USE_LEFT_ARM and 'left_wrist' in self.cameras:
            images['left_wrist'] = self.cameras['left_wrist'].read()
        if USE_RIGHT_ARM and 'right_wrist' in self.cameras:
            images['right_wrist'] = self.cameras['right_wrist'].read()

        return np.array(state, dtype=np.float32), images

    def get_key_nonblocking(self):
        """Get key press without blocking."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def print_status(self):
        """Print current status."""
        status = "RECORDING" if self.is_recording else "PAUSED"
        print(f"\r[{status}] Episode: {self.episode_count} | Frames: {self.frame_count} | "
              f"Press SPACE to {'stop' if self.is_recording else 'start'}, "
              f"ENTER to save, R to reset, Q to quit", end="", flush=True)

    def run(self):
        """Main recording loop."""
        print("\n" + "="*60)
        print("KINESTHETIC TEACHING")
        print("="*60)
        print(f"Task: {TASK}")
        print(f"Dataset: {DATASET_REPO}")
        print(f"FPS: {FPS}")
        print("="*60)

        # Initialize
        self.connect_robot()
        self.connect_cameras()
        self.dataset = self.init_dataset()

        print("\n" + "-"*60)
        print("CONTROLS:")
        print("  SPACE  - Start/stop recording")
        print("  ENTER  - Save episode and start new one")
        print("  R      - Reset (discard current episode)")
        print("  Q      - Quit and save all data")
        print("-"*60)
        print("\nReady! Move the robot arm to starting position, then press SPACE.\n")

        frame_duration = 1.0 / FPS

        # Set terminal to raw mode for non-blocking input
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        try:
            while not self.shutdown_event.is_set():
                loop_start = time.time()

                # Handle keyboard input
                key = self.get_key_nonblocking()
                if key == ' ':  # Space
                    self.is_recording = not self.is_recording
                    if self.is_recording:
                        print("\n>>> Recording started!")
                    else:
                        print(f"\n>>> Recording paused ({self.frame_count} frames)")

                elif key == '\n' or key == '\r':  # Enter
                    if self.frame_count >= MIN_EPISODE_LENGTH:
                        print(f"\n>>> Saving episode {self.episode_count} ({self.frame_count} frames)...")
                        self.save_episode()
                        self.episode_count += 1
                        self.frame_count = 0
                        self.is_recording = False
                        print(f">>> Episode saved! Ready for episode {self.episode_count}")
                    else:
                        print(f"\n>>> Episode too short ({self.frame_count} < {MIN_EPISODE_LENGTH})")

                elif key == 'r' or key == 'R':  # Reset
                    print(f"\n>>> Discarding current episode ({self.frame_count} frames)")
                    self.frame_count = 0
                    self.is_recording = False

                elif key == 'q' or key == 'Q':  # Quit
                    print("\n>>> Quitting...")
                    if self.frame_count >= MIN_EPISODE_LENGTH:
                        print(f">>> Saving final episode ({self.frame_count} frames)...")
                        self.save_episode()
                    self.shutdown_event.set()
                    break

                # Record frame if recording
                if self.is_recording:
                    state, images = self.get_observation()

                    # Build frame
                    frame = {
                        'action': state,  # For kinesthetic teaching, action = next state
                        'observation.state': state,
                        'observation.images.head': images['head'],
                        'task': TASK,
                    }
                    if USE_LEFT_ARM and 'left_wrist' in images:
                        frame['observation.images.left_wrist'] = images['left_wrist']
                    if USE_RIGHT_ARM and 'right_wrist' in images:
                        frame['observation.images.right_wrist'] = images['right_wrist']

                    self.dataset.add_frame(frame)
                    self.frame_count += 1

                    # Check max length
                    if self.frame_count >= MAX_EPISODE_LENGTH:
                        print(f"\n>>> Max episode length reached, auto-saving...")
                        self.save_episode()
                        self.episode_count += 1
                        self.frame_count = 0
                        self.is_recording = False

                self.print_status()

                # Maintain FPS
                elapsed = time.time() - loop_start
                if elapsed < frame_duration:
                    time.sleep(frame_duration - elapsed)

        except KeyboardInterrupt:
            print("\n>>> Interrupted!")

        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.cleanup()

    def save_episode(self):
        """Save current episode to dataset."""
        if self.frame_count > 0:
            self.dataset.save_episode()
            self.dataset.image_writer.wait_until_done()

    def cleanup(self):
        """Cleanup and push to hub."""
        print("\nCleaning up...")

        # Wait for any pending writes
        if self.dataset:
            self.dataset.image_writer.wait_until_done()

            print(f"\nTotal episodes recorded: {self.episode_count}")
            print(f"Dataset saved locally: {self.dataset.root}")

        # Disconnect cameras
        for name, cam in self.cameras.items():
            try:
                cam.disconnect()
            except:
                pass

        # Re-enable torque and disconnect robot
        if self.robot:
            try:
                print("Re-enabling motor torque...")
                self.robot.bus1.enable_torque()
                self.robot.bus2.enable_torque()
            except:
                pass
            self.robot.disconnect()

        print("Done!")


if __name__ == "__main__":
    teacher = KinestheticTeacher()
    teacher.run()
