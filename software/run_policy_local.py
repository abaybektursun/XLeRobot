"""
Run π0.5 policy inference on laptop, controlling XLeRobot over LAN.

Architecture:
    [Raspberry Pi]              [This Laptop]
    xlerobot_host.py    <--->   run_policy_local.py
    - motors                    - π0.5 model (7GB)
    - cameras                   - GPU inference
    - ZMQ server                - ZMQ client

Usage:
    # 1. On Raspberry Pi:
    python -m lerobot.robots.xlerobot.xlerobot_host

    # 2. On this laptop (with GPU):
    python run_policy_local.py --pi-ip 192.168.1.100
"""

import argparse
import time
import numpy as np
import torch
import cv2
from pathlib import Path

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


# Model paths
MODEL_PATH = "/home/abay/Projects/xle_robot/XLeRobot/software/trained_model_pi05/pretrained_model"

# π0.5 expects 224x224 images
PI05_IMAGE_SIZE = (224, 224)

# Task description for π0.5 VLM
TASK_DESCRIPTION = "Pick up the towel"

# Motor names (must match training)
RIGHT_ARM_MOTORS = [
    "right_arm_shoulder_pan",
    "right_arm_shoulder_lift",
    "right_arm_elbow_flex",
    "right_arm_wrist_flex",
    "right_arm_wrist_roll",
    "right_arm_gripper",
]

# Safety limits
ACTION_LIMITS = {
    "right_arm_shoulder_pan": (90, 100),
    "right_arm_shoulder_lift": (25, 45),
    "right_arm_elbow_flex": (-70, -55),
    "right_arm_wrist_flex": (85, 100),
    "right_arm_wrist_roll": (-30, 30),
    "right_arm_gripper": (0, 100),
}


def load_policy(model_path: str, device: str = "cuda"):
    """Load trained π0.5 policy."""
    print(f"Loading π0.5 policy from {model_path}")

    policy = PI05Policy.from_pretrained(model_path)

    if device == "cuda" and torch.cuda.is_available():
        policy.to(device, dtype=torch.bfloat16)
    else:
        policy.to(device)

    policy.eval()
    print(f"Policy loaded on {device}")

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        model_path, config_filename="policy_preprocessor.json"
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        model_path, config_filename="policy_postprocessor.json"
    )

    return policy, preprocessor, postprocessor


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Convert image to tensor for π0.5."""
    if image is None:
        return torch.zeros(3, 480, 640)

    # BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # CHW format, normalize to [0, 1]
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(image)


def run_inference_loop(
    policy: PI05Policy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    robot: XLerobotClient,
    num_steps: int = 500,
    fps: float = 30.0,
):
    """Run policy inference loop."""
    print(f"\nStarting inference loop: {num_steps} steps at {fps} FPS")
    print("Press Ctrl+C to stop\n")

    step_duration = 1.0 / fps
    device = next(policy.parameters()).device

    try:
        for step in range(num_steps):
            step_start = time.time()

            # Get observation from Pi (via ZMQ)
            obs = robot.get_observation()

            # Extract state
            raw_state = np.array([
                obs.get(f"{motor}.pos", 0.0) for motor in RIGHT_ARM_MOTORS
            ], dtype=np.float32)

            # Get camera frame (sent as numpy array from Pi)
            camera_frame = obs.get("head")
            if camera_frame is None:
                print(f"Step {step}: No camera frame, skipping")
                continue

            image_tensor = preprocess_image(camera_frame)
            state_tensor = torch.from_numpy(raw_state)

            # Build observation dict
            observation = {
                "observation.images.head": image_tensor,
                "observation.state": state_tensor,
                "task": TASK_DESCRIPTION,
            }

            # Preprocess
            processed_batch = preprocessor(observation)

            # Convert to bfloat16
            for key in processed_batch:
                if isinstance(processed_batch[key], torch.Tensor):
                    if processed_batch[key].is_floating_point():
                        processed_batch[key] = processed_batch[key].to(device, dtype=torch.bfloat16)
                    else:
                        processed_batch[key] = processed_batch[key].to(device)

            # Inference
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                action = policy.select_action(processed_batch)

            # Postprocess
            action_dict = {"action": action}
            processed_action = postprocessor(action_dict)
            action_values = processed_action["action"].float().cpu().numpy()

            if action_values.ndim > 1:
                action_values = action_values[0]

            # Safety clamp
            clamped = []
            for i, motor in enumerate(RIGHT_ARM_MOTORS):
                val = float(action_values[i])
                min_val, max_val = ACTION_LIMITS[motor]
                clamped.append(max(min_val, min(max_val, val)))

            # Send action to Pi
            robot_action = {
                f"{motor}.pos": clamped[i]
                for i, motor in enumerate(RIGHT_ARM_MOTORS)
            }
            robot.send_action(robot_action)

            # Log every 50 steps
            if step % 50 == 0:
                print(f"Step {step}/{num_steps}")
                print(f"  State:  {raw_state[:3]}...")
                print(f"  Action: {clamped[:3]}...")

            # Maintain timing
            elapsed = time.time() - step_start
            if elapsed < step_duration:
                time.sleep(step_duration - elapsed)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.disconnect()
        print("Disconnected.")


def main():
    parser = argparse.ArgumentParser(description="Run π0.5 policy over LAN")
    parser.add_argument("--pi-ip", required=True, help="Raspberry Pi IP address")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to trained model")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps")
    parser.add_argument("--fps", type=float, default=30.0, help="Control frequency")
    parser.add_argument("--dry-run", action="store_true", help="Test without Pi")
    args = parser.parse_args()

    # Load policy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    policy, preprocessor, postprocessor = load_policy(args.model_path, device)

    if args.dry_run:
        print("\n--- DRY RUN ---")
        dummy_image = torch.randn(3, 480, 640)
        dummy_state = torch.randn(6)
        observation = {
            "observation.images.head": dummy_image,
            "observation.state": dummy_state,
            "task": TASK_DESCRIPTION,
        }
        processed = preprocessor(observation)
        for k in processed:
            if isinstance(processed[k], torch.Tensor) and processed[k].is_floating_point():
                processed[k] = processed[k].to(device, dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            action = policy.select_action(processed)
        result = postprocessor({"action": action})["action"].float().cpu().numpy()
        print(f"Output action shape: {result.shape}")
        print(f"Output action: {result[0] if result.ndim > 1 else result}")
        print("Dry run OK!")
        return

    # Connect to Pi
    print(f"\nConnecting to Pi at {args.pi_ip}...")
    config = XLerobotClientConfig(
        remote_ip=args.pi_ip,
        cameras={
            "head": OpenCVCameraConfig(
                index_or_path=0,  # Placeholder, actual camera is on Pi
                fps=30,
                width=640,
                height=480,
            )
        }
    )
    robot = XLerobotClient(config)
    robot.connect()
    print("Connected!")

    # Run inference
    run_inference_loop(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot=robot,
        num_steps=args.steps,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
