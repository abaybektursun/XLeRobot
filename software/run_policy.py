"""
Run trained π0.5 policy on XLeRobot.

!!! IMPORTANT !!!
Uses π0.5 (pi05) foundation model - NOT ACT!
π0.5 works with 10-50 demos and is a vision-language-action model.

This script loads the trained model and runs inference on the robot
to validate the towel pickup behavior.
"""

import time
import numpy as np
import torch
import cv2
from pathlib import Path
import json
from safetensors.torch import load_file

# LeRobot imports
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import PolicyProcessorPipeline

# XLeRobot imports
from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


# π0.5 model path (7GB model)
MODEL_PATH = "/home/abay/Projects/xle_robot/XLeRobot/software/trained_model_104ep/pretrained_model"
PREPROCESSOR_CONFIG = "policy_preprocessor.json"
POSTPROCESSOR_CONFIG = "policy_postprocessor.json"

# π0.5 expects 224x224 images
PI05_IMAGE_SIZE = (224, 224)

# Task description for π0.5 VLM (language prompt)
TASK_DESCRIPTION = "Pick up the towel"

# Safety limits for each motor (min, max) - prevent arm dropping
ACTION_LIMITS = {
    "right_arm_shoulder_pan": (90, 100),      # Keep centered
    "right_arm_shoulder_lift": (25, 45),      # Tighter - prevent dropping
    "right_arm_elbow_flex": (-70, -55),       # Tighter elbow range
    "right_arm_wrist_flex": (85, 100),        # Keep wrist upright
    "right_arm_wrist_roll": (-30, 30),        # Limit rotation more
    "right_arm_gripper": (0, 100),            # Full gripper range
}


def load_normalization_stats(model_path: str):
    """Load normalization stats from the trained model."""
    # Load from safetensors files
    preprocessor_path = Path(model_path) / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    postprocessor_path = Path(model_path) / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

    state_mean = None
    state_std = None
    action_mean = None
    action_std = None

    if preprocessor_path.exists():
        data = load_file(preprocessor_path)
        for key in data:
            if "mean" in key:
                state_mean = data[key].numpy()
            elif "std" in key:
                state_std = data[key].numpy()

    if postprocessor_path.exists():
        data = load_file(postprocessor_path)
        for key in data:
            if "mean" in key:
                action_mean = data[key].numpy()
            elif "std" in key:
                action_std = data[key].numpy()

    return state_mean, state_std, action_mean, action_std


def load_policy(model_path: str, device: str = "cpu"):
    """Load trained π0.5 policy with preprocessor and postprocessor."""
    print(f"Loading π0.5 policy from {model_path}")

    # Load the policy
    policy = PI05Policy.from_pretrained(model_path)

    # π0.5 uses bfloat16
    if device == "cuda" and torch.cuda.is_available():
        policy.to(device, dtype=torch.bfloat16)
    else:
        policy.to(device)

    policy.eval()
    print(f"π0.5 policy loaded on {device}")

    # Load preprocessor and postprocessor
    print("Loading preprocessor and postprocessor...")
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        model_path, config_filename=PREPROCESSOR_CONFIG
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        model_path, config_filename=POSTPROCESSOR_CONFIG
    )
    print("Preprocessor and postprocessor loaded")

    return policy, preprocessor, postprocessor


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Convert OpenCV image to tensor format expected by π0.5 policy."""
    # OpenCV gives BGR, convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # π0.5 expects 224x224 images
    if image.shape[:2] != PI05_IMAGE_SIZE:
        image = cv2.resize(image, PI05_IMAGE_SIZE)
    # Convert to CHW format and normalize to [0, 1]
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(image)


# Motor names for the right arm (matches dataset)
RIGHT_ARM_MOTORS = [
    "right_arm_shoulder_pan",
    "right_arm_shoulder_lift",
    "right_arm_elbow_flex",
    "right_arm_wrist_flex",
    "right_arm_wrist_roll",
    "right_arm_gripper",
]

# Starting pose from training data (first frame of episode 0)
TRAINING_START_POSE = np.array([100.0, 31.13553, -65.811966, 100.0, -69.23077, 3.1516743], dtype=np.float32)


def move_to_start_pose(robot: "XLerobot", target_pose: np.ndarray, duration: float = 3.0, fps: float = 30.0):
    """Smoothly move arm to target pose over duration seconds."""
    print(f"\nMoving arm to starting pose over {duration}s...")

    obs = robot.get_observation()
    current_pose = np.array([obs[f"{motor}.pos"] for motor in RIGHT_ARM_MOTORS], dtype=np.float32)

    print(f"  Current pose: {current_pose}")
    print(f"  Target pose:  {target_pose}")

    num_steps = int(duration * fps)
    for i in range(num_steps + 1):
        t = i / num_steps
        # Smooth interpolation
        interp_pose = current_pose + t * (target_pose - current_pose)

        action = {f"{motor}.pos": float(interp_pose[j]) for j, motor in enumerate(RIGHT_ARM_MOTORS)}
        robot.send_action(action)
        time.sleep(1.0 / fps)

    print("  Done moving to start pose.")


def run_inference_loop(
    policy: PI05Policy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    robot: XLerobot,
    num_steps: int = 500,
    fps: float = 30.0,
):
    """Run the π0.5 policy in a control loop."""
    print(f"\nStarting π0.5 inference loop for {num_steps} steps at {fps} FPS")
    print("Press Ctrl+C to stop\n")

    step_duration = 1.0 / fps

    try:
        for step in range(num_steps):
            step_start = time.time()

            # Get current observation from robot (includes camera)
            obs = robot.get_observation()

            # Extract state (6 DOF for right arm) - robot uses "{motor}.pos" format
            raw_state = np.array([
                obs[f"{motor}.pos"] for motor in RIGHT_ARM_MOTORS
            ], dtype=np.float32)

            # Get camera frame and preprocess - robot uses camera key directly
            camera_frame = obs["head"]
            image_tensor = preprocess_image(camera_frame)
            state_tensor = torch.from_numpy(raw_state)

            # Create observation dict for preprocessor
            observation = {
                "observation.images.head": image_tensor,
                "observation.state": state_tensor,
                "task": TASK_DESCRIPTION,
            }

            # Apply preprocessor (handles batching, normalization, tokenization)
            processed_batch = preprocessor(observation)

            # Convert tensors to bfloat16 to match model dtype
            for key in processed_batch:
                if isinstance(processed_batch[key], torch.Tensor) and processed_batch[key].is_floating_point():
                    processed_batch[key] = processed_batch[key].to(torch.bfloat16)

            # Run inference with autocast to handle dtype conversion
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                action = policy.select_action(processed_batch)

            # Apply postprocessor to unnormalize action
            action_dict = {"action": action}
            processed_action = postprocessor(action_dict)
            robot_action_values = processed_action["action"].float().cpu().numpy()

            # Handle batch dimension if present
            if robot_action_values.ndim > 1:
                robot_action_values = robot_action_values[0]

            # Apply safety clamping to prevent arm dropping
            clamped_action_values = []
            for i, motor in enumerate(RIGHT_ARM_MOTORS):
                val = float(robot_action_values[i])
                min_val, max_val = ACTION_LIMITS[motor]
                clamped_val = max(min_val, min(max_val, val))
                clamped_action_values.append(clamped_val)

            # Convert action to robot format: {motor}.pos
            robot_action = {
                f"{motor}.pos": clamped_action_values[i]
                for i, motor in enumerate(RIGHT_ARM_MOTORS)
            }

            # Send action to robot
            robot.send_action(robot_action)

            # Print progress every 50 steps
            if step % 50 == 0:
                print(f"Step {step}/{num_steps}")
                print(f"  Robot state: {raw_state}")
                print(f"  Raw action:  {robot_action_values[:6]}")
                print(f"  Clamped:     {clamped_action_values}")

            # Maintain loop timing
            elapsed = time.time() - step_start
            if elapsed < step_duration:
                time.sleep(step_duration - elapsed)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop robot if connected
        if robot.is_connected:
            robot.disconnect()
        print("Done.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run trained policy on XLeRobot")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to trained model")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps to run")
    parser.add_argument("--fps", type=float, default=30.0, help="Control loop FPS")
    parser.add_argument("--dry-run", action="store_true", help="Test without robot")
    parser.add_argument("--go-to-start", action="store_true", help="Move arm to training start pose first")
    args = parser.parse_args()

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, preprocessor, postprocessor = load_policy(args.model_path, device)

    if args.dry_run:
        print("\n--- DRY RUN MODE ---")
        print("Testing π0.5 policy with dummy data...")

        # Create dummy observation - preprocessor handles batching and dtype
        # Images should be 480x640 (original camera size) - preprocessor resizes to 224x224
        dummy_image = torch.randn(3, 480, 640)
        dummy_state = torch.randn(6)
        observation = {
            "observation.images.head": dummy_image,
            "observation.state": dummy_state,
            "task": TASK_DESCRIPTION,
        }

        print(f"Input image shape: {dummy_image.shape}")
        print(f"Input state shape: {dummy_state.shape}")
        print(f"Task: {TASK_DESCRIPTION}")

        # Apply preprocessor
        print("\nApplying preprocessor...")
        processed_batch = preprocessor(observation)
        print(f"Processed batch keys: {list(processed_batch.keys())}")

        # Convert tensors to bfloat16 to match model dtype
        for key in processed_batch:
            if isinstance(processed_batch[key], torch.Tensor) and processed_batch[key].is_floating_point():
                processed_batch[key] = processed_batch[key].to(torch.bfloat16)

        # Run inference with autocast to handle dtype conversion
        print("\nRunning inference...")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            action_result = policy.select_action(processed_batch)

        print(f"Action result type: {type(action_result)}")
        if isinstance(action_result, dict):
            print(f"Action dict keys: {action_result.keys()}")
            actions = action_result["action"]
        else:
            actions = action_result

        print(f"Action shape: {actions.shape}")
        print(f"Action dtype: {actions.dtype}")

        # Apply postprocessor
        print("\nApplying postprocessor...")
        action_dict = {"action": actions}
        processed_action = postprocessor(action_dict)
        final_action = processed_action["action"].float().cpu().numpy()
        if final_action.ndim > 1:
            final_action = final_action[0]

        print(f"Final action (first 6 dims): {final_action[:6]}")
        print("\n✓ π0.5 policy works correctly!")
        return

    # Initialize robot with camera config
    # Default config uses RANGE_M100_100 normalization (same as data collection)
    print("Initializing robot...")
    config = XLerobotConfig(
        port1="/dev/ttyACM0",
        port2="/dev/ttyACM1",
        cameras={
            "head": OpenCVCameraConfig(
                index_or_path="/dev/video0",
                fps=30,
                width=640,
                height=480,
            )
        }
    )
    robot = XLerobot(config)
    robot.connect()

    try:
        # Optionally move to training start pose first
        if args.go_to_start:
            move_to_start_pose(robot, TRAINING_START_POSE)

        # Run inference loop
        run_inference_loop(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot=robot,
            num_steps=args.steps,
            fps=args.fps,
        )
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
