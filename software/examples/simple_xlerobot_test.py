#!/usr/bin/env python3
"""
Simple XLeRobot test - connects to robot and reads motor positions
"""

from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
import time

def main():
    print("="*70)
    print("XLEROBOT SIMPLE TEST")
    print("="*70)

    # Create config
    config = XLerobotConfig()
    print(f"\nPort 1 (Left arm + Head): {config.port1}")
    print(f"Port 2 (Right arm + Base): {config.port2}")

    # Create robot
    robot = XLerobot(config)

    # Connect without calibration
    print("\nConnecting to robot (skipping calibration)...")
    try:
        robot.connect(calibrate=False)
        print("✓ Robot connected successfully!")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print("\nTry power cycling the robot and run again")
        return

    # Get observation
    print("\nReading motor positions...")
    obs = robot.get_observation()

    print("\nLeft Arm:")
    for key in sorted(obs.keys()):
        if key.startswith("left_arm") and key.endswith(".pos"):
            print(f"  {key:30} = {obs[key]:.1f}")

    print("\nRight Arm:")
    for key in sorted(obs.keys()):
        if key.startswith("right_arm") and key.endswith(".pos"):
            print(f"  {key:30} = {obs[key]:.1f}")

    print("\nHead:")
    for key in sorted(obs.keys()):
        if key.startswith("head") and key.endswith(".pos"):
            print(f"  {key:30} = {obs[key]:.1f}")

    print("\nBase:")
    for key in sorted(obs.keys()):
        if key.startswith("base") and key.endswith(".pos"):
            print(f"  {key:30} = {obs[key]:.1f}")

    # Disconnect
    robot.disconnect()
    print("\n✓ Test complete!")

if __name__ == "__main__":
    main()
