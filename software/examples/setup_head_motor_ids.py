#!/usr/bin/env python3
"""
Setup HEAD motor IDs ONLY (Motor IDs 7 and 8 on /dev/ttyACM0)
IMPORTANT: This does NOT change arm motor IDs (1-6)
"""

import time
import logging
import scservo_sdk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_motor():
    """Find what ID the currently connected motor has"""
    port = "/dev/ttyACM0"

    portHandler = scservo_sdk.PortHandler(port)
    packetHandler = scservo_sdk.PacketHandler(protocol_end=0)

    if not portHandler.openPort():
        logger.error(f"Failed to open {port}")
        return None

    portHandler.setBaudRate(1000000)

    logger.info("Scanning for motor IDs 1-20...")
    found = []

    for motor_id in range(1, 21):
        model, result, error = packetHandler.ping(portHandler, motor_id)
        if result == scservo_sdk.COMM_SUCCESS:
            found.append(motor_id)
            logger.info(f"  Found motor at ID {motor_id} (model: {model})")

    portHandler.closePort()
    return found

def set_motor_id(target_id):
    """Set the motor ID"""
    port = "/dev/ttyACM0"

    portHandler = scservo_sdk.PortHandler(port)
    packetHandler = scservo_sdk.PacketHandler(protocol_end=0)

    if not portHandler.openPort():
        logger.error(f"Failed to open {port}")
        return False

    portHandler.setBaudRate(1000000)

    # Find the motor
    logger.info("Looking for the connected motor...")
    current_id = None

    for motor_id in range(1, 21):
        model, result, error = packetHandler.ping(portHandler, motor_id)
        if result == scservo_sdk.COMM_SUCCESS:
            current_id = motor_id
            logger.info(f"✓ Found motor at ID {current_id}")
            break

    if current_id is None:
        logger.error("✗ No motor found!")
        portHandler.closePort()
        return False

    # Change the ID
    logger.info(f"Changing motor ID from {current_id} to {target_id}...")

    # Write to ID register (address 5)
    result, error = packetHandler.write1ByteTxRx(portHandler, current_id, 5, target_id)

    if result != scservo_sdk.COMM_SUCCESS:
        logger.error(f"✗ Failed to change ID: {packetHandler.getTxRxResult(result)}")
        portHandler.closePort()
        return False

    time.sleep(0.5)

    # Verify the change
    model, result, error = packetHandler.ping(portHandler, target_id)
    if result == scservo_sdk.COMM_SUCCESS:
        logger.info(f"✓ Successfully set motor ID to {target_id}")
        portHandler.closePort()
        return True
    else:
        logger.error(f"✗ Could not verify new ID {target_id}")
        portHandler.closePort()
        return False

def main():
    logger.info("="*70)
    logger.info("HEAD MOTOR ID SETUP (/dev/ttyACM0)")
    logger.info("="*70)
    logger.info("\nThis script will set head motor IDs to 7 and 8")
    logger.info("The arm motors (IDs 1-6) should already be connected in daisy chain")
    logger.info("="*70)

    # Check current state
    logger.info("\nChecking currently connected motors on /dev/ttyACM0...")
    existing = find_motor()

    if existing:
        logger.info(f"\nCurrently detected motor IDs: {existing}")
        if 1 in existing and 2 in existing and 3 in existing and 4 in existing and 5 in existing and 6 in existing:
            logger.info("✓ Arm motors (1-6) detected")
        else:
            logger.warning("⚠ Arm motors may not be fully connected")

    logger.info("\n" + "="*70)
    logger.info("SETUP HEAD MOTOR 1 (ID 7)")
    logger.info("="*70)
    logger.info("\nInstructions:")
    logger.info("1. Disconnect head motor 2 (if connected)")
    logger.info("2. Connect ONLY head motor 1 to the END of the arm daisy chain")
    logger.info("3. The full chain should be: Arm motors 1-6 → Head motor 1")
    logger.info("")

    input("Press ENTER when ready...")
    time.sleep(2)

    if set_motor_id(7):
        logger.info("✓ Head motor 1 configured as ID 7")
    else:
        logger.error("✗ Failed to set head motor 1 ID")
        return

    logger.info("\n" + "="*70)
    logger.info("SETUP HEAD MOTOR 2 (ID 8)")
    logger.info("="*70)
    logger.info("\nInstructions:")
    logger.info("1. Keep head motor 1 connected")
    logger.info("2. Connect head motor 2 to the END of the chain")
    logger.info("3. The full chain should be: Arm motors 1-6 → Head motor 1 (ID 7) → Head motor 2")
    logger.info("")

    input("Press ENTER when ready...")
    time.sleep(2)

    if set_motor_id(8):
        logger.info("✓ Head motor 2 configured as ID 8")
    else:
        logger.error("✗ Failed to set head motor 2 ID")
        return

    logger.info("\n" + "="*70)
    logger.info("HEAD MOTOR SETUP COMPLETE!")
    logger.info("="*70)
    logger.info("\nFinal verification...")
    final = find_motor()
    logger.info(f"Motors on /dev/ttyACM0: {final}")

    if final == [1, 2, 3, 4, 5, 6, 7, 8]:
        logger.info("✓ All motors configured correctly!")
        logger.info("\nNext: Setup base wheel motors on /dev/ttyACM1")
    else:
        logger.warning("⚠ Unexpected motor configuration")

    logger.info("="*70)

if __name__ == "__main__":
    main()
