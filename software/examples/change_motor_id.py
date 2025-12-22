#!/usr/bin/env python3
"""
Change a single motor's ID
"""

import time
import logging
import scservo_sdk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def change_motor_id(port, current_id, new_id):
    """Change motor ID from current_id to new_id"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Changing motor ID: {current_id} → {new_id}")
    logger.info(f"Port: {port}")
    logger.info(f"{'='*70}")

    try:
        portHandler = scservo_sdk.PortHandler(port)
        packetHandler = scservo_sdk.PacketHandler(protocol_end=0)

        if not portHandler.openPort():
            logger.error(f"✗ Failed to open {port}")
            return False

        portHandler.setBaudRate(1000000)

        # Verify current motor exists
        logger.info(f"\n1. Verifying motor at ID {current_id}...")
        model, result, error = packetHandler.ping(portHandler, current_id)

        if result != scservo_sdk.COMM_SUCCESS:
            logger.error(f"✗ No motor found at ID {current_id}")
            portHandler.closePort()
            return False

        logger.info(f"✓ Found motor at ID {current_id} (model: {model})")

        # Change ID
        logger.info(f"\n2. Writing new ID {new_id} to motor...")
        result, error = packetHandler.write1ByteTxRx(portHandler, current_id, 5, new_id)

        if result != scservo_sdk.COMM_SUCCESS:
            logger.error(f"✗ Failed to write new ID: {packetHandler.getTxRxResult(result)}")
            portHandler.closePort()
            return False

        logger.info(f"✓ ID write command sent")

        time.sleep(0.5)

        # Verify new ID
        logger.info(f"\n3. Verifying motor at new ID {new_id}...")
        model, result, error = packetHandler.ping(portHandler, new_id)

        if result == scservo_sdk.COMM_SUCCESS:
            logger.info(f"✓✓✓ SUCCESS! Motor now responds at ID {new_id}")
            portHandler.closePort()
            return True
        else:
            logger.error(f"✗ Motor does not respond at new ID {new_id}")
            portHandler.closePort()
            return False

    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return False

def main():
    logger.info("\n" + "="*70)
    logger.info("MOTOR ID CHANGER")
    logger.info("="*70)

    port = input("Enter port (default /dev/ttyACM0): ").strip() or "/dev/ttyACM0"

    try:
        current_id = int(input("Enter CURRENT motor ID: ").strip())
        new_id = int(input("Enter NEW motor ID: ").strip())
    except ValueError:
        logger.error("Invalid ID entered")
        return

    logger.info(f"\n⚠ WARNING: About to change motor ID {current_id} → {new_id}")
    confirm = input("Continue? (yes/no): ").strip().lower()

    if confirm != "yes":
        logger.info("Cancelled")
        return

    success = change_motor_id(port, current_id, new_id)

    if success:
        logger.info("\n" + "="*70)
        logger.info("✓ Motor ID changed successfully!")
        logger.info("="*70)
    else:
        logger.error("\n" + "="*70)
        logger.error("✗ Failed to change motor ID")
        logger.error("="*70)

if __name__ == "__main__":
    main()
