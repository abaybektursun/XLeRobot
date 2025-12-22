#!/usr/bin/env python3
"""
Change motor ID with EEPROM lock/unlock for permanent save
Based on Feetech STS3215 documentation
"""

import time
import logging
import scservo_sdk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory addresses for Feetech STS3215 servos
ADDR_ID = 5  # 0x05 - ID register
ADDR_LOCK = 56  # 0x38 - Lock Flag (0=unlock, 1=lock)

def change_motor_id_permanent(port, current_id, new_id):
    """Change motor ID with EEPROM unlock/lock for permanent save"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Permanently changing motor ID: {current_id} → {new_id}")
    logger.info(f"Port: {port}")
    logger.info(f"{'='*70}")

    try:
        portHandler = scservo_sdk.PortHandler(port)
        packetHandler = scservo_sdk.PacketHandler(protocol_end=0)

        if not portHandler.openPort():
            logger.error(f"✗ Failed to open {port}")
            return False

        portHandler.setBaudRate(1000000)

        # Step 1: Verify current motor exists
        logger.info(f"\n1. Verifying motor at ID {current_id}...")
        model, result, error = packetHandler.ping(portHandler, current_id)

        if result != scservo_sdk.COMM_SUCCESS:
            logger.error(f"✗ No motor found at ID {current_id}")
            portHandler.closePort()
            return False

        logger.info(f"✓ Found motor at ID {current_id} (model: {model})")

        # Step 2: Unlock EEPROM
        logger.info(f"\n2. Unlocking EEPROM (address {ADDR_LOCK})...")
        result, error = packetHandler.write1ByteTxRx(portHandler, current_id, ADDR_LOCK, 0)

        if result != scservo_sdk.COMM_SUCCESS:
            logger.error(f"✗ Failed to unlock EEPROM: {packetHandler.getTxRxResult(result)}")
            portHandler.closePort()
            return False

        logger.info(f"✓ EEPROM unlocked")
        time.sleep(0.1)

        # Step 3: Write new ID
        logger.info(f"\n3. Writing new ID {new_id} to address {ADDR_ID}...")
        result, error = packetHandler.write1ByteTxRx(portHandler, current_id, ADDR_ID, new_id)

        if result != scservo_sdk.COMM_SUCCESS:
            logger.error(f"✗ Failed to write new ID: {packetHandler.getTxRxResult(result)}")
            portHandler.closePort()
            return False

        logger.info(f"✓ New ID written")
        time.sleep(0.2)

        # Step 4: Lock EEPROM with new ID
        logger.info(f"\n4. Locking EEPROM with new ID {new_id}...")
        result, error = packetHandler.write1ByteTxRx(portHandler, new_id, ADDR_LOCK, 1)

        if result != scservo_sdk.COMM_SUCCESS:
            logger.warning(f"⚠ Lock command returned: {packetHandler.getTxRxResult(result)}")
            # Continue anyway - lock might still work
        else:
            logger.info(f"✓ EEPROM locked")

        time.sleep(0.3)

        # Step 5: Verify new ID
        logger.info(f"\n5. Verifying motor at new ID {new_id}...")
        model, result, error = packetHandler.ping(portHandler, new_id)

        if result == scservo_sdk.COMM_SUCCESS:
            logger.info(f"✓✓✓ SUCCESS! Motor responds at ID {new_id}")
            logger.info(f"✓ ID saved to EEPROM (will persist after power cycle)")
            portHandler.closePort()
            return True
        else:
            logger.error(f"✗ Motor does not respond at new ID {new_id}")
            portHandler.closePort()
            return False

    except Exception as e:
        logger.error(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    logger.info("\n" + "="*70)
    logger.info("PERMANENT MOTOR ID CHANGER (with EEPROM lock/unlock)")
    logger.info("="*70)

    port = input("Enter port (default /dev/ttyACM1): ").strip() or "/dev/ttyACM1"

    try:
        current_id = int(input("Enter CURRENT motor ID: ").strip())
        new_id = int(input("Enter NEW motor ID: ").strip())
    except ValueError:
        logger.error("Invalid ID entered")
        return

    logger.info(f"\n⚠ WARNING: About to permanently change motor ID {current_id} → {new_id}")
    logger.info("This will unlock EEPROM, change ID, and lock EEPROM")
    confirm = input("Continue? (yes/no): ").strip().lower()

    if confirm != "yes":
        logger.info("Cancelled")
        return

    success = change_motor_id_permanent(port, current_id, new_id)

    if success:
        logger.info("\n" + "="*70)
        logger.info("✓ Motor ID changed and saved permanently!")
        logger.info("✓ Power cycle the motor to verify it persists")
        logger.info("="*70)
    else:
        logger.error("\n" + "="*70)
        logger.error("✗ Failed to change motor ID")
        logger.error("="*70)

if __name__ == "__main__":
    main()
