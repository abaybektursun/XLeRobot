#!/usr/bin/env python3
"""
Debug script for changing motor ID with detailed logging
"""

import time
import logging
import scservo_sdk

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

ADDR_ID = 5      # 0x05 - ID register
ADDR_LOCK = 55   # 0x37 - Lock register (common for Feetech servos) - 0=unlock, 1=lock

def change_id_with_debug(port, current_id, new_id):
    """Change ID with detailed debugging"""
    logger.info(f"\n{'='*70}")
    logger.info(f"DEBUG: Changing motor ID {current_id} → {new_id}")
    logger.info(f"{'='*70}")

    portHandler = scservo_sdk.PortHandler(port)
    packetHandler = scservo_sdk.PacketHandler(protocol_end=0)

    if not portHandler.openPort():
        logger.error("Failed to open port")
        return False

    portHandler.setBaudRate(1000000)

    # Step 1: Verify motor
    logger.info(f"\n1. Pinging motor at ID {current_id}...")
    model, result, error = packetHandler.ping(portHandler, current_id)
    if result != scservo_sdk.COMM_SUCCESS:
        logger.error(f"Motor not found")
        portHandler.closePort()
        return False
    logger.info(f"✓ Motor found (model: {model})")

    # Step 2: Read current lock status
    logger.info(f"\n2. Reading current lock status (addr {ADDR_LOCK})...")
    lock_value, result, error = packetHandler.read1ByteTxRx(portHandler, current_id, ADDR_LOCK)
    if result == scservo_sdk.COMM_SUCCESS:
        logger.info(f"Current lock value: {lock_value} (0=unlocked, 1=locked)")
    else:
        logger.warning(f"Could not read lock status")

    # Step 3: Unlock EEPROM
    logger.info(f"\n3. Unlocking EEPROM (writing 0 to addr {ADDR_LOCK})...")
    result, error = packetHandler.write1ByteTxRx(portHandler, current_id, ADDR_LOCK, 0)
    if result != scservo_sdk.COMM_SUCCESS:
        logger.error(f"Unlock failed: {packetHandler.getTxRxResult(result)}")
        portHandler.closePort()
        return False
    logger.info(f"✓ Unlock command sent")
    time.sleep(0.5)

    # Step 3b: Verify unlock
    lock_value, result, error = packetHandler.read1ByteTxRx(portHandler, current_id, ADDR_LOCK)
    if result == scservo_sdk.COMM_SUCCESS:
        logger.info(f"Lock value after unlock: {lock_value}")

    # Step 4: Write new ID
    logger.info(f"\n4. Writing new ID {new_id} to addr {ADDR_ID}...")
    result, error = packetHandler.write1ByteTxRx(portHandler, current_id, ADDR_ID, new_id)
    if result != scservo_sdk.COMM_SUCCESS:
        logger.error(f"ID write failed: {packetHandler.getTxRxResult(result)}")
        portHandler.closePort()
        return False
    logger.info(f"✓ ID write command sent")
    time.sleep(0.5)

    # Step 5: Verify new ID responds
    logger.info(f"\n5. Pinging motor at new ID {new_id}...")
    model, result, error = packetHandler.ping(portHandler, new_id)
    if result != scservo_sdk.COMM_SUCCESS:
        logger.error(f"Motor does not respond at new ID")
        portHandler.closePort()
        return False
    logger.info(f"✓ Motor responds at new ID {new_id}")

    # Step 6: Lock EEPROM with new ID
    logger.info(f"\n6. Locking EEPROM (writing 1 to addr {ADDR_LOCK} on ID {new_id})...")
    result, error = packetHandler.write1ByteTxRx(portHandler, new_id, ADDR_LOCK, 1)
    if result != scservo_sdk.COMM_SUCCESS:
        logger.warning(f"Lock command: {packetHandler.getTxRxResult(result)}")
    else:
        logger.info(f"✓ Lock command sent")
    time.sleep(0.5)

    # Step 7: Verify lock
    lock_value, result, error = packetHandler.read1ByteTxRx(portHandler, new_id, ADDR_LOCK)
    if result == scservo_sdk.COMM_SUCCESS:
        logger.info(f"Lock value after lock: {lock_value}")

    # Step 8: Read ID to verify
    logger.info(f"\n7. Reading ID register to verify...")
    id_value, result, error = packetHandler.read1ByteTxRx(portHandler, new_id, ADDR_ID)
    if result == scservo_sdk.COMM_SUCCESS:
        logger.info(f"ID register value: {id_value}")

    portHandler.closePort()

    logger.info(f"\n{'='*70}")
    logger.info(f"✓ Process complete!")
    logger.info(f"Now power cycle the motor and scan to verify persistence")
    logger.info(f"{'='*70}")

    return True

if __name__ == "__main__":
    port = "/dev/ttyACM0"
    current_id = 1
    new_id = 9  # Third base wheel motor (right wheel)

    change_id_with_debug(port, current_id, new_id)
