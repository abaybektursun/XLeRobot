#!/usr/bin/env python3
"""
Detailed scan of a single port
"""

import logging
import scservo_sdk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scan_port_detailed(port):
    """Scan a port with retries for each motor ID"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Detailed scan: {port}")
    logger.info(f"{'='*70}")

    portHandler = scservo_sdk.PortHandler(port)
    packetHandler = scservo_sdk.PacketHandler(protocol_end=0)

    if not portHandler.openPort():
        logger.error(f"Failed to open {port}")
        return

    if not portHandler.setBaudRate(1000000):
        logger.error(f"Failed to set baudrate")
        portHandler.closePort()
        return

    logger.info(f"\nScanning IDs 1-10 with 3 retries each...\n")
    logger.info(f"{'ID':<6} {'Model':<10} {'Position':<12} {'Retries':<10} {'Status'}")
    logger.info("-" * 70)

    found_motors = []

    for motor_id in range(1, 11):
        model = None
        retries_needed = 0

        # Try up to 3 times
        for attempt in range(3):
            result_model, result, error = packetHandler.ping(portHandler, motor_id)

            if result == scservo_sdk.COMM_SUCCESS:
                model = result_model
                retries_needed = attempt + 1
                break

        if model is not None:
            # Try to read position
            position, pos_result, pos_error = packetHandler.read2ByteTxRx(
                portHandler, motor_id, 56
            )

            pos_str = str(position) if pos_result == scservo_sdk.COMM_SUCCESS else "N/A"
            logger.info(f"{motor_id:<6} {model:<10} {pos_str:<12} {retries_needed:<10} ✓ Found")
            found_motors.append(motor_id)
        else:
            logger.info(f"{motor_id:<6} {'N/A':<10} {'N/A':<12} {'3':<10} ✗ Not found")

    portHandler.closePort()

    logger.info("-" * 70)
    logger.info(f"✓ Found {len(found_motors)} motor(s): {found_motors}")
    logger.info(f"✗ Missing: {[i for i in range(1, 11) if i not in found_motors]}")
    logger.info(f"{'='*70}\n")

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("DETAILED PORT SCANNER")
    logger.info("="*70)

    scan_port_detailed("/dev/ttyACM0")
