#!/usr/bin/env python3
"""
Scan and display all motors on a port
"""

import logging
import scservo_sdk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scan_port(port):
    """Scan a port for all connected motors"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Scanning port: {port}")
    logger.info(f"{'='*60}")

    try:
        portHandler = scservo_sdk.PortHandler(port)
        packetHandler = scservo_sdk.PacketHandler(protocol_end=0)

        if not portHandler.openPort():
            logger.error(f"✗ Failed to open port {port}")
            return []

        if not portHandler.setBaudRate(1000000):
            logger.error(f"✗ Failed to set baudrate on {port}")
            portHandler.closePort()
            return []

        logger.info(f"✓ Port {port} opened successfully")
        logger.info(f"\nScanning for motors (IDs 1-20)...\n")
        logger.info(f"{'ID':<6} {'Model':<10} {'Position':<12} {'Status'}")
        logger.info("-" * 60)

        found_motors = []

        for motor_id in range(1, 21):
            model, result, error = packetHandler.ping(portHandler, motor_id)

            if result == scservo_sdk.COMM_SUCCESS:
                # Try to read position
                position, pos_result, pos_error = packetHandler.read2ByteTxRx(
                    portHandler, motor_id, 56  # Present_Position address
                )

                if pos_result == scservo_sdk.COMM_SUCCESS:
                    logger.info(f"{motor_id:<6} {model:<10} {position:<12} ✓ OK")
                else:
                    logger.info(f"{motor_id:<6} {model:<10} {'N/A':<12} ✓ Detected")

                found_motors.append(motor_id)

        portHandler.closePort()

        logger.info("-" * 60)
        if len(found_motors) == 0:
            logger.warning("⚠ No motors detected!")
        else:
            logger.info(f"✓ Found {len(found_motors)} motor(s): {found_motors}")

        return found_motors

    except Exception as e:
        logger.error(f"✗ Error scanning {port}: {e}")
        return []

def main():
    logger.info("\n" + "="*70)
    logger.info("MOTOR SCANNER")
    logger.info("="*70)

    # Scan both ports
    logger.info("\n")
    motors_acm0 = scan_port('/dev/ttyACM0')

    logger.info("\n")
    motors_acm1 = scan_port('/dev/ttyACM1')

    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"/dev/ttyACM0: {len(motors_acm0)} motor(s) - IDs: {motors_acm0}")
    logger.info(f"/dev/ttyACM1: {len(motors_acm1)} motor(s) - IDs: {motors_acm1}")
    logger.info("="*70 + "\n")

if __name__ == "__main__":
    main()
