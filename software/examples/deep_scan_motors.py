#!/usr/bin/env python3
"""
Deep scan with multiple baudrates and protocols
Some motors might be configured with different baudrates
"""

import logging
import scservo_sdk
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deep_scan(port):
    """Try different baudrates to find motors"""
    logger.info(f"\n{'='*70}")
    logger.info(f"DEEP SCAN: {port}")
    logger.info(f"{'='*70}")

    # Try different baudrates
    baudrates = [1000000, 115200, 57600, 9600]

    for baudrate in baudrates:
        logger.info(f"\n--- Testing baudrate: {baudrate} ---")

        try:
            portHandler = scservo_sdk.PortHandler(port)
            packetHandler = scservo_sdk.PacketHandler(protocol_end=0)

            if not portHandler.openPort():
                logger.error(f"✗ Failed to open {port}")
                continue

            if not portHandler.setBaudRate(baudrate):
                logger.error(f"✗ Failed to set baudrate {baudrate}")
                portHandler.closePort()
                continue

            found = []

            # Scan IDs 1-20
            for motor_id in range(1, 21):
                model, result, error = packetHandler.ping(portHandler, motor_id)

                if result == scservo_sdk.COMM_SUCCESS:
                    found.append(motor_id)
                    logger.info(f"  ✓ Motor ID {motor_id} found (model: {model})")

            if found:
                logger.info(f"\n✓✓✓ SUCCESS at {baudrate} baud! Found motors: {found}")
            else:
                logger.info(f"  No motors found at {baudrate}")

            portHandler.closePort()
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error at {baudrate}: {e}")

def test_broadcast_ping(port):
    """Try broadcast ping to detect any motors"""
    logger.info(f"\n{'='*70}")
    logger.info(f"BROADCAST PING TEST: {port}")
    logger.info(f"{'='*70}")

    try:
        portHandler = scservo_sdk.PortHandler(port)
        packetHandler = scservo_sdk.PacketHandler(protocol_end=0)

        if not portHandler.openPort():
            logger.error(f"✗ Failed to open {port}")
            return

        portHandler.setBaudRate(1000000)

        logger.info("Sending broadcast ping...")

        # Try to ping broadcast address (0xFE)
        model, result, error = packetHandler.ping(portHandler, 0xFE)

        if result == scservo_sdk.COMM_SUCCESS:
            logger.info(f"✓ Got broadcast response")
        else:
            logger.info(f"No broadcast response")

        portHandler.closePort()

    except Exception as e:
        logger.error(f"Broadcast test error: {e}")

def main():
    logger.info("\n" + "="*70)
    logger.info("DEEP MOTOR SCANNER")
    logger.info("="*70)
    logger.info("This will try multiple baudrates to find motors")
    logger.info("="*70)

    # Test both ports
    for port in ['/dev/ttyACM0', '/dev/ttyACM1']:
        deep_scan(port)
        test_broadcast_ping(port)

    logger.info("\n" + "="*70)
    logger.info("SCAN COMPLETE")
    logger.info("="*70)

if __name__ == "__main__":
    main()
