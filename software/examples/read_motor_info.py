#!/usr/bin/env python3
"""
Read detailed motor information including firmware version
"""

import logging
import scservo_sdk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_motor_info(port, motor_id):
    """Read all available motor information"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Reading Motor Information - ID {motor_id} on {port}")
    logger.info(f"{'='*70}")

    portHandler = scservo_sdk.PortHandler(port)
    packetHandler = scservo_sdk.PacketHandler(protocol_end=0)

    if not portHandler.openPort():
        logger.error("Failed to open port")
        return

    portHandler.setBaudRate(1000000)

    # Ping motor
    model, result, error = packetHandler.ping(portHandler, motor_id)
    if result != scservo_sdk.COMM_SUCCESS:
        logger.error(f"Motor not found at ID {motor_id}")
        portHandler.closePort()
        return

    logger.info(f"\n✓ Motor found at ID {motor_id}")
    logger.info(f"Model Number: {model}")

    # Common register addresses for Feetech servos
    registers = {
        "ID": 5,
        "Baud Rate": 6,
        "Return Delay": 7,
        "Firmware Version": 3,  # Common address
        "Model Number Low": 0,
        "Model Number High": 1,
        "Lock Flag": 69,
    }

    logger.info(f"\nRegister Values:")
    logger.info("-" * 70)

    for name, addr in registers.items():
        value, result, error = packetHandler.read1ByteTxRx(portHandler, motor_id, addr)
        if result == scservo_sdk.COMM_SUCCESS:
            logger.info(f"{name:20} (addr {addr:3}): {value}")
        else:
            logger.info(f"{name:20} (addr {addr:3}): Failed to read")

    # Try reading 2-byte values
    logger.info(f"\n2-Byte Register Values:")
    logger.info("-" * 70)

    two_byte_registers = {
        "Model Number": 0,
        "Firmware Version": 2,
        "Present Position": 56,
        "Present Speed": 58,
        "Present Load": 60,
        "Present Voltage": 62,
        "Present Temperature": 63,
    }

    for name, addr in two_byte_registers.items():
        value, result, error = packetHandler.read2ByteTxRx(portHandler, motor_id, addr)
        if result == scservo_sdk.COMM_SUCCESS:
            logger.info(f"{name:20} (addr {addr:3}): {value}")
        else:
            logger.info(f"{name:20} (addr {addr:3}): Failed to read")

    portHandler.closePort()
    logger.info(f"\n{'='*70}\n")

if __name__ == "__main__":
    port = "/dev/ttyACM1"
    motor_id = 1

    logger.info("\n" + "="*70)
    logger.info("STS3215 MOTOR INFORMATION READER")
    logger.info("="*70)
    logger.info("This will read firmware version and other motor details")
    logger.info("="*70)

    read_motor_info(port, motor_id)
