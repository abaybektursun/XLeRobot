# XLeRobot Motor ID Setup - Work Log & Lessons Learned

**Date:** October 4, 2025
**Goal:** Configure motor IDs for XLeRobot head and base wheel motors

---

## Hardware Overview

### Expected Configuration (per xlerobot.py)
- **Bus 1 (/dev/ttyACM0):** Left arm (IDs 1-6) + Head motors (IDs 7-8)
- **Bus 2 (/dev/ttyACM1):** Right arm (IDs 1-6) + Base wheels (IDs 7-9)

### Motors Available
- **2 Head motors:** TOP (pan/yaw) and BOTTOM (tilt) - both STS3215 servos
- **3 Base wheel motors:** Left, Back, Right - all STS3215 servos
- **2 Arm motors:** Left and Right SO-100/SO-101 arms (pre-configured IDs 1-6)

---

## Key Findings

### 1. SO-100/SO-101 Arms
- **Pre-configured at factory** with IDs 1-6
- **DO NOT change these IDs** - they are permanent and correct
- Each arm has 6 motors daisy-chained together

### 2. Head & Base Motors (STS3215)
- Come with **default ID 1** (unconfigured)
- Require **individual ID assignment** one motor at a time
- IDs **persist after power cycle** when using correct lock address (55)

### 3. Motor ID Persistence Issue - SOLVED ✓
**Problem:** Motor IDs revert to default (ID 1) after power cycling.

**Attempted Solutions:**
- ✗ Writing to lock address 48 (incorrect address)
- ✗ Writing to lock address 56/0x38 (wrong register - position data)
- ✗ Writing to lock address 69/0x45 (wrong register - lock didn't change)
- ✓ **Writing to lock address 55/0x37 (CORRECT - IDs now persist!)**

**Solution Found:** Lock register is at address **55 (0x37)**, not 69 as initially thought. With this address, the lock flag properly changes (0→1) and motor IDs persist after power cycling.

---

## STS3215 Memory Map (Verified)

| Register | Address (Hex) | Address (Dec) | Function |
|----------|---------------|---------------|----------|
| ID | 0x05 | 5 | Motor ID on bus |
| Lock Flag | **0x37** | **55** | EEPROM write lock (0=unlock, 1=lock) ✓ CORRECT |
| ~~Lock Flag~~ | ~~0x45~~ | ~~69~~ | ~~Wrong register - doesn't work~~ |

**Lock Flag Values:**
- Write `0`: Unlocks EEPROM, allows parameter changes
- Write `1`: Locks EEPROM, saves parameters permanently

**Note:** The lock register at address 55 (0x37) is the correct one for STS3215 servos. Address 69 mentioned in some documentation does not work.

---

## Correct ID Change Procedure ✓

```python
# 1. Unlock EEPROM
write_byte(motor_id, address=55, value=0)

# 2. Write new ID
write_byte(old_id, address=5, value=new_id)

# 3. Lock EEPROM
write_byte(new_id, address=55, value=1)
```

**Result:** Lock register properly changes from 1→0→1, and motor IDs persist after power cycle!

---

## Scripts Created

### Diagnostic Scripts
1. **scan_motors.py** - Scan both ports for all connected motors
2. **deep_scan_motors.py** - Test multiple baudrates to find motors
3. **change_motor_id.py** - Basic ID change (no EEPROM lock)
4. **change_motor_id_permanent.py** - ID change with lock address 56
5. **change_id_debug.py** - Detailed debugging with lock address 69

### Usage
```bash
cd /home/abay/Projects/xle_robot/XLeRobot/software
conda activate lerobot
PYTHONPATH=src python examples/scan_motors.py
```

---

## Lessons Learned

### 1. USB Permissions
- Permissions reset after each power cycle
- Must run: `sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1`
- For permanent fix: Add udev rules and add user to dialout group

### 2. Motor Detection
- **Only 1 motor at a time** for ID configuration
- Multiple motors with same ID = ID conflict (only 1 responds)
- Motor must be powered and connected to be detected

### 3. Port Detection Issues
- **/dev/ttyACM0** had consistent detection problems
- **/dev/ttyACM1** worked reliably
- May indicate hardware issue with one control board

### 4. Lock Register Discovery Process
- Address 48: Incorrect guess
- Address 56 (0x38): Wrong register (position data)
- Address 69 (0x45): Wrong register (found in ST3215 memory map, but didn't work)
- **Address 55 (0x37): CORRECT!** (found via web search for Feetech servo lock register)

### 5. Power Cycling Effects
- With correct lock address (55): IDs persist after power cycle ✓
- Lock register properly changes: 1 (locked) → 0 (unlocked) → 1 (locked)
- EEPROM writes work correctly when using address 55

### 6. Hot-Plugging Damage
- **Critical:** Hot-plugging (connecting/disconnecting while powered) can permanently damage STS3215 servos
- Causes: Voltage transients, ESD damage, incorrect pin sequencing
- **Always power down before connecting/disconnecting** motors
- Motor ID 1 failure likely caused by hot-plugging during initial troubleshooting

---

## Current Status

### Working ✓
- ✓ Motor detection and scanning
- ✓ Permanent ID saves to EEPROM (using address 55)
- ✓ IDs persisting after power cycle
- ✓ Motor communication at 1000000 baud
- ✓ Reading motor positions and status
- ✓ Lock register value changes properly

### Completed ✓
**Head Motors (on /dev/ttyACM1):**
- ✓ TOP head motor configured as ID 7 (persists after power cycle)
- ✓ BOTTOM head motor configured as ID 8 (persists after power cycle)

**Base Wheel Motors (on /dev/ttyACM0):**
- ✓ Base wheel 1 (left) configured as ID 7 (persists after power cycle)
- ✓ Base wheel 2 (back) configured as ID 8 (persists after power cycle)
- ✓ Base wheel 3 (right) configured as ID 9 (persists after power cycle)

**All 5 STS3215 motors successfully configured!**

---

## Technical Notes

### Baudrate
- All motors communicate at **1000000 baud**
- No motors detected at 115200, 57600, or 9600 baud

### Power Requirements
- Motors require **12V 8A** power supply
- Control boards have LED indicators when powered

### Daisy Chain Order (Actual Configuration)
**Configuration:**
- **/dev/ttyACM0**: Left arm (partial) + Head motors
  - Detected: [2, 3, 4, 5, 6, 7, 8]
  - Missing: Motor ID 1 (shoulder_pan)
- **/dev/ttyACM1**: Right arm + Base wheels (complete) ✓
  - Detected: [1, 2, 3, 4, 5, 6, 7, 8, 9]

### Left Arm Manufacturing Defects Identified

**Critical Finding:** Left arm has **reversed daisy-chain wiring** (6→5→4→3→2→1 instead of 1→6)

**Three Distinct Faults:**
1. **Reversed Wiring (Manufacturing Defect):** Chain wired backwards from logical IDs
   - Explains why motors 2-6 work without motor 1
   - Motor 1 is at END of physical chain, not beginning

2. **Motor ID 1 Permanent Failure:**
   - Never detected in any configuration
   - **Likely cause:** Electrical damage from hot-plugging during initial troubleshooting
   - Corroborated by community reports of "burnt" STS3215 servos
   - **Action required:** Motor 1 replacement needed

3. **Claw Assembly Short Circuit:**
   - Claw connected → Motor 6 disappears (bus corrupted)
   - Claw disconnected → Motor 6 appears (bus restored)
   - **Cause:** Short circuit in claw wiring at head of reversed chain
   - **Action required:** Claw inspection/repair or replacement

**Supplier Contact Required:** Reversed wiring and claw short are manufacturing defects requiring replacement/repair.

---

## Resources

### Documentation
- [Waveshare ST3215 Wiki](https://www.waveshare.com/wiki/ST3215_Servo)
- [ST3215 Memory Register Map](https://files.waveshare.com/upload/2/27/ST3215%20memory%20register%20map-EN.xls)
- [Feetech Software Downloads](https://www.feetechrc.com/software.html)
- [XLeRobot Docs](https://xlerobot.readthedocs.io)

### Code References
- xlerobot.py:78-92 - Bus 1 motor definitions (left arm + head)
- xlerobot.py:107-123 - Bus 2 motor definitions (right arm + base)
