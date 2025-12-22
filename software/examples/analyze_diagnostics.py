#!/usr/bin/env python3
"""
Analyze diagnostic logs to identify lockup causes
"""

import json
import sys


def analyze_log(log_file):
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC LOG ANALYSIS: {log_file}")
    print(f"{'='*70}\n")

    samples = []
    comm_errors = []

    with open(log_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            try:
                timestamp, event_type, data_str = line.strip().split(',', 2)
                data = json.loads(data_str)

                if event_type == "SAMPLE":
                    samples.append((float(timestamp), data))
                elif event_type == "COMM_ERROR":
                    comm_errors.append(float(timestamp))
            except:
                continue

    if not samples:
        print("⚠️  No samples found in log")
        return

    print(f"📊 Session Statistics:")
    print(f"   Total samples: {len(samples)}")
    print(f"   Communication errors: {len(comm_errors)}")
    print(f"   Duration: {samples[-1][0] - samples[0][0]:.1f} seconds\n")

    # Analyze FPS stability
    print("🎯 Control Loop Performance:")
    fps_values = [s[1]['loop_stats']['avg_fps'] for s in samples if 'loop_stats' in s[1]]
    if fps_values:
        print(f"   Average FPS: {sum(fps_values)/len(fps_values):.1f}")
        print(f"   Min FPS: {min(fps_values):.1f}")
        print(f"   Max FPS: {max(fps_values):.1f}")
        if min(fps_values) < 20:
            print(f"   ⚠️  WARNING: FPS dropped below 20 - control loop slowdown detected\n")
        else:
            print()

    # Analyze temperature trends
    print("🌡️  Temperature Analysis:")
    max_temps = {}
    for timestamp, data in samples:
        for motor, status in data.get('motors', {}).items():
            temp = status.get('temp', 0)
            if motor not in max_temps or temp > max_temps[motor]:
                max_temps[motor] = temp

    for motor, temp in sorted(max_temps.items()):
        status = "🔥 CRITICAL" if temp > 70 else "⚠️  HIGH" if temp > 60 else "✓ OK"
        print(f"   {motor}: {temp}°C - {status}")

    if max(max_temps.values()) > 70:
        print(f"\n   🔥 THERMAL SHUTDOWN LIKELY - Motors exceeded 70°C")
    elif max(max_temps.values()) > 60:
        print(f"\n   ⚠️  OVERHEATING RISK - Motors approaching thermal limit")
    print()

    # Analyze torque disable events
    print("⚙️  Torque Disable Events:")
    torque_disabled = []
    for timestamp, data in samples:
        for motor, status in data.get('motors', {}).items():
            if status.get('torque', 1) == 0:
                torque_disabled.append((timestamp, motor))

    if torque_disabled:
        print(f"   Found {len(torque_disabled)} torque disable events:")
        for ts, motor in torque_disabled[-5:]:  # Show last 5
            print(f"   - {motor} at t={ts:.1f}s")
        print(f"\n   🚨 MOTORS DISABLED - Likely thermal protection or error state")
    else:
        print(f"   ✓ No torque disable events detected")
    print()

    # Analyze position errors
    print("📏 Position Error Trends:")
    for arm in ['left_arm', 'right_arm']:
        errors = [s[1]['error_stats'].get(arm, {}).get('max', 0)
                  for s in samples if 'error_stats' in s[1]]
        if errors:
            avg_error = sum(errors) / len(errors)
            max_error = max(errors)
            print(f"   {arm}: avg={avg_error:.1f}°, max={max_error:.1f}°", end="")
            if max_error > 50:
                print(" - ⚠️  EXCESSIVE (possible oscillation/fighting)")
            elif max_error > 30:
                print(" - ⚠️  HIGH")
            else:
                print(" - ✓ OK")

    print(f"\n{'='*70}")
    print("CONCLUSION:")
    print(f"{'='*70}")

    # Determine root cause
    if torque_disabled:
        print("🚨 ROOT CAUSE: Motors disabled torque (thermal protection or error)")
    elif max(max_temps.values()) > 65:
        print("🔥 ROOT CAUSE: Thermal shutdown (motors overheating)")
    elif len(comm_errors) > 10:
        print("📡 ROOT CAUSE: Communication errors (bus overload or interference)")
    elif min(fps_values) < 15:
        print("⏱️  ROOT CAUSE: Control loop slowdown (CPU/timing issue)")
    else:
        print("✓ No obvious root cause detected - may need longer test")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "xlerobot_diagnostics.log"
    try:
        analyze_log(log_file)
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found")
        print("Usage: python analyze_diagnostics.py [log_file]")
