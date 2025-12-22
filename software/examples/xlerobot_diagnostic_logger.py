#!/usr/bin/env python3
"""
Diagnostic logger for XLeRobot to identify lockup causes
"""

import time
import json
from collections import deque
import scservo_sdk


class DiagnosticLogger:
    def __init__(self, log_file="xlerobot_diagnostics.log", sample_interval=2.0):
        self.log_file = log_file
        self.sample_interval = sample_interval
        self.last_sample_time = time.time()

        # Circular buffers for aggregation
        self.loop_times = deque(maxlen=100)
        self.position_errors = {
            'left_arm': deque(maxlen=100),
            'right_arm': deque(maxlen=100),
        }

        # Critical event counters
        self.comm_errors = 0
        self.total_loops = 0

        # Motor bus handlers for direct queries
        self.port_handlers = {}
        self.packet_handler = scservo_sdk.PacketHandler(protocol_end=0)

        # Open log file
        self.f = open(log_file, 'w')
        self.f.write("timestamp,event_type,data\n")
        self.f.flush()

        print(f"[DIAG] Logging to {log_file}")

    def init_motor_bus(self, port1, port2):
        """Initialize direct motor communication for diagnostics"""
        for name, port in [('port1', port1), ('port2', port2)]:
            ph = scservo_sdk.PortHandler(port)
            if ph.openPort():
                ph.setBaudRate(1000000)
                self.port_handlers[name] = {'handler': ph, 'port': port}
                print(f"[DIAG] Initialized {name} ({port})")

    def record_loop_time(self, dt):
        """Record control loop timing"""
        self.loop_times.append(dt)
        self.total_loops += 1

    def record_position_error(self, arm, error):
        """Record position error magnitude"""
        self.position_errors[arm].append(error)

    def record_comm_error(self):
        """Record communication error"""
        self.comm_errors += 1
        self._log_event("COMM_ERROR", {"total": self.comm_errors})

    def should_sample(self):
        """Check if it's time to sample motor status"""
        return (time.time() - self.last_sample_time) >= self.sample_interval

    def sample_motor_status(self):
        """Sample motor temperatures and torque status"""
        if not self.should_sample():
            return

        self.last_sample_time = time.time()

        data = {
            'loop_stats': self._get_loop_stats(),
            'error_stats': self._get_error_stats(),
            'motors': {}
        }

        # Sample critical motors (avoid checking all 17 to reduce overhead)
        critical_motors = [
            ('port1', [1, 2, 3]),  # Left arm shoulder/elbow
            ('port2', [1, 2, 3]),  # Right arm shoulder/elbow
        ]

        for port_name, motor_ids in critical_motors:
            if port_name not in self.port_handlers:
                continue

            ph = self.port_handlers[port_name]['handler']
            for motor_id in motor_ids:
                # Read temperature (addr 63)
                temp, result, _ = self.packet_handler.read1ByteTxRx(ph, motor_id, 63)
                # Read torque enable (addr 40)
                torque, result2, _ = self.packet_handler.read1ByteTxRx(ph, motor_id, 40)

                if result == scservo_sdk.COMM_SUCCESS:
                    key = f"{port_name}_m{motor_id}"
                    data['motors'][key] = {
                        'temp': temp,
                        'torque': torque
                    }

        self._log_event("SAMPLE", data)

    def _get_loop_stats(self):
        """Get loop timing statistics"""
        if not self.loop_times:
            return {}

        times = list(self.loop_times)
        return {
            'avg_fps': 1.0 / (sum(times) / len(times)) if times else 0,
            'min_fps': 1.0 / max(times) if times else 0,
            'max_fps': 1.0 / min(times) if times else 0,
        }

    def _get_error_stats(self):
        """Get position error statistics"""
        stats = {}
        for arm, errors in self.position_errors.items():
            if errors:
                stats[arm] = {
                    'avg': sum(errors) / len(errors),
                    'max': max(errors),
                }
        return stats

    def _log_event(self, event_type, data):
        """Write event to log file"""
        entry = {
            'timestamp': time.time(),
            'event': event_type,
            'data': data
        }
        self.f.write(f"{time.time()},{event_type},{json.dumps(data)}\n")
        self.f.flush()

        # Print critical events to console
        if event_type == "COMM_ERROR":
            print(f"[DIAG] Communication error #{data['total']}")
        elif event_type == "SAMPLE":
            # Print temperature warnings
            for motor, status in data.get('motors', {}).items():
                temp = status.get('temp', 0)
                torque = status.get('torque', 1)
                if temp > 60:
                    print(f"[DIAG] ⚠️  {motor} temp: {temp}°C")
                if torque == 0:
                    print(f"[DIAG] ⚠️  {motor} torque disabled!")

    def close(self):
        """Close logger and cleanup"""
        # Write final summary
        summary = {
            'total_loops': self.total_loops,
            'total_comm_errors': self.comm_errors,
            'error_rate': self.comm_errors / self.total_loops if self.total_loops else 0
        }
        self._log_event("SUMMARY", summary)

        print(f"[DIAG] Session summary: {summary}")

        self.f.close()
        for port_data in self.port_handlers.values():
            port_data['handler'].closePort()
