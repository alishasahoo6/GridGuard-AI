"""Virtual smart-meter telemetry and anomaly scenario simulator.

Planned responsibilities:
- Generation of synthetic multi-feeder smart meter consumption streams.
- Injection of parameterized anomaly profiles (meter bypass, load reduction,
  neutral disconnect, reverse feed, intermittent connection loss).
- Continuous real-time streaming harness (via SQLite / MQTT / in-memory queues).
"""

