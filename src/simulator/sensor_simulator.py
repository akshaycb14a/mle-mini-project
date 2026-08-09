import argparse
import json
import random
import time
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
except Exception:  # pragma: no cover - optional for non-MQTT use
    mqtt = None


class SensorSimulator:
    """
    Simulates refrigerated truck sensor readings.
    """

    VALID_ANOMALIES = {"none", "temp_drift", "vibration", "combined"}

    def __init__(self, truck_id="TRUCK_001", anomaly="none", mode=None):
        self.truck_id = truck_id
        self.anomaly = (anomaly or mode or "none").lower()
        if self.anomaly == "random":
            self.anomaly = "none"
        if self.anomaly not in self.VALID_ANOMALIES:
            raise ValueError(
                "anomaly must be one of: none, temp_drift, vibration, combined"
            )
        self.temperature_drift = 0.0
        self.reading_index = 0

    def _sample_temperature(self):
        baseline = random.gauss(4.0, 0.3)
        if self.anomaly in {"temp_drift", "combined"}:
            value = baseline + self.temperature_drift
            self.temperature_drift += 0.08
            return value
        return baseline

    def _sample_vibration(self):
        if self.anomaly in {"vibration", "combined"}:
            return random.gauss(1.2, 0.15)
        return random.gauss(0.45, 0.05)

    def _sample_door_event(self):
        open_probability = 0.05
        if self.anomaly == "temp_drift":
            open_probability = 0.08
        elif self.anomaly == "vibration":
            open_probability = 0.10
        elif self.anomaly == "combined":
            open_probability = 0.15

        door_open = random.random() < open_probability
        return door_open, "OPEN" if door_open else "CLOSE"

    def generate_reading(self):
        temperature = round(self._sample_temperature(), 2)
        vibration = round(self._sample_vibration(), 2)
        door_open, door_event = self._sample_door_event()
        status = {
            "none": "normal",
            "temp_drift": "warning",
            "vibration": "warning",
            "combined": "critical",
        }[self.anomaly]

        return {
            "timestamp": datetime.now().isoformat(),
            "truck_id": self.truck_id,
            "temperature": temperature,
            "vibration": vibration,
            "door_event": door_event,
            "door_open": door_open,
            "status": status,
            "anomaly": self.anomaly,
        }


def build_parser():
    parser = argparse.ArgumentParser(description="Sensor simulator and MQTT publisher")
    parser.add_argument(
        "--anomaly",
        choices=sorted(SensorSimulator.VALID_ANOMALIES),
        default="none",
        help="Sensor mode to simulate",
    )
    parser.add_argument("--truck-id", default="TRUCK_001")
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of readings to publish. 0 means keep running.",
    )
    return parser


def run_mqtt_stream(args):
    if mqtt is None:
        raise RuntimeError("paho-mqtt is required to publish to MQTT")

    simulator = SensorSimulator(truck_id=args.truck_id, anomaly=args.anomaly)
    topic = args.topic or f"logibridge/trucks/{args.truck_id}/sensors"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(args.broker, args.port)
    client.loop_start()

    print(
        f"Publishing anomaly={args.anomaly} to mqtt://{args.broker}:{args.port}/{topic}"
    )

    try:
        sent = 0
        while True:
            reading = simulator.generate_reading()
            client.publish(topic, json.dumps(reading)).wait_for_publish()
            print(reading)
            sent += 1
            if args.count and sent >= args.count:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


def main():
    args = build_parser().parse_args()
    run_mqtt_stream(args)


if __name__ == "__main__":
    main()
