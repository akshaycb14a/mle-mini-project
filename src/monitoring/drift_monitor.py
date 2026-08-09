import argparse
import json
import os
import sqlite3
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.inference.predictor import EdgePredictor
from src.preprocessing.pipeline import PreprocessingPipeline
from src.simulator.sensor_simulator import SensorSimulator


BIN_EDGES = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
REFERENCE_OUTPUT = Path("reference_dist.json")
DEFAULT_DB_PATH = Path("logiedge.db")


def confidence_to_bin(confidence):
    confidence = float(np.clip(confidence, 0.0, 1.0))
    if confidence < 0.25:
        return 0
    if confidence < 0.50:
        return 1
    if confidence < 0.75:
        return 2
    return 3


def distribution_from_scores(scores):
    scores = np.asarray(list(scores), dtype=np.float32)
    if scores.size == 0:
        raise ValueError("scores cannot be empty")

    counts = np.zeros(4, dtype=np.float32)
    for score in scores:
        counts[confidence_to_bin(score)] += 1.0
    return counts / counts.sum(), counts


def psi(reference_dist, current_dist, epsilon=1e-6):
    reference_dist = np.asarray(reference_dist, dtype=np.float32)
    current_dist = np.asarray(current_dist, dtype=np.float32)

    reference_dist = np.clip(reference_dist, epsilon, 1.0)
    current_dist = np.clip(current_dist, epsilon, 1.0)
    return float(np.sum((current_dist - reference_dist) * np.log(current_dist / reference_dist)))


def save_reference_distribution(scores, output_path=REFERENCE_OUTPUT):
    dist, counts = distribution_from_scores(scores)
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "sample_count": int(len(scores)),
        "bin_edges": BIN_EDGES.tolist(),
        "bin_labels": [
            "[0,0.25)",
            "[0.25,0.50)",
            "[0.50,0.75)",
            "[0.75,1.0]",
        ],
        "distribution": dist.tolist(),
        "counts": counts.astype(int).tolist(),
    }
    Path(output_path).write_text(json.dumps(payload, indent=2))
    return payload


def load_reference_distribution(path=REFERENCE_OUTPUT):
    return json.loads(Path(path).read_text())


def load_recent_confidences(db_path=DEFAULT_DB_PATH, limit=100):
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT confidence
            FROM inference_results
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()

    scores = [float(row[0]) for row in reversed(rows) if row[0] is not None]
    return scores


def build_clean_reference_scores(sample_target=300):
    simulator = SensorSimulator(anomaly="none")
    pipeline = PreprocessingPipeline()
    predictor = EdgePredictor()
    scores = []

    while len(scores) < sample_target:
        reading = simulator.generate_reading()
        features = pipeline.process(reading)
        if features is None:
            continue
        result = predictor.predict(features)
        scores.append(float(result["confidence"]))

    return scores


@dataclass
class PSIDriftMonitor:
    reference_dist: list
    window_size: int = 100

    def __post_init__(self):
        self.window = deque(maxlen=self.window_size)

    def observe(self, confidence):
        self.window.append(float(confidence))

    def progress(self):
        return len(self.window), self.window.maxlen

    def current_psi(self):
        if len(self.window) < self.window.maxlen:
            return None
        current_dist, _ = distribution_from_scores(self.window)
        return psi(self.reference_dist, current_dist)


def build_parser():
    parser = argparse.ArgumentParser(description="PSI drift monitor")
    parser.add_argument("--reference-path", default=str(REFERENCE_OUTPUT))
    parser.add_argument("--build-reference", action="store_true")
    parser.add_argument("--mqtt", action="store_true", help="Listen to MQTT inference events")
    parser.add_argument("--broker", default=os.getenv("MQTT_BROKER", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument(
        "--inference-topic",
        default=os.getenv("MQTT_INFERENCE_TOPIC", "logibridge/trucks/TRUCK_001/inference"),
    )
    parser.add_argument("--truck-id", default=os.getenv("TRUCK_ID", "TRUCK_001"))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--demo-phases",
        default="none,combined,none",
        help="Comma-separated anomaly phases for demo mode",
    )
    parser.add_argument(
        "--demo-phase-samples",
        default="120,120,120",
        help="Comma-separated sample counts per demo phase",
    )
    parser.add_argument("--demo-report-every", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--check-interval", type=int, default=60)
    return parser


def ensure_reference(path):
    path = Path(path)
    if path.exists():
        try:
            return load_reference_distribution(path)
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    scores = build_clean_reference_scores(300)
    return save_reference_distribution(scores, path)


def run_mqtt_monitor(args, reference_dist):
    monitor = PSIDriftMonitor(reference_dist=reference_dist, window_size=args.window_size)
    backfill_scores = load_recent_confidences(args.db_path, args.window_size)
    for score in backfill_scores:
        monitor.observe(score)

    def on_connect(client, userdata, flags, rc, properties=None):
        print(f"Connected to MQTT broker (rc={rc})")
        client.subscribe(args.inference_topic)

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            confidence = float(payload["confidence"])
            monitor.observe(confidence)
        except Exception as exc:
            print(f"Monitor payload error: {exc}")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker, args.port)
    client.loop_start()

    print(f"Listening on {args.inference_topic}")
    if backfill_scores:
        seen, target = monitor.progress()
        print(f"Backfilled PSI window with {seen}/{target} recent confidence scores")
    try:
        while True:
            time.sleep(args.check_interval)
            current = monitor.current_psi()
            if current is None:
                seen, target = monitor.progress()
                print(f"Current PSI: waiting for {target} confidence scores ({seen}/{target})")
                continue
            print(f"Current PSI: {current:.4f}")
            if current > 0.25:
                print(f"[LOGIBRIDGE DRIFT ALERT] PSI={current:.4f}")
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


def collect_demo_scores(predictor, phases, counts):
    for anomaly, sample_count in zip(phases, counts):
        print(f"Demo phase: {anomaly} ({sample_count} windows)")
        simulator = SensorSimulator(anomaly=anomaly)
        pipeline = PreprocessingPipeline()
        phase_scores = []
        while len(phase_scores) < sample_count:
            reading = simulator.generate_reading()
            features = pipeline.process(reading)
            if features is None:
                continue
            result = predictor.predict(features)
            score = float(result["confidence"])
            phase_scores.append(score)
        yield anomaly, phase_scores


def run_demo_monitor(args, reference_dist):
    phases = [part.strip() for part in args.demo_phases.split(",") if part.strip()]
    counts = [int(part.strip()) for part in args.demo_phase_samples.split(",") if part.strip()]
    if len(phases) != len(counts):
        raise ValueError("--demo-phases and --demo-phase-samples must have the same length")

    predictor = EdgePredictor()
    monitor = PSIDriftMonitor(reference_dist=reference_dist, window_size=args.window_size)

    total_seen = 0
    for anomaly, phase_scores in collect_demo_scores(predictor, phases, counts):
        for score in phase_scores:
            monitor.observe(score)
            total_seen += 1
            if len(monitor.window) < monitor.window.maxlen:
                continue
            if total_seen % args.demo_report_every != 0:
                continue
            current = monitor.current_psi()
            if current is None:
                continue
            print(f"Current PSI: {current:.4f}")
            if current > 0.25:
                print(f"[LOGIBRIDGE DRIFT ALERT] PSI={current:.4f}")
            elif current < 0.10:
                print(f"PSI recovered below 0.10: {current:.4f}")

        current = monitor.current_psi()
        if current is not None:
            print(f"Current PSI: {current:.4f}")
            if current > 0.25:
                print(f"[LOGIBRIDGE DRIFT ALERT] PSI={current:.4f}")
            elif current < 0.10:
                print(f"PSI recovered below 0.10: {current:.4f}")


def main():
    args = build_parser().parse_args()
    if args.build_reference:
        scores = build_clean_reference_scores(300)
        reference_payload = save_reference_distribution(scores, args.reference_path)
        print(f"Reference distribution saved to {args.reference_path}")
        print(json.dumps(reference_payload, indent=2))
        return

    reference_payload = ensure_reference(args.reference_path)
    reference_dist = reference_payload["distribution"]

    if args.demo:
        run_demo_monitor(args, reference_dist)
        return

    if args.mqtt:
        run_mqtt_monitor(args, reference_dist)
        return

    print("Choose either --mqtt for live monitoring or --demo for a fast demo.")


if __name__ == "__main__":
    main()
