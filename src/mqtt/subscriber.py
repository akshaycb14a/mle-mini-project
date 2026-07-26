import os
import json
import paho.mqtt.client as mqtt

from data.database.sqlite_manager import SQLiteManager
from data.database.repository import Repository

from src.preprocessing.pipeline import PreprocessingPipeline
from src.inference.predictor import EdgePredictor
from src.alerts.alert_engine import AlertEngine
from src.logging.event_logger import (
    log_telemetry,
    log_inference,
    log_alert,
)
from src.monitoring.health_monitor import HealthMonitor


BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = 1883
TOPIC = "logiedge/truck001/sensors"

pipeline = PreprocessingPipeline()

predictor = EdgePredictor()

alert_engine = AlertEngine()

health_monitor = HealthMonitor()

db = SQLiteManager()
db.connect()

repo = Repository(db)


def on_connect(client, userdata, flags, rc, properties=None):
    print(f"Connected to MQTT broker (rc={rc})")
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    print("Topic:", msg.topic, flush=True)
    print("Payload:", msg.payload, flush=True)

    try:
        data = json.loads(msg.payload.decode())

        print("=" * 60)
        print("Received Sensor Reading")
        print(data)

        features = pipeline.process(data)

        if features is None:
            return

        result = predictor.predict(features)

        prediction = result["prediction"]
        confidence = float(result["confidence"])
        anomaly_score = 1.0 - confidence if prediction != "normal" else 0.0

        print("\nEDGE AI Prediction")
        print("-" * 40)
        print(f"Prediction : {prediction}")
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"Probabilities: {result['probabilities']}")
        print("-" * 40)

        repo.insert_telemetry(
            data["truck_id"],
            data["temperature"],
            data.get("vibration", 0.0),
            anomaly_score,
        )

        repo.insert_inference(prediction, confidence)

        log_telemetry(
            data["truck_id"],
            data["temperature"],
            data.get("vibration", 0.0),
        )
        log_inference(prediction, confidence)

        alerts = alert_engine.check(
            data["temperature"],
            data.get("vibration", 0.0),
            anomaly_score,
        )

        for alert in alerts:
            repo.insert_alert("WARNING", alert["message"])
            log_alert("WARNING", alert["message"])
            print("ALERT:", alert["message"])

        try:
            print("Health:", health_monitor.get_health())
        except Exception:
            pass

    except Exception as exc:
        print(f"Processing error: {exc}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)


client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT)

client.loop_forever()
