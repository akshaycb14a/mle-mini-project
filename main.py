from data.database.sqlite_manager import SQLiteManager
from data.database.repository import Repository
from src.alerts.alert_engine import AlertEngine
from src.logging.event_logger import (
    log_telemetry,
    log_inference,
    log_alert,
)

def main():
    db = SQLiteManager()
    db.connect()
    repo = Repository(db)
    engine = AlertEngine()

    # Example data (replace with MQTT + TFLite pipeline)
    device_id = "sensor-001"
    temperature = 9.2
    vibration = 1.15
    prediction = "Normal"
    confidence = 0.91
    anomaly_score = 0.87

    repo.insert_telemetry(device_id, temperature, vibration, anomaly_score)
    repo.insert_inference(prediction, confidence)

    log_telemetry(device_id, temperature, vibration)
    log_inference(prediction, confidence)

    alerts = engine.check(temperature, vibration, anomaly_score)
    for alert in alerts:
        repo.insert_alert("WARNING", alert["message"])
        log_alert("WARNING", alert["message"])
        print(alert)

    db.close()

if __name__ == "__main__":
    main()
