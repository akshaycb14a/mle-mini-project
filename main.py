from database.sqlite_manager import SQLiteManager
from database.repository import Repository
from alerts.alert_engine import AlertEngine
from logging.event_logger import (
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
    humidity = 83.5
    prediction = "Normal"
    confidence = 0.91
    anomaly_score = 0.87

    repo.insert_telemetry(device_id, temperature, humidity, anomaly_score)
    repo.insert_inference(prediction, confidence)

    log_telemetry(device_id, temperature, humidity)
    log_inference(prediction, confidence)

    alerts = engine.check(temperature, humidity, anomaly_score)
    for alert in alerts:
        repo.insert_alert("WARNING", alert["message"])
        log_alert("WARNING", alert["message"])
        print(alert)

    db.close()

if __name__ == "__main__":
    main()
