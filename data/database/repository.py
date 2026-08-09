from datetime import datetime

class Repository:
    def __init__(self, db):
        self.db = db

    def insert_telemetry(self, device_id, temperature, vibration, anomaly_score):
        self.db.execute(
            "INSERT INTO telemetry(timestamp,device_id,temperature,vibration,anomaly_score) VALUES(?,?,?,?,?)",
            (datetime.utcnow().isoformat(), device_id, temperature, vibration, anomaly_score),
        )

    def insert_inference(self, prediction, confidence):
        self.db.execute(
            "INSERT INTO inference_results(timestamp,prediction,confidence) VALUES(?,?,?)",
            (datetime.utcnow().isoformat(), prediction, confidence),
        )

    def insert_alert(self, severity, message):
        self.db.execute(
            "INSERT INTO alerts(timestamp,severity,message) VALUES(?,?,?)",
            (datetime.utcnow().isoformat(), severity, message),
        )

    def get_latest_telemetry(self):
        return self.db.execute(
            "SELECT * FROM telemetry ORDER BY id DESC LIMIT 1"
        ).fetchone()

    def get_alerts(self):
        return self.db.execute(
            "SELECT * FROM alerts ORDER BY id DESC"
        ).fetchall()
