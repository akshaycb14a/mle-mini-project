from datetime import datetime

class AlertEngine:
    def __init__(self,
                 temp_min=2.0,
                 temp_max=8.0,
                 humidity_max=80.0,
                 anomaly_threshold=0.80):
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.humidity_max = humidity_max
        self.anomaly_threshold = anomaly_threshold

    def check(self, temperature, humidity, anomaly_score):
        alerts = []

        if temperature < self.temp_min:
            alerts.append(self._alert(
                "LOW_TEMPERATURE",
                f"Temperature below limit: {temperature:.2f}°C"
            ))

        if temperature > self.temp_max:
            alerts.append(self._alert(
                "HIGH_TEMPERATURE",
                f"Temperature above limit: {temperature:.2f}°C"
            ))

        if humidity > self.humidity_max:
            alerts.append(self._alert(
                "HIGH_HUMIDITY",
                f"Humidity above limit: {humidity:.2f}%"
            ))

        if anomaly_score >= self.anomaly_threshold:
            alerts.append(self._alert(
                "ANOMALY_DETECTED",
                f"Anomaly score: {anomaly_score:.3f}"
            ))

        return alerts

    @staticmethod
    def _alert(alert_type, message):
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "type": alert_type,
            "message": message,
        }


if __name__ == "__main__":
    engine = AlertEngine()
    sample = engine.check(
        temperature=10.5,
        humidity=85.0,
        anomaly_score=0.91
    )

    for alert in sample:
        print(alert)
